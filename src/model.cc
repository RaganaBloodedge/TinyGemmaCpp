#include "model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#define TG_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#define TG_THREAD_ID omp_get_thread_num()
#define TG_MAX_THREADS omp_get_max_threads()
#else
#define TG_OMP_PARALLEL_FOR
#define TG_THREAD_ID 0
#define TG_MAX_THREADS 1
#endif

namespace tg {
namespace {

constexpr int64_t kWarmupPositions = 512;

// ---- 分阶段计时 ----
// 计时器本身是一次 steady_clock::now()（约 20 ns），只有在 prof.on 时才执行。
// 注意：所有 timer 都在并行区**外侧**构造/析构，累加是单线程的，不需要锁。
double ProfNow() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count() * 1e3;
}

struct ScopedTimer {
  double* acc;
  double t0;
  bool active;
  ScopedTimer(double* a, bool on) : acc(a), t0(on ? ProfNow() : 0.0), active(on) {}
  ~ScopedTimer() {
    if (active) *acc += ProfNow() - t0;
  }
};

// 缓存里的 K/V 是连续 float，直接手写点积，比走 Matrix 少一层类型分支。
inline float DotF32Vec(const float* a, const float* b, int64_t n) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i + 0] * b[i + 0];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

// 写 .npy（version 1.0，小端 f32，C 序），Python 侧 np.load 直接读。
bool WriteNpy(const std::string& path, const float* data, int64_t rows,
              int64_t cols) {
  char header[256];
  int len = std::snprintf(header, sizeof(header),
                          "{'descr': '<f4', 'fortran_order': False, 'shape': "
                          "(%lld, %lld), }",
                          static_cast<long long>(rows),
                          static_cast<long long>(cols));
  if (len <= 0) return false;
  const int prefix = 10;  // magic(6) + version(2) + header_len(2)
  int total = prefix + len + 1;  // +1 是结尾换行
  int pad = (64 - (total % 64)) % 64;
  for (int i = 0; i < pad; ++i) header[len + i] = ' ';
  header[len + pad] = '\n';
  const uint16_t hlen = static_cast<uint16_t>(len + pad + 1);

  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return false;
  const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  std::fwrite(magic, 1, 6, f);
  const uint8_t ver[2] = {1, 0};
  std::fwrite(ver, 1, 2, f);
  std::fwrite(&hlen, 2, 1, f);
  std::fwrite(header, 1, hlen, f);
  std::fwrite(data, sizeof(float), static_cast<size_t>(rows * cols), f);
  std::fclose(f);
  return true;
}

}  // namespace

Model::~Model() = default;

bool Model::Load(const std::string& sbs_path, const ModelConfig& config,
                 std::string* err) {
  config_ = config;
  file_ = std::make_unique<SbsFile>();
  if (!file_->Open(sbs_path, err)) return false;
  if (!weights_.Build(*file_, config_, err)) return false;

  // RoPE 表：先预热 512 个位置，之后按需扩。
  rope_.Init(MakeInvTimescale(config_.layer.qkv_dim, 10000.0),
             kWarmupPositions);
  return true;
}

namespace {
void ResizeIfSmaller(std::vector<float>* v, size_t n) {
  if (v->size() < n) v->resize(n);
}
}  // namespace

void Model::Forward(const std::vector<int32_t>& tokens, int64_t start_pos,
                    KVCache& kv, std::vector<float>* logits) {
  const LayerConfig& lc = config_.layer;
  const int64_t n = static_cast<int64_t>(tokens.size());
  const int64_t D = lc.model_dim;
  const int64_t qkv_w = lc.QRows() + lc.KvRows();
  // 一次吃多个 token 的调用是 prefill，单 token 的是 decode，分开记账。
  PhaseProfile& pf = (n > 1) ? prof_prefill_ : prof_decode_;
  const bool prof = pf.on;
  const double t_begin = prof ? ProfNow() : 0.0;
  pf.tokens += n;
  ++pf.calls;

  ResizeIfSmaller(&x_, static_cast<size_t>(n * D));
  ResizeIfSmaller(&normed_, static_cast<size_t>(n * D));
  ResizeIfSmaller(&qkv_, static_cast<size_t>(n * qkv_w));
  ResizeIfSmaller(&att_out_, static_cast<size_t>(n * lc.QRows()));
  ResizeIfSmaller(&proj_, static_cast<size_t>(n * D));
  ResizeIfSmaller(&c1_, static_cast<size_t>(n * lc.ff_hidden_dim));
  ResizeIfSmaller(&c2_, static_cast<size_t>(n * lc.ff_hidden_dim));
  ResizeIfSmaller(&logits_, static_cast<size_t>(config_.vocab_size));
  // 注意力打分临时区：每个线程一份，避免并行时竞争。
  ResizeIfSmaller(&scores_, static_cast<size_t>(std::max<int64_t>(1, kv.seq())) *
                                static_cast<size_t>(TG_MAX_THREADS));

  // ---- 词嵌入（并乘 sqrt(model_dim)）----
  {
    ScopedTimer tm(&pf.embed_ms, prof);
    const float emb_scale = config_.EmbeddingScale();
    for (int64_t t = 0; t < n; ++t) {
      const int32_t tok = tokens[static_cast<size_t>(t)];
      float* row = x_.data() + t * D;
      for (int64_t d = 0; d < D; ++d) {
        row[d] = weights_.embed.At(tok, d) * emb_scale;
      }
    }
  }
  if (!dump_dir_.empty() && dump_once_) Dump("00_embed", x_.data(), n, D);

  // ---- 26 层 ----
  for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
    ForwardLayer(layer, n, start_pos, kv, pf);
  }

  {
    ScopedTimer tm(&pf.logits_ms, prof);
    // ---- final norm 只对最后一个 token 做 ----
    const float* last = x_.data() + (n - 1) * D;
    std::vector<float> norm_last(static_cast<size_t>(D));
    RMSNorm(last, D, weights_.final_norm.data(), config_.rms_eps, norm_last.data());
    if (!dump_dir_.empty() && dump_once_) Dump("90_final_norm", norm_last.data(), 1, D);

    MatMulBT(logits_.data(), norm_last.data(), 1, D, weights_.embed);
    const float final_cap = 30.0f;
    SoftCapInplace(logits_.data(), config_.vocab_size, final_cap);
  }
  if (!dump_dir_.empty() && dump_once_) {
    Dump("91_logits", logits_.data(), 1, config_.vocab_size);
    dump_once_ = false;
  }

  logits->assign(logits_.begin(),
                 logits_.begin() + static_cast<size_t>(config_.vocab_size));
  if (prof) pf.total_ms += ProfNow() - t_begin;
}

void Model::ForwardLayer(int64_t layer, int64_t n, int64_t start_pos,
                         KVCache& kv, PhaseProfile& pf) {
  const LayerConfig& lc = config_.layer;
  const LayerWeights& lw = weights_.layers[static_cast<size_t>(layer)];
  const int64_t D = lc.model_dim;
  const int64_t QD = lc.qkv_dim;
  const int64_t H = lc.heads;
  const int64_t KH = lc.kv_heads;
  const int64_t G = lc.QHeadGroup();
  const int64_t QR = lc.QRows();
  const int64_t KVR = lc.KvRows();
  const float eps = config_.rms_eps;

  [[maybe_unused]] std::string tag = "L" + std::to_string(layer) + "_";
  const bool prof = pf.on;

  // ---------- Attention ----------
  // 1) 注意力前的 RMSNorm
  {
    ScopedTimer tm(&pf.norm_ms, prof);
    for (int64_t t = 0; t < n; ++t) {
      RMSNorm(x_.data() + t * D, D, lw.pre_att_norm.data(), eps,
              normed_.data() + t * D);
    }
  }
  if (!dump_dir_.empty() && dump_once_ && layer == 0)
    Dump(tag + "pre_att_norm", normed_.data(), n, D);

  // 2) qkv 投影：[n, 4096]
  {
    ScopedTimer tm(&pf.qkv_ms, prof);
    MatMulBT(qkv_.data(), normed_.data(), n, D, lw.qkv);
  }

  // 3) 写 KV Cache，并给 q/k 加 RoPE。
  //    qkv 行布局： [h0..h7 各 256] <- q；随后 (k0,v0,k1,v1,k2,v2,k3,v3) 各 256。
  {
    ScopedTimer tm(&pf.rope_ms, prof);
    const float qscale = config_.QueryScale();
    for (int64_t t = 0; t < n; ++t) {
      const int64_t pos = start_pos + t;
      rope_.Ensure(pos);
      float* row = qkv_.data() + t * (QR + KVR);
      for (int64_t h = 0; h < H; ++h) {
        RopeInplace(row + h * QD, QD, rope_, pos, qscale);
      }
      float* kvrow = row + QR;
      float* kbase = kv.KBase(layer, pos);
      float* vbase = kv.VBase(layer, pos);
      for (int64_t h = 0; h < KH; ++h) {
        float* kd = kbase + h * QD;
        float* vd = vbase + h * QD;
        std::memcpy(kd, kvrow + h * 2 * QD, static_cast<size_t>(QD) * sizeof(float));
        std::memcpy(vd, kvrow + h * 2 * QD + QD, static_cast<size_t>(QD) * sizeof(float));
        RopeInplace(kd, QD, rope_, pos, 1.0f);
      }
    }
  }

  // 4) Q·K -> soft-cap -> softmax -> 加权求和 V
  const int64_t window = config_.WindowOf(layer);
  const int64_t seq_cap = kv.seq();
  const int64_t score_stride = std::max<int64_t>(1, seq_cap);

  {
    ScopedTimer tm(&pf.attn_ms, prof);
    TG_OMP_PARALLEL_FOR
    for (int64_t task = 0; task < n * H; ++task) {
      const int64_t t = task / H;
      const int64_t h = task - t * H;
      const int64_t pos = start_pos + t;
      const int64_t start = pos - std::min(window - 1, pos);
      const int64_t len = pos - start + 1;
      const int64_t kvh = h / G;

      float* sc = scores_.data() + static_cast<size_t>(TG_THREAD_ID) * score_stride;
      const float* qh = qkv_.data() + t * (QR + KVR) + h * QD;
      for (int64_t p = start; p <= pos; ++p) {
        sc[p - start] = DotF32Vec(qh, kv.KHead(layer, p, kvh), QD);
      }
      SoftCapInplace(sc, len, 50.0f);  // att_cap = 50.0，在 softmax 之前
      SoftmaxInplace(sc, len);

      float* out = att_out_.data() + t * QR + h * QD;
      std::memset(out, 0, static_cast<size_t>(QD) * sizeof(float));
      for (int64_t p = start; p <= pos; ++p) {
        const float w = sc[p - start];
        const float* vh = kv.VHead(layer, p, kvh);
        for (int64_t d = 0; d < QD; ++d) out[d] += w * vh[d];
      }
    }
  }

  // 5) 各头输出拼起来过 o_proj。
  //    att_ein 在文件里是 [head][model_dim][qkv_dim]，即行 = h*model_dim + m，
  //    列 = d。所以 out[t][m] = Σ_h Σ_d att_out[t][h*qkv_dim + d] * att[h*model_dim+m][d]。
  //
  //    注意朴素写法（固定 m、遍历 h）会让相邻 head 之间跳 model_dim*qkv_dim
  //    字节，实测只跑到 2.4 GB/s、占 decode 的 42%。AttEinSum 按 model_dim 分块
  //    重排了遍历顺序，让每次读到的权重都是连续段。
  {
    ScopedTimer tm(&pf.att_ein_ms, prof);
    AttEinSum(proj_.data(), att_out_.data(), n, QR, H, QD, D, lw.att);
  }
  if (!dump_dir_.empty() && dump_once_ && layer == 0)
    Dump(tag + "att_sums_raw", proj_.data(), n, D);

  // 6) PostNorm：先 norm 分支输出，再加回主干（顺序不能反）
  {
    ScopedTimer tm(&pf.norm_ms, prof);
    for (int64_t t = 0; t < n; ++t) {
      float* p = proj_.data() + t * D;
      RMSNorm(p, D, lw.post_att_norm.data(), eps, p);
      float* xx = x_.data() + t * D;
      for (int64_t d = 0; d < D; ++d) xx[d] += p[d];
    }
  }
  if (!dump_dir_.empty() && dump_once_ && layer == 0)
    Dump(tag + "x_after_attn", x_.data(), n, D);

  // ---------- FFN (GeGLU) ----------
  const int64_t F = lc.ff_hidden_dim;
  {
    ScopedTimer tm(&pf.norm_ms, prof);
    for (int64_t t = 0; t < n; ++t) {
      RMSNorm(x_.data() + t * D, D, lw.pre_ff_norm.data(), eps,
              normed_.data() + t * D);
    }
  }
  // gating_ein 上半是 gate，下半是 up；这里按行切片，不复制权重。
  {
    ScopedTimer tm(&pf.ff_up_ms, prof);
    MatMulBTRows(c1_.data(), normed_.data(), n, D, lw.gating, 0, F);
    MatMulBTRows(c2_.data(), normed_.data(), n, D, lw.gating, F, 2 * F);
  }
  {
    ScopedTimer tm(&pf.gelu_ms, prof);
    for (int64_t t = 0; t < n; ++t) {
      GeluMulInplace(c1_.data() + t * F, c2_.data() + t * F, F);
    }
  }
  {
    ScopedTimer tm(&pf.ff_down_ms, prof);
    MatMulBT(proj_.data(), c1_.data(), n, F, lw.linear);
  }

  {
    ScopedTimer tm(&pf.norm_ms, prof);
    for (int64_t t = 0; t < n; ++t) {
      float* p = proj_.data() + t * D;
      RMSNorm(p, D, lw.post_ff_norm.data(), eps, p);
      float* xx = x_.data() + t * D;
      for (int64_t d = 0; d < D; ++d) xx[d] += p[d];
    }
  }
  if (!dump_dir_.empty() && dump_once_ && layer == 0)
    Dump(tag + "x_out", x_.data(), n, D);
}

void PhaseProfile::Report(const char* title) const {
  if (!on || tokens == 0 || calls == 0) return;
  const double per = 1.0 / static_cast<double>(tokens);
  const double counted = (embed_ms + norm_ms + qkv_ms + rope_ms + attn_ms +
                          att_ein_ms + ff_up_ms + gelu_ms + ff_down_ms +
                          logits_ms) * per;
  const double total = total_ms * per;
  auto row = [&](const char* name, double ms) {
    const double v = ms * per;
    std::fprintf(stderr, "  %-14s %8.2f ms  %5.1f%%\n", name, v,
                 total > 0 ? 100.0 * v / total : 0.0);
  };
  std::fprintf(stderr,
               "\n[profile] %s 每 token 平均 %.2f ms"
               "（%lld 次 Forward / %lld token，这段合计 %.1f ms）\n",
               title, total, static_cast<long long>(calls),
               static_cast<long long>(tokens), total_ms);
  row("embed", embed_ms);
  row("RMSNorm", norm_ms);
  row("qkv", qkv_ms);
  row("rope+kv", rope_ms);
  row("attention", attn_ms);
  row("att_ein", att_ein_ms);
  row("ff gate/up", ff_up_ms);
  row("geglu", gelu_ms);
  row("ff down", ff_down_ms);
  row("logits", logits_ms);
  row("其余/调度", total - counted);
}

void Model::Dump(const std::string& name, const float* data, int64_t rows,
                 int64_t cols) const {
  if (dump_dir_.empty()) return;
  const std::string path = dump_dir_ + "/" + name + ".npy";
  WriteNpy(path, data, rows, cols);
}

// ---------------- 采样 ----------------

int32_t SampleGreedy(const float* logits, int64_t n) {
  int32_t best = 0;
  float bv = logits[0];
  for (int64_t i = 1; i < n; ++i) {
    if (logits[i] > bv) {
      bv = logits[i];
      best = static_cast<int32_t>(i);
    }
  }
  return best;
}

int32_t SampleTopK(const float* logits, int64_t n, float temperature,
                   int64_t top_k, uint64_t* rng_state) {
  if (temperature <= 0.0f || top_k <= 1) return SampleGreedy(logits, n);
  const float inv_t = 1.0f / temperature;

  std::vector<float> tmp(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) tmp[static_cast<size_t>(i)] = logits[i] * inv_t;
  float max_v = tmp[0];
  for (int64_t i = 1; i < n; ++i) max_v = std::max(max_v, tmp[static_cast<size_t>(i)]);
  for (int64_t i = 0; i < n; ++i) {
    tmp[static_cast<size_t>(i)] = std::exp(tmp[static_cast<size_t>(i)] - max_v);
  }

  const int64_t k = std::min<int64_t>(top_k, n);
  std::vector<int32_t> idx(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::partial_sort(
      idx.begin(), idx.begin() + static_cast<size_t>(k), idx.end(),
      [&](int32_t a, int32_t b) {
        return tmp[static_cast<size_t>(a)] > tmp[static_cast<size_t>(b)];
      });

  float acc = 0.f;
  for (int64_t i = 0; i < k; ++i) acc += tmp[static_cast<size_t>(idx[static_cast<size_t>(i)])];

  // xorshift64*：不引第三方 RNG，结果可复现。
  uint64_t s = *rng_state ? *rng_state : 0x9E3779B97F4A7C15ull;
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  *rng_state = s;
  const double u = static_cast<double>(s * 0x2545F4914F6CDD1Dull >> 11) /
                   static_cast<double>(1ull << 53) * static_cast<double>(acc);

  double cdf = 0.0;
  for (int64_t i = 0; i < k; ++i) {
    const int32_t tok = idx[static_cast<size_t>(i)];
    cdf += static_cast<double>(tmp[static_cast<size_t>(tok)]);
    if (u < cdf) return tok;
  }
  return idx[static_cast<size_t>(k - 1)];
}

}  // namespace tg
