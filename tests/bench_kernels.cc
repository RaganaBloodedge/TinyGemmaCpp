// TinyGemmaCpp —— GEMM 内核微基准
//
// 不加载 3.2 GB 真权重，用合成数据把 decode/batch=1 的每个投影单独计时。
// 目的是把「端到端 tok/s 变了」这种模糊信号，拆成「哪个形状的内核、跑到了
// 多少 GB/s、离访存上限还有多远」的可量化信号。
//
//   build/bench_kernels                 # 全部形状 x 默认线程
//   build/bench_kernels --threads 1,8,24
//   build/bench_kernels --rep 7
//
// 报告口径：
//   bytes = n*k*elem      （m=1 时权重只读一遍，这就是必须搬运的字节数）
//   GB/s  = bytes / time
//   权重常驻带宽上限由 --bw 的纯读循环给出，是优化能拿到的天花板。

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#endif

#include "dtypes.h"
#include "matmul.h"
#include "weights.h"

using namespace tg;

namespace {

const char* kThreadsEnvHelp =
    "  --threads LIST   逗号分隔的线程数（默认 1,4,8,12,16,24）\n"
    "  --rep N          每个用例重复次数，取最小值（默认 5）\n"
    "  --bw             只测纯顺序读带宽上限\n"
    "  --scalar         只测标量参考内核（对比用）\n";

double NowS() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<int> ParseIntList(const std::string& s) {
  std::vector<int> out;
  std::string cur;
  for (size_t i = 0; i <= s.size(); ++i) {
    const char c = (i == s.size()) ? ',' : s[i];
    if (c == ',' || c == ' ') {
      if (!cur.empty()) { out.push_back(std::atoi(cur.c_str())); cur.clear(); }
    } else {
      cur.push_back(c);
    }
  }
  return out;
}

int MaxThreads() {
#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void SetThreads(int n) {
#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
  omp_set_num_threads(n);
#else
  (void)n;
#endif
}

// 合成权重：填成"看起来像真权重"的字节，避免全 0 让解码路径短路。
void FillSynthetic(uint8_t* p, int64_t bytes, DType t) {
  uint32_t s = 0x12345678u;
  auto next = [&]() {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
  };
  switch (t) {
    case DType::kSFP:
      // 合法 sfp：1..254，跳过 0 / 0x80（零值）。
      for (int64_t i = 0; i < bytes; ++i) {
        uint8_t v = static_cast<uint8_t>(next() & 0xFF);
        p[i] = (v == 0 || v == 0x80) ? 0x21 : v;
      }
      break;
    case DType::kBF16: {
      // 量级接近真实激活：指数落在 [-8, 2) 附近。
      for (int64_t i = 0; i < bytes / 2; ++i) {
        const uint16_t bits = static_cast<uint16_t>((next() & 0x83FFu) | 0x3900u);
        std::memcpy(p + i * 2, &bits, 2);
      }
      break;
    }
    default: {
      for (int64_t i = 0; i < bytes / 4; ++i) {
        const float f = static_cast<float>(static_cast<int32_t>(next() >> 8)) / 1e7f;
        std::memcpy(p + i * 4, &f, 4);
      }
      break;
    }
  }
}

struct Case {
  const char* name;
  DType type;
  int64_t n;  // 输出行数
  int64_t k;  // 规约维（= model_dim 或 ff_hidden_dim）
  int64_t layer_uses;  // 一个 token 要跑几次（用于折算端到端）
};

// Gemma 2 2B 的 decode 投影清单（m=1）。
const Case kCases[] = {
    {"qkv      (4096x2304)", DType::kSFP, 4096, 2304, 26},
    {"gate     (9216x2304)", DType::kSFP, 9216, 2304, 26},
    {"up       (9216x2304)", DType::kSFP, 9216, 2304, 26},
    {"linear   (2304x9216)", DType::kSFP, 2304, 9216, 26},
    {"logits   (256000x2304)", DType::kBF16, 256000, 2304, 1},
};

// 纯顺序读带宽上限：8 字节一组做 XOR 累加，读满整块内存。
double MeasureReadBandwidth(int64_t bytes, int threads) {
  const int64_t words = bytes / 8;
  std::vector<uint64_t> buf(static_cast<size_t>(words), 0x9E3779B97F4A7C15ull);
  SetThreads(threads);
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    const double t0 = NowS();
    uint64_t acc = 0;
#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(^ : acc)
#endif
    for (int64_t i = 0; i < words; ++i) acc ^= buf[static_cast<size_t>(i)];
    const double dt = NowS() - t0;
    // 防优化掉：把累加结果落到 volatile，编译器就不能把整个读循环删掉。
    volatile uint64_t sink = acc;
    (void)sink;
    best = std::min(best, dt);
  }
  return static_cast<double>(bytes) / best / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> threads_list = {1, 4, 8, 12, 16, 24};
  int rep = 5;
  bool bw_only = false;
  bool scalar_only = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) threads_list = ParseIntList(argv[++i]);
    else if (a == "--rep" && i + 1 < argc) rep = std::atoi(argv[++i]);
    else if (a == "--bw") bw_only = true;
    else if (a == "--scalar") scalar_only = true;
    else if (a == "-h" || a == "--help") { std::fputs(kThreadsEnvHelp, stdout); return 0; }
  }
  if (rep < 1) rep = 1;

  std::printf("机器: %d 线程可用\n", MaxThreads());
#if defined(__AVX2__)
  std::printf("编译期: AVX2 已启用%s\n", ""
#if defined(__FMA__)
      "，FMA 已启用"
#endif
  );
#else
  std::printf("编译期: 未启用 AVX2（内核会回退标量）\n");
#endif
  std::printf("%s\n", scalar_only ? "内核: 仅标量参考" : "内核: 自动选择（AVX2 可用则用 AVX2）");

  // ---- 天花板：纯顺序读 ----
  std::printf("\n=== 纯顺序读带宽上限（8 字节 XOR 累加，512 MB）===\n");
  std::printf("%-8s %10s\n", "线程", "GB/s");
  double peak_bw = 0.0;
  for (int t : threads_list) {
    const double bw = MeasureReadBandwidth(512LL << 20, t);
    peak_bw = std::max(peak_bw, bw);
    std::printf("%-8d %10.2f\n", t, bw);
  }

  if (bw_only) return 0;

  // ---- 各投影形状 ----
  const float mb = 1024.0f * 1024.0f;
  double total_bytes_per_token = 0.0;
  double total_time_per_token_ms = 0.0;

  for (const Case& c : kCases) {
    const int64_t elem = DTypeElementBytes(c.type);
    const int64_t bytes = c.n * c.k * elem;
    std::vector<uint8_t> wbuf(static_cast<size_t>(bytes));
    FillSynthetic(wbuf.data(), bytes, c.type);

    std::vector<float> a(static_cast<size_t>(c.k));
    FillSynthetic(reinterpret_cast<uint8_t*>(a.data()), c.k * 4, DType::kF32);
    std::vector<float> out(static_cast<size_t>(c.n));

    Matrix w;
    w.data = wbuf.data();
    w.type = c.type;
    w.rows = c.n;
    w.cols = c.k;

    std::printf("\n=== %s  %s  %.1f MB/token ===\n", c.name, DTypeName(c.type),
                bytes / mb);
    std::printf("%-8s %10s %10s %12s %12s\n", "线程", "ms", "GB/s", "Gelem/s",
                "带宽占比");

    double best_ms_overall = 1e30;
    for (int t : threads_list) {
      SetThreads(t);
      double best = 1e30;
      for (int r = 0; r < rep; ++r) {
        const double t0 = NowS();
        if (scalar_only) {
          MatMulBTRef(out.data(), a.data(), 1, c.k, w);
        } else {
          MatMulBT(out.data(), a.data(), 1, c.k, w);
        }
        const double dt = NowS() - t0;
        best = std::min(best, dt);
      }
      const double ms = best * 1e3;
      const double gbs = static_cast<double>(bytes) / best / 1e9;
      const double gel = static_cast<double>(c.n) * c.k / best / 1e9;
      std::printf("%-8d %10.3f %10.2f %12.2f %11.0f%%\n", t, ms, gbs, gel,
                  peak_bw > 0 ? 100.0 * gbs / peak_bw : 0.0);
      best_ms_overall = std::min(best_ms_overall, ms);
    }
    total_bytes_per_token += static_cast<double>(bytes) * c.layer_uses;
    total_time_per_token_ms += best_ms_overall * c.layer_uses;

    // 校验：解引用一次输出，防止整段被优化掉。
    volatile float sink = out[0];
    (void)sink;
  }

  std::printf("\n=== 折算（每个 decode token 的投影总量）===\n");
  std::printf("权重搬运   : %.2f GB\n", total_bytes_per_token / 1e9);
  std::printf("投影耗时   : %.1f ms\n", total_time_per_token_ms);
  std::printf("等效上限   : %.2f tok/s\n",
              1000.0 / std::max(1e-9, total_time_per_token_ms));
  std::printf("访存天花板 : %.2f tok/s  （按 %.2f GB/s 纯读上限）\n",
              peak_bw * 1e9 / total_bytes_per_token, peak_bw);
  return 0;
}
