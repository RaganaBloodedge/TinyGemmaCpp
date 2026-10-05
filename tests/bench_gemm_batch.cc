// TinyGemmaCpp —— prefill（m>1）GEMM 微基准
//
// bench_kernels.cc 只测 m=1 的 decode 形状，那里一切都被访存带宽锁死，
// 看不出内核本身的效率。prefill 是计算受限的，要单独量。
//
// 报告口径：
//   MACs   = m*n*k
//   GMAC/s = MACs / time
//   峰值%  = 相对同机 --fma 测出的纯 FMA 吞吐
//
//   build/bench_gemm_batch                      # 默认形状 x 默认线程
//   build/bench_gemm_batch --m 301 --k 2304 --n 9216 --type sfp
//   build/bench_gemm_batch --threads 1,8,16 --rep 5
//   build/bench_gemm_batch --fma                # 只测纯 FMA 峰值

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "dtypes.h"
#include "matmul.h"
#include "weights.h"

using namespace tg;

namespace {

// 防止累加结果被优化掉：写一个 volatile 全局，比 fprintf 干净。
volatile float g_sink = 0.0f;

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

// 合成权重。跟 bench_kernels 一样填"像真权重"的字节，别让解码短路。
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

DType ParseType(const std::string& s) {
  if (s == "bf16") return DType::kBF16;
  if (s == "f32") return DType::kF32;
  return DType::kSFP;
}

// 纯 FMA 吞吐上限：所有操作数常驻寄存器，不碰内存。
// 这是这台机器上"内核最多能跑多快"的硬天花板。
double MeasureFmaPeak(int threads, double seconds) {
#if defined(__AVX2__) && defined(__FMA__)
  const int64_t iters = static_cast<int64_t>(seconds * 1e9 / 4.0);
  double total_gmac = 0.0;
#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#pragma omp parallel reduction(+ : total_gmac)
#endif
  {
    __m256 a0 = _mm256_set1_ps(1.0001f), a1 = _mm256_set1_ps(1.0002f);
    __m256 a2 = _mm256_set1_ps(1.0003f), a3 = _mm256_set1_ps(1.0004f);
    __m256 a4 = _mm256_set1_ps(1.0005f), a5 = _mm256_set1_ps(1.0006f);
    __m256 a6 = _mm256_set1_ps(1.0007f), a7 = _mm256_set1_ps(1.0008f);
    const __m256 b = _mm256_set1_ps(0.99999f);
    const __m256 c = _mm256_set1_ps(1e-9f);
    const double t0 = NowS();
    for (int64_t i = 0; i < iters; ++i) {
      a0 = _mm256_fmadd_ps(a0, b, c);
      a1 = _mm256_fmadd_ps(a1, b, c);
      a2 = _mm256_fmadd_ps(a2, b, c);
      a3 = _mm256_fmadd_ps(a3, b, c);
      a4 = _mm256_fmadd_ps(a4, b, c);
      a5 = _mm256_fmadd_ps(a5, b, c);
      a6 = _mm256_fmadd_ps(a6, b, c);
      a7 = _mm256_fmadd_ps(a7, b, c);
    }
    const double dt = NowS() - t0;
    // 8 条 FMA × 8 个 float。/2 是把 flop 折成 MAC。
    total_gmac += 8.0 * 8.0 * static_cast<double>(iters) / dt / 1e9;
    const float sink = _mm_cvtss_f32(_mm256_castps256_ps128(
        _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))));
    g_sink = sink;
  }
  (void)threads;
  return total_gmac;
#else
  (void)threads;
  (void)seconds;
  return 0.0;
#endif
}

struct Shape {
  const char* name;
  DType type;
  int64_t m, k, n;
};

// Gemma 2 2B 的 prefill 投影清单（m 取 301 = 一个真实长 prompt）。
const Shape kShapes[] = {
    {"ff gate/up  (9216x2304)", DType::kSFP, 301, 2304, 9216},
    {"ff linear   (2304x9216)", DType::kSFP, 301, 9216, 2304},
    {"qkv         (4096x2304)", DType::kSFP, 301, 2304, 4096},
    {"att_ein     (2304x2048)", DType::kSFP, 301, 2048, 2304},
};

double RunOne(const Shape& s, int threads, int rep, bool ref) {
  const int64_t elem = DTypeElementBytes(s.type);
  const int64_t wbytes = s.n * s.k * elem;
  std::vector<uint8_t> wbuf(static_cast<size_t>(wbytes));
  FillSynthetic(wbuf.data(), wbytes, s.type);

  std::vector<float> a(static_cast<size_t>(s.m * s.k));
  FillSynthetic(reinterpret_cast<uint8_t*>(a.data()),
                static_cast<int64_t>(a.size()) * 4, DType::kF32);
  std::vector<float> out(static_cast<size_t>(s.m * s.n));

  Matrix w;
  w.data = wbuf.data();
  w.type = s.type;
  w.rows = s.n;
  w.cols = s.k;

  SetThreads(threads);
  double best = 1e30;
  for (int r = 0; r < rep; ++r) {
    const double t0 = NowS();
    if (ref) {
      MatMulBTRef(out.data(), a.data(), s.m, s.k, w);
    } else {
      MatMulBT(out.data(), a.data(), s.m, s.k, w);
    }
    const double dt = NowS() - t0;
    best = std::min(best, dt);
  }
  if (out[0] != out[0]) std::fprintf(stderr, "nan\n");
  g_sink = out[0];
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> threads_list = {1, 8, 16};
  int rep = 3;
  bool fma_only = false;
  bool ref = false;
  int64_t om = -1, ok = -1, on = -1;
  DType otype = DType::kSFP;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) threads_list = ParseIntList(argv[++i]);
    else if (a == "--rep" && i + 1 < argc) rep = std::atoi(argv[++i]);
    else if (a == "--fma") fma_only = true;
    else if (a == "--ref") ref = true;
    else if (a == "--m" && i + 1 < argc) om = std::atoll(argv[++i]);
    else if (a == "--k" && i + 1 < argc) ok = std::atoll(argv[++i]);
    else if (a == "--n" && i + 1 < argc) on = std::atoll(argv[++i]);
    else if (a == "--type" && i + 1 < argc) otype = ParseType(argv[++i]);
    else if (a == "-h" || a == "--help") {
      std::fputs(
          "  --m/--k/--n N    自定义形状（三者都要给）\n"
          "  --type sfp|bf16|f32\n"
          "  --threads LIST   默认 1,8,16\n"
          "  --rep N          每个配置重复次数，取最小值（默认 3）\n"
          "  --ref            用标量参考内核\n"
          "  --fma            只测纯 FMA 峰值\n",
          stdout);
      return 0;
    }
  }
  if (rep < 1) rep = 1;

  std::printf("机器: %d 线程可用\n", MaxThreads());

  // ---- 天花板：纯 FMA（不碰内存）----
  std::printf("\n=== 纯 FMA 峰值（操作数常驻寄存器）===\n");
  std::printf("%-8s %14s\n", "线程", "GMAC/s");
  double peak = 0.0;
  for (int t : threads_list) {
    const double g = MeasureFmaPeak(t, 0.15);
    peak = std::max(peak, g);
    std::printf("%-8d %14.1f\n", t, g);
  }
  if (fma_only) return 0;

  std::vector<Shape> shapes;
  if (om > 0 && ok > 0 && on > 0) {
    shapes.push_back({"custom", otype, om, ok, on});
  } else {
    for (const Shape& s : kShapes) shapes.push_back(s);
  }

  for (const Shape& s : shapes) {
    const double macs = static_cast<double>(s.m) * static_cast<double>(s.k) *
                        static_cast<double>(s.n);
    std::printf("\n=== %s  m=%lld k=%lld n=%lld  %.2f GMAC%s ===\n", s.name,
                static_cast<long long>(s.m), static_cast<long long>(s.k),
                static_cast<long long>(s.n), macs / 1e9,
                ref ? "  [标量参考]" : "");
    std::printf("%-8s %10s %12s %10s\n", "线程", "ms", "GMAC/s", "峰值%");
    for (int t : threads_list) {
      const double dt = RunOne(s, t, rep, ref);
      const double g = macs / dt / 1e9;
      std::printf("%-8d %10.3f %12.1f %9.1f%%\n", t, dt * 1e3, g,
                  peak > 0 ? 100.0 * g / peak : 0.0);
    }
  }
  return 0;
}
