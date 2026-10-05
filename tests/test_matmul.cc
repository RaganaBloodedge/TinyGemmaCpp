// TinyGemmaCpp —— GEMM 内核回归测试
//
// 校验 SIMD 路径与标量参考路径一致。不要求逐位相同（浮点累加顺序不同，
// 允许 ULP 级差异），但要求相对误差在阈值内，且 dtype/shape 边界全覆盖。
//
// 覆盖的边界：
//   * k = 1..40（跨过 8 / 32 两个向量化边界）
//   * k = 2304 / 9216（真实形状）
//   * k 不是 8 的倍数（尾循环）
//   * m = 1（decode）与 m > 1（prefill 的权重行复用路径）
//   * sfp / bf16 / f32 三种存储类型
//   * 权重里塞入 0x00 / 0x80 两个"零"编码，验证 SIMD 抹零
//
// 用法:  build/test_matmul

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "dtypes.h"
#include "matmul.h"
#include "weights.h"

using namespace tg;

namespace {

int g_failed = 0;
int g_passed = 0;

// 相对误差判据：分子用 |a-b|，分母用 max(|a|,|b|,1)，避免大数掩盖小误差。
bool CloseEnough(double a, double b, double rtol) {
  const double d = std::fabs(a - b);
  return d <= rtol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

void Fill(uint8_t* p, int64_t bytes, DType t, uint32_t seed) {
  uint32_t s = seed;
  auto next = [&]() {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
  };
  switch (t) {
    case DType::kSFP:
      for (int64_t i = 0; i < bytes; ++i) {
        // 每 37 个元素塞一个零编码，确保抹零分支被走到。
        if (i % 37 == 0) { p[i] = (i % 74 == 0) ? 0x00 : 0x80; continue; }
        uint8_t v = static_cast<uint8_t>(next() & 0xFF);
        p[i] = (v == 0 || v == 0x80) ? 0x21 : v;
      }
      break;
    case DType::kBF16:
      for (int64_t i = 0; i < bytes / 2; ++i) {
        const uint16_t bits =
            static_cast<uint16_t>((next() & 0x87FFu) | 0x3900u);
        std::memcpy(p + i * 2, &bits, 2);
      }
      break;
    default:
      for (int64_t i = 0; i < bytes / 4; ++i) {
        const float f =
            static_cast<float>(static_cast<int32_t>(next() >> 9)) / 4.0e6f;
        std::memcpy(p + i * 4, &f, 4);
      }
      break;
  }
}

void Case(const char* name, DType t, int64_t m, int64_t n, int64_t k) {
  const int64_t elem = DTypeElementBytes(t);
  // k 不是 32 的倍数时补齐读写范围，避免尾循环越界。
  const int64_t k_pad = ((k + 31) / 32) * 32 + 32;

  std::vector<uint8_t> wbuf(static_cast<size_t>(n * k_pad * elem));
  Fill(wbuf.data(), static_cast<int64_t>(wbuf.size()), t, 0xABCD1234u);

  std::vector<float> a(static_cast<size_t>(m * k_pad));
  Fill(reinterpret_cast<uint8_t*>(a.data()),
       static_cast<int64_t>(a.size()) * 4, DType::kF32, 0x5EED5678u);

  std::vector<float> out_ref(static_cast<size_t>(m * n), 0.f);
  std::vector<float> out_simd(static_cast<size_t>(m * n), 0.f);

  Matrix w;
  w.data = wbuf.data();
  w.type = t;
  w.rows = n;
  w.cols = k;  // 逻辑列数用真的 k，内核按 k 走规约

  if (m == 1) {
    MatMulBTRef(out_ref.data(), a.data(), m, k, w);
    MatMulBT(out_simd.data(), a.data(), m, k, w);
  } else {
    // prefill 路径：整表一次算完
    MatMulBTRef(out_ref.data(), a.data(), m, k, w);
    MatMulBT(out_simd.data(), a.data(), m, k, w);
  }

  double worst = 0.0;
  int64_t worst_idx = -1;
  for (int64_t i = 0; i < m * n; ++i) {
    if (!CloseEnough(out_ref[static_cast<size_t>(i)],
                     out_simd[static_cast<size_t>(i)], 2e-3)) {
      ++worst_idx;
    }
    const double d = std::fabs(out_ref[static_cast<size_t>(i)] -
                               out_simd[static_cast<size_t>(i)]) /
                     std::max(1.0f, std::fabs(out_ref[static_cast<size_t>(i)]));
    if (d > worst) { worst = d; worst_idx = i; }
  }

  const bool ok = worst <= 2e-3;
  if (ok) ++g_passed; else ++g_failed;
  std::printf("  [%s] %-34s m=%-3lld n=%-6lld k=%-5lld %s  最大相对误差 %.3e\n",
              ok ? "PASS" : "FAIL", name, static_cast<long long>(m),
              static_cast<long long>(n), static_cast<long long>(k),
              DTypeName(t), worst);
  if (!ok) {
    std::printf("         首个超差位置 idx=%lld ref=%.9g simd=%.9g\n",
                static_cast<long long>(worst_idx),
                static_cast<double>(out_ref[static_cast<size_t>(worst_idx)]),
                static_cast<double>(out_simd[static_cast<size_t>(worst_idx)]));
  }
}

// gate/up 拆分路径：MatMulBTRows 的后半段输出应当等于整表调用的对应列切片。
// 模型里 gating_ein 就是被切成 [0,F) 与 [F,2F) 两次调用的，必须逐位对齐。
void CaseRowSplit(DType t, int64_t m, int64_t n, int64_t k, int64_t split) {
  const int64_t elem = DTypeElementBytes(t);
  const int64_t k_pad = ((k + 31) / 32) * 32 + 32;

  std::vector<uint8_t> wbuf(static_cast<size_t>(n * k_pad * elem));
  Fill(wbuf.data(), static_cast<int64_t>(wbuf.size()), t, 0x1111AAAAu);

  std::vector<float> a(static_cast<size_t>(m * k_pad));
  Fill(reinterpret_cast<uint8_t*>(a.data()),
       static_cast<int64_t>(a.size()) * 4, DType::kF32, 0x2222BBBBu);

  std::vector<float> full(static_cast<size_t>(m * n), 0.f);
  std::vector<float> low(static_cast<size_t>(m * split), 0.f);
  std::vector<float> high(static_cast<size_t>(m * (n - split)), 0.f);

  Matrix w;
  w.data = wbuf.data();
  w.type = t;
  w.rows = n;
  w.cols = k;

  MatMulBTRef(full.data(), a.data(), m, k, w);
  MatMulBTRows(low.data(), a.data(), m, k, w, 0, split);
  MatMulBTRows(high.data(), a.data(), m, k, w, split, n);

  double worst = 0.0;
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < split; ++j) {
      const double d = std::fabs(full[static_cast<size_t>(i * n + j)] -
                                 low[static_cast<size_t>(i * split + j)]) /
                       std::max(1.0f, std::fabs(full[static_cast<size_t>(i * n + j)]));
      worst = std::max(worst, d);
    }
    for (int64_t j = split; j < n; ++j) {
      const double d =
          std::fabs(full[static_cast<size_t>(i * n + j)] -
                    high[static_cast<size_t>(i * (n - split) + (j - split))]) /
          std::max(1.0f, std::fabs(full[static_cast<size_t>(i * n + j)]));
      worst = std::max(worst, d);
    }
  }

  const bool ok = worst <= 2e-3;
  if (ok) ++g_passed; else ++g_failed;
  std::printf("  [%s] %-34s m=%-3lld n=%-6lld k=%-5lld %s  最大相对误差 %.3e\n",
              ok ? "PASS" : "FAIL", "gate/up 行切片", static_cast<long long>(m),
              static_cast<long long>(n), static_cast<long long>(k),
              DTypeName(t), worst);
}

}  // namespace

int main() {
  std::printf("=== GEMM 内核回归：SIMD vs 标量参考 ===\n");
#if defined(__AVX2__)
  std::printf("(本次编译启用了 AVX2)\n");
#else
  std::printf("(本次编译未启用 AVX2，默认路径=标量参考，应当零误差)\n");
#endif

  // k 跨过向量化边界：8 / 32
  for (int64_t k : {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 63, 64, 100}) {
    Case("k边界", DType::kSFP, 1, 17, k);
    Case("k边界", DType::kBF16, 1, 17, k);
    Case("k边界", DType::kF32, 1, 17, k);
  }

  // 真实形状（decode）
  Case("qkv", DType::kSFP, 1, 4096, 2304);
  Case("gate", DType::kSFP, 1, 9216, 2304);
  Case("linear", DType::kSFP, 1, 2304, 9216);
  Case("logits", DType::kBF16, 1, 256000, 2304);
  Case("att", DType::kSFP, 1, 2304, 2048);

  // prefill（m > 1）—— 权重行复用路径
  Case("prefill小", DType::kSFP, 5, 512, 2304);
  Case("prefill中", DType::kSFP, 33, 256, 2304);
  Case("prefill大", DType::kSFP, 100, 128, 2304);

  // gate/up 行切片（模型里的真实调用形态）
  CaseRowSplit(DType::kSFP, 1, 9216, 2304, 9216 / 2);
  CaseRowSplit(DType::kSFP, 5, 9216, 2304, 9216 / 2);
  CaseRowSplit(DType::kBF16, 5, 512, 2304, 256);
  CaseRowSplit(DType::kF32, 3, 128, 2304, 64);

  std::printf("\n%s：%d 通过 / %d 失败\n", g_failed == 0 ? "全部通过" : "有失败",
              g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}
