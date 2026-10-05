#include "ops.h"

#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#define TG_OPS_HAVE_AVX2 1
#else
#define TG_OPS_HAVE_AVX2 0
#endif

namespace tg {

#if TG_OPS_HAVE_AVX2
namespace {

// 向量化 e^x，Cephes/avx_mathfun 的经典做法，相对误差约 1 ulp。
//
// e^x = 2^y，y = x*log2(e)。把 y 取到最近的整数 n，余数 z = x - n*ln2
// 落在 [-0.35, 0.35]，e^z 用 6 阶泰勒展开就够了；2^n 不进浮点，直接用
// 指数域拼位模式。ln2 拆成 hi/lo 两段是为了抵消 n*ln2 的舍入。
//
// 输入先夹到 ±87：再大一点 n 会顶到指数域上界，2^n 变成 inf。
inline __m256 ExpPs(__m256 x) {
  x = _mm256_min_ps(x, _mm256_set1_ps(87.0f));
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));

  const __m256 y = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
  const __m256 fx = _mm256_floor_ps(_mm256_add_ps(y, _mm256_set1_ps(0.5f)));
  // z = x - fx*ln2_hi + fx*ln2_lo
  __m256 z = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  z = _mm256_fmadd_ps(fx, _mm256_set1_ps(2.12194440e-4f), z);

  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(5.0000001201e-1f));
  p = _mm256_fmadd_ps(p, _mm256_mul_ps(z, z), z);  // e^z ~ 1 + z + z^2*p
  p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

  const __m256i e = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(0x7f)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

// sigmoid(x) = 1/(1+e^-x)。负数一侧同样要先夹，否则 e^-x 溢出成 inf 后
// 结果是 0 而不是"接近 0"，两者在后续乘法里行为不同。
inline __m256 SigmoidPs(__m256 x) {
  return _mm256_div_ps(
      _mm256_set1_ps(1.0f),
      _mm256_add_ps(_mm256_set1_ps(1.0f),
                    ExpPs(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

// tanh(x) = 2*sigmoid(2x) - 1
inline __m256 TanhPs(__m256 x) {
  const __m256 s = SigmoidPs(_mm256_mul_ps(_mm256_set1_ps(2.0f), x));
  return _mm256_fmsub_ps(_mm256_set1_ps(2.0f), s, _mm256_set1_ps(1.0f));
}

}  // namespace
#endif  // TG_OPS_HAVE_AVX2

void RMSNorm(const float* x, int64_t n, const float* w, float eps, float* out) {
  // gemma.cpp 的 RMSNormMul：l2 用 f32 累加，再 1/sqrtf(l2/n + eps)。
  float l2 = 0.f;
  for (int64_t i = 0; i < n; ++i) l2 += x[i] * x[i];
  const float mul = 1.0f / std::sqrt(l2 / static_cast<float>(n) + eps);
  for (int64_t i = 0; i < n; ++i) out[i] = (1.0f + w[i]) * (mul * x[i]);
}

std::vector<float> MakeInvTimescale(int64_t qkv_dim, double base_frequency) {
  const int64_t half = qkv_dim / 2;
  std::vector<float> inv(static_cast<size_t>(half));
  for (int64_t d = 0; d < half; ++d) {
    const double exponent = static_cast<double>(2 * d) / static_cast<double>(qkv_dim);
    inv[static_cast<size_t>(d)] =
        static_cast<float>(1.0 / std::pow(base_frequency, exponent));
  }
  return inv;
}

void RopeTable::Init(std::vector<float> inv, int64_t initial_positions) {
  inv_timescale = std::move(inv);
  half = static_cast<int64_t>(inv_timescale.size());
  built = 0;
  const int64_t want = initial_positions > 0 ? initial_positions : 1;
  cos_t.assign(static_cast<size_t>(half * want), 0.0f);
  sin_t.assign(static_cast<size_t>(half * want), 0.0f);
  Ensure(want - 1);
}

void RopeTable::Ensure(int64_t pos) {
  if (pos < built) return;
  const int64_t want = pos + 1;
  if (static_cast<int64_t>(cos_t.size()) < half * want) {
    cos_t.resize(static_cast<size_t>(half * want));
    sin_t.resize(static_cast<size_t>(half * want));
  }
  for (int64_t p = built; p < want; ++p) {
    float* c = &cos_t[static_cast<size_t>(p * half)];
    float* s = &sin_t[static_cast<size_t>(p * half)];
    // 与 gemma.cpp 一致：theta = pos * inv_timescale[d]，用 double 的 pow 预计算
    // inv_timescale 后，这里只做一次 float 乘法。
    for (int64_t d = 0; d < half; ++d) {
      const double theta =
          static_cast<double>(p) * static_cast<double>(inv_timescale[static_cast<size_t>(d)]);
      c[d] = static_cast<float>(std::cos(theta));
      s[d] = static_cast<float>(std::sin(theta));
    }
  }
  built = want;
}

void RopeInplace(float* x, int64_t qkv_dim, const RopeTable& table, int64_t pos,
                 float mul) {
  const int64_t half = qkv_dim / 2;
  const float* c = &table.cos_t[static_cast<size_t>(pos * half)];
  const float* s = &table.sin_t[static_cast<size_t>(pos * half)];
  for (int64_t d = 0; d < half; ++d) {
    const float x0 = x[d];
    const float x1 = x[d + half];
    x[d] = mul * (x0 * c[d] - x1 * s[d]);
    x[d + half] = mul * (x0 * s[d] + x1 * c[d]);
  }
}

void GeluMulInplace(float* c1, const float* c2, int64_t n) {
  constexpr float kMul = 0.03567740813636141f;
  constexpr float kSqrt2OverPi = 0.797884560804236f;
#if TG_OPS_HAVE_AVX2
  // 标量版每元素一次 std::tanh，那是 libm 调用、带分支、编译器没法向量化。
  // 实测它占 prefill 的 12%（每 token 1.5 ms，而这块访存只需要 0.1 ms）。
  //
  // 化简：gelu_tanh(v) = 0.5*v*(1 + tanh(arg)) = v * sigmoid(2*arg)
  // （把 tanh 展开成 (e^a-e^-a)/(e^a+e^-a) 就能约掉成 sigmoid），
  // 于是只需要一次 exp 和一次除法，全部可以放在 AVX2 里。
  const __m256 vmul = _mm256_set1_ps(kMul);
  const __m256 v_s2p = _mm256_set1_ps(kSqrt2OverPi);
  const __m256 v_two = _mm256_set1_ps(2.0f);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 v = _mm256_loadu_ps(c1 + i);
    const __m256 arg =
        _mm256_mul_ps(v, _mm256_fmadd_ps(vmul, _mm256_mul_ps(v, v), v_s2p));
    const __m256 cdf = SigmoidPs(_mm256_mul_ps(v_two, arg));
    _mm256_storeu_ps(
        c1 + i, _mm256_mul_ps(_mm256_mul_ps(v, cdf), _mm256_loadu_ps(c2 + i)));
  }
  for (; i < n; ++i) {
    const float v = c1[i];
    const float arg = v * (kSqrt2OverPi + kMul * v * v);
    c1[i] = v * (0.5f + 0.5f * std::tanh(arg)) * c2[i];
  }
  return;
#endif
  for (int64_t i = 0; i < n; ++i) {
    const float v = c1[i];
    const float arg = v * (kSqrt2OverPi + kMul * v * v);
    const float cdf = 0.5f + 0.5f * std::tanh(arg);
    c1[i] = v * cdf * c2[i];
  }
}

void SoftCapInplace(float* x, int64_t n, float cap) {
  if (cap == 0.0f) return;
  const float inv = 1.0f / cap;
#if TG_OPS_HAVE_AVX2
  // 这里同样在 256000 维 logits 上逐元素调 tanh，decode 每 token 都要跑一遍，
  // 值得一起向量化。
  const __m256 v_cap = _mm256_set1_ps(cap);
  const __m256 v_inv = _mm256_set1_ps(inv);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(
        x + i, _mm256_mul_ps(v_cap,
                             TanhPs(_mm256_mul_ps(_mm256_loadu_ps(x + i), v_inv))));
  }
  for (; i < n; ++i) x[i] = cap * std::tanh(x[i] * inv);
  return;
#endif
  for (int64_t i = 0; i < n; ++i) x[i] = cap * std::tanh(x[i] * inv);
}

void SoftmaxInplace(float* x, int64_t n) {
  float max_v = x[0];
  for (int64_t i = 1; i < n; ++i) max_v = x[i] > max_v ? x[i] : max_v;
  float sum = 0.f;
  for (int64_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - max_v);
    sum += x[i];
  }
  const float inv = 1.0f / sum;
  for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

}  // namespace tg
