// TinyGemmaCpp —— GEMM 内核
//
// 两条路径：
//   * *Ref  —— 标量参考实现，不做任何平台假设，是正确性基准。
//   * 默认  —— 编译期探到 AVX2/FMA 就走 SIMD 内核，否则自动退回标量。
//
// decode（m=1）的性能模型：权重每个元素 1~2 字节，必须整表从内存搬到核里，
// 所以上限就是访存带宽。优化目标不是"算得快"，而是**每字节权重少花指令**，
// 让端口/L1 查表不再是瓶颈，把时间让给内存。
//
// 下面三个 AVX2 内核都按 8 元素/向量、4 路展开（32 元素/迭代）写，
// 4 个独立累加器是为了打断 FMA 的延迟链（Raptor Lake 上 FMA 延迟 4 cycle）。

#include "matmul.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#include <omp.h>
// dynamic 分块：这颗 14650HX 是 8 P-core + 8 E-core 的异构片，
// static 会让 P-core 早早做完干等，E-core 拖长尾部。
#define TG_OMP_FOR_ROWS _Pragma("omp parallel for schedule(dynamic, 8)")
#else
#define TG_OMP_FOR_ROWS
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define TG_HAVE_AVX2 1
#else
#define TG_HAVE_AVX2 0
#endif

namespace tg {
namespace {

// ===========================================================================
// 标量参考内核
// ===========================================================================

// sfp 只有 256 种取值，标量路径退化成一次 L1 查表 + 一次 FMA。
const std::array<float, 256>& SfpLut() {
  static const std::array<float, 256> table = [] {
    std::array<float, 256> t{};
    for (int i = 0; i < 256; ++i) {
      t[static_cast<size_t>(i)] =
          Bf16ToF32(SfpDecodeToBf16(static_cast<uint8_t>(i)));
    }
    return t;
  }();
  return table;
}

inline float DotBf16Ref(const float* a, const uint8_t* w, int64_t k) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int64_t t = 0;
  for (; t + 4 <= k; t += 4) {
    s0 += a[t + 0] * Bf16ToF32(LoadU16(w + (t + 0) * 2));
    s1 += a[t + 1] * Bf16ToF32(LoadU16(w + (t + 1) * 2));
    s2 += a[t + 2] * Bf16ToF32(LoadU16(w + (t + 2) * 2));
    s3 += a[t + 3] * Bf16ToF32(LoadU16(w + (t + 3) * 2));
  }
  for (; t < k; ++t) s0 += a[t] * Bf16ToF32(LoadU16(w + t * 2));
  return (s0 + s1) + (s2 + s3);
}

inline float DotSfpRef(const float* a, const uint8_t* w, int64_t k) {
  const float* lut = SfpLut().data();
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int64_t t = 0;
  for (; t + 4 <= k; t += 4) {
    s0 += a[t + 0] * lut[w[t + 0]];
    s1 += a[t + 1] * lut[w[t + 1]];
    s2 += a[t + 2] * lut[w[t + 2]];
    s3 += a[t + 3] * lut[w[t + 3]];
  }
  for (; t < k; ++t) s0 += a[t] * lut[w[t]];
  return (s0 + s1) + (s2 + s3);
}

inline float DotF32Ref(const float* a, const uint8_t* w, int64_t k) {
  const float* wp = reinterpret_cast<const float*>(w);
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int64_t t = 0;
  for (; t + 4 <= k; t += 4) {
    s0 += a[t + 0] * wp[t + 0];
    s1 += a[t + 1] * wp[t + 1];
    s2 += a[t + 2] * wp[t + 2];
    s3 += a[t + 3] * wp[t + 3];
  }
  for (; t < k; ++t) s0 += a[t] * wp[t];
  return (s0 + s1) + (s2 + s3);
}

float DotRef(const float* a, const uint8_t* w, int64_t k, DType t) {
  switch (t) {
    case DType::kBF16: return DotBf16Ref(a, w, k);
    case DType::kSFP: return DotSfpRef(a, w, k);
    case DType::kF32: return DotF32Ref(a, w, k);
    default: return 0.0f;
  }
}

// 把一整行权重解码成 f32。prefill 用：一行解码一次，然后拿解码结果
// 去点乘所有 token，避免解码被重复做 m 遍。
void DecodeRowToF32(const uint8_t* src, int64_t k, DType t, float* dst) {
  switch (t) {
    case DType::kBF16:
      for (int64_t i = 0; i < k; ++i) dst[i] = Bf16ToF32(LoadU16(src + i * 2));
      break;
    case DType::kSFP: {
      // 标量路径本来就是查表，一行 2304 次查表远比再做 m 遍便宜。
      const float* lut = SfpLut().data();
      for (int64_t i = 0; i < k; ++i) dst[i] = lut[src[i]];
      break;
    }
    case DType::kF32:
      std::memcpy(dst, src, static_cast<size_t>(k) * sizeof(float));
      break;
    default:
      for (int64_t i = 0; i < k; ++i) dst[i] = 0.0f;
      break;
  }
}

// 权重已经是 f32 时的点积（解码缓冲 / 原生 f32 权重共用）。
inline float DotF32VecRef(const float* a, const float* w, int64_t k) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int64_t t = 0;
  for (; t + 4 <= k; t += 4) {
    s0 += a[t + 0] * w[t + 0];
    s1 += a[t + 1] * w[t + 1];
    s2 += a[t + 2] * w[t + 2];
    s3 += a[t + 3] * w[t + 3];
  }
  for (; t < k; ++t) s0 += a[t] * w[t];
  return (s0 + s1) + (s2 + s3);
}

// ===========================================================================
// AVX2 内核
// ===========================================================================
#if TG_HAVE_AVX2

// 128-bit 水平求和 -> 标量。
inline float HSum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

// bf16 -> f32 在 AVX2 上是纯粹的左移 16 位（没有舍入），
// 所以解码就是一个 cvtepu16 + slli，编译器在自动向量化时也常能生成，
// 但手写能保证 4 路累加器不被拆散。
inline __m256 Bf16x8ToF32x8(const uint8_t* p) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

// sfp 8 字节 -> f32x8。
//
// 原始解码（见 dtypes.h SfpDecodeToBf16）是按 small/large 两条分支拼
// (hi<<8)|lo，逐元素有分支。这里把它化简成无分支闭式：
//
//   v   = x & 0x7F
//   bf16 = (v > 63) ? 0x3800 + (v << 4)      // large 分支
//                   : 0x3400 + (v << 5)      // small 分支
//   f32  = (bf16 << 16) | (sign << 16)
//
// 推导：large 时 hi = 0x38 + (v>>4)、lo = (v<<4)&0xFF，
//       ((0x38+v>>4)<<8) | (v<<4 & 0xFF) = 0x3800 + 16v（因为 v<<4 的高位
//       恰好被 v>>4 的进位吸收）；small 同理得到 0x3400 + 32v。
// 已验证 256 个字节里全部 254 个非零编码与分支版逐位相同（tools/ 下留了脚本）。
//
// 唯一要另外处理的是 v == 0：闭式会给出 2^-23 而不是 0，必须显式抹掉。
inline __m256 Sfp8ToF32x8(const uint8_t* p) {
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
  const __m256i x = _mm256_cvtepu8_epi32(raw);
  const __m256i v = _mm256_and_si256(x, _mm256_set1_epi32(0x7F));
  const __m256i large = _mm256_cmpgt_epi32(v, _mm256_set1_epi32(63));
  const __m256i vv = _mm256_blendv_epi8(_mm256_slli_epi32(v, 5),
                                        _mm256_slli_epi32(v, 4), large);
  const __m256i base = _mm256_blendv_epi8(_mm256_set1_epi32(0x3400),
                                          _mm256_set1_epi32(0x3800), large);
  const __m256i bits = _mm256_or_si256(
      _mm256_slli_epi32(_mm256_add_epi32(vv, base), 16),
      _mm256_slli_epi32(_mm256_and_si256(x, _mm256_set1_epi32(0x80)), 24));
  __m256 f = _mm256_castsi256_ps(bits);
  // v == 0 才是真零；两个零编码（0x00 / 0x80）都落在这一支。
  return _mm256_andnot_ps(
      _mm256_castsi256_ps(_mm256_cmpeq_epi32(v, _mm256_setzero_si256())), f);
}

// 16 个 sfp 字节 -> 两路 f32x8，一条指令处理 16 个元素。
//
// 为什么能这么做：sfp 解出来的 bf16 只有 16 位，上面 Sfp8ToF32x8 里从
// `vv + base` 到 `<<16` 之间的所有中间量（vv <= 0x7F0、base <= 0x3800、
// 和 <= 0x3FF0）都装得下 16 位。所以把整个解码放进 16 bit 通道，每条
// 向量指令处理的元素数从 8 变成 16，解码的指令数直接减半。
//
// 代价是输出顺序被打乱：unpack 是按 128 位车道做的，所以
//   out_lo 对应字节 [0..3] 与 [8..11]
//   out_hi 对应字节 [4..7] 与 [12..15]
// 调用方必须用同样的顺序去取 a[]（见 DotSfpAvx2 里的 permute2f128）。
inline void Sfp16ToF32x8x2(const uint8_t* p, __m256* out_lo, __m256* out_hi) {
  const __m256i x =
      _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  const __m256i v = _mm256_and_si256(x, _mm256_set1_epi16(0x7F));
  const __m256i large = _mm256_cmpgt_epi16(v, _mm256_set1_epi16(63));
  const __m256i vv = _mm256_blendv_epi8(_mm256_slli_epi16(v, 5),
                                        _mm256_slli_epi16(v, 4), large);
  const __m256i base = _mm256_blendv_epi8(_mm256_set1_epi16(0x3400),
                                          _mm256_set1_epi16(0x3800), large);
  __m256i u = _mm256_add_epi16(vv, base);
  // 符号（字节 bit7）挪到 16 位模式的 bit15。中间位的垃圾必须掩掉，
  // 否则会污染指数与尾数。
  u = _mm256_or_si256(u,
                      _mm256_and_si256(_mm256_slli_epi16(x, 8),
                                       _mm256_set1_epi16(static_cast<short>(0x8000))));
  // v == 0 才是真零（0x00 / 0x80 都要被清成 0）。
  u = _mm256_andnot_si256(_mm256_cmpeq_epi16(v, _mm256_setzero_si256()), u);
  // 把 16 位模式搬到 32 位的高半部分，就是 bf16 -> f32。
  const __m256i z = _mm256_setzero_si256();
  *out_lo = _mm256_castsi256_ps(_mm256_unpacklo_epi16(z, u));
  *out_hi = _mm256_castsi256_ps(_mm256_unpackhi_epi16(z, u));
}

inline float DotBf16Avx2(const float* a, const uint8_t* w, int64_t k) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  int64_t t = 0;
  for (; t + 32 <= k; t += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t), Bf16x8ToF32x8(w + t * 2), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 8), Bf16x8ToF32x8(w + (t + 8) * 2), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 16), Bf16x8ToF32x8(w + (t + 16) * 2), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 24), Bf16x8ToF32x8(w + (t + 24) * 2), acc3);
  }
  for (; t + 8 <= k; t += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t), Bf16x8ToF32x8(w + t * 2), acc0);
  }
  float r = HSum(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                               _mm256_add_ps(acc2, acc3)));
  for (; t < k; ++t) r += a[t] * Bf16ToF32(LoadU16(w + t * 2));
  return r;
}

inline float DotSfpAvx2(const float* a, const uint8_t* w, int64_t k) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  int64_t t = 0;
  // 主循环：每次吃掉 32 个元素（两批 16），4 个累加器。
  for (; t + 32 <= k; t += 32) {
    __m256 d0, d1, d2, d3;
    Sfp16ToF32x8x2(w + t, &d0, &d1);
    Sfp16ToF32x8x2(w + t + 16, &d2, &d3);
    const __m256 av0 = _mm256_loadu_ps(a + t);
    const __m256 aw0 = _mm256_loadu_ps(a + t + 8);
    const __m256 av1 = _mm256_loadu_ps(a + t + 16);
    const __m256 aw1 = _mm256_loadu_ps(a + t + 24);
    // d0 管字节 [0..3]∪[8..11]，d1 管 [4..7]∪[12..15]，a 要按同样顺序取。
    acc0 = _mm256_fmadd_ps(_mm256_permute2f128_ps(av0, aw0, 0x20), d0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_permute2f128_ps(av0, aw0, 0x31), d1, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_permute2f128_ps(av1, aw1, 0x20), d2, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_permute2f128_ps(av1, aw1, 0x31), d3, acc3);
  }
  for (; t + 8 <= k; t += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t), Sfp8ToF32x8(w + t), acc0);
  }
  float r = HSum(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                               _mm256_add_ps(acc2, acc3)));
  const float* lut = SfpLut().data();
  for (; t < k; ++t) r += a[t] * lut[w[t]];
  return r;
}

inline float DotF32Avx2(const float* a, const uint8_t* w, int64_t k) {
  const float* wp = reinterpret_cast<const float*>(w);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  int64_t t = 0;
  for (; t + 32 <= k; t += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t), _mm256_loadu_ps(wp + t), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 8), _mm256_loadu_ps(wp + t + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 16), _mm256_loadu_ps(wp + t + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 24), _mm256_loadu_ps(wp + t + 24), acc3);
  }
  for (; t + 8 <= k; t += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t), _mm256_loadu_ps(wp + t), acc0);
  }
  float r = HSum(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                               _mm256_add_ps(acc2, acc3)));
  for (; t < k; ++t) r += a[t] * wp[t];
  return r;
}

// 权重已经是 f32 时（解码缓冲或原生 f32 权重）的 AVX2 点积。
// 与 DotF32Avx2 同构，只是省掉 uint8_t* 的不必要转换。
inline float DotF32VecAvx2(const float* a, const float* w, int64_t k) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  int64_t t = 0;
  for (; t + 32 <= k; t += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t), _mm256_loadu_ps(w + t), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 8), _mm256_loadu_ps(w + t + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 16), _mm256_loadu_ps(w + t + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t + 24), _mm256_loadu_ps(w + t + 24), acc3);
  }
  for (; t + 8 <= k; t += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + t), _mm256_loadu_ps(w + t), acc0);
  }
  float r = HSum(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                               _mm256_add_ps(acc2, acc3)));
  for (; t < k; ++t) r += a[t] * w[t];
  return r;
}

// 2 token × 2 权重行 的寄存器分块微内核。prefill 的核心。
//
// 单点积内核每个 k 步长要 8 次向量加载（4×a + 4×w）才换 4 次 FMA，
// 即 2 loads/FMA。而机器只有 2 个 load 端口和 2 个 FMA 端口，
// 于是 FMA 只用了一半 —— 加载端口先饱和。
//
// 这里让 2 个 a 向量和 2 个 w 向量喂出 4 次 FMA：4 loads / 4 FMA = 1 load/FMA，
// 每 32 字节加载对应 8 个 MAC，正好 4 B/MAC，与 64 B/cycle 的加载带宽
// 和 16 MAC/cycle 的 FMA 能力同时打平，两边都不浪费。
//
// 4 个累加器（s00/s01/s10/s11）彼此独立，也顺便把 FMA 的 4 cycle 延迟藏住了。
//
// 语义是 **累加**（*o += ...）：调用方负责先清零。prefill 路径要按 k 分块
// 累加，AttEinSum 要跨 head 累加，都需要这个语义。
inline void Gemm2x2Avx2(const float* a0, const float* a1, const float* w0,
                        const float* w1, int64_t k, float* r00, float* r01,
                        float* r10, float* r11) {
  __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
  __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
  int64_t t = 0;
  for (; t + 8 <= k; t += 8) {
    const __m256 av0 = _mm256_loadu_ps(a0 + t);
    const __m256 av1 = _mm256_loadu_ps(a1 + t);
    const __m256 wv0 = _mm256_loadu_ps(w0 + t);
    const __m256 wv1 = _mm256_loadu_ps(w1 + t);
    s00 = _mm256_fmadd_ps(av0, wv0, s00);
    s01 = _mm256_fmadd_ps(av0, wv1, s01);
    s10 = _mm256_fmadd_ps(av1, wv0, s10);
    s11 = _mm256_fmadd_ps(av1, wv1, s11);
  }
  float f00 = HSum(s00), f01 = HSum(s01), f10 = HSum(s10), f11 = HSum(s11);
  for (; t < k; ++t) {  // k 不是 8 的倍数时的尾巴
    const float x0 = a0[t], x1 = a1[t];
    f00 += x0 * w0[t];
    f01 += x0 * w1[t];
    f10 += x1 * w0[t];
    f11 += x1 * w1[t];
  }
  *r00 += f00;
  *r01 += f01;
  *r10 += f10;
  *r11 += f11;
}

// 4 token × 2 行 的微内核。比 2x2 多一倍独立累加链。
//
// 为什么需要更多链：Raptor Lake 上 FMA 延迟 4 周期，而每个 P 核有两个 FMA
// 端口。要让两个端口都满载，得有 8 条互不依赖的 FMA 在飞。2x2 只有 4 条，
// 于是每个周期最多发 4 条，FMA 吞吐只能到一半 —— 实测 FFN 就卡在 43% 峰值。
// 这里 8 个累加器正好把延迟完全藏住。
//
// 加载比也更好：每个 k 步长 4 个 a 向量 + 2 个 w 向量 = 6 次加载喂 8 次 FMA，
// 0.75 loads/FMA（2x2 是 1.0），低于加载端口 1:1 的极限。
inline void Gemm4x2Avx2(const float* a0, const float* a1, const float* a2,
                        const float* a3, const float* w0, const float* w1,
                        int64_t k, float* o00, float* o01, float* o10,
                        float* o11, float* o20, float* o21, float* o30,
                        float* o31) {
  __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
  __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
  __m256 s20 = _mm256_setzero_ps(), s21 = _mm256_setzero_ps();
  __m256 s30 = _mm256_setzero_ps(), s31 = _mm256_setzero_ps();
  int64_t t = 0;
  for (; t + 8 <= k; t += 8) {
    const __m256 av0 = _mm256_loadu_ps(a0 + t);
    const __m256 av1 = _mm256_loadu_ps(a1 + t);
    const __m256 av2 = _mm256_loadu_ps(a2 + t);
    const __m256 av3 = _mm256_loadu_ps(a3 + t);
    const __m256 wv0 = _mm256_loadu_ps(w0 + t);
    const __m256 wv1 = _mm256_loadu_ps(w1 + t);
    s00 = _mm256_fmadd_ps(av0, wv0, s00);
    s01 = _mm256_fmadd_ps(av0, wv1, s01);
    s10 = _mm256_fmadd_ps(av1, wv0, s10);
    s11 = _mm256_fmadd_ps(av1, wv1, s11);
    s20 = _mm256_fmadd_ps(av2, wv0, s20);
    s21 = _mm256_fmadd_ps(av2, wv1, s21);
    s30 = _mm256_fmadd_ps(av3, wv0, s30);
    s31 = _mm256_fmadd_ps(av3, wv1, s31);
  }
  float f00 = HSum(s00), f01 = HSum(s01);
  float f10 = HSum(s10), f11 = HSum(s11);
  float f20 = HSum(s20), f21 = HSum(s21);
  float f30 = HSum(s30), f31 = HSum(s31);
  for (; t < k; ++t) {  // k 不是 8 的倍数时的尾巴
    const float x0 = a0[t], x1 = a1[t], x2 = a2[t], x3 = a3[t];
    const float y0 = w0[t], y1 = w1[t];
    f00 += x0 * y0;
    f01 += x0 * y1;
    f10 += x1 * y0;
    f11 += x1 * y1;
    f20 += x2 * y0;
    f21 += x2 * y1;
    f30 += x3 * y0;
    f31 += x3 * y1;
  }
  *o00 += f00;
  *o01 += f01;
  *o10 += f10;
  *o11 += f11;
  *o20 += f20;
  *o21 += f21;
  *o30 += f30;
  *o31 += f31;
}

float DotAvx2(const float* a, const uint8_t* w, int64_t k, DType t) {
  switch (t) {
    case DType::kBF16: return DotBf16Avx2(a, w, k);
    case DType::kSFP: return DotSfpAvx2(a, w, k);
    case DType::kF32: return DotF32Avx2(a, w, k);
    default: return 0.0f;
  }
}

#endif  // TG_HAVE_AVX2

// f32 权重点积的默认派发。解码缓冲走这条路。
inline float DotF32VecDefault(const float* a, const float* w, int64_t k) {
#if TG_HAVE_AVX2
  return DotF32VecAvx2(a, w, k);
#else
  return DotF32VecRef(a, w, k);
#endif
}

// 2x2 微内核的默认派发。没有 AVX2 时退化成 4 次标量点积，
// 语义完全一致，只是没有寄存器复用的好处。
inline void GemmMicro2x2(const float* a0, const float* a1, const float* w0,
                         const float* w1, int64_t k, float* r00, float* r01,
                         float* r10, float* r11) {
#if TG_HAVE_AVX2
  Gemm2x2Avx2(a0, a1, w0, w1, k, r00, r01, r10, r11);
#else
  *r00 += DotF32VecRef(a0, w0, k);
  *r01 += DotF32VecRef(a0, w1, k);
  *r10 += DotF32VecRef(a1, w0, k);
  *r11 += DotF32VecRef(a1, w1, k);
#endif
}

// 4x2 微内核的默认派发。没有 AVX2 时退化成一堆标量点积。
inline void GemmMicro4x2(const float* a0, const float* a1, const float* a2,
                         const float* a3, const float* w0, const float* w1,
                         int64_t k, float* o00, float* o01, float* o10,
                         float* o11, float* o20, float* o21, float* o30,
                         float* o31) {
#if TG_HAVE_AVX2
  Gemm4x2Avx2(a0, a1, a2, a3, w0, w1, k, o00, o01, o10, o11, o20, o21, o30,
              o31);
#else
  *o00 += DotF32VecRef(a0, w0, k);
  *o01 += DotF32VecRef(a0, w1, k);
  *o10 += DotF32VecRef(a1, w0, k);
  *o11 += DotF32VecRef(a1, w1, k);
  *o20 += DotF32VecRef(a2, w0, k);
  *o21 += DotF32VecRef(a2, w1, k);
  *o30 += DotF32VecRef(a3, w0, k);
  *o31 += DotF32VecRef(a3, w1, k);
#endif
}

// 默认路径的派发点：有 AVX2 就用 AVX2，否则退回标量参考。
inline float DotDefault(const float* a, const uint8_t* w, int64_t k, DType t) {
#if TG_HAVE_AVX2
  return DotAvx2(a, w, k, t);
#else
  return DotRef(a, w, k, t);
#endif
}

template <bool kRef>
inline float DotDispatch(const float* a, const uint8_t* w, int64_t k, DType t) {
  return kRef ? DotRef(a, w, k, t) : DotDefault(a, w, k, t);
}

// ---- 性能调参旋钮 ----
//
// 三个参数都能用环境变量覆盖，方便不重编译就扫参。默认值是实测扫出来的：
//
//   TG_MBLOCK_KB  每块激活的 KB，决定权重被重读多少遍（默认 2048）
//                 m=301 的 gate 形状：512KB -> 368 GMAC/s，2048KB -> 404 GMAC/s
//   TG_KC         k 方向分块宽度，决定微内核工作集能否留在 L1（默认 1024）
//                 见 MatMulBTRowsImpl 里的推导
//   TG_USCHED     OpenMP 行对循环的调度块（默认 4），实测影响很小
//
// 这几个是给性能调优用的，正常跑不需要设。
int64_t ReadEnv(const char* name, int64_t def) {
  const char* e = std::getenv(name);
  if (e == nullptr) return def;
  const int64_t v = std::atoll(e);
  return v > 0 ? v : def;
}

int64_t MBlockKB() {
  static const int64_t v = ReadEnv("TG_MBLOCK_KB", 2048);
  return v;
}
int64_t KCValue() {
  static const int64_t v = ReadEnv("TG_KC", 1024);
  return v < 8 ? 8 : v;
}
int64_t USchedChunk() {
  static const int64_t v = ReadEnv("TG_USCHED", 4);
  return v;
}

template <bool kRef>
void MatMulBTRowsImpl(float* out, const float* a, int64_t m, int64_t k,
                      const Matrix& w, int64_t row_begin, int64_t row_end) {
  const int64_t n = row_end - row_begin;
  if (n <= 0) return;
  const int64_t elem = DTypeElementBytes(w.type);
  if (elem == 0) {
    std::memset(out, 0, static_cast<size_t>(m * n) * sizeof(float));
    return;
  }
  const uint8_t* base = w.data + row_begin * k * elem;

  if (m == 1) {
    // decode：按权重行并行。原来是按 (i,j) 展平再逐元素做 idx/n，
    // 那是一次 64 位整数除法／输出元素；改成两层循环后彻底没有除法。
    TG_OMP_FOR_ROWS
    for (int64_t j = 0; j < n; ++j) {
      out[j] = DotDispatch<kRef>(a, base + j * k * elem, k, w.type);
    }
    return;
  }

  // 参考实现：朴素双循环，解码内联在点积里。慢，但语义最直白，作为对拍基准。
  if (kRef) {
    TG_OMP_FOR_ROWS
    for (int64_t j = 0; j < n; ++j) {
      const uint8_t* wr = base + j * k * elem;
      for (int64_t i = 0; i < m; ++i) {
        out[i * n + j] = DotRef(a + i * k, wr, k, w.type);
      }
    }
    return;
  }

  // prefill 优化路径。原始写法"权重行在外、token 在内"有两个致命开销：
  //
  // 1) 解码被重复 m 遍。解码内联在 DotXxx 里，所以每换一个 token 就把同一行
  //    权重重新解一次。sfp 要 7 条位运算、bf16 要一次左移，乘上 m*n*k 就是
  //    每层几十亿次解码，而这些位运算和 FMA 抢的是同一批执行端口。
  // 2) 激活 a 被重复扫 n 遍。每读一行权重就要把全部 m 个 token 的 a 重扫，
  //    a 的流量是 n*m*k*4 字节 —— 256 token、9216 行时每层 21.7 GB，
  //    26 层共 565 GB，正好撞满内存带宽（实测 61.5 GB/s，机器上限 70）。
  //    权重流还会不停冲刷 L3（30 MB），把 a 挤出缓存逼回内存。
  //
  // 改法：按 token 分块，块内把一行权重**解码一次**存进 L1，再对块里所有
  // token 复用；块本身只有 MB*k*4 字节，常驻每核的私有 L2。代价是权重被
  // 重读 m/MB 遍，但那只有几十 MB 量级。
  const int64_t a_bytes_per_token = k * 4;
  // 2 MB 是实测的甜点：再小则权重被重读太多遍，再大则激活块挤爆 L3。
  // m=301 的 gate 形状：512KB -> 368 GMAC/s，2048KB -> 404 GMAC/s，3072KB -> 376。
  // 另加一道 4096 token 的上限：k 很小时按字节预算算出的块会大到让下面的
  // 累加缓冲失控，虽然实际形状不会走到那一步，但留个保险。
  const int64_t m_cap = std::min<int64_t>(m, 4096);
  const int64_t m_block = std::max<int64_t>(
      1, std::min<int64_t>(m_cap, (MBlockKB() * 1024) / a_bytes_per_token));
  const bool need_decode = (w.type != DType::kF32);
  // 行也成对处理：一行一份解码缓冲，两份共 2*k 个 f32（典型 18 KB，L1 装得下）。
  const int64_t n_row_units = (n + 1) / 2;

  // ---- k 方向的分块宽度 ----
  //
  // 上面把「解码一次」和「激活分块」解决了之后，剩下的瓶颈是**一级缓存放不下**。
  // 微内核一次要吃「4 个 token 的整条激活」+「2 行的整条权重」：k=2304 时是
  // 36 KB + 18 KB = 54 KB，超过 P 核 48 KB 的 L1D。后果是：从一行对扫到下一个
  // token 块时，权重行已经被激活流挤出 L1，只能回 L2 重读。而权重行正是最内层
  // 的复用点，这个代价要被放大 n/2 倍。
  //
  // 实测（m=301、n=9216、16 线程，只改 k 看工作集大小的影响）：
  //   k=512    12 KB 工作集  -> 464 GMAC/s
  //   k=1024   24 KB         -> 535 GMAC/s   <- 峰值
  //   k=2304   54 KB         -> 381 GMAC/s
  // 也就是说，只要让每个 k 分块的工作集留在 L1 里，就能拿回约 40%。
  //
  // 做法：在行对内部再把 k 切成 kc 段。这样「同一段权重再次被用到」之间，
  // 只流过 kc*4 字节/行的激活，权重段（2*kc*4 字节）稳稳留在 L1。
  // 代价是输出要多一次累加缓冲中转 —— 每个行对先把整块 token 的结果攒在
  // accbuf 里，k 分块跑完再写回 out。accbuf 只有 2*m_block 个 float，
  // 而且是从「按 n 步长散写 out」变成「连续写 accbuf」，实际不亏。
  const int64_t kc = std::max<int64_t>(8, std::min<int64_t>(k, KCValue()));

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<float> wbuf(need_decode ? static_cast<size_t>(k) * 2 : 0);
    std::vector<float> accbuf(static_cast<size_t>(2 * m_block));

    // 取第 j 行的 f32 权重：非 f32 类型先解码到本线程的缓冲里。
    const auto row_ptr = [&](int64_t j, int64_t slot) -> const float* {
      const uint8_t* wr = base + j * k * elem;
      if (!need_decode) return reinterpret_cast<const float*>(wr);
      float* dst = wbuf.data() + static_cast<size_t>(slot) * k;
      DecodeRowToF32(wr, k, w.type, dst);
      return dst;
    };

    for (int64_t i0 = 0; i0 < m; i0 += m_block) {
      const int64_t i_end = std::min(m, i0 + m_block);
      const int64_t cnt = i_end - i0;
#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#pragma omp for schedule(dynamic, USchedChunk())
#endif
      for (int64_t u = 0; u < n_row_units; ++u) {
        const int64_t j0 = u * 2;
        const bool has_pair = (j0 + 1 < n);
        const float* w0 = row_ptr(j0, 0);
        const float* w1 = has_pair ? row_ptr(j0 + 1, 1) : nullptr;

        float* acc = accbuf.data();
        std::fill(acc, acc + 2 * cnt, 0.0f);

        for (int64_t kh = 0; kh < k; kh += kc) {
          const int64_t kk = std::min(kc, k - kh);
          int64_t i = i0;
          if (has_pair) {
            // 主路径：一次吃 4 个 token（8 条独立累加链，见 Gemm4x2Avx2）。
            for (; i + 3 < i_end; i += 4) {
              const int64_t o = (i - i0) * 2;
              GemmMicro4x2(a + i * k + kh, a + (i + 1) * k + kh,
                           a + (i + 2) * k + kh, a + (i + 3) * k + kh,
                           w0 + kh, w1 + kh, kk, acc + o, acc + o + 1,
                           acc + o + 2, acc + o + 3, acc + o + 4, acc + o + 5,
                           acc + o + 6, acc + o + 7);
            }
            for (; i + 1 < i_end; i += 2) {
              const int64_t o = (i - i0) * 2;
              GemmMicro2x2(a + i * k + kh, a + (i + 1) * k + kh, w0 + kh,
                           w1 + kh, kk, acc + o, acc + o + 1, acc + o + 2,
                           acc + o + 3);
            }
            if (i < i_end) {  // token 数的尾巴（1~3 个）
              const int64_t o = (i - i0) * 2;
              acc[o] += DotF32VecDefault(a + i * k + kh, w0 + kh, kk);
              acc[o + 1] += DotF32VecDefault(a + i * k + kh, w1 + kh, kk);
            }
          } else {
            for (; i < i_end; ++i) {
              acc[(i - i0) * 2] +=
                  DotF32VecDefault(a + i * k + kh, w0 + kh, kk);
            }
          }
        }

        // 写回：out 是行主序，每个 token 的 (j0, j0+1) 相邻。
        for (int64_t i = i0; i < i_end; ++i) {
          const int64_t o = (i - i0) * 2;
          out[i * n + j0] = acc[o];
          if (has_pair) out[i * n + j0 + 1] = acc[o + 1];
        }
      }
    }
  }
}

}  // namespace

float DotRow(const float* a, const Matrix& w, int64_t row, int64_t k) {
  const uint8_t* p = w.data + row * k * DTypeElementBytes(w.type);
  return DotDefault(a, p, k, w.type);
}

void MatMulBTRows(float* out, const float* a, int64_t m, int64_t k,
                  const Matrix& w, int64_t row_begin, int64_t row_end) {
  MatMulBTRowsImpl<false>(out, a, m, k, w, row_begin, row_end);
}

void MatMulBT(float* out, const float* a, int64_t m, int64_t k, const Matrix& w) {
  MatMulBTRows(out, a, m, k, w, 0, w.rows);
}

float DotRowRef(const float* a, const Matrix& w, int64_t row, int64_t k) {
  const uint8_t* p = w.data + row * k * DTypeElementBytes(w.type);
  return DotRef(a, p, k, w.type);
}

void MatMulBTRowsRef(float* out, const float* a, int64_t m, int64_t k,
                     const Matrix& w, int64_t row_begin, int64_t row_end) {
  MatMulBTRowsImpl<true>(out, a, m, k, w, row_begin, row_end);
}

void MatMulBTRef(float* out, const float* a, int64_t m, int64_t k,
                 const Matrix& w) {
  MatMulBTRowsRef(out, a, m, k, w, 0, w.rows);
}

void AttEinSum(float* out, const float* a, int64_t m, int64_t a_stride,
               int64_t heads, int64_t head_dim, int64_t out_dim,
               const Matrix& w) {
  const int64_t elem = DTypeElementBytes(w.type);
  if (elem == 0 || out_dim <= 0 || heads <= 0 || head_dim <= 0) return;

  // w 的布局是 [heads][out_dim][head_dim] 连续。
  const int64_t head_bytes = out_dim * head_dim * elem;
  const bool need_decode = (w.type != DType::kF32);

  // decode（m=1）走融合路径：每个权重元素只用一次，先解码到 f32 再点乘
  // 等于白白多绕一趟，而且那趟解码是标量 LUT，比向量化的融合解码慢。
  // 只有 m > 1 时"解码一次、多 token 复用"才划算。
  if (m <= 1) {
    constexpr int64_t kMBlock = 32;
#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t mb = 0; mb < out_dim; mb += kMBlock) {
      const int64_t m_end = std::min(out_dim, mb + kMBlock);
      const int64_t bw = m_end - mb;
      float acc[kMBlock];
      for (int64_t j = 0; j < bw; ++j) acc[j] = 0.0f;
      for (int64_t h = 0; h < heads; ++h) {
        const uint8_t* base = w.data + h * head_bytes + mb * head_dim * elem;
        const float* ah = a + h * head_dim;
        for (int64_t j = 0; j < bw; ++j) {
          acc[j] += DotDefault(ah, base + j * head_dim * elem, head_dim, w.type);
        }
      }
      float* orow = out + mb;
      for (int64_t j = 0; j < bw; ++j) orow[j] = acc[j];
    }
    return;
  }

  // 行块宽度取 8：
  //   1) 解码缓冲 = heads*bw*head_dim*4 字节，8 头 8 行是 64 KB，带得动；
  //   2) out_dim/bw = 2304/8 = 288 个块，并行度够。
  // 原来的 32 会让缓冲涨到 256 KB，反而把热点数据挤出去。
  constexpr int64_t kBW = 8;

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#pragma omp parallel
#endif
  {
    // 每线程一份。放在并行区外，避免 288 个行块各分配一次。
    std::vector<float> wbuf;
    std::vector<float> acc;

#if defined(TG_HAVE_OPENMP) || defined(_OPENMP)
#pragma omp for schedule(dynamic, 1)
#endif
    for (int64_t mb = 0; mb < out_dim; mb += kBW) {
      const int64_t m_end = std::min(out_dim, mb + kBW);
      const int64_t bw = m_end - mb;

      // 把本块所有 head 的权重**解码一次**。原实现把解码内联在点积里，
      // 于是同一行权重被解了 m 遍 —— m=301 时每层 14 亿次，纯浪费。
      if (need_decode) {
        wbuf.resize(static_cast<size_t>(heads) * static_cast<size_t>(bw) *
                    static_cast<size_t>(head_dim));
        for (int64_t h = 0; h < heads; ++h) {
          for (int64_t j = 0; j < bw; ++j) {
            DecodeRowToF32(w.data + h * head_bytes + (mb + j) * head_dim * elem,
                           head_dim, w.type,
                           wbuf.data() + (h * bw + j) * head_dim);
          }
        }
      }
      // 取 (head h, 块内第 j 行) 的 f32 权重。
      const auto wrow = [&](int64_t h, int64_t j) -> const float* {
        if (need_decode) return wbuf.data() + (h * bw + j) * head_dim;
        return reinterpret_cast<const float*>(
            w.data + h * head_bytes + (mb + j) * head_dim * elem);
      };

      acc.assign(static_cast<size_t>(m) * static_cast<size_t>(bw), 0.0f);
      for (int64_t h = 0; h < heads; ++h) {
        int64_t i = 0;
        // 2 token × 2 行：4 次加载喂 4 次 FMA，省掉一半加载端口。
        for (; i + 1 < m; i += 2) {
          const float* a0 = a + i * a_stride + h * head_dim;
          const float* a1 = a + (i + 1) * a_stride + h * head_dim;
          float* r0 = acc.data() + i * bw;
          float* r1 = r0 + bw;
          int64_t j = 0;
          for (; j + 1 < bw; j += 2) {
            // 微内核本身就是累加语义，直接写进 acc。
            GemmMicro2x2(a0, a1, wrow(h, j), wrow(h, j + 1), head_dim, r0 + j,
                         r0 + j + 1, r1 + j, r1 + j + 1);
          }
          for (; j < bw; ++j) {
            r0[j] += DotF32VecDefault(a0, wrow(h, j), head_dim);
            r1[j] += DotF32VecDefault(a1, wrow(h, j), head_dim);
          }
        }
        if (i < m) {  // token 数为奇数时的尾巴
          const float* a0 = a + i * a_stride + h * head_dim;
          float* r0 = acc.data() + i * bw;
          for (int64_t j = 0; j < bw; ++j) {
            r0[j] += DotF32VecDefault(a0, wrow(h, j), head_dim);
          }
        }
      }
      // 写回：out 行主序，每个 token 是连续的一段 j。
      for (int64_t i = 0; i < m; ++i) {
        std::memcpy(out + i * out_dim + mb, acc.data() + i * bw,
                    static_cast<size_t>(bw) * sizeof(float));
      }
    }
  }
}

}  // namespace tg
