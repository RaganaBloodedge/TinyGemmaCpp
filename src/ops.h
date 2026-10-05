// TinyGemmaCpp —— 逐元素算子
//
// 这里的每一条都对应 Gemma 2 架构里一个容易写错的细节，注释里标注了依据。

#ifndef TINYGEMMA_OPS_H_
#define TINYGEMMA_OPS_H_

#include <cstdint>
#include <vector>

namespace tg {

// RMSNorm：out = x * rsqrt(mean(x^2) + eps) * (1 + w)
//
// ⚠️ 是 (1 + w) 而不是 w。依据 gemma.cpp ops/ops-inl.h:236-238
//   `const VF m = Mul(mul, vx); return MulAdd(m, vw, m);`  // (1+w)*m
// 如果按教材写法漏掉 1，输出会退化成噪声。
void RMSNorm(const float* x, int64_t n, const float* w, float eps, float* out);

// 预计算 RoPE 的 cos/sin 表，避免每个 (layer, token, dim) 重算三角函数。
// 表按需扩展：只算到实际用到的最大位置。
struct RopeTable {
  int64_t half = 0;   // qkv_dim / 2
  int64_t built = 0;  // 已覆盖的位置数
  std::vector<float> inv_timescale;
  std::vector<float> cos_t;  // [positions, half]，行主序
  std::vector<float> sin_t;

  void Init(std::vector<float> inv_timescale, int64_t initial_positions);
  void Ensure(int64_t pos);
};

// 构造 inv_timescale[d] = 1 / base^(2d/qkv_dim)，d in [0, qkv_dim/2)。
std::vector<float> MakeInvTimescale(int64_t qkv_dim, double base_frequency = 10000.0);

// RoPE，就地旋转，并乘上 mul。
//
// ⚠️ Gemma 的配对被改了：旋转的是 (d, d + half) 这一对，不是相邻的 (2d, 2d+1)。
// 依据 ops/ops-inl.h 的注释 "in the Gemma implementation we choose to rotate
// the pairs of dimensions v_{i} and v_{i + d//2} instead."
//
//   x[d]        = mul * (x[d]*cos - x[d+half]*sin)
//   x[d+half]   = mul * (x[d]*sin + x[d+half]*cos)
void RopeInplace(float* x, int64_t qkv_dim, const RopeTable& table, int64_t pos,
                 float mul);

// GeGLU 的激活部分：c1[i] = gelu_tanh(c1[i]) * c2[i]
//
// ⚠️ 用 tanh 近似，不是 erf。对应 HF 的 gelu_pytorch_tanh。
void GeluMulInplace(float* c1, const float* c2, int64_t n);

// Logits soft-cap：x = cap * tanh(x / cap)
//
// ⚠️ Gemma 2 有两处：注意力 logits 用 50.0，最终 logits 用 30.0。
void SoftCapInplace(float* x, int64_t n, float cap);

// 数值稳定的 softmax（减最大值再 exp）。
void SoftmaxInplace(float* x, int64_t n);

}  // namespace tg

#endif  // TINYGEMMA_OPS_H_
