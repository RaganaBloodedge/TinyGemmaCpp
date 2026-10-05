// 算子单元测试：不依赖权重文件，只验算 Gemma 特有的那几条约定。
// 编译：cmake --build build && ./build/test_ops

#include <cmath>
#include <cstdio>
#include <vector>

#include "ops.h"

namespace {

int g_failed = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    ++g_failed;
    std::printf("  [FAIL] %s\n", what);
  } else {
    std::printf("  [ ok ] %s\n", what);
  }
}

void CheckNear(float got, float want, float tol, const char* what) {
  const bool ok = std::fabs(got - want) <= tol;
  if (!ok) {
    ++g_failed;
    std::printf("  [FAIL] %s: got %.8g want %.8g\n", what, got, want);
  } else {
    std::printf("  [ ok ] %s (%.8g)\n", what, got);
  }
}

void TestRMSNorm() {
  std::printf("RMSNorm\n");
  const int64_t n = 4;
  const float x[4] = {1.f, 2.f, 3.f, 4.f};

  // mean(x^2) = (1+4+9+16)/4 = 7.5；mul = 1/sqrt(7.5+1e-6)
  const float mul = 1.0f / std::sqrt(7.5f + 1e-6f);

  {  // w = 0 -> 输出 = mul * x（系数是 1+0）
    const float w[4] = {0.f, 0.f, 0.f, 0.f};
    float out[4];
    tg::RMSNorm(x, n, w, 1e-6f, out);
    CheckNear(out[2], mul * 3.f, 1e-6f, "w=0 时系数为 1");
  }
  {  // w = 1 -> 输出 = 2 * mul * x。这是 (1+w) 约定最直接的检验：
     // 如果实现写成 w 而不是 (1+w)，w=1 会得到与 w=0 完全相同的结果。
    const float w[4] = {1.f, 1.f, 1.f, 1.f};
    float out[4];
    tg::RMSNorm(x, n, w, 1e-6f, out);
    CheckNear(out[2], 2.f * mul * 3.f, 1e-6f, "(1+w) 而不是 w");
  }
  {  // 全零输入不应产生 nan
    const float z[4] = {0.f, 0.f, 0.f, 0.f};
    const float w[4] = {1.f, 1.f, 1.f, 1.f};
    float out[4];
    tg::RMSNorm(z, n, w, 1e-6f, out);
    Check(std::isfinite(out[0]), "零输入不产生 NaN");
  }
}

void TestRope() {
  std::printf("RoPE\n");
  const int64_t qkv_dim = 8;
  const int64_t half = qkv_dim / 2;
  auto inv = tg::MakeInvTimescale(qkv_dim, 10000.0);
  tg::RopeTable table;
  table.Init(inv, 8);

  {  // pos=0 -> cos=1, sin=0 -> 恒等
    float x[8];
    for (int i = 0; i < 8; ++i) x[i] = 0.1f * (i + 1);
    float ref[8];
    for (int i = 0; i < 8; ++i) ref[i] = x[i];
    tg::RopeInplace(x, qkv_dim, table, 0, 1.0f);
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok = ok && std::fabs(x[i] - ref[i]) < 1e-7f;
    Check(ok, "pos=0 是恒等变换");
  }
  {  // 配对的必须是 (d, d+half)，不是 (2d, 2d+1)。
     // inv[0] = 1，所以 d=0 那对的旋转角就是 pos。
    const int64_t pos = 3;
    float x[8] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tg::RopeInplace(x, qkv_dim, table, pos, 1.0f);
    const float theta = static_cast<float>(pos) * inv[0];
    CheckNear(x[0], std::cos(theta), 1e-6f, "x[0] = x[0]*cos (pair = 0 与 4)");
    CheckNear(x[half], std::sin(theta), 1e-6f, "x[4] = x[0]*sin");
    // 若误用相邻配对 (2d, 2d+1)，x[4] 会保持不变。
    Check(std::fabs(x[4]) > 1e-6f, "相邻配对写法在此会失败");
  }
  {  // 旋转保持每对的模长，乘 mul 则整体缩放
    const int64_t pos = 5;
    float x[8] = {0.5f, -1.5f, 2.f, 0.25f, 1.f, -0.75f, 0.125f, 3.f};
    float y[8];
    for (int i = 0; i < 8; ++i) y[i] = x[i];
    tg::RopeInplace(y, qkv_dim, table, pos, 1.0f);
    bool ok = true;
    for (int64_t d = 0; d < half; ++d) {
      const float n0 = x[d] * x[d] + x[d + half] * x[d + half];
      const float n1 = y[d] * y[d] + y[d + half] * y[d + half];
      ok = ok && std::fabs(n0 - n1) < 1e-5f;
    }
    Check(ok, "旋转保持模长");

    float z[8];
    for (int i = 0; i < 8; ++i) z[i] = x[i];
    tg::RopeInplace(z, qkv_dim, table, pos, 0.0625f);
    CheckNear(z[0], y[0] * 0.0625f, 1e-6f, "mul 参数同时做 query scale");
  }
}

void TestSoftCap() {
  std::printf("soft-cap\n");
  float x[3] = {0.f, 30.f, 1000.f};
  tg::SoftCapInplace(x, 3, 30.0f);
  CheckNear(x[0], 0.f, 1e-7f, "cap(0) = 0");
  CheckNear(x[1], 30.f * std::tanh(1.f), 1e-4f, "cap(30) = 30*tanh(1)");
  Check(x[2] <= 30.0f && x[2] > 29.9f, "cap 把极端 logit 压到 30 以内");
}

void TestGelu() {
  std::printf("GeGLU\n");
  float c1[3] = {-1.f, 0.f, 1.f};
  const float c2[3] = {1.f, 1.f, 1.f};
  tg::GeluMulInplace(c1, c2, 3);
  CheckNear(c1[1], 0.f, 1e-7f, "gelu(0) = 0");
  // gelu_tanh(1) = 0.5*1*(1+tanh(0.7978845608*(1+0.044715))) = 0.84119199
  CheckNear(c1[2], 0.84119199f, 1e-5f, "gelu_tanh(1) = 0.841192（不是 erf 版 0.841345）");
  CheckNear(c1[0], -0.15880796f, 1e-5f, "gelu_tanh(-1)");
}

// 向量化 gelu 把 tanh 换成了 sigmoid 的形式，再展开成 exp + 除法。
// 定点值的那两条用例覆盖不到它在中段的偏差，这里扫一遍取值范围，
// 拿双精度算的 tanh 当参考，把最大绝对误差量出来。
//
// 为什么可以接受近似值：gelu 的输出会被后续的 down 投影线性组合，
// 误差不会放大；而端到端的 token 序列与官方逐 token 一致（tools/run.py），
// 说明这点偏差不影响 argmax。
void TestGeluSweep() {
  std::printf("gelu_tanh 向量化近似的精度\n");
  constexpr float kMul = 0.03567740813636141f;
  constexpr float kSqrt2OverPi = 0.797884560804236f;
  constexpr int kSteps = 1601;  // v 从 -20 到 20，步长 0.025
  constexpr float kStep = 0.025f;

  std::vector<float> c1(static_cast<size_t>(kSteps), 0.0f);
  std::vector<float> c2(static_cast<size_t>(kSteps), 1.0f);
  std::vector<double> want(static_cast<size_t>(kSteps), 0.0);

  for (int i = 0; i < kSteps; ++i) {
    const float v = (static_cast<float>(i) - kSteps / 2) * kStep;
    c1[static_cast<size_t>(i)] = v;
    const double arg =
        static_cast<double>(v) *
        (static_cast<double>(kSqrt2OverPi) +
         static_cast<double>(kMul) * static_cast<double>(v) * v);
    want[static_cast<size_t>(i)] = 0.5 * static_cast<double>(v) * (1.0 + std::tanh(arg));
  }

  tg::GeluMulInplace(c1.data(), c2.data(), kSteps);

  double max_abs = 0.0;
  float worst_v = 0.0f;
  for (int i = 0; i < kSteps; ++i) {
    const double d = std::fabs(static_cast<double>(c1[static_cast<size_t>(i)]) -
                               want[static_cast<size_t>(i)]);
    if (d > max_abs) {
      max_abs = d;
      worst_v = (static_cast<float>(i) - kSteps / 2) * kStep;
    }
  }
  std::printf("  最大绝对误差 %.3e（在 v≈%.2f 处）；|gelu| 在 v=20 时约 20\n",
              max_abs, static_cast<double>(worst_v));
  Check(max_abs < 1e-4, "向量化 gelu 最大绝对误差 < 1e-4");
}

void TestSoftmax() {
  std::printf("softmax\n");
  float x[4] = {1.f, 2.f, 3.f, 4.f};
  tg::SoftmaxInplace(x, 4);
  float s = 0.f;
  for (float v : x) s += v;
  CheckNear(s, 1.f, 1e-5f, "概率和为 1");
  Check(x[3] > x[2] && x[2] > x[1] && x[1] > x[0], "保持单调");
}

}  // namespace

int main() {
  TestRMSNorm();
  TestRope();
  TestSoftCap();
  TestGelu();
  TestGeluSweep();
  TestSoftmax();
  if (g_failed == 0) {
    std::printf("\n全部通过\n");
    return 0;
  }
  std::printf("\n%d 项失败\n", g_failed);
  return 1;
}
