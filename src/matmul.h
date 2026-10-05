// TinyGemmaCpp —— GEMM
//
// 约定：权重永远是 [out_features, in_features] 行主序，
// 所以 out = x @ W^T，即 out[i][j] = Σ_k x[i][k] * W[j][k]。
// 这与 gemma.cpp 的 CallMatMul(x, w, ..., out) 语义一致。
//
// 本文件同时提供两条路径：
//   * *Ref  —— 标量参考实现。不做任何平台假设，是正确性基准。
//   * 默认 —— 编译期探测 AVX2/FMA 后启用 SIMD 内核。
// 两者必须给出（近）一致的结果，test_matmul 会逐用例比对。
//
// decode（m=1）的内存模型：权重每个元素一个字节/两字节，必须全部从内存搬到
// 核里，所以性能上限就是访存带宽。优化目标不是"算得更快"，而是"每字节权重
// 少花指令"：让 FMA 端口和 L1 查表不再是瓶颈。

#ifndef TINYGEMMA_MATMUL_H_
#define TINYGEMMA_MATMUL_H_

#include <cstdint>

#include "dtypes.h"
#include "weights.h"

namespace tg {

// 点积：a[0..k) 与 w 的第 row 行。权重按自身 DType 解码。
float DotRow(const float* a, const Matrix& w, int64_t row, int64_t k);

// out[m, n] = a[m, k] @ w[n, k]^T。OpenMP 存在时按行并行。
void MatMulBT(float* out, const float* a, int64_t m, int64_t k, const Matrix& w);

// 与 MatMulBT 相同，但权重只取 [row_begin, row_end) 这些行（用于 gate/up 拆分，
// 避免为了切开 gating_ein 而复制 42 MB 权重）。
void MatMulBTRows(float* out, const float* a, int64_t m, int64_t k,
                  const Matrix& w, int64_t row_begin, int64_t row_end);

// ---- 标量参考实现（正确性基准，GPU 无关，无 SIMD 假设）----
float DotRowRef(const float* a, const Matrix& w, int64_t row, int64_t k);
void MatMulBTRef(float* out, const float* a, int64_t m, int64_t k,
                 const Matrix& w);
void MatMulBTRowsRef(float* out, const float* a, int64_t m, int64_t k,
                     const Matrix& w, int64_t row_begin, int64_t row_end);

// ---- att_ein（o_proj）专用收缩 ----
//
//   out[t][j] = Σ_h Σ_d a[t*a_stride + h*head_dim + d] * w[h][j][d]
//
// 权重布局是 [heads][out_dim][head_dim] 连续。按 (j, h) 顺序索引会让相邻 head
// 之间跳 out_dim*head_dim 字节（Gemma 2B 上约 590 KB），硬件预取器和 TLB 都跟不上
// —— 实测这条路径只跑到 2.4 GB/s，占总 decode 时间的 42%。
//
// 所以这里按 out_dim 分块：每个线程拿一块连续的 j，块内按 (h, j) 遍历，
// 每次读到的都是 kMBlock*head_dim 字节的连续段。
void AttEinSum(float* out, const float* a, int64_t m, int64_t a_stride,
               int64_t heads, int64_t head_dim, int64_t out_dim, const Matrix& w);

}  // namespace tg

#endif  // TINYGEMMA_MATMUL_H_
