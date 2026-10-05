// TinyGemmaCpp —— 权重容器
//
// 权重不复制、不解码到 f32：Matrix 只是 SbsFile 大缓冲上的一个视图 + 存储类型。
// 只有归一化权重（每个才 2304 个元素）会在加载时展开成 f32 以便算子直接用。

#ifndef TINYGEMMA_WEIGHTS_H_
#define TINYGEMMA_WEIGHTS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "dtypes.h"
#include "sbs_reader.h"

namespace tg {

// 行主序矩阵视图：w[j][k]，j in [0, rows)，k in [0, cols)。
struct Matrix {
  const uint8_t* data = nullptr;
  DType type = DType::kUnknown;
  int64_t rows = 0;
  int64_t cols = 0;

  bool Valid() const { return data != nullptr; }
  float At(int64_t j, int64_t k) const {
    return LoadAsF32(type, data, j * cols + k);
  }
};

struct LayerWeights {
  Matrix qkv;     // [(heads + 2*kv_heads)*qkv_dim, model_dim]
  Matrix att;     // [heads, model_dim, qkv_dim]（按 head 分块的行主序）
  Matrix gating;  // [2*ff_hidden_dim, model_dim]：上半 gate，下半 up
  Matrix linear;  // [model_dim, ff_hidden_dim]

  std::vector<float> pre_att_norm;   // [model_dim]
  std::vector<float> post_att_norm;  // [model_dim]
  std::vector<float> pre_ff_norm;    // [model_dim]
  std::vector<float> post_ff_norm;   // [model_dim]
};

struct Weights {
  Matrix embed;                    // [vocab_size, model_dim]
  std::vector<float> final_norm;   // [model_dim]
  std::vector<LayerWeights> layers;

  // 统计信息，加载后打印用于自查。
  struct Stats {
    int64_t num_blobs = 0;
    int64_t bytes_f32 = 0, bytes_bf16 = 0, bytes_sfp = 0;
    int64_t elements = 0;
  } stats;

  // 从已打开的 .sbs 构建所有权重视图。失败返回 false。
  bool Build(const SbsFile& file, const ModelConfig& config, std::string* err);

  int64_t ResidentBytes() const {
    return stats.bytes_f32 + stats.bytes_bf16 + stats.bytes_sfp;
  }
};

}  // namespace tg

#endif  // TINYGEMMA_WEIGHTS_H_
