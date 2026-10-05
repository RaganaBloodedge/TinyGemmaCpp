// TinyGemmaCpp —— 模型配置
//
// 引擎不硬编码任何模型常量，全部走 ModelConfig。目前只提供 Gemma 2 2B，
// 但同一份 forward 代码可以跑更小/更大的同架构配置（对拍用的调试档就靠这个）。

#ifndef TINYGEMMA_CONFIG_H_
#define TINYGEMMA_CONFIG_H_

#include <cmath>
#include <cstdint>
#include <string>

namespace tg {

// 单层配置。
struct LayerConfig {
  int64_t model_dim = 2304;
  int64_t ff_hidden_dim = 9216;
  int64_t heads = 8;      // query 头数
  int64_t kv_heads = 4;   // key/value 头数（GQA）
  int64_t qkv_dim = 256;  // 每个头的维度

  // q 投影占 heads 组，kv 投影按 (k,v) 交错占 2*kv_heads 组。
  int64_t QkvRows() const { return (heads + 2 * kv_heads) * qkv_dim; }
  int64_t QRows() const { return heads * qkv_dim; }
  int64_t KvRows() const { return 2 * kv_heads * qkv_dim; }
  // GQA：连续 QHeadGroup 个 query 头共享一组 kv。
  int64_t QHeadGroup() const { return heads / kv_heads; }
};

struct ModelConfig {
  LayerConfig layer;
  int64_t num_layers = 26;
  int64_t vocab_size = 256000;
  int64_t max_seq_len = 8192;
  int64_t sliding_window = 4096;  // 偶数层用局部窗口，奇数层全局
  float rms_eps = 1e-6f;

  int64_t eos_id = 1;
  int64_t secondary_eos_id = 107;

  // 偶数层局部窗口，奇数层全局（gemma.cpp: RepeatedAttentionWindowSizes<26,2>({4096, 8192})）
  int64_t WindowOf(int64_t layer) const {
    return (layer % 2 == 0) ? sliding_window : max_seq_len;
  }
  bool IsGlobalLayer(int64_t layer) const { return WindowOf(layer) >= max_seq_len; }

  // QueryScaleType::SqrtKeySize -> 1/sqrt(256) = 1/16
  float QueryScale() const {
    return 1.0f / std::sqrt(static_cast<float>(layer.qkv_dim));
  }

  // 词向量缩放 = bf16(sqrt(model_dim))。gemma.cpp 先把 sqrt 舍到 bf16 再乘，
  // 这里必须照做才能逐位对齐：bf16(sqrt(2304)) == 48.0f，恰好是整数。
  float EmbeddingScale() const;

  static ModelConfig Gemma2_2B();
  std::string ToString() const;
};

}  // namespace tg

#endif  // TINYGEMMA_CONFIG_H_
