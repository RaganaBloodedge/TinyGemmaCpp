// TinyGemmaCpp —— KV Cache
//
// 没有 KV Cache 时，生成第 t 个 token 要把前面 t-1 个位置全部重算一遍，
// 总计算量 O(T^2)；有了它每次只需要算新 token 并复用历史 K/V，总计算量 O(T)。
//
// 布局：[layer][pos][kv_head][dim]，K 和 V 分开存，避免下标的奇偶判断。

#ifndef TINYGEMMA_KV_CACHE_H_
#define TINYGEMMA_KV_CACHE_H_

#include <cstdint>
#include <vector>

namespace tg {

class KVCache {
 public:
  void Init(int64_t layers, int64_t seq, int64_t kv_heads, int64_t dim) {
    layers_ = layers;
    seq_ = seq;
    kv_heads_ = kv_heads;
    dim_ = dim;
    const size_t n = static_cast<size_t>(layers * seq * kv_heads * dim);
    k_.assign(n, 0.0f);
    v_.assign(n, 0.0f);
  }

  float* KBase(int64_t layer, int64_t pos) {
    return k_.data() + ((layer * seq_ + pos) * kv_heads_) * dim_;
  }
  float* VBase(int64_t layer, int64_t pos) {
    return v_.data() + ((layer * seq_ + pos) * kv_heads_) * dim_;
  }
  const float* KHead(int64_t layer, int64_t pos, int64_t head) const {
    return k_.data() + ((layer * seq_ + pos) * kv_heads_ + head) * dim_;
  }
  const float* VHead(int64_t layer, int64_t pos, int64_t head) const {
    return v_.data() + ((layer * seq_ + pos) * kv_heads_ + head) * dim_;
  }

  int64_t seq() const { return seq_; }
  size_t bytes() const { return (k_.size() + v_.size()) * sizeof(float); }

 private:
  int64_t layers_ = 0, seq_ = 0, kv_heads_ = 0, dim_ = 0;
  std::vector<float> k_, v_;
};

}  // namespace tg

#endif  // TINYGEMMA_KV_CACHE_H_
