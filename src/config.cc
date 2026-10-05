#include "config.h"

#include <sstream>

#include "dtypes.h"

namespace tg {

float ModelConfig::EmbeddingScale() const {
  // gemma.cpp 的 EmbeddingScaling()：先算 sqrtf，再舍入到 bf16，再还原成 float。
  // bf16(sqrt(2304)) == 48.0f，正好是整数；但对其它 model_dim 不能省这一步。
  return Bf16ToF32(F32ToBf16(std::sqrt(static_cast<float>(layer.model_dim))));
}

ModelConfig ModelConfig::Gemma2_2B() {
  ModelConfig c;
  c.layer.model_dim = 2304;
  c.layer.ff_hidden_dim = 8 * 2304 / 2;  // 9216
  c.layer.heads = 8;
  c.layer.kv_heads = 4;
  c.layer.qkv_dim = 256;
  c.num_layers = 26;
  c.vocab_size = 256000;
  c.max_seq_len = 8192;
  c.sliding_window = 4096;
  c.rms_eps = 1e-6f;
  c.eos_id = 1;
  c.secondary_eos_id = 107;
  return c;
}

std::string ModelConfig::ToString() const {
  std::ostringstream os;
  os << "ModelConfig{layers=" << num_layers << ", model_dim=" << layer.model_dim
     << ", ff_hidden_dim=" << layer.ff_hidden_dim << ", heads=" << layer.heads
     << ", kv_heads=" << layer.kv_heads << ", qkv_dim=" << layer.qkv_dim
     << ", vocab=" << vocab_size << ", max_seq=" << max_seq_len
     << ", window=" << sliding_window << ", query_scale=" << QueryScale()
     << ", emb_scale=" << EmbeddingScale() << "}";
  return os.str();
}

}  // namespace tg
