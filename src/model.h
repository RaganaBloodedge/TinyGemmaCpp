// TinyGemmaCpp —— 前向传播与生成

#ifndef TINYGEMMA_MODEL_H_
#define TINYGEMMA_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "kv_cache.h"
#include "matmul.h"
#include "ops.h"
#include "sbs_reader.h"
#include "weights.h"

namespace tg {

struct ForwardStats {
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  int64_t prefill_tokens = 0;
  int64_t decoded_tokens = 0;
};

// 分阶段计时。默认关闭（计时器本身有 steady_clock 的固定开销，
// 每个 token 会多几百次 now()，不开时性能数据才干净）。
//
// prefill 与 decode 分开记账：两者的 batch 形状差一个数量级（prefill 一次
// 吃下整个 prompt，decode 每次只 1 个 token），混在一起平均会互相稀释，
// 两边的瓶颈都看不出来。两段都按"每个 token 平均毫秒"呈现，但有各自的
// 分母，所以可以直接横向比较。
struct PhaseProfile {
  bool on = false;

  double embed_ms = 0;    // 词嵌入查表 + 缩放
  double norm_ms = 0;     // 全部 RMSNorm（每层 4 次）
  double qkv_ms = 0;      // qkv 投影
  double rope_ms = 0;     // RoPE + 写 KV Cache
  double attn_ms = 0;     // Q·K / softmax / 加权 V
  double att_ein_ms = 0;  // 各头输出拼接后过 o_proj
  double ff_up_ms = 0;    // gate / up 投影
  double gelu_ms = 0;     // GeGLU 激活
  double ff_down_ms = 0;  // down 投影
  double logits_ms = 0;   // final norm + logits 投影 + soft-cap

  double total_ms = 0;    // 整个 Forward 的墙钟
  double sync_ms = 0;     // total 减去上面各段，即 OpenMP fork/join 与循环调度开销
  int64_t tokens = 0;     // 累计处理的 token 数（不是 Forward 次数）
  int64_t calls = 0;      // 累计 Forward 次数

  void Reset() {
    const bool was = on;
    *this = PhaseProfile{};
    on = was;
  }
  void Report(const char* title) const;  // 打印到 stderr
};

class Model {
 public:
  Model() = default;
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // 打开 .sbs、校验全部 210 个张量、建立视图。
  bool Load(const std::string& sbs_path, const ModelConfig& config,
            std::string* err);

  const ModelConfig& config() const { return config_; }
  const Weights& weights() const { return weights_; }
  const Weights::Stats& stats() const { return weights_.stats; }

  // 输入 tokens 放在 [start_pos, start_pos + tokens.size()) 位置，
  // 把它们的 K/V 写入 kv，并返回**最后一个 token** 的 logits（length = vocab）。
  //
  // v0 只算最后一个 token 的 logits：prefill 阶段前面的 token 的输出没人看，
  // 而 vocab 是 256000，对每个 prefill token 都算一遍要多花几十倍时间。
  void Forward(const std::vector<int32_t>& tokens, int64_t start_pos,
               KVCache& kv, std::vector<float>* logits);

  // 打开中间张量导出（--dump），供与参考实现对拍。
  void SetDumpDir(const std::string& dir) { dump_dir_ = dir; }

  // 分阶段计时的开关与读取。prefill / decode 各自一本账：
  // 判据是单次 Forward 的 token 数，>1 算 prefill，==1 算 decode。
  void SetProfile(bool on) {
    prof_prefill_.on = on;
    prof_decode_.on = on;
  }
  PhaseProfile& prof_prefill() { return prof_prefill_; }
  PhaseProfile& prof_decode() { return prof_decode_; }
  const PhaseProfile& prof_prefill() const { return prof_prefill_; }
  const PhaseProfile& prof_decode() const { return prof_decode_; }

 private:
  void ForwardLayer(int64_t layer, int64_t n_tokens, int64_t start_pos,
                    KVCache& kv, PhaseProfile& prof);
  void Dump(const std::string& name, const float* data, int64_t rows,
            int64_t cols) const;

  std::unique_ptr<SbsFile> file_;
  Weights weights_;
  ModelConfig config_;
  RopeTable rope_;
  std::string dump_dir_;
  bool dump_once_ = true;  // 只在第一次 Forward（prefill）导出中间张量

  // 工作缓冲，加载后一次性分配，避免每层反复 malloc。
  std::vector<float> x_;         // [n, model_dim] 主干残差流
  std::vector<float> normed_;    // [n, model_dim]
  std::vector<float> qkv_;       // [n, (heads + 2*kv_heads)*qkv_dim]
  std::vector<float> att_out_;   // [n, heads*qkv_dim]
  std::vector<float> proj_;      // [n, model_dim]
  std::vector<float> c1_, c2_;   // [n, ff_hidden_dim]
  std::vector<float> logits_;    // [vocab_size]
  std::vector<float> scores_;    // [seq] 注意力打分临时区
  PhaseProfile prof_prefill_;
  PhaseProfile prof_decode_;
};

// 贪心/采样。logits 已做过 final soft-cap。
int32_t SampleGreedy(const float* logits, int64_t n);
int32_t SampleTopK(const float* logits, int64_t n, float temperature, int64_t top_k,
                   uint64_t* rng_state);

}  // namespace tg

#endif  // TINYGEMMA_MODEL_H_
