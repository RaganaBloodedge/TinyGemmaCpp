#include "weights.h"

#include <cstring>

namespace tg {
namespace {

// 把一个 blob 绑定成 [rows, cols] 的矩阵视图，并校验元素数。
bool BindMatrix(const SbsFile& file, const std::string& name, int64_t rows,
                int64_t cols, Matrix* out, std::string* err) {
  const Blob* b = file.Find(name);
  if (b == nullptr) {
    *err = "权重文件缺少张量: " + name;
    return false;
  }
  const int64_t want = rows * cols;
  if (b->Elements() != want) {
    *err = "张量 " + name + " 元素数不符: 文件 " +
           std::to_string(b->Elements()) + " vs 期望 " + std::to_string(want);
    return false;
  }
  out->data = file.Data(*b);
  out->type = b->type;
  out->rows = rows;
  out->cols = cols;
  return true;
}

// 一维 bf16/f32 权重 -> 展开成 f32 向量。
bool LoadVector(const SbsFile& file, const std::string& name, int64_t n,
                std::vector<float>* out, std::string* err) {
  const Blob* b = file.Find(name);
  if (b == nullptr) {
    *err = "权重文件缺少张量: " + name;
    return false;
  }
  if (b->Elements() != n) {
    *err = "张量 " + name + " 元素数不符";
    return false;
  }
  const uint8_t* p = file.Data(*b);
  out->resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) (*out)[static_cast<size_t>(i)] = LoadAsF32(b->type, p, i);
  return true;
}

std::string LayerSuffix(int64_t i) { return "_" + std::to_string(i); }

}  // namespace

bool Weights::Build(const SbsFile& file, const ModelConfig& config,
                    std::string* err) {
  const LayerConfig& lc = config.layer;

  if (!BindMatrix(file, "c_embedding", config.vocab_size, lc.model_dim, &embed,
                  err)) {
    return false;
  }
  if (!LoadVector(file, "c_final_norm", lc.model_dim, &final_norm, err)) {
    return false;
  }

  layers.resize(static_cast<size_t>(config.num_layers));
  for (int64_t i = 0; i < config.num_layers; ++i) {
    const std::string s = LayerSuffix(i);
    LayerWeights& lw = layers[static_cast<size_t>(i)];

    if (!BindMatrix(file, "qkv_ein" + s, lc.QkvRows(), lc.model_dim, &lw.qkv,
                    err)) {
      return false;
    }
    // att_ein 在文件里是 [heads, model_dim, qkv_dim]：每个 head 一块
    // [model_dim, qkv_dim] 的输出投影。
    if (!BindMatrix(file, "att_ein" + s, lc.heads * lc.model_dim, lc.qkv_dim,
                    &lw.att, err)) {
      return false;
    }
    if (!BindMatrix(file, "gating_ein" + s, 2 * lc.ff_hidden_dim, lc.model_dim,
                    &lw.gating, err)) {
      return false;
    }
    if (!BindMatrix(file, "linear_w" + s, lc.model_dim, lc.ff_hidden_dim,
                    &lw.linear, err)) {
      return false;
    }
    if (!LoadVector(file, "pre_att_ns" + s, lc.model_dim, &lw.pre_att_norm, err) ||
        !LoadVector(file, "post_att_ns" + s, lc.model_dim, &lw.post_att_norm, err) ||
        !LoadVector(file, "pre_ff_ns" + s, lc.model_dim, &lw.pre_ff_norm, err) ||
        !LoadVector(file, "post_ff_ns" + s, lc.model_dim, &lw.post_ff_norm, err)) {
      return false;
    }
  }

  // 统计（不含 norms，它们已被展开成 f32 常驻）。
  for (const Blob& b : file.blobs()) {
    ++stats.num_blobs;
    switch (b.type) {
      case DType::kF32: stats.bytes_f32 += b.range.bytes; break;
      case DType::kBF16: stats.bytes_bf16 += b.range.bytes; break;
      case DType::kSFP: stats.bytes_sfp += b.range.bytes; break;
      default: break;
    }
  }
  stats.elements = stats.bytes_bf16 / 2 + stats.bytes_sfp + stats.bytes_f32 / 4;
  return true;
}

}  // namespace tg
