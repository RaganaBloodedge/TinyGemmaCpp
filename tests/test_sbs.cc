// .sbs 读取与解码自检。
//
//   ./build/test_sbs --sbs PATH            打印 blob 统计
//   ./build/test_sbs --sbs PATH --list     列出全部 210 个 blob
//   ./build/test_sbs --sbs PATH --dump-hex c_final_norm 64
//       按存储类型解码前 64 个元素，输出 8 位十六进制，
//       交给 tools/verify_decode.py 用 Python 参考实现逐位比对。
//   ./build/test_sbs --sbs PATH --check    按 Gemma 2 2B 的形状校验所有张量

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "config.h"
#include "dtypes.h"
#include "sbs_reader.h"
#include "weights.h"

namespace {

void PrintBlob(const tg::Blob& b) {
  const int64_t per = tg::DTypeElementBytes(b.type);
  std::printf("%-20s %-5s off=%-14llu bytes=%-12llu elems=%lld\n",
              b.key.c_str(), tg::DTypeName(b.type),
              static_cast<unsigned long long>(b.range.offset),
              static_cast<unsigned long long>(b.range.bytes),
              static_cast<long long>(per ? b.range.bytes / per : 0));
}

}  // namespace

int main(int argc, char** argv) {
  std::string sbs_path;
  std::string dump_blob;
  int64_t dump_n = 16;
  int64_t dump_offset = 0;
  bool list = false;
  bool check = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--sbs" && i + 1 < argc) sbs_path = argv[++i];
    else if (a == "--list") list = true;
    else if (a == "--check") check = true;
    else if (a == "--dump-hex" && i + 1 < argc) dump_blob = argv[++i];
    else if (a == "--n" && i + 1 < argc) dump_n = std::atoll(argv[++i]);
    else if (a == "--offset" && i + 1 < argc) dump_offset = std::atoll(argv[++i]);
    else {
      std::fprintf(stderr, "未知参数 %s\n", a.c_str());
      return 2;
    }
  }
  if (sbs_path.empty()) {
    const char* env = std::getenv("TG_SBS");
    if (env != nullptr) sbs_path = env;
  }
  if (sbs_path.empty()) {
    // 没有权重路径就跳过，而不是报错退出：本仓库不带权重（见 README），
    // ctest 与 CI 上不该因为这个判失败。
    std::fprintf(stderr,
                 "跳过：需要 .sbs 权重。用 --sbs PATH 或环境变量 TG_SBS 指定。\n");
    return 77;  // CTest 的 SKIP_RETURN_CODE
  }

  tg::SbsFile file;
  std::string err;
  if (!file.Open(sbs_path, &err)) {
    std::fprintf(stderr, "打开失败: %s\n", err.c_str());
    return 1;
  }

  // --dump-hex 模式下 stdout 只允许出现十六进制数据（供管道 / 对拍脚本消费），
  // 其余诊断信息一律走 stderr。
  FILE* info = dump_blob.empty() ? stdout : stderr;

  std::fprintf(info, "布局      : %s\n", file.layout().c_str());
  std::fprintf(info, "blob 数   : %lld\n", static_cast<long long>(file.num_blobs()));
  std::fprintf(info, "file_bytes: %lld (%.2f GiB)\n", static_cast<long long>(file.file_bytes()),
               file.file_bytes() / 1073741824.0);

  int64_t by_type[4] = {0, 0, 0, 0};
  for (const tg::Blob& b : file.blobs()) {
    switch (b.type) {
      case tg::DType::kF32: by_type[0] += b.range.bytes; break;
      case tg::DType::kBF16: by_type[1] += b.range.bytes; break;
      case tg::DType::kSFP: by_type[2] += b.range.bytes; break;
      default: by_type[3] += b.range.bytes; break;
    }
  }
  std::fprintf(info, "按类型    : f32 %lld / bf16 %lld / sfp %lld / 其它 %lld\n\n",
               static_cast<long long>(by_type[0]),
               static_cast<long long>(by_type[1]),
               static_cast<long long>(by_type[2]),
               static_cast<long long>(by_type[3]));

  if (list) {
    for (const tg::Blob& b : file.blobs()) PrintBlob(b);
    std::printf("\n");
  }

  const tg::Blob* b = file.Find(dump_blob);
  if (!dump_blob.empty()) {
    if (b == nullptr) {
      std::fprintf(stderr, "找不到 blob: %s\n", dump_blob.c_str());
      return 1;
    }
    const uint8_t* p = file.Data(*b);
    std::fprintf(stderr, "decoded %s (%s), elements [%lld, %lld) as f32 bit patterns\n",
                 b->key.c_str(), tg::DTypeName(b->type),
                 static_cast<long long>(dump_offset),
                 static_cast<long long>(dump_offset + dump_n));
    for (int64_t i = 0; i < dump_n; ++i) {
      const int64_t idx = dump_offset + i;
      if (idx >= b->Elements()) break;
      const float v = tg::LoadAsF32(b->type, p, idx);
      uint32_t bits;
      std::memcpy(&bits, &v, 4);
      std::printf("%08x\n", bits);
    }
    return 0;
  }

  if (check) {
    const tg::ModelConfig config = tg::ModelConfig::Gemma2_2B();
    tg::Weights w;
    if (!w.Build(file, config, &err)) {
      std::fprintf(stderr, "构建权重失败: %s\n", err.c_str());
      return 1;
    }
    std::printf("权重校验通过：\n");
    std::printf("  c_embedding   : [%lld, %lld]\n",
                static_cast<long long>(w.embed.rows),
                static_cast<long long>(w.embed.cols));
    std::printf("  c_final_norm  : [%lld]\n", static_cast<long long>(w.final_norm.size()));
    std::printf("  层数          : %lld\n", static_cast<long long>(w.layers.size()));
    for (int64_t i = 0; i < static_cast<int64_t>(w.layers.size()); i += 25) {
      const tg::LayerWeights& lw = w.layers[static_cast<size_t>(i)];
      std::printf("  layer %-2lld       : qkv[%lld,%lld] att[%lld,%lld] "
                  "gating[%lld,%lld] linear[%lld,%lld]\n",
                  static_cast<long long>(i),
                  static_cast<long long>(lw.qkv.rows), static_cast<long long>(lw.qkv.cols),
                  static_cast<long long>(lw.att.rows), static_cast<long long>(lw.att.cols),
                  static_cast<long long>(lw.gating.rows), static_cast<long long>(lw.gating.cols),
                  static_cast<long long>(lw.linear.rows), static_cast<long long>(lw.linear.cols));
    }
    std::printf("  常驻          : %.2f GiB\n", w.ResidentBytes() / 1073741824.0);
    std::printf("\n全部张量形状正确\n");
    return 0;
  }

  // 默认：打印前若干个 blob 概览
  const int64_t show = file.num_blobs() < 12 ? file.num_blobs() : 12;
  for (int64_t i = 0; i < show; ++i) PrintBlob(file.blobs()[static_cast<size_t>(i)]);
  std::printf("... 共 %lld 个（用 --list 看全部）\n",
              static_cast<long long>(file.num_blobs()));
  return 0;
}
