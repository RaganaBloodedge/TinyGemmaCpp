// TinyGemmaCpp —— .sbs (BlobStore) 容器解析
//
// 格式（参考 google/gemma.cpp 的 io/blob_store.h/.cc，Apache-2.0；此处为独立重写）：
//
//   V1: [Header 16B][Keys N*16][Ranges N*16][Pad->256][Payload][Pad->64KiB]
//   V2: [Header 16B][Pad->256][Payload][Pad][Layout][Header]
//
//   Header : magic(u32)="SBS\n" + layout(u32) + num_blobs(u32) + file_bytes(u64)
//   Key    : 16 字节 ASCII 明文（不足补 \0），首字符是类型前缀 F/B/$/2/I
//   Range  : offset(u64) + bytes(u64)
//
// 本文件把整个 .sbs 一次性读进内存，所有张量都是这块内存上的视图（零拷贝）。

#ifndef TINYGEMMA_SBS_READER_H_
#define TINYGEMMA_SBS_READER_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "dtypes.h"

namespace tg {

struct BlobRange {
  uint64_t offset = 0;
  uint64_t bytes = 0;
};

struct Blob {
  std::string key;      // 含类型前缀，如 "$qkv_ein_0"
  std::string name;     // 去掉前缀，如 "qkv_ein_0"
  DType type = DType::kUnknown;
  BlobRange range;
  int64_t Elements() const { return static_cast<int64_t>(range.bytes) /
                                    (DTypeElementBytes(type) ? DTypeElementBytes(type) : 1); }
};

class SbsFile {
 public:
  SbsFile() = default;
  ~SbsFile();
  SbsFile(const SbsFile&) = delete;
  SbsFile& operator=(const SbsFile&) = delete;

  // 读入整个文件并解析目录；失败时返回 false 并写入 err。
  bool Open(const std::string& path, std::string* err);

  const std::vector<Blob>& blobs() const { return blobs_; }
  const Blob* Find(std::string_view name) const;

  // blob 指向的原始字节（不解码）。
  const uint8_t* Data(const Blob& b) const { return file_ + b.range.offset; }

  int64_t file_bytes() const { return file_bytes_; }
  int64_t num_blobs() const { return static_cast<int64_t>(blobs_.size()); }
  const std::string& layout() const { return layout_; }

 private:
  const uint8_t* file_ = nullptr;
  int64_t file_bytes_ = 0;
  std::string layout_;
  std::vector<Blob> blobs_;
  std::unordered_map<std::string, size_t> by_name_;
};

}  // namespace tg

#endif  // TINYGEMMA_SBS_READER_H_
