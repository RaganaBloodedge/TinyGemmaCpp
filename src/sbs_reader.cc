#include "sbs_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

namespace tg {
namespace {

constexpr uint32_t kMagic = 0x0A534253u;  // 磁盘上即 ASCII "SBS\n"
constexpr uint64_t kBlobAlign = 256;

uint32_t ReadU32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
uint64_t ReadU64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

uint64_t RoundUp(uint64_t x, uint64_t align) {
  return (x + align - 1) / align * align;
}

}  // namespace

SbsFile::~SbsFile() { std::free(const_cast<uint8_t*>(file_)); }

bool SbsFile::Open(const std::string& path, std::string* err) {
  // 注意：不能用 ftell()。Windows 上 long 只有 32 位，3.2 GB 的文件会被截断，
  // 表现成"文件太小"。用 std::filesystem 拿 64 位大小。
  std::error_code ec;
  const uint64_t size_u = std::filesystem::file_size(path, ec);
  if (ec) {
    *err = "无法读取文件大小: " + path + " (" + ec.message() + ")";
    return false;
  }
  const int64_t size = static_cast<int64_t>(size_u);
  if (size < 16) {
    *err = "文件太小，不是 .sbs";
    return false;
  }

  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    *err = "无法打开文件: " + path;
    return false;
  }

  uint8_t* buf = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(size)));
  if (buf == nullptr) {
    *err = "无法分配 " + std::to_string(size) + " 字节";
    std::fclose(f);
    return false;
  }
  size_t got = 0;
  while (got < static_cast<size_t>(size)) {
    const size_t n = std::fread(buf + got, 1, static_cast<size_t>(size) - got, f);
    if (n == 0) break;
    got += n;
  }
  std::fclose(f);
  if (got != static_cast<size_t>(size)) {
    std::free(buf);
    *err = "读取字节数不符";
    return false;
  }
  file_ = buf;
  file_bytes_ = size;

  // ---- 解析 Header ----
  uint32_t num_blobs = 0;
  uint64_t dir_off = 0;
  uint64_t payload_off = 0;

  if (ReadU32(buf) != kMagic) {
    std::free(buf);
    file_ = nullptr;
    *err = "magic 不匹配，不是 .sbs 文件";
    return false;
  }

  const uint32_t head_num_blobs = ReadU32(buf + 4);
  if (head_num_blobs != 0) {
    // V1：目录紧跟头部。
    num_blobs = head_num_blobs;
    layout_ = "V1";
    dir_off = 16;
    payload_off = RoundUp(16 + 32ull * num_blobs, kBlobAlign);
  } else {
    // V2：头部在文件末尾，num_blobs 在最后一个 header 里。
    const uint8_t* tail = buf + size - 16;
    if (ReadU32(tail) != kMagic) {
      std::free(buf);
      file_ = nullptr;
      *err = "V2 尾部 header magic 不匹配";
      return false;
    }
    num_blobs = ReadU32(tail + 4);
    dir_off = static_cast<uint64_t>(size) - 16 - 32ull * num_blobs;
    payload_off = kBlobAlign;
    layout_ = "V2";
  }

  if (dir_off + 32ull * num_blobs > static_cast<uint64_t>(size)) {
    std::free(buf);
    file_ = nullptr;
    *err = "目录越界";
    return false;
  }

  // ---- 解析目录 ----
  blobs_.reserve(num_blobs);
  by_name_.reserve(num_blobs * 2);
  for (uint32_t i = 0; i < num_blobs; ++i) {
    const uint8_t* key_p = buf + dir_off + i * 16;
    const uint8_t* range_p = buf + dir_off + 16ull * num_blobs + i * 16;

    Blob b;
    char key[17] = {0};
    std::memcpy(key, key_p, 16);
    b.key = key;  // 遇 \0 自然截断
    b.type = DTypeFromChar(b.key.empty() ? '?' : b.key[0]);
    b.name = b.key.empty() ? std::string() : b.key.substr(1);
    b.range.offset = ReadU64(range_p);
    b.range.bytes = ReadU64(range_p + 8);

    if (b.range.offset + b.range.bytes > static_cast<uint64_t>(size)) {
      std::free(buf);
      file_ = nullptr;
      *err = "blob " + b.key + " 越界";
      return false;
    }
    by_name_.emplace(b.name, blobs_.size());
    blobs_.push_back(std::move(b));
  }

  (void)payload_off;
  return true;
}

const Blob* SbsFile::Find(std::string_view name) const {
  auto it = by_name_.find(std::string(name));
  if (it == by_name_.end()) return nullptr;
  return &blobs_[it->second];
}

}  // namespace tg
