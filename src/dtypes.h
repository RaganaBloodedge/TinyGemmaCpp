// TinyGemmaCpp —— 权重的存储类型与解码
//
// 权重文件里一个元素可能是 f32 / bf16 / sfp（8bit switched float）。
// 引擎策略：**保持文件里的存储宽度，不展开成 f32**。
//   - bf16 张量：原地 2 字节/元素
//   - sfp  张量：原地 1 字节/元素，需要时在计算内核里解码
// 这样 3.0 GiB 的文件常驻内存约 3.2 GiB，而不是展开成 f32 的 ~10 GiB。
//
// sfp 解码算法参考 google/gemma.cpp（compression/sfp-inl.h，Apache-2.0）的
// 通用路径，此处为标量重写，去掉了 Highway SIMD 依赖。

#ifndef TINYGEMMA_DTYPES_H_
#define TINYGEMMA_DTYPES_H_

#include <bit>
#include <cstdint>
#include <cstring>

namespace tg {

enum class DType : uint8_t { kUnknown = 0, kF32, kBF16, kSFP };

// 文件里 blob key 的首字符就是类型标记。
inline DType DTypeFromChar(char c) {
  switch (c) {
    case 'F': return DType::kF32;
    case 'B': return DType::kBF16;
    case '$': return DType::kSFP;
    default: return DType::kUnknown;
  }
}

inline const char* DTypeName(DType t) {
  switch (t) {
    case DType::kF32: return "f32";
    case DType::kBF16: return "bf16";
    case DType::kSFP: return "sfp";
    default: return "?";
  }
}

inline int64_t DTypeElementBytes(DType t) {
  switch (t) {
    case DType::kF32: return 4;
    case DType::kBF16: return 2;
    case DType::kSFP: return 1;
    default: return 0;
  }
}

// bf16 -> f32：就是左移 16 位，没有舍入误差，是精确的。
inline float Bf16ToF32(uint16_t v) {
  return std::bit_cast<float>(static_cast<uint32_t>(v) << 16);
}

// f32 -> bf16：round-to-nearest-even。
inline uint16_t F32ToBf16(float f) {
  uint32_t u = std::bit_cast<uint32_t>(f);
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

// sfp 单字节 -> bf16 位模式。
//
// 字节布局：bit7 = 符号，bit6 = 指数高位（=1 表示"大"指数，保留 3 位尾数），
// 指数占 4 bit，尾数占 2（小）或 3（大）bit。
//   小指数（e<64）：v = 2^(e-23) * (1 + m/4)，e = v>>2, m = v&3
//   大指数（e>=64）：v = 2^(e-15) * (1 + m/8)，e = v>>3, m = v&7
// 可表示范围约 [1.19e-7, 1.875]，所以 sfp 张量的 |x| 必须 <= 1.875。
// 实测 Gemma 2B 的 sfp 张量 scale 全为 1.0（见 python/convert_from_safetensors.py
// 的 compute_scale：max(1.0, max|x|/1.875)），无需额外缩放。
inline uint16_t SfpDecodeToBf16(uint8_t e) {
  const uint8_t sign = e & 0x80u;
  const uint8_t v = e & 0x7Fu;
  if (v == 0) return 0;  // 零（-0 被保留不用）
  const bool small = v < 64;
  // 小指数时等价于 v<<1 再整体 <<4，即尾数落在 bit7..5。
  const uint8_t shl = small ? static_cast<uint8_t>(v << 1) : v;
  const uint8_t lo = static_cast<uint8_t>(shl << 4);
  const uint8_t hi = static_cast<uint8_t>((small ? 0x34 : 0x38) + (shl >> 4));
  return static_cast<uint16_t>((static_cast<uint8_t>(hi | sign) << 8) | lo);
}

inline float SfpToF32(uint8_t e) { return Bf16ToF32(SfpDecodeToBf16(e)); }

inline uint16_t LoadU16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline float LoadF32(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

// 把第 idx 个元素解码为 f32。
inline float LoadAsF32(DType t, const uint8_t* p, int64_t idx) {
  switch (t) {
    case DType::kF32: return LoadF32(p + idx * 4);
    case DType::kBF16: return Bf16ToF32(LoadU16(p + idx * 2));
    case DType::kSFP: return Bf16ToF32(SfpDecodeToBf16(p[idx]));
    default: return 0.0f;
  }
}

}  // namespace tg

#endif  // TINYGEMMA_DTYPES_H_
