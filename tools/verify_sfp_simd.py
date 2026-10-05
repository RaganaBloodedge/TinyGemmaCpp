#!/usr/bin/env python3
"""核对 sfp 的 SIMD 闭式解码与 dtypes.h 里的分支版逐位一致。

src/matmul.cc 的 Sfp8ToF32x8 用了一个无分支闭式：

    v    = x & 0x7F
    bf16 = (v > 63) ? 0x3800 + (v << 4) : 0x3400 + (v << 5)
    f32bits = (bf16 << 16) | (sign << 16)

闭式的来历：把 SfpDecodeToBf16 的两条分支各自展开成位串，
发现 (hi<<8)|lo 恰好等于「常数 + v 左移若干位」，因为 v<<n 的高位
正好被 hi 的进位吸收。这里把 256 个字节全枚举一遍做等价性证明。

用法:  python tools/verify_sfp_simd.py
"""

import struct
import sys


def reference(e: int) -> int:
    """逐位复刻 dtypes.h:SfpDecodeToBf16 的分支实现。"""
    sign = e & 0x80
    v = e & 0x7F
    if v == 0:
        return 0
    small = v < 64
    shl = (v << 1) if small else v
    lo = (shl << 4) & 0xFF
    hi = ((0x34 if small else 0x38) + (shl >> 4)) & 0xFF
    return ((hi | sign) << 8) | lo


def closed_form(e: int) -> int:
    """与 Sfp8ToF32x8 等价的标量写法。"""
    x = e
    v = x & 0x7F
    large = v > 63
    vv = (v << 4) if large else (v << 5)
    base = 0x3800 if large else 0x3400
    bf16 = (vv + base) | ((x & 0x80) << 8)
    if v == 0:
        bf16 = 0  # SIMD 里由 cmpeq(v,0) + andnot 抹掉
    return bf16 & 0xFFFF


def bf16_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b << 16))[0]


def main() -> int:
    bad = [(e, reference(e), closed_form(e)) for e in range(256)
           if reference(e) != closed_form(e)]
    if bad:
        print("不一致：")
        for e, r, c in bad[:10]:
            print(f"  e={e:#04x}  分支={r:#06x}  闭式={c:#06x}")
        print(f"共 {len(bad)} 个字节不一致")
        return 1

    print("256 个字节全部逐位一致（含两个零编码 0x00 / 0x80）")

    print("\n抽样核对数值：")
    for e in (0x01, 0x04, 0x21, 0x40, 0x60, 0x7F):
        v = bf16_to_f32(closed_form(e))
        print(f"  e={e:#04x} -> {v:.6g}")

    # 顺带核对可表示范围（dtypes.h 注释里写的 [1.19e-7, 1.875]）
    vals = [bf16_to_f32(closed_form(e)) for e in range(256) if (e & 0x7F)]
    print(f"\n非零取值: {len(set(vals))} 个，范围 [{min(vals):.6g}, {max(vals):.6g}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
