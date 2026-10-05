#!/usr/bin/env python3
"""TinyGemmaCpp 权重层的 Python 参考实现。

存在的意义只有一个：**给 C++ 那份实现当对照**。
两边独立写同一份算法（.sbs 目录解析 + sfp 解码），解出来的浮点位模式必须逐位一致。
不逐位一致就说明有一边理解错了格式，而不是"浮点误差"。

用法：
    python tools/sbs.py WEIGHTS.sbs --list
    python tools/sbs.py WEIGHTS.sbs --decode c_final_norm --n 32
"""

from __future__ import annotations

import argparse
import mmap
import struct
import sys

MAGIC = 0x0A534253  # 磁盘上即 "SBS\n"
BLOB_ALIGN = 256
TYPE_BYTES = {"F": 4, "B": 2, "$": 1, "2": 0.5, "I": 1}
TYPE_NAMES = {"F": "f32", "B": "bf16", "$": "sfp", "2": "nuq", "I": "i8"}


def sfp_to_f32(e: int) -> float:
    """sfp 单字节 -> f32。与 src/dtypes.h 的 SfpDecodeToBf16 是同一份算法。"""
    sign = e & 0x80
    v = e & 0x7F
    if v == 0:
        return 0.0
    small = v < 64
    shl = (v << 1) & 0xFF if small else v
    lo = (shl << 4) & 0xFF
    hi = ((0x34 if small else 0x38) + (shl >> 4)) & 0xFF
    bf16 = ((hi | sign) << 8) | lo
    return struct.unpack("<f", struct.pack("<I", bf16 << 16))[0]


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


class BlobStore:
    """只 mmap，不把 3 GB 读进内存 —— 按需分页，取哪段读哪段。"""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self._size = self._f.seek(0, 2)
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse()

    def close(self) -> None:
        self._mm.close()
        self._f.close()

    def _parse(self) -> None:
        # Header 固定 16 字节：magic(u32) + num_blobs(u32) + file_bytes(u64)。
        # 注意别写成 "<IIQQ"（24 字节）——那样会把 file_bytes 的低 32 位
        # 当成 num_blobs，得到一个 32 亿量级的数字，循环到内存耗尽。
        mm = self._mm
        magic, head_num_blobs = struct.unpack_from("<II", mm, 0)
        if magic != MAGIC:
            raise ValueError(f"magic 不匹配: {magic:#x}")

        if head_num_blobs != 0:
            # V1：目录紧跟头部。
            self.layout = "V1"
            num_blobs = head_num_blobs
            dir_off = 16
            file_bytes = struct.unpack_from("<Q", mm, 8)[0]
        else:
            # V2：真正的 header 在文件末尾。
            self.layout = "V2"
            tail = self._size - 16
            magic2, num_blobs = struct.unpack_from("<II", mm, tail)
            if magic2 != MAGIC:
                raise ValueError("V2 尾部 header magic 不匹配")
            file_bytes = struct.unpack_from("<Q", mm, tail + 8)[0]
            dir_off = self._size - 16 - 32 * num_blobs

        if dir_off + 32 * num_blobs > self._size:
            raise ValueError("目录越界，文件可能损坏")

        self.num_blobs = num_blobs
        self.file_bytes = file_bytes

        self.keys: list[str] = []
        self.ranges: list[tuple[int, int]] = []
        for i in range(num_blobs):
            raw = mm[dir_off + i * 16: dir_off + (i + 1) * 16]
            self.keys.append(raw.split(b"\0")[0].decode("ascii", "replace"))
        base = dir_off + 16 * num_blobs
        for i in range(num_blobs):
            off, nbytes = struct.unpack_from("<QQ", mm, base + i * 16)
            self.ranges.append((off, nbytes))
        self._index = {k[1:]: i for i, k in enumerate(self.keys)}

    # ------------------------------------------------------------------
    @property
    def type_bytes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in range(self.num_blobs):
            t = TYPE_NAMES.get(self.keys[i][0], "?")
            out[t] = out.get(t, 0) + self.ranges[i][1]
        return out

    def find(self, name: str) -> int | None:
        return self._index.get(name)

    def elements(self, name: str) -> int:
        i = self._index[name]
        per = TYPE_BYTES[self.keys[i][0]]
        return int(self.ranges[i][1] / per)

    def raw(self, name: str, start: int = 0, n: int | None = None) -> bytes:
        """只取需要的区间，不复制整个 blob。"""
        i = self._index[name]
        per = TYPE_BYTES[self.keys[i][0]]
        off, nbytes = self.ranges[i]
        total = int(nbytes / per)
        if n is None:
            n = total - start
        lo = off + int(start * per)
        hi = lo + int(n * per)
        return self._mm[lo:hi]

    def decode(self, name: str, start: int = 0, n: int | None = None) -> list[float]:
        """按存储类型解码为 f32 列表。"""
        i = self._index[name]
        tchar = self.keys[i][0]
        if n is None:
            n = self.elements(name) - start
        raw = self.raw(name, start, n)
        if tchar == "F":
            return list(struct.unpack_from(f"<{n}f", raw, 0))
        if tchar == "B":
            bits = struct.unpack_from(f"<{n}H", raw, 0)
            return [bf16_to_f32(b) for b in bits]
        if tchar == "$":
            return [sfp_to_f32(b) for b in raw[:n]]
        raise ValueError(f"不支持的类型 {tchar}")


def to_hex_bits(values: list[float]) -> list[str]:
    return [f"{struct.unpack('<I', struct.pack('<f', v))[0]:08x}" for v in values]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sbs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--decode", default=None, help="blob 名")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    bs = BlobStore(args.sbs)
    print(f"布局      : {bs.layout}")
    print(f"blob 数   : {bs.num_blobs}")
    print(f"file_bytes: {bs.file_bytes:,} ({bs.file_bytes / 2**30:.2f} GiB)")
    for t, nb in sorted(bs.type_bytes.items(), key=lambda x: -x[1]):
        print(f"  {t:<5} {nb:>14,} 字节 ({nb / bs.file_bytes * 100:5.2f}%)")

    if args.list:
        print()
        for i in range(bs.num_blobs):
            off, nbytes = bs.ranges[i]
            print(f"{i:>4}  {bs.keys[i]:<20} {TYPE_NAMES.get(bs.keys[i][0], '?'):<5} "
                  f"{off:>14,} {nbytes:>14,}")

    if args.decode:
        vals = bs.decode(args.decode, 0, args.n)
        for h in to_hex_bits(vals):
            print(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
