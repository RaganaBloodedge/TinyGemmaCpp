#!/usr/bin/env python3
"""C++ 与 Python 两条 .sbs 解码路径的逐位对拍。

同一个算法两边各写一遍，如果对格式的理解有出入，一定会在这里暴露。
比较的是 f32 的**位模式**而不是数值 —— 解码是纯位运算，不存在"浮点误差"，
任何一位不同都说明实现有 bug。

    python tools/verify_decode.py --sbs PATH --bin build/test_sbs
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbs import BlobStore, to_hex_bits  # noqa: E402

# C++ 侧的诊断信息走 stderr，stdout 只留数据；这里再收紧一道，
# 只认 8 位十六进制，避免任何意外输出混进比较。
_HEX8 = re.compile(r"^[0-9a-fA-F]{8}$")

# (blob, 起点, 元素数) —— 覆盖三种存储类型，并特意取一段中间偏移，
# 以确认不是"只有开头对"。
CASES = [
    ("c_final_norm", 0, 2304),
    ("c_embedding", 0, 4096),
    ("c_embedding", 589_824_000 - 4096, 4096),  # embedding 末尾
    ("post_att_ns_0", 0, 2304),
    ("qkv_ein_0", 0, 8192),
    ("qkv_ein_0", 4_718_592, 8192),
    ("qkv_ein_25", 9_437_184 - 4096, 4096),
    ("att_ein_0", 0, 8192),
    ("att_ein_7", 2_359_296, 8192),
    ("gating_ein_12", 21_233_664, 8192),
    ("linear_w_25", 21_233_664 - 4096, 4096),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbs", required=True)
    ap.add_argument("--bin", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "build", "test_sbs.exe"))
    args = ap.parse_args()
    if not os.path.exists(args.bin):
        args.bin = args.bin[:-4]

    bs = BlobStore(args.sbs)
    print(f"权重: {args.sbs}")
    print(f"blob: {bs.num_blobs}  布局: {bs.layout}")
    print()

    failed = 0
    total = 0
    for name, offset, n in CASES:
        if bs.find(name) is None:
            print(f"  [skip] {name} 不存在")
            continue
        proc = subprocess.run(
            [args.bin, "--sbs", args.sbs, "--dump-hex", name,
             "--n", str(n), "--offset", str(offset)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  [FAIL] {name} C++ 侧退出码 {proc.returncode}: {proc.stderr.strip()}")
            failed += 1
            continue
        cpp = [ln.strip() for ln in proc.stdout.splitlines()
               if _HEX8.match(ln.strip())]
        py = to_hex_bits(bs.decode(name, offset, n))
        total += len(py)
        if len(cpp) != len(py):
            print(f"  [FAIL] {name:<16} off={offset:<12} n={n:<6} "
                  f"长度不符：cpp={len(cpp)} py={len(py)}")
            failed += 1
            continue
        if cpp == py:
            print(f"  [ ok ] {name:<16} off={offset:<12} n={n:<6} 逐位一致")
        else:
            diff = [i for i, (a, b) in enumerate(zip(cpp, py)) if a != b]
            print(f"  [FAIL] {name:<16} off={offset:<12} n={n:<6} "
                  f"{len(diff)} 个元素不同，首个在 #{diff[0] if diff else '?'}")
            for i in diff[:3]:
                print(f"         idx {offset + i}: cpp={cpp[i]} py={py[i]}")
            failed += 1

    print()
    if failed == 0:
        print(f"全部通过：{len(CASES)} 组 / {total} 个元素，位模式完全一致")
        return 0
    print(f"{failed} 组失败")
    return 1


if __name__ == "__main__":
    sys.exit(main())
