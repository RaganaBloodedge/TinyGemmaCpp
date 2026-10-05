#!/usr/bin/env python3
"""测 prefill（TTFT）随 prompt 长度的变化。

关键假设：decode 是访存受限的，但 prefill 只要权重行被复用，搬运量就与
prompt 长度无关 —— TTFT 应该几乎不随长度增长，直到 attention 的 O(n^2)
开始占主导。

    python tools/bench_prefill.py --weights weights/2.0-2b-it-sfp.sbs
    python tools/bench_prefill.py --lengths 6,32,128,512 --threads 16
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spm import SentencePieceModel  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 重复用的语料，保证 token 分布接近自然文本。
CORPUS = (
    "The history of computing spans several centuries, from mechanical calculators "
    "and punched cards to the transistor, the integrated circuit, and the modern "
    "microprocessor. Each transition changed not only what machines could do, but "
    "also how people thought about problems. Software followed hardware: assembly "
    "gave way to structured programming, then to object orientation, and later to "
    "the functional and concurrent styles that dominate distributed systems today. "
    "Machine learning added another layer, where programs are fitted to data rather "
    "than written by hand, and inference engines became as important as compilers. "
)


def parse_ttft(out: str) -> tuple[float, float, int] | None:
    """从 --bench 输出里抓 prefill 耗时与 tok/s。"""
    m = re.search(
        r"\[bench\]\s+prefill\s+(\d+)\s+tok\s+([\d.]+)\s+s\s+->\s+([\d.]+)\s+tok/s",
        out,
    )
    if not m:
        return None
    toks = int(m.group(1))
    secs = float(m.group(2))
    tps = float(m.group(3))
    return secs * 1000.0, tps, toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "gemma.cpp-main", "weights", "tokenizer.spm"))
    ap.add_argument("--bin", default=os.path.join(HERE, "build", "tinygemma.exe"))
    ap.add_argument("--lengths", default="6,32,64,128,256,512")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=2, help="每档取最快的一次")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        args.bin = os.path.join(HERE, "build", "tinygemma")

    m = SentencePieceModel.load(args.tokenizer)
    bos = m._piece_to_id["<bos>"]

    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"{'prompt':>8} {'TTFT(ms)':>10} {'prefill tok/s':>14} {'ms/token':>10}")
    for want in lengths:
        # 拼到足够长再截断，保证每档都是自然文本。
        text = CORPUS
        while len(m.encode(text)) < want:
            text += " " + CORPUS
        ids = [bos] + m.encode(text)[: want - 1]

        best = None
        for _ in range(args.repeat):
            cmd = [args.bin, "--weights", args.weights,
                   "--prompt-ids", ",".join(str(i) for i in ids),
                   "--max-tokens", "1", "--bench"]
            if args.threads > 0:
                cmd += ["--threads", str(args.threads)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            got = parse_ttft(proc.stdout + proc.stderr)
            if got is None:
                print(f"  [{want}] 解析失败:\n{proc.stderr[-500:]}", file=sys.stderr)
                return 1
            best = got if best is None or got[0] < best[0] else best

        ms, tps, n = best
        print(f"{n:>8} {ms:>10.1f} {tps:>14.2f} {ms / max(1, n):>10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
