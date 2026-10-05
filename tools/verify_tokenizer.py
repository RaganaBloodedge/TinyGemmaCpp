#!/usr/bin/env python3
"""用官方 sentencepiece 校验我们的 Python 分词器。

同一个模型文件、同一批文本，两边各切一次，比较 token id 序列。
官方那边是 WSL 里的 tools/spm_probe（链接 gemma.cpp 构建出的 libsentencepiece）。

    python tools/verify_tokenizer.py                 # 跑内置用例
    python tools/verify_tokenizer.py --probe PATH    # 指定探针可执行文件

探针与模型都在 WSL 内，路径用环境变量给（见下面的常量）。

⚠️ Windows 的 Git Bash 下要同时设 MSYS_NO_PATHCONV=1：MSYS 会把 /home/... 这类值
改写成 Windows 路径，传给 WSL 就成了不存在的文件，现象是"探针无输出"。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find_asset  # noqa: E402
from spm import SentencePieceModel  # noqa: E402

# 官方侧跑在 WSL 里，探针与模型都是 WSL 内的路径，随机器而变，用环境变量给：
#   TG_WSL_DISTRO    WSL 发行版名（默认 Ubuntu）
#   TG_SPM_PROBE     tools/spm_probe.cc 编出的可执行文件在 WSL 内的路径
#   TG_MODEL_IN_WSL  同一份 tokenizer.spm 在 WSL 内的路径
WSL_DISTRO = os.environ.get("TG_WSL_DISTRO", "Ubuntu")
PROBE_DEFAULT = os.environ.get("TG_SPM_PROBE", "")
MODEL_IN_WSL = os.environ.get("TG_MODEL_IN_WSL", "")

CASES = [
    # 普通英文
    "The capital of France is",
    " The capital of France is",
    "Hello world!",
    "Once upon a time, there was a girl.",
    "The quick brown fox jumps over the lazy dog.",
    "Paris is the capital of France.",
    "What is the capital of France? Answer in one sentence.",
    "I love programming in C++",
    "I love programming in C++ and Python.",
    "Machine learning is a subset of artificial intelligence.",
    # 代码
    "def main():\n    pass\n",
    "#include <vector>\nint main() { return 0; }\n",
    "for i in range(10):\n    print(i)\n",
    "SELECT * FROM users WHERE id = 1;",
    # 数字与符号
    "1234567890",
    "3.14159265358979",
    "$1,234.56",
    "a-b_c*d/e",
    "    ",
    "",
    # 大小写与重复
    "AAAAAAAAAAAAAAAA",
    "the the the the the",
    "TOKENIZATION",
    # 中文与多语言
    "你好，世界",
    "今天天气不错，我们去公园散步吧。",
    "日本語のテスト",
    "한국어 테스트",
    "Привет, мир!",
    # emoji / 特殊
    "Hello 👋 world 🌍",
    "café naïve résumé",
    # 长文本
    "In a distant future, humanity has spread across the stars, and the "
    "search for meaning continues unabated.",
]


def official(probe: str, text: str) -> list[int] | None:
    cmd = ["wsl.exe", "-d", WSL_DISTRO, "--", probe, MODEL_IN_WSL, "-"]
    # 必须走二进制管道：文本模式会在 Windows 侧把 \n 翻译成 \r\n，
    # 于是官方那边看到的是 CRLF，和我们的输入不一致。
    proc = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
    out = (proc.stdout or b"").decode("utf-8", "replace")
    for ln in out.splitlines():
        if ln.startswith("IDS"):
            # 形如 "IDS  (5): 651 6037 ..."，取 "):" 之后的部分
            return [int(x) for x in ln.split("):", 1)[1].split()]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=PROBE_DEFAULT)
    ap.add_argument("--model", default=find_asset("TG_TOKENIZER", "tokenizer.spm"))
    args = ap.parse_args()

    if not args.probe or not MODEL_IN_WSL:
        print("需要官方侧的探针与模型路径（编译方式见 tools/spm_probe.cc 头部注释）：")
        print("  TG_SPM_PROBE=<WSL 内路径> TG_MODEL_IN_WSL=<WSL 内路径> \\")
        print("      python tools/verify_tokenizer.py")
        return 2

    m = SentencePieceModel.load(args.model)
    print(f"model_type = {m.model_type_name}  pieces = {len(m)}")
    print()

    bad = 0
    for t in CASES:
        off = official(args.probe, t)
        ours = m.encode(t)
        if off is None:
            print(f"  [ERR ] 探针无输出: {t!r}")
            bad += 1
            continue
        show = t.replace("\n", "\\n")
        show = show[:44] + ("…" if len(show) > 44 else "")
        if off == ours:
            print(f"  [ ok ] {show!r:<48} {len(ours):>3} tok")
        else:
            bad += 1
            print(f"  [FAIL] {show!r:<48}")
            print(f"         官方: {m.to_pieces(off)}")
            print(f"         我们: {m.to_pieces(ours)}")

    print()
    if bad == 0:
        print(f"全部通过：{len(CASES)} 条用例与官方 sentencepiece 完全一致")
        return 0
    print(f"{bad}/{len(CASES)} 条不一致")
    return 1


if __name__ == "__main__":
    sys.exit(main())
