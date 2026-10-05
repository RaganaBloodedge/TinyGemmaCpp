#!/usr/bin/env python3
"""端到端跑一遍 TinyGemmaCpp。

    python tools/run.py --question "What is the capital of France?"
    python tools/run.py --prompt "Once upon a time" --style raw --max-tokens 40

流程：text --spm--> token ids --stdin--> tinygemma --stdout--> ids --spm--> text
引擎本体不认识文本，两边都靠 tools/spm.py 转换。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import HERE, find_asset  # noqa: E402
from spm import SentencePieceModel  # noqa: E402

DEFAULT_BIN = os.path.join(HERE, "build", "tinygemma.exe")
if not os.path.exists(DEFAULT_BIN):
    DEFAULT_BIN = os.path.join(HERE, "build", "tinygemma")


def build_chat_prompt(m: SentencePieceModel, user_text: str) -> list[int]:
    """Gemma 2 instruction-tuned 的对话模板。"""
    pid = m._piece_to_id
    bos, start_turn, end_turn = pid["<bos>"], pid["<start_of_turn>"], pid["<end_of_turn>"]
    ids = [bos, start_turn] + m.encode("user\n" + user_text) + [end_turn]
    ids += m.encode("\n") + [start_turn] + m.encode("model\n")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--weights", default=find_asset("TG_WEIGHTS", "2.0-2b-it-sfp.sbs"))
    ap.add_argument("--tokenizer", default=find_asset("TG_TOKENIZER", "tokenizer.spm"))
    ap.add_argument("--question", default=None, help="用对话模板包成 prompt")
    ap.add_argument("--prompt", default=None, help="原始文本续写")
    ap.add_argument("--style", choices=["chat", "raw"], default="chat")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--show-ids", action="store_true")
    args = ap.parse_args()

    m = SentencePieceModel.load(args.tokenizer)

    if args.question:
        prompt_ids = build_chat_prompt(m, args.question) if args.style == "chat" \
            else m.encode(args.question)
    elif args.prompt:
        if args.style == "chat":
            prompt_ids = build_chat_prompt(m, args.prompt)
        else:
            # raw：原样分词再加 BOS。这份 tokenizer 的 add_dummy_prefix=0，
            # 所以不要自己补前导空格，否则会多出一个 ▁。
            prompt_ids = [m._piece_to_id["<bos>"]] + m.encode(args.prompt)
    else:
        ap.error("需要 --question 或 --prompt")

    if args.show_ids:
        print(f"[ids] prompt ({len(prompt_ids)}): {prompt_ids}")

    cmd = [args.bin, "--weights", args.weights,
           "--prompt-ids", ",".join(str(i) for i in prompt_ids),
           "--max-tokens", str(args.max_tokens),
           "--temperature", str(args.temperature),
           "--top-k", str(args.top_k)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    if args.bench:
        cmd += ["--bench"]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.stderr.write(f"[run.py] 引擎退出码 {proc.returncode}\n")
        return proc.returncode

    out_ids = [int(x) for x in proc.stdout.strip().split(",") if x.strip()]
    if args.show_ids:
        print(f"[ids] out ({len(out_ids)}): {out_ids}")

    print("=" * 68)
    print(f"prompt  : {args.question or args.prompt}")
    print("-" * 68)
    print(m.decode(out_ids))
    print("=" * 68)
    print(f"[run.py] {len(out_ids)} token, 端到端 {wall:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
