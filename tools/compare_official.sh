#!/usr/bin/env bash
# 在同一台机器、同一份权重、同样的线程数下，对比 TinyGemmaCpp 与官方 gemma.cpp。
#
# 只在 WSL 里跑：两边都是 Linux 原生可执行文件，排除 WSL/Windows 的系统差异。
#
#   wsl -d Ubuntu -- bash <仓库路径>/tools/compare_official.sh
#
# 可用环境变量覆盖：
#   PROMPT_TEXT=...   一段英文文本；脚本用 tools/spm.py 算出它的 token id
#   TOKENS=24         生成多少 token
#   THREADS=16        线程数（两边相同）
#   WEIGHTS=... TOKENIZER=...   默认取仓库同级 gemma.cpp-main/weights/ 下的文件
#   OFFICIAL=...      自己构建的官方 gemma 可执行文件（必填，见 gemma.cpp 的构建说明）
#
# 为什么默认 16 而不是 24：在 8P+8E / 24 逻辑核的异构片上，Linux 的 libgomp
# 在 24 线程下会过度订阅，实测比 16 线程慢一倍且不稳定。
# 16 是两边都能正常发挥的值，比较才公平。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

# 本仓库不带权重（见 README），默认从同级的 gemma.cpp-main 下取。
SIBLING="$(dirname "$ROOT")/gemma.cpp-main/weights"
WEIGHTS="${WEIGHTS:-$SIBLING/2.0-2b-it-sfp.sbs}"
TOKENIZER="${TOKENIZER:-$SIBLING/tokenizer.spm}"
OFFICIAL="${OFFICIAL:-}"
TOKENS="${TOKENS:-24}"
THREADS="${THREADS:-16}"
PROMPT_TEXT="${PROMPT_TEXT:-The capital of France is}"
# 长 prompt 建议走文件，绕开命令行引号转义（尤其是从 Windows 侧调 wsl.exe 时）。
if [ -n "${PROMPT_FILE:-}" ] && [ -f "$PROMPT_FILE" ]; then
  PROMPT_TEXT="$(tr -d '\r\n' < "$PROMPT_FILE")"
fi

OURS="$ROOT/build/tinygemma_linux"

# 用与引擎同一套 tokenizer（已验证与官方 sentencepiece 逐 id 一致）把文本
# 转成 id，保证两边吃进去的是同一个 token 序列。
PROMPT_IDS="$(python3 - "$ROOT" "$TOKENIZER" "$PROMPT_TEXT" <<'PY'
import sys
root, tok, text = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root + "/tools")
from spm import SentencePieceModel
m = SentencePieceModel.load(tok)
ids = [m._piece_to_id["<bos>"]] + m.encode(text)
print(",".join(str(i) for i in ids))
PY
)"
if [ -z "$PROMPT_IDS" ]; then
  echo "无法计算 prompt token id（检查 python3 与 $TOKENIZER）" >&2
  exit 1
fi
N_TOK=$(awk -F, '{print NF}' <<<"$PROMPT_IDS")

if [ ! -x "$OURS" ]; then
  echo "[build] 缺少 $OURS，正在编译 ..."
  g++ -std=c++20 -O3 -fopenmp -march=native -Isrc \
    "$ROOT"/src/config.cc "$ROOT"/src/sbs_reader.cc "$ROOT"/src/weights.cc \
    "$ROOT"/src/matmul.cc "$ROOT"/src/ops.cc "$ROOT"/src/model.cc \
    "$ROOT"/src/threads.cc "$ROOT"/src/main.cc -o "$OURS" || exit 1
fi

echo "权重      : $WEIGHTS"
echo "线程      : $THREADS   （两边相同）"
echo "prompt    : ${PROMPT_TEXT:0:60}"
echo "prompt 长 : $N_TOK 个 token"
echo

echo "===== TinyGemmaCpp ====="
"$OURS" --weights "$WEIGHTS" --prompt-ids "$PROMPT_IDS" \
  --max-tokens "$TOKENS" --threads "$THREADS" --bench 2>&1 |
  grep -E "^\[bench\]|^\[tinygemma\] 就绪" || true

echo
echo "===== 官方 gemma.cpp ====="
if [ -n "$OFFICIAL" ] && [ -x "$OFFICIAL" ]; then
  "$OFFICIAL" --tokenizer "$TOKENIZER" --weights "$WEIGHTS" \
    --wrapping 0 --deterministic 1 --num_threads "$THREADS" --top_k 1 \
    --max_generated_tokens "$TOKENS" --prompt "$PROMPT_TEXT" --verbosity 2 2>&1 |
    grep -E "Timing info" || true
else
  echo "未指定官方可执行文件。构建 gemma.cpp 后这样调用："
  echo "  OFFICIAL=<gemma 可执行文件路径> bash tools/compare_official.sh"
fi
