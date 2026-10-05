#!/usr/bin/env bash
# 排查 Linux 侧 OpenMP 行为对 prefill 的影响。
#
# 背景：同一份源码，Windows (MSYS2 GCC) 编译的 prefill 比 WSL (Linux GCC)
# 快约 30%。怀疑差异出在 libgomp 的线程亲和/自旋策略上——早先在 24 线程下
# 已经观察到 WSL 会过度订阅到慢一倍。
#
# 用法： wsl -d Ubuntu -- bash <仓库路径>/tools/diag_omp.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/tinygemma_linux}"
SIBLING="$(dirname "$ROOT")/gemma.cpp-main/weights"
WEIGHTS="${WEIGHTS:-$SIBLING/2.0-2b-it-sfp.sbs}"
TOKENIZER="${TOKENIZER:-$SIBLING/tokenizer.spm}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/tools/prompts/long_text.txt}"
THREADS="${THREADS:-16}"

if [ ! -x "$BIN" ]; then
  echo "找不到 $BIN，先编译" >&2
  exit 1
fi
if [ ! -f "$PROMPT_FILE" ]; then
  echo "找不到 prompt 文件 $PROMPT_FILE" >&2
  exit 1
fi

IDS="$(python3 - "$ROOT" "$TOKENIZER" "$PROMPT_FILE" <<'PY'
import sys
root, tok, pf = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root + "/tools")
from spm import SentencePieceModel
m = SentencePieceModel.load(tok)
text = open(pf, encoding="utf-8").read().strip()
print(",".join(str(i) for i in [m._piece_to_id["<bos>"]] + m.encode(text)))
PY
)"

echo "编译器: $(g++ --version | head -1)"
echo "线程  : $THREADS"
echo

run() {
  local label="$1"; shift
  printf "  %-38s " "$label"
  # 每次都只跑一次；表里的目的是比较相对差异，不是绝对值。
  env "$@" "$BIN" --weights "$WEIGHTS" --prompt-ids "$IDS" \
    --max-tokens 2 --threads "$THREADS" --bench 2>&1 |
    grep -oE "prefill [0-9]+ tok  [0-9.]+ s" | tr '\n' ' '
  echo
}

echo "prefill："
run "默认"                          __UNSET__=1
run "PROC_BIND=true PLACES=cores"   OMP_PROC_BIND=true OMP_PLACES=cores
run "PROC_BIND=close"               OMP_PROC_BIND=close
run "PROC_BIND=false"               OMP_PROC_BIND=false
run "WAIT_POLICY=passive"           OMP_WAIT_POLICY=passive
run "WAIT_POLICY=active"            OMP_WAIT_POLICY=active
run "GOMP_SPINCOUNT=0"              GOMP_SPINCOUNT=0
run "OMP_DYNAMIC=false"             OMP_DYNAMIC=false
