#!/usr/bin/env bash
# 排查：同样在 WSL 里，为什么 TinyGemmaCpp 比官方慢一个数量级。
# 分别扫线程数与 OMP_PROC_BIND，看是否符合"线程调度/亲和性"的猜想。
#
#   wsl -d Ubuntu -- bash <仓库路径>/tools/diag_wsl.sh
#
# OFFICIAL_DIR 指向自己构建的 gemma.cpp 目录；不设就只跑本引擎那两组。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="${WEIGHTS_DIR:-$(dirname "$ROOT")/gemma.cpp-main/weights}"
IDS=2,651,6037,576,6081,603

echo "== 线程数扫描 =="
for T in 1 4 8 16 24; do
  printf "threads=%-3s " "$T"
  "$ROOT/build/tinygemma_linux" --weights "$W/2.0-2b-it-sfp.sbs" \
    --prompt-ids "$IDS" --max-tokens 8 --threads "$T" --bench 2>&1 |
    grep -E "^\[bench\] decode" | tr '\n' ' '
  echo
done

echo
echo "== OMP_PROC_BIND 影响（24 线程）=="
for PB in false true spread close; do
  printf "proc_bind=%-7s " "$PB"
  OMP_PROC_BIND=$PB "$ROOT/build/tinygemma_linux" --weights "$W/2.0-2b-it-sfp.sbs" \
    --prompt-ids "$IDS" --max-tokens 8 --threads 24 --bench 2>&1 |
    grep -E "^\[bench\] decode" | tr '\n' ' '
  echo
done

echo
echo "== 官方 gemma.cpp 同样 8 token =="
if [ -n "${OFFICIAL_DIR:-}" ] && [ -x "$OFFICIAL_DIR/gemma" ]; then
  (cd "$OFFICIAL_DIR" && ./gemma --tokenizer "$W/tokenizer.spm" \
    --weights "$W/2.0-2b-it-sfp.sbs" --wrapping 0 --deterministic 1 \
    --num_threads 24 --top_k 1 --max_generated_tokens 8 \
    --prompt 'The capital of France is' --verbosity 2 2>&1 | grep -E "Timing")
else
  echo "未指定 OFFICIAL_DIR（自己构建的 gemma.cpp 目录），跳过。"
fi
