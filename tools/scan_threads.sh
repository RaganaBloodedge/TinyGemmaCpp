#!/usr/bin/env bash
# 线程数扫描：同一二进制，逐个线程数跑 REPS 次，取最快的一次。
#
# 用法：
#   tools/scan_threads.sh <二进制> [权重路径] [线程列表] [token 数]
# 例：
#   tools/scan_threads.sh ./build/tinygemma.exe
#   tools/scan_threads.sh ./build/tinygemma_linux "" "8 12 16 24" 32
#
# 不传权重时按同级的 gemma.cpp-main/weights/ 找，也可用 WEIGHTS 环境变量指定。
#
# 取最快值而不是平均值：机器上还有别的负载，平均值会把干扰算进来，
# 而"最快的一次"更接近这台机器在无人抢 CPU 时的真实能力。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

BIN="${1:?用法: scan_threads.sh <二进制> [权重] [线程列表] [token 数]}"
W="${2:-${WEIGHTS:-$(dirname "$ROOT")/gemma.cpp-main/weights/2.0-2b-it-sfp.sbs}}"
THREADS="${3:-4 8 12 16 20 24}"
NGEN="${4:-32}"
REPS="${REPS:-3}"
IDS="${IDS:-2,651,6037,576,6081,603}"

if [ ! -x "$BIN" ]; then
  echo "找不到可执行文件: $BIN" >&2
  exit 1
fi

# 声明式前缀，跨平台一致（Linux 用 env，Windows 下本来就是 bash 也能用）
run_once() {
  "$BIN" --weights "$W" --prompt-ids "$IDS" --max-tokens "$NGEN" \
         --threads "$1" --bench 2>&1 |
    grep -E "^\[bench\] decode" |
    sed -E 's/.*\(([0-9.]+) ms\/tok\).*/\1/'
}

printf "binary   : %s\n" "$BIN"
printf "threads  : %s  (reps=%s, gen=%s tok)\n\n" "$THREADS" "$REPS" "$NGEN"

for t in $THREADS; do
  best=""
  all=""
  for _ in $(seq "$REPS"); do
    v="$(run_once "$t")"
    [ -z "$v" ] && continue
    all="$all $v"
    if [ -z "$best" ] || awk -v a="$v" -v b="$best" 'BEGIN{exit !(a<b)}'; then
      best="$v"
    fi
  done
  if [ -z "$best" ]; then
    printf "  threads=%-3s  失败\n" "$t"
  else
    printf "  threads=%-3s  best %7s ms/tok   (%s tok/s)   [%s ]\n" \
      "$t" "$best" "$(awk -v x="$best" 'BEGIN{printf "%.2f", 1000/x}')" "$all"
  fi
done
