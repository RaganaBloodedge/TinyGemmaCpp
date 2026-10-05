#!/usr/bin/env bash
# 定位 WSL 下 24 线程崩溃的原因。
# 猜想：libgomp 默认 OMP_WAIT_POLICY=active，屏障处自旋；24 个线程挤在
# 不到 24 个真正可用的 vCPU 上时，自旋会互相抢 CPU，造成雪崩式退化。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/tinygemma_linux}"
W="${WEIGHTS:-$(dirname "$ROOT")/gemma.cpp-main/weights/2.0-2b-it-sfp.sbs}"
IDS=2,651,6037,576,6081,603

run() {
  local label="$1"; shift
  printf "%-34s " "$label"
  env "$@" "$BIN" --weights "$W" --prompt-ids "$IDS" --max-tokens 8 --threads 24 --bench 2>&1 |
    grep -E "^\[bench\] decode" | sed 's/.*-> //' | tr '\n' ' '
  echo
}

echo "全部 24 线程，只改环境变量："
run "默认"                    DUMMY=1
run "OMP_WAIT_POLICY=passive" OMP_WAIT_POLICY=passive
run "OMP_WAIT_POLICY=active"  OMP_WAIT_POLICY=active
run "OMP_PROC_BIND=false"     OMP_PROC_BIND=false
run "OMP_WAIT_POLICY=passive + PROC_BIND=false" OMP_WAIT_POLICY=passive OMP_PROC_BIND=false
run "GOMP_SPINCOUNT=0"        GOMP_SPINCOUNT=0
