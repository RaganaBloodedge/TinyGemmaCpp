#!/usr/bin/env bash
# 直接编译（不需要 cmake）。Git Bash / Linux / WSL 都能跑。
#
#   ./build.sh            # 编译到 build/tinygemma
#   ./build.sh --tests    # 额外编译单元测试
#   ./build.sh --bench    # 额外编译 GEMM 内核微基准
#   CXX=clang++ ./build.sh
#   ./build.sh clean
set -euo pipefail

cd "$(dirname "$0")"

if [ "${1:-}" = "clean" ]; then
  rm -rf build
  echo "已清理 build/"
  exit 0
fi

CXX="${CXX:-g++}"
OUT_DIR="build"
BIN="$OUT_DIR/tinygemma"

FLAGS=(-std=c++20 -O3 -fopenmp -Wall -Wextra -Isrc)
# -march=native 让编译器用 AVX2/AVX-512/FMA 自动向量化，是本项目最大的单点提速。
# 只在能编译时才加，避免移植到老平台上直接失败。
if echo 'int main(){return 0;}' | "$CXX" -std=c++20 -march=native -fsyntax-only -x c++ - 2>/dev/null; then
  FLAGS+=(-march=native)
else
  echo "[build] 警告: 不支持 -march=native，退化为通用指令集"
fi

mkdir -p "$OUT_DIR"
echo "[build] $CXX ${FLAGS[*]}"
"$CXX" "${FLAGS[@]}" \
  src/config.cc src/sbs_reader.cc src/weights.cc src/matmul.cc src/ops.cc \
  src/model.cc src/threads.cc src/main.cc \
  -o "$BIN"

echo "[build] 完成 -> $BIN"

# 单元测试（可选，--tests 时编译）
if [ "${1:-}" = "--tests" ] || [ "${1:-}" = "tests" ]; then
  "$CXX" "${FLAGS[@]}" src/config.cc src/sbs_reader.cc src/weights.cc \
    src/matmul.cc src/ops.cc src/model.cc tests/test_ops.cc -o "$OUT_DIR/test_ops"
  "$CXX" "${FLAGS[@]}" src/config.cc src/sbs_reader.cc src/weights.cc \
    src/matmul.cc src/ops.cc src/model.cc tests/test_sbs.cc -o "$OUT_DIR/test_sbs"
  "$CXX" "${FLAGS[@]}" src/config.cc src/sbs_reader.cc src/weights.cc \
    src/matmul.cc src/ops.cc src/model.cc tests/test_matmul.cc -o "$OUT_DIR/test_matmul"
  "$CXX" "${FLAGS[@]}" src/threads.cc tests/test_threads.cc -o "$OUT_DIR/test_threads"
  echo "[build] 测试程序 -> $OUT_DIR/test_ops, test_sbs, test_matmul, test_threads"
fi

# 内核微基准（不需要权重文件，秒级出结果）
if [ "${1:-}" = "--bench" ] || [ "${1:-}" = "bench" ]; then
  "$CXX" "${FLAGS[@]}" src/config.cc src/sbs_reader.cc src/weights.cc \
    src/matmul.cc src/ops.cc src/model.cc tests/bench_kernels.cc \
    -o "$OUT_DIR/bench_kernels"
  "$CXX" "${FLAGS[@]}" src/config.cc src/sbs_reader.cc src/weights.cc \
    src/matmul.cc src/ops.cc src/model.cc tests/bench_gemm_batch.cc \
    -o "$OUT_DIR/bench_gemm_batch"
  echo "[build] 微基准 -> $OUT_DIR/bench_kernels, bench_gemm_batch"
fi
