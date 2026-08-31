#!/usr/bin/env bash
# Single build entry for the COVE GPU SpMV binaries.
#
#   ./build.sh [ARCH]      ARCH default = auto (detect from GPU 0 via nvidia-smi)
#                          explicit: sm_120a (Blackwell), sm_90a (Hopper), sm_80 (Ampere)
#   CUDA_HOME=/path ./build.sh    to point at a specific CUDA toolkit
#                          (default: /usr/local/cuda, else newest /usr/local/cuda-*)
#
# GPU binaries need nvcc; host-only tools (struct_analyze/struct_spmv/struct_tests)
# are built via CMake (structural/CMakeLists.txt).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# -- CUDA toolkit: CUDA_HOME > /usr/local/cuda > newest /usr/local/cuda-* > PATH
if [ -z "${CUDA_HOME:-}" ] || [ ! -d "${CUDA_HOME:-}" ]; then
    CUDA_HOME=/usr/local/cuda
    [ -d "$CUDA_HOME" ] || CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 || true)"
fi
NVCC="${CUDA_HOME:+$CUDA_HOME/bin/nvcc}"
[ -x "${NVCC:-/nonexistent}" ] || NVCC="$(command -v nvcc 2>/dev/null || true)"
NVCC_REL="$([ -x "${NVCC:-/nonexistent}" ] && "$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' || true)"

# -- GPU arch: detect compute capability of GPU 0 unless given explicitly
ARCH="${1:-auto}"
if [ "$ARCH" = auto ]; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .' || true)"
    case "${CC:-}" in
        120|121)  ARCH="sm_${CC}a" ;;            # Blackwell workstation / Thor
        90)       ARCH=sm_90a ;;                 # Hopper
        80|86|89) ARCH="sm_$CC" ;;               # Ampere / Ada
        '')       ARCH=sm_120a
                  echo "WARN: could not detect a GPU (nvidia-smi); defaulting to $ARCH" ;;
        *)        ARCH="sm_$CC"
                  echo "WARN: untested compute capability ${CC}; trying $ARCH" ;;
    esac
fi
COMPUTE="compute_${ARCH#sm_}"
INC="-Istructural/include"
OUT="structural/build"
mkdir -p "$OUT"

NVFLAGS=(-O3 -std=c++17 --extended-lambda -gencode "arch=${COMPUTE},code=${ARCH}" $INC)

echo "== COVE build (arch=${ARCH}, nvcc=${NVCC:-none}${NVCC_REL:+, cuda-$NVCC_REL}) =="

gpu() {  # gpu <out-name> <src> [extra link flags...]
    local name="$1" src="$2"; shift 2
    printf '  [nvcc] %-28s\n' "$name"
    "$NVCC" "${NVFLAGS[@]}" -o "$OUT/$name" "$src" "$@"
}

if [ -n "${NVCC:-}" ] && [ -x "$NVCC" ]; then
    # ---- core operators ----
    gpu struct_gpu_csr_bitbsr_spmv structural/src/main_gpu_csr_bitbsr_spmv.cu -lcusparse  # hybrid (position) + JOINT value-position operator
    gpu struct_gpu_spmv            structural/src/main_gpu_spmv.cu            -lcusparse  # family runner: BitBSR-LB + value codecs + cuSPARSE baseline
    gpu struct_gpu_convert         structural/src/main_gpu_convert.cu         -lcusparse  # format-conversion tool
    # ---- evaluation harnesses (cited evidence) ----
    gpu bench_accuracy_robust      structural/src/bench_accuracy_robust.cu    -lcusparse            # value-axis accuracy (synthesis §13)
    gpu bench_cg_solve             structural/src/bench_cg_solve.cu           -lcusparse -lcublas   # iterative-solver (CG / IR / FCG) study
    gpu check_outlier_globalnorm   structural/src/check_outlier_globalnorm.cu -lcusparse            # global-norm output-error metric (§12)
else
    echo "  (no nvcc found -> skipping GPU binaries)"
fi

# ---- host-only analysis tool ----
printf '  [g++ ] %-28s\n' analyze_value_freq
g++ -O2 -std=c++17 $INC -o "$OUT/analyze_value_freq" structural/src/analyze_value_freq.cpp  # value-frequency concentration

echo "== done -> $OUT =="
