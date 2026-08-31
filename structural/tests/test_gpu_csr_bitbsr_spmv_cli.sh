#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${ROOT_DIR}/structural/build/struct_gpu_csr_bitbsr_spmv"
TMP_MTX="$(mktemp)"
OUT_FILE="$(mktemp)"
ERR_FILE="$(mktemp)"
trap 'rm -f "${TMP_MTX}" "${OUT_FILE}" "${ERR_FILE}"' EXIT

printf '%s\n' \
  '%%MatrixMarket matrix coordinate real general' \
  '4 8 5' \
  '1 1 1.0' \
  '1 8 2.0' \
  '2 4 3.0' \
  '3 2 4.0' \
  '4 7 5.0' >"${TMP_MTX}"

result="$("${BIN}" "${TMP_MTX}" --residual-threshold 2 --warmup 0 --iters 1 --verify-cpu)"

grep -q '^spmv_effective_layout: csr_bitbsr_fused_lb$' <<<"${result}"
grep -q '^block: 8 x 4$' <<<"${result}"
grep -q '^residual_threshold: 2$' <<<"${result}"
grep -q '^split_policy: cost_gated$' <<<"${result}"
grep -q '^kept_blocks: 1$' <<<"${result}"
grep -q '^residual_blocks: 1$' <<<"${result}"
grep -q '^verify_cpu: PASS,' <<<"${result}"

two_pass_result="$("${BIN}" "${TMP_MTX}" --mode two_pass --residual-threshold 2 \
  --split-policy threshold \
  --warmup 0 --iters 1 --verify-cpu)"

grep -q '^spmv_effective_layout: csr_bitbsr_two_pass$' <<<"${two_pass_result}"
grep -q '^verify_cpu: PASS,' <<<"${two_pass_result}"

cost_gated_result="$("${BIN}" "${TMP_MTX}" --split-policy cost_gated \
  --residual-threshold 3 --cost-bitmap-block 2 --cost-csr-nnz 1 \
  --warmup 0 --iters 1 --verify-cpu)"

grep -q '^split_policy: cost_gated$' <<<"${cost_gated_result}"
grep -q '^kept_blocks: 2$' <<<"${cost_gated_result}"
grep -q '^residual_blocks: 0$' <<<"${cost_gated_result}"
grep -q '^cost_skipped_blocks: 2$' <<<"${cost_gated_result}"
grep -q '^verify_cpu: PASS,' <<<"${cost_gated_result}"

brcoo_result="$("${BIN}" "${TMP_MTX}" --residual-format brcoo \
  --residual-threshold 2 --warmup 0 --iters 1 --verify-cpu)"

grep -q '^spmv_effective_layout: csr_bitbsr_brcoo_fused_lb$' <<<"${brcoo_result}"
grep -q '^residual_format: brcoo$' <<<"${brcoo_result}"
grep -q '^verify_cpu: PASS,' <<<"${brcoo_result}"

sell_result="$("${BIN}" "${TMP_MTX}" --residual-format sell \
  --residual-threshold 2 --warmup 0 --iters 1 --verify-cpu)"

grep -q '^spmv_effective_layout: csr_bitbsr_sell_fused_lb$' <<<"${sell_result}"
grep -q '^residual_format: sell$' <<<"${sell_result}"
grep -q '^verify_cpu: PASS,' <<<"${sell_result}"

if "${BIN}" "${TMP_MTX}" --residual-threshold -1 --warmup 0 --iters 1 \
  >"${OUT_FILE}" 2>"${ERR_FILE}"; then
  echo "expected negative residual threshold to be rejected" >&2
  exit 1
fi
grep -q 'residual-threshold must be non-negative' "${ERR_FILE}"
