#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${ROOT_DIR}/structural/build/struct_gpu_spmv"
TMP_MTX="$(mktemp)"
TMP_PATTERN_MTX="$(mktemp)"
TMP_SIGN_MTX="$(mktemp)"
TMP_CONST_MTX="$(mktemp)"
TMP_SIGN_SCALE_MTX="$(mktemp)"
TMP_UNIQUE_MTX="$(mktemp)"
OUT_FILE="$(mktemp)"
ERR_FILE="$(mktemp)"
trap 'rm -f "${TMP_MTX}" "${TMP_PATTERN_MTX}" "${TMP_SIGN_MTX}" "${TMP_CONST_MTX}" "${TMP_SIGN_SCALE_MTX}" "${TMP_UNIQUE_MTX}" "${OUT_FILE}" "${ERR_FILE}"' EXIT

printf '%s\n' \
  '%%MatrixMarket matrix coordinate real general' \
  '4 8 5' \
  '1 1 1.0' \
  '1 8 2.0' \
  '2 4 3.0' \
  '3 2 4.0' \
  '4 7 5.0' >"${TMP_MTX}"

printf '%s\n' \
  '%%MatrixMarket matrix coordinate pattern general' \
  '4 8 5' \
  '1 1' \
  '1 8' \
  '2 4' \
  '3 2' \
  '4 7' >"${TMP_PATTERN_MTX}"

printf '%s\n' \
  '%%MatrixMarket matrix coordinate real general' \
  '8 4 32' >"${TMP_SIGN_MTX}"
for r in $(seq 1 8); do
  for c in $(seq 1 4); do
    if (( (r + c) % 2 == 0 )); then
      printf '%s %s 1.0\n' "${r}" "${c}" >>"${TMP_SIGN_MTX}"
    else
      printf '%s %s -1.0\n' "${r}" "${c}" >>"${TMP_SIGN_MTX}"
    fi
  done
done

printf '%s\n' \
  '%%MatrixMarket matrix coordinate real general' \
  '8 4 32' >"${TMP_CONST_MTX}"
for r in $(seq 1 8); do
  for c in $(seq 1 4); do
    printf '%s %s 2.5\n' "${r}" "${c}" >>"${TMP_CONST_MTX}"
  done
done

printf '%s\n' \
  '%%MatrixMarket matrix coordinate real general' \
  '8 4 32' >"${TMP_SIGN_SCALE_MTX}"
for r in $(seq 1 8); do
  for c in $(seq 1 4); do
    if (( (r + c) % 2 == 0 )); then
      printf '%s %s 2.5\n' "${r}" "${c}" >>"${TMP_SIGN_SCALE_MTX}"
    else
      printf '%s %s -2.5\n' "${r}" "${c}" >>"${TMP_SIGN_SCALE_MTX}"
    fi
  done
done

printf '%s\n' \
  '%%MatrixMarket matrix coordinate real general' \
  '8 4 32' >"${TMP_UNIQUE_MTX}"
for r in $(seq 1 8); do
  for c in $(seq 1 4); do
    case $(( (r + c) % 3 )) in
      0) value='2.0' ;;
      1) value='-3.0' ;;
      *) value='5.0' ;;
    esac
    printf '%s %s %s\n' "${r}" "${c}" "${value}" >>"${TMP_UNIQUE_MTX}"
  done
done

assert_report_contract() {
  local result="$1"
  grep -q '^report_schema: structural_spmv_cli_v3$' <<<"${result}"
}

run_allowed_layout() {
  local layout="$1"
  local operator_name="$2"
  local operator_family="$3"
  local operator_promotion_status="$4"
  local operator_runner="$5"
  local result
  result="$("${BIN}" "${TMP_MTX}" --layout "${layout}" --warmup 0 --iters 1 --verify-cpu)"
  assert_report_contract "${result}"
  grep -q "^operator_name: ${operator_name}$" <<<"${result}"
  grep -q "^operator_family: ${operator_family}$" <<<"${result}"
  grep -q "^operator_promotion_status: ${operator_promotion_status}$" <<<"${result}"
  grep -q "^operator_runner: ${operator_runner}$" <<<"${result}"
  grep -q "^spmv_effective_layout: ${layout}$" <<<"${result}"
  grep -q '^value_codec: original$' <<<"${result}"
  grep -q '^value_codec_exact: true$' <<<"${result}"
  grep -q '^value_codec_codebook_size: 0$' <<<"${result}"
  grep -q '^value_codec_payload_bytes: 40$' <<<"${result}"
  grep -q '^value_codec_bytes_per_nnz: 8$' <<<"${result}"
  grep -q '^verify_gate: exact_abs_or_rel$' <<<"${result}"
  grep -q '^timing_scope: event_excludes_output_init$' <<<"${result}"
  grep -q '^verify_cpu: PASS,' <<<"${result}"
}

run_allowed_layout col_bitmap64 original_nolb non_load_balanced_traversal buildable_needs_report bitbsr_original_traversal
run_allowed_layout col_bitmap64_flag_lb original_lb load_balanced_traversal promoted bitbsr_original_traversal

if "${BIN}" /tmp/structural_missing_deprecated_block.mtx --layout block_meta --warmup 0 --iters 1 >"${OUT_FILE}" 2>"${ERR_FILE}"; then
  echo 'expected block_meta layout to be rejected before matrix loading' >&2
  exit 1
fi
grep -q 'layout must be col_bitmap64, col_bitmap64_flag_lb, or cusparse_csr' "${ERR_FILE}"

if "${BIN}" /tmp/structural_missing_deprecated_delta.mtx --layout col_delta_u16_raw_lb --warmup 0 --iters 1 >"${OUT_FILE}" 2>"${ERR_FILE}"; then
  echo 'expected col_delta_u16_raw_lb layout to be rejected before matrix loading' >&2
  exit 1
fi
grep -q 'layout must be col_bitmap64, col_bitmap64_flag_lb, or cusparse_csr' "${ERR_FILE}"

if "${BIN}" /tmp/structural_missing_operator_spec_policy.mtx --layout col_bitmap64 --value-codec original --lb-policy binary_implicit_one --warmup 0 --iters 1 >"${OUT_FILE}" 2>"${ERR_FILE}"; then
  echo 'expected binary_implicit_one policy on non-LB original layout to be rejected before matrix loading' >&2
  exit 1
fi
grep -q 'lb-policy binary_implicit_one requires col_bitmap64_flag_lb layout and all_one value codec' "${ERR_FILE}"

if "${BIN}" /tmp/structural_missing_operator_spec_policy_value.mtx --layout col_bitmap64_flag_lb --value-codec original --lb-policy binary_implicit_one --warmup 0 --iters 1 >"${OUT_FILE}" 2>"${ERR_FILE}"; then
  echo 'expected binary_implicit_one policy with original value codec to be rejected before matrix loading' >&2
  exit 1
fi
grep -q 'lb-policy binary_implicit_one requires col_bitmap64_flag_lb layout and all_one value codec' "${ERR_FILE}"

k256_result="$("${BIN}" "${TMP_MTX}" --layout col_bitmap64_flag_lb --value-codec codebook256_u8 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${k256_result}"
grep -q '^operator_name: codebook256_u8_lb$' <<<"${k256_result}"
grep -q '^operator_family: controlled_lossy_value_specialized$' <<<"${k256_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${k256_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${k256_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${k256_result}"
grep -q '^value_codec: codebook256_u8$' <<<"${k256_result}"
grep -q '^verify_gate: spmv_max_rel$' <<<"${k256_result}"
grep -q '^timing_scope: event_excludes_output_init$' <<<"${k256_result}"
grep -q '^verify_cpu: PASS,' <<<"${k256_result}"

k256_nolb_result="$("${BIN}" "${TMP_MTX}" --layout col_bitmap64 --value-codec codebook256_u8 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${k256_nolb_result}"
grep -q '^operator_name: codebook256_u8_nolb$' <<<"${k256_nolb_result}"
grep -q '^operator_family: controlled_lossy_value_specialized$' <<<"${k256_nolb_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${k256_nolb_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${k256_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${k256_nolb_result}"
grep -q '^verify_gate: spmv_max_rel$' <<<"${k256_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${k256_nolb_result}"

run_lossy_codec() {
  local codec="$1"
  local expected_codebook_size="$2"
  local result
  result="$("${BIN}" "${TMP_SIGN_MTX}" --layout col_bitmap64_flag_lb --value-codec "${codec}" --warmup 0 --iters 1 --verify-cpu)"
  assert_report_contract "${result}"
  grep -q "^operator_name: ${codec}_lb$" <<<"${result}"
  grep -q '^operator_family: controlled_lossy_value_specialized$' <<<"${result}"
  grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${result}"
  grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${result}"
  grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${result}"
  grep -q "^value_codec: ${codec}$" <<<"${result}"
  grep -q '^value_codec_exact: false$' <<<"${result}"
  grep -q "^value_codec_codebook_size: ${expected_codebook_size}$" <<<"${result}"
  grep -q '^verify_gate: spmv_max_rel$' <<<"${result}"
  grep -q '^verify_cpu: PASS,' <<<"${result}"

  result="$("${BIN}" "${TMP_SIGN_MTX}" --layout col_bitmap64 --value-codec "${codec}" --warmup 0 --iters 1 --verify-cpu)"
  assert_report_contract "${result}"
  grep -q "^operator_name: ${codec}_nolb$" <<<"${result}"
  grep -q '^operator_family: controlled_lossy_value_specialized$' <<<"${result}"
  grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${result}"
  grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${result}"
  grep -q "^value_codec: ${codec}$" <<<"${result}"
  grep -q '^value_codec_exact: false$' <<<"${result}"
  grep -q "^value_codec_codebook_size: ${expected_codebook_size}$" <<<"${result}"
  grep -q '^verify_gate: spmv_max_rel$' <<<"${result}"
  grep -q '^verify_cpu: PASS,' <<<"${result}"
}

run_lossy_codec fp16 0
run_lossy_codec bf16 0
run_lossy_codec global_logkmeans16 2
run_lossy_codec global_logkmeans32 2
run_lossy_codec global_logkmeans64 2
run_lossy_codec global_logkmeans128 2

implicit_one_result="$("${BIN}" "${TMP_PATTERN_MTX}" --layout col_bitmap64_flag_lb --value-codec all_one --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${implicit_one_result}"
grep -q '^operator_name: all_one_lb$' <<<"${implicit_one_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${implicit_one_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${implicit_one_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${implicit_one_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${implicit_one_result}"
grep -q '^value_codec: all_one$' <<<"${implicit_one_result}"
grep -q '^value_codec_exact: true$' <<<"${implicit_one_result}"
grep -q '^value_codec_codebook_size: 0$' <<<"${implicit_one_result}"
grep -q '^value_codec_payload_bytes: 0$' <<<"${implicit_one_result}"
grep -q '^value_codec_bytes_per_nnz: 0$' <<<"${implicit_one_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${implicit_one_result}"
grep -q '^timing_scope: event_excludes_output_init$' <<<"${implicit_one_result}"
grep -q '^verify_cpu: PASS,' <<<"${implicit_one_result}"

implicit_one_nolb_result="$("${BIN}" "${TMP_PATTERN_MTX}" --layout col_bitmap64 --value-codec all_one --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${implicit_one_nolb_result}"
grep -q '^operator_name: all_one_nolb$' <<<"${implicit_one_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${implicit_one_nolb_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${implicit_one_nolb_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${implicit_one_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${implicit_one_nolb_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${implicit_one_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${implicit_one_nolb_result}"

implicit_one_policy_result="$("${BIN}" "${TMP_PATTERN_MTX}" --layout col_bitmap64_flag_lb --value-codec all_one --lb-policy binary_implicit_one --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${implicit_one_policy_result}"
grep -q '^operator_name: all_one_lb$' <<<"${implicit_one_policy_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${implicit_one_policy_result}"
grep -q '^operator_promotion_status: promoted$' <<<"${implicit_one_policy_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${implicit_one_policy_result}"
grep -q '^load_balance_policy: binary_implicit_one$' <<<"${implicit_one_policy_result}"
grep -q '^load_balance_chunk_blocks: 8$' <<<"${implicit_one_policy_result}"
grep -q '^load_balance_split_threshold_blocks: 8$' <<<"${implicit_one_policy_result}"
grep -q '^verify_cpu: PASS,' <<<"${implicit_one_policy_result}"

implicit_one_manual_lb_result="$("${BIN}" "${TMP_PATTERN_MTX}" --layout col_bitmap64_flag_lb --value-codec all_one --lb-policy binary_implicit_one --lb-chunk-blocks 64 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${implicit_one_manual_lb_result}"
grep -q '^operator_name: all_one_lb$' <<<"${implicit_one_manual_lb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${implicit_one_manual_lb_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${implicit_one_manual_lb_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${implicit_one_manual_lb_result}"
grep -q '^load_balance_policy: manual$' <<<"${implicit_one_manual_lb_result}"
grep -q '^load_balance_chunk_blocks: 64$' <<<"${implicit_one_manual_lb_result}"
grep -q '^load_balance_split_threshold_blocks: 64$' <<<"${implicit_one_manual_lb_result}"
grep -q '^verify_cpu: PASS,' <<<"${implicit_one_manual_lb_result}"

const_c_result="$("${BIN}" "${TMP_CONST_MTX}" --layout col_bitmap64_flag_lb --value-codec const_c --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${const_c_result}"
grep -q '^operator_name: const_c_lb$' <<<"${const_c_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${const_c_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${const_c_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${const_c_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${const_c_result}"
grep -q '^value_codec: const_c$' <<<"${const_c_result}"
grep -q '^value_codec_exact: true$' <<<"${const_c_result}"
grep -q '^value_codec_codebook_size: 1$' <<<"${const_c_result}"
grep -q '^value_codec_payload_bytes: 8$' <<<"${const_c_result}"
grep -q '^value_codec_bytes_per_nnz: 0.25$' <<<"${const_c_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${const_c_result}"
grep -q '^verify_cpu: PASS,' <<<"${const_c_result}"

const_c_nolb_result="$("${BIN}" "${TMP_CONST_MTX}" --layout col_bitmap64 --value-codec const_c --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${const_c_nolb_result}"
grep -q '^operator_name: const_c_nolb$' <<<"${const_c_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${const_c_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${const_c_nolb_result}"
grep -q '^value_codec: const_c$' <<<"${const_c_nolb_result}"
grep -q '^value_codec_exact: true$' <<<"${const_c_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${const_c_nolb_result}"

sign2_result="$("${BIN}" "${TMP_SIGN_MTX}" --layout col_bitmap64_flag_lb --value-codec sign2 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${sign2_result}"
grep -q '^operator_name: sign2_lb$' <<<"${sign2_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${sign2_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${sign2_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${sign2_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${sign2_result}"
grep -q '^value_codec: sign2$' <<<"${sign2_result}"
grep -q '^value_codec_exact: true$' <<<"${sign2_result}"
grep -q '^value_codec_codebook_size: 0$' <<<"${sign2_result}"
grep -q '^value_codec_payload_bytes: 4$' <<<"${sign2_result}"
grep -q '^value_codec_bytes_per_nnz: 0.125$' <<<"${sign2_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${sign2_result}"
grep -q '^timing_scope: event_excludes_output_init$' <<<"${sign2_result}"
grep -q '^verify_cpu: PASS,' <<<"${sign2_result}"

sign2_nolb_result="$("${BIN}" "${TMP_SIGN_MTX}" --layout col_bitmap64 --value-codec sign2 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${sign2_nolb_result}"
grep -q '^operator_name: sign2_nolb$' <<<"${sign2_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${sign2_nolb_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${sign2_nolb_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${sign2_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${sign2_nolb_result}"
grep -q '^value_codec: sign2$' <<<"${sign2_nolb_result}"
grep -q '^value_codec_exact: true$' <<<"${sign2_nolb_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${sign2_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${sign2_nolb_result}"

sign2_scale_result="$("${BIN}" "${TMP_SIGN_SCALE_MTX}" --layout col_bitmap64_flag_lb --value-codec sign2_scale --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${sign2_scale_result}"
grep -q '^operator_name: sign2_scale_lb$' <<<"${sign2_scale_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${sign2_scale_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${sign2_scale_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${sign2_scale_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${sign2_scale_result}"
grep -q '^value_codec: sign2_scale$' <<<"${sign2_scale_result}"
grep -q '^value_codec_exact: true$' <<<"${sign2_scale_result}"
grep -q '^value_codec_codebook_size: 1$' <<<"${sign2_scale_result}"
grep -q '^value_codec_payload_bytes: 12$' <<<"${sign2_scale_result}"
grep -q '^value_codec_bytes_per_nnz: 0.375$' <<<"${sign2_scale_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${sign2_scale_result}"
grep -q '^verify_cpu: PASS,' <<<"${sign2_scale_result}"

sign2_scale_nolb_result="$("${BIN}" "${TMP_SIGN_SCALE_MTX}" --layout col_bitmap64 --value-codec sign2_scale --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${sign2_scale_nolb_result}"
grep -q '^operator_name: sign2_scale_nolb$' <<<"${sign2_scale_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${sign2_scale_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${sign2_scale_nolb_result}"
grep -q '^value_codec: sign2_scale$' <<<"${sign2_scale_nolb_result}"
grep -q '^value_codec_exact: true$' <<<"${sign2_scale_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${sign2_scale_nolb_result}"

dict2_result="$("${BIN}" "${TMP_SIGN_MTX}" --layout col_bitmap64_flag_lb --value-codec dict2 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${dict2_result}"
grep -q '^operator_name: dict2_lb$' <<<"${dict2_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${dict2_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${dict2_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${dict2_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${dict2_result}"
grep -q '^value_codec: dict2$' <<<"${dict2_result}"
grep -q '^value_codec_exact: true$' <<<"${dict2_result}"
grep -q '^value_codec_codebook_size: 2$' <<<"${dict2_result}"
grep -q '^value_codec_payload_bytes: 20$' <<<"${dict2_result}"
grep -q '^value_codec_bytes_per_nnz: 0.625$' <<<"${dict2_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${dict2_result}"
grep -q '^verify_cpu: PASS,' <<<"${dict2_result}"

dict2_nolb_result="$("${BIN}" "${TMP_SIGN_MTX}" --layout col_bitmap64 --value-codec dict2 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${dict2_nolb_result}"
grep -q '^operator_name: dict2_nolb$' <<<"${dict2_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${dict2_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${dict2_nolb_result}"
grep -q '^value_codec: dict2$' <<<"${dict2_nolb_result}"
grep -q '^value_codec_exact: true$' <<<"${dict2_nolb_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${dict2_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${dict2_nolb_result}"

unique16_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64_flag_lb --value-codec dict16 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${unique16_result}"
grep -q '^operator_name: dict16_lb$' <<<"${unique16_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${unique16_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${unique16_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${unique16_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${unique16_result}"
grep -q '^value_codec: dict16$' <<<"${unique16_result}"
grep -q '^value_codec_exact: true$' <<<"${unique16_result}"
grep -q '^value_codec_codebook_size: 3$' <<<"${unique16_result}"
grep -q '^value_codec_payload_bytes: 40$' <<<"${unique16_result}"
grep -q '^value_codec_bytes_per_nnz: 1.25$' <<<"${unique16_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${unique16_result}"
grep -q '^timing_scope: event_excludes_output_init$' <<<"${unique16_result}"
grep -q '^verify_cpu: PASS,' <<<"${unique16_result}"

unique16_nolb_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64 --value-codec dict16 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${unique16_nolb_result}"
grep -q '^operator_name: dict16_nolb$' <<<"${unique16_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${unique16_nolb_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${unique16_nolb_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${unique16_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${unique16_nolb_result}"
grep -q '^value_codec: dict16$' <<<"${unique16_nolb_result}"
grep -q '^value_codec_exact: true$' <<<"${unique16_nolb_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${unique16_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${unique16_nolb_result}"

unique4_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64_flag_lb --value-codec dict4 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${unique4_result}"
grep -q '^operator_name: dict4_lb$' <<<"${unique4_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${unique4_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${unique4_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${unique4_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${unique4_result}"
grep -q '^value_codec: dict4$' <<<"${unique4_result}"
grep -q '^value_codec_exact: true$' <<<"${unique4_result}"
grep -q '^value_codec_codebook_size: 3$' <<<"${unique4_result}"
grep -q '^value_codec_payload_bytes: 32$' <<<"${unique4_result}"
grep -q '^value_codec_bytes_per_nnz: 1$' <<<"${unique4_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${unique4_result}"
grep -q '^timing_scope: event_excludes_output_init$' <<<"${unique4_result}"
grep -q '^verify_cpu: PASS,' <<<"${unique4_result}"

unique4_nolb_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64 --value-codec dict4 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${unique4_nolb_result}"
grep -q '^operator_name: dict4_nolb$' <<<"${unique4_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${unique4_nolb_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${unique4_nolb_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${unique4_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${unique4_nolb_result}"
grep -q '^value_codec: dict4$' <<<"${unique4_nolb_result}"
grep -q '^value_codec_exact: true$' <<<"${unique4_nolb_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${unique4_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${unique4_nolb_result}"

dict256_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64_flag_lb --value-codec dict256_u8 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${dict256_result}"
grep -q '^operator_name: dict256_u8_lb$' <<<"${dict256_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${dict256_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${dict256_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${dict256_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${dict256_result}"
grep -q '^value_codec: dict256_u8$' <<<"${dict256_result}"
grep -q '^value_codec_exact: true$' <<<"${dict256_result}"
grep -q '^value_codec_codebook_size: 3$' <<<"${dict256_result}"
grep -q '^value_codec_payload_bytes: 56$' <<<"${dict256_result}"
grep -q '^value_codec_bytes_per_nnz: 1.75$' <<<"${dict256_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${dict256_result}"
grep -q '^verify_cpu: PASS,' <<<"${dict256_result}"

dict256_nolb_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64 --value-codec dict256_u8 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${dict256_nolb_result}"
grep -q '^operator_name: dict256_u8_nolb$' <<<"${dict256_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${dict256_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${dict256_nolb_result}"
grep -q '^value_codec: dict256_u8$' <<<"${dict256_nolb_result}"
grep -q '^value_codec_exact: true$' <<<"${dict256_nolb_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${dict256_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${dict256_nolb_result}"

dict65536_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64_flag_lb --value-codec dict65536_u16 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${dict65536_result}"
grep -q '^operator_name: dict65536_u16_lb$' <<<"${dict65536_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${dict65536_result}"
grep -q '^operator_promotion_status: buildable_needs_report$' <<<"${dict65536_result}"
grep -q '^operator_runner: bitbsr_value_specialized$' <<<"${dict65536_result}"
grep -q '^spmv_effective_layout: col_bitmap64_flag_lb$' <<<"${dict65536_result}"
grep -q '^value_codec: dict65536_u16$' <<<"${dict65536_result}"
grep -q '^value_codec_exact: true$' <<<"${dict65536_result}"
grep -q '^value_codec_codebook_size: 3$' <<<"${dict65536_result}"
grep -q '^value_codec_payload_bytes: 88$' <<<"${dict65536_result}"
grep -q '^value_codec_bytes_per_nnz: 2.75$' <<<"${dict65536_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${dict65536_result}"
grep -q '^verify_cpu: PASS,' <<<"${dict65536_result}"

dict65536_nolb_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64 --value-codec dict65536_u16 --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${dict65536_nolb_result}"
grep -q '^operator_name: dict65536_u16_nolb$' <<<"${dict65536_nolb_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${dict65536_nolb_result}"
grep -q '^spmv_effective_layout: col_bitmap64$' <<<"${dict65536_nolb_result}"
grep -q '^value_codec: dict65536_u16$' <<<"${dict65536_nolb_result}"
grep -q '^value_codec_exact: true$' <<<"${dict65536_nolb_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${dict65536_nolb_result}"
grep -q '^verify_cpu: PASS,' <<<"${dict65536_nolb_result}"

auto_pattern_result="$("${BIN}" "${TMP_PATTERN_MTX}" --layout col_bitmap64_flag_lb --value-codec auto_lossless --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${auto_pattern_result}"
grep -q '^operator_name: all_one_lb$' <<<"${auto_pattern_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${auto_pattern_result}"
grep -q '^value_codec: auto_lossless$' <<<"${auto_pattern_result}"
grep -q '^value_codec_effective: all_one$' <<<"${auto_pattern_result}"
grep -q '^value_codec_auto_selected: true$' <<<"${auto_pattern_result}"
grep -q '^value_codec_auto_unique_count: 1$' <<<"${auto_pattern_result}"
grep -q '^value_codec_auto_candidate_payload_bytes: 0$' <<<"${auto_pattern_result}"
grep -q 'value_codec_auto_select_event=' <<<"${auto_pattern_result}"
grep -q '^verify_cpu: PASS,' <<<"${auto_pattern_result}"

auto_const_result="$("${BIN}" "${TMP_CONST_MTX}" --layout col_bitmap64_flag_lb --value-codec auto_lossless --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${auto_const_result}"
grep -q '^operator_name: const_c_lb$' <<<"${auto_const_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${auto_const_result}"
grep -q '^value_codec_effective: const_c$' <<<"${auto_const_result}"
grep -q '^value_codec_auto_selected: true$' <<<"${auto_const_result}"
grep -q '^value_codec_auto_unique_count: 1$' <<<"${auto_const_result}"
grep -q '^value_codec_auto_candidate_payload_bytes: 8$' <<<"${auto_const_result}"
grep -q 'value_codec_auto_select_event=' <<<"${auto_const_result}"
grep -q '^verify_cpu: PASS,' <<<"${auto_const_result}"

auto_sign_result="$("${BIN}" "${TMP_SIGN_MTX}" --layout col_bitmap64_flag_lb --value-codec auto_lossless --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${auto_sign_result}"
grep -q '^operator_name: sign2_lb$' <<<"${auto_sign_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${auto_sign_result}"
grep -q '^value_codec_effective: sign2$' <<<"${auto_sign_result}"
grep -q '^value_codec_auto_selected: true$' <<<"${auto_sign_result}"
grep -q '^value_codec_auto_unique_count: 2$' <<<"${auto_sign_result}"
grep -q '^value_codec_auto_candidate_payload_bytes: 4$' <<<"${auto_sign_result}"
grep -q 'value_codec_auto_select_event=' <<<"${auto_sign_result}"
grep -q '^verify_cpu: PASS,' <<<"${auto_sign_result}"

auto_sign_scale_result="$("${BIN}" "${TMP_SIGN_SCALE_MTX}" --layout col_bitmap64_flag_lb --value-codec auto_lossless --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${auto_sign_scale_result}"
grep -q '^operator_name: sign2_scale_lb$' <<<"${auto_sign_scale_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${auto_sign_scale_result}"
grep -q '^value_codec_effective: sign2_scale$' <<<"${auto_sign_scale_result}"
grep -q '^value_codec_auto_selected: true$' <<<"${auto_sign_scale_result}"
grep -q '^value_codec_auto_unique_count: 2$' <<<"${auto_sign_scale_result}"
grep -q '^value_codec_auto_candidate_payload_bytes: 12$' <<<"${auto_sign_scale_result}"
grep -q 'value_codec_auto_select_event=' <<<"${auto_sign_scale_result}"
grep -q '^verify_cpu: PASS,' <<<"${auto_sign_scale_result}"

auto_unique_result="$("${BIN}" "${TMP_UNIQUE_MTX}" --layout col_bitmap64_flag_lb --value-codec auto_lossless --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${auto_unique_result}"
grep -q '^operator_name: dict4_lb$' <<<"${auto_unique_result}"
grep -q '^operator_family: lossless_value_specialized$' <<<"${auto_unique_result}"
grep -q '^value_codec_effective: dict4$' <<<"${auto_unique_result}"
grep -q '^value_codec_codebook_size: 3$' <<<"${auto_unique_result}"
grep -q '^value_codec_auto_selected: true$' <<<"${auto_unique_result}"
grep -q '^value_codec_auto_unique_count: 3$' <<<"${auto_unique_result}"
grep -q '^value_codec_auto_candidate_payload_bytes: 32$' <<<"${auto_unique_result}"
grep -q 'value_codec_auto_select_event=' <<<"${auto_unique_result}"
grep -q '^verify_cpu: PASS,' <<<"${auto_unique_result}"

auto_raw_result="$("${BIN}" "${TMP_MTX}" --layout col_bitmap64_flag_lb --value-codec auto_lossless --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${auto_raw_result}"
grep -q '^operator_name: original_lb$' <<<"${auto_raw_result}"
grep -q '^operator_family: load_balanced_traversal$' <<<"${auto_raw_result}"
grep -q '^value_codec: auto_lossless$' <<<"${auto_raw_result}"
grep -q '^value_codec_effective: original$' <<<"${auto_raw_result}"
grep -q '^value_codec_auto_selected: true$' <<<"${auto_raw_result}"
grep -q '^value_codec_auto_unique_count: 5$' <<<"${auto_raw_result}"
grep -q '^value_codec_auto_candidate_payload_bytes: 40$' <<<"${auto_raw_result}"
grep -q 'value_codec_auto_select_event=' <<<"${auto_raw_result}"
grep -q '^verify_cpu: PASS,' <<<"${auto_raw_result}"

cusparse_result="$("${BIN}" "${TMP_MTX}" --layout cusparse_csr --warmup 0 --iters 1 --verify-cpu)"
assert_report_contract "${cusparse_result}"
grep -q '^operator_name: cusparse_csr$' <<<"${cusparse_result}"
grep -q '^operator_family: external_baseline$' <<<"${cusparse_result}"
grep -q '^operator_promotion_status: external_baseline$' <<<"${cusparse_result}"
grep -q '^operator_runner: cusparse_csr$' <<<"${cusparse_result}"
grep -q '^spmv_effective_layout: cusparse_csr$' <<<"${cusparse_result}"
grep -q '^value_codec: original$' <<<"${cusparse_result}"
grep -q '^value_codec_exact: true$' <<<"${cusparse_result}"
grep -q '^value_codec_codebook_size: 0$' <<<"${cusparse_result}"
grep -q '^value_codec_payload_bytes: 40$' <<<"${cusparse_result}"
grep -q '^value_codec_bytes_per_nnz: 8$' <<<"${cusparse_result}"
grep -q '^verify_gate: exact_abs_or_rel$' <<<"${cusparse_result}"
grep -q '^timing_scope: cusparse_event_includes_beta_zero$' <<<"${cusparse_result}"
grep -q '^verify_cpu: PASS,' <<<"${cusparse_result}"

for layout in \
  soa \
  delta_u16 \
  delta_packed \
  delta_packed_lb \
  delta_packed_lb_auto \
  col_bitmap64_lb \
  col_hybrid64 \
  col_hybrid64_flag_lb \
  auto; do
  if "${BIN}" "${TMP_MTX}" --layout "${layout}" --warmup 0 --iters 1 >"${OUT_FILE}" 2>"${ERR_FILE}"; then
    echo "expected --layout ${layout} to be rejected" >&2
    exit 1
  fi
  grep -q 'layout must be' "${ERR_FILE}"
done
