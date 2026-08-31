#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${ROOT_DIR}/structural/build/struct_gpu_convert"
TMP_MTX="$(mktemp)"
DIRECT_OUT="$(mktemp)"
DIRECT_ERR="$(mktemp)"
SEGMENTED_OUT="$(mktemp)"
SEGMENTED_ERR="$(mktemp)"
HOST_OUT="$(mktemp)"
HOST_ERR="$(mktemp)"
GLOBAL_OUT="$(mktemp)"
GLOBAL_ERR="$(mktemp)"
WARP8_OUT="$(mktemp)"
WARP8_ERR="$(mktemp)"
SHAPE_OUT="$(mktemp)"
SHAPE_ERR="$(mktemp)"
trap 'rm -f "${TMP_MTX}" "${DIRECT_OUT}" "${DIRECT_ERR}" "${SEGMENTED_OUT}" "${SEGMENTED_ERR}" "${HOST_OUT}" "${HOST_ERR}" "${GLOBAL_OUT}" "${GLOBAL_ERR}" "${WARP8_OUT}" "${WARP8_ERR}" "${SHAPE_OUT}" "${SHAPE_ERR}"' EXIT

cat > "${TMP_MTX}" <<'MTX'
%%MatrixMarket matrix coordinate real general
4 5 4
1 1 1.0
2 4 2.0
3 2 3.0
4 5 4.0
MTX

default_result="$("${BIN}" "${TMP_MTX}")"

grep -q '^gpu_convert_mode: warp_direct$' <<< "${default_result}"
grep -q '^gpu_output_residency: device$' <<< "${default_result}"
grep -q '^verify_cpu: SKIP_DEVICE_RESIDENT$' <<< "${default_result}"

if "${BIN}" "${TMP_MTX}" --verify-cpu >"${HOST_OUT}" 2>"${HOST_ERR}"; then
  echo "expected --verify-cpu host materialization to be retired" >&2
  exit 1
fi
grep -q 'high-performance conversion path is device-resident only' "${HOST_ERR}"

if "${BIN}" "${TMP_MTX}" --host-output >"${HOST_OUT}" 2>"${HOST_ERR}"; then
  echo "expected --host-output to be retired" >&2
  exit 1
fi
grep -q 'high-performance conversion path is device-resident only' "${HOST_ERR}"

if "${BIN}" "${TMP_MTX}" --materialize >"${HOST_OUT}" 2>"${HOST_ERR}"; then
  echo "expected --materialize to be retired" >&2
  exit 1
fi
grep -q 'high-performance conversion path is device-resident only' "${HOST_ERR}"

if "${BIN}" "${TMP_MTX}" --mode global_sort >"${GLOBAL_OUT}" 2>"${GLOBAL_ERR}"; then
  echo "expected --mode global_sort to be retired" >&2
  exit 1
fi
grep -q 'only warp_direct 8x4 is exposed' "${GLOBAL_ERR}"

if "${BIN}" "${TMP_MTX}" --br 8 --bc 8 --mode warp_direct_8x8 >"${WARP8_OUT}" 2>"${WARP8_ERR}"; then
  echo "expected --mode warp_direct_8x8 to be retired" >&2
  exit 1
fi
grep -q 'only warp_direct 8x4 is exposed' "${WARP8_ERR}"

if "${BIN}" "${TMP_MTX}" --br 4 --bc 4 >"${SHAPE_OUT}" 2>"${SHAPE_ERR}"; then
  echo "expected non-8x4 conversion shape to be retired" >&2
  exit 1
fi
grep -q 'only warp_direct 8x4 is exposed' "${SHAPE_ERR}"

if "${BIN}" "${TMP_MTX}" --mode direct --verify-cpu >"${DIRECT_OUT}" 2>"${DIRECT_ERR}"; then
  echo "expected --mode direct to be unsupported" >&2
  exit 1
fi
grep -q 'unsupported GPU conversion mode' "${DIRECT_ERR}"

if "${BIN}" "${TMP_MTX}" --mode segmented --verify-cpu >"${SEGMENTED_OUT}" 2>"${SEGMENTED_ERR}"; then
  echo "expected --mode segmented to be unsupported" >&2
  exit 1
fi
grep -q 'unsupported GPU conversion mode' "${SEGMENTED_ERR}"
