# COVE structural and value operators

This directory contains the COVE sparse-format implementation, GPU operators,
tests, frozen matrix denominators, and measured CSV evidence.

## Representation

COVE uses an 8x4 unit mapped to one 32-lane warp. The lossless block position
word is:

```text
bits 0..30   absolute block column
bit 31       scheduling flag
bits 32..63  8x4 occupancy bitmap
```

Rows that do not pass the block-format cost gate remain in CSR residual chunks.
The value path independently selects exact or error-bounded codecs and decodes
their payload in the same traversal when both gates pass.

The four implementation families are:

- load-balanced original-value traversal;
- non-load-balanced original-value traversal;
- exact value-specialized traversal;
- controlled-lossy value-specialized traversal.

The public headers are under `include/structural/`; concrete 8x4 kernels are
under `include/structural/operators/`.

## Build and host tests

From the repository root:

```bash
make structural-build
make structural-test
```

This builds `struct_analyze`, `struct_spmv`, and `struct_tests` without requiring
a CUDA toolkit.

## GPU build and correctness checks

The top-level build script compiles all CUDA binaries and accepts an explicit
architecture:

```bash
./build.sh sm_120a
./build.sh sm_90a
./build.sh sm_80
```

Principal binaries:

| Binary | Purpose |
|---|---|
| `struct_gpu_csr_bitbsr_spmv` | complete CSR/BitBSR hybrid and joint operator |
| `struct_gpu_spmv` | position/value family runner and selector |
| `struct_gpu_convert` | device-side CSR-to-COVE conversion |
| `bench_accuracy_robust` | multi-input value-codec robustness |
| `bench_cg_solve` | conjugate-gradient end-to-end study |

Use `--verify-cpu` for correctness-facing smoke tests:

```bash
structural/build/struct_gpu_csr_bitbsr_spmv data/cant.mtx --verify-cpu
structural/build/struct_gpu_spmv data/cant.mtx --precision double --verify-cpu
```

The Makefile also exposes individual GPU build and CLI-test targets such as
`structural-gpu-test`, `structural-gpu-spmv-cli-test`, and
`structural-gpu-csr-bitbsr-spmv-cli-test`.

## Matrix denominators

The repository stores matrix lists, not MatrixMarket payloads. Downloaded
matrices should use a user-selected `${DATA_ROOT}` and retain the
`Group/Entry/file.mtx` layout:

```bash
structural/scripts/download_matrices.sh \
  structural/matrix_lists/suitesparse_auto_select_value1000_2026-06-07.txt \
  "${DATA_ROOT}"
```

Tracked denominators include:

- `canonical25.txt`: small correctness/debug subset;
- `suitesparse_auto_select_value1000_2026-06-07.txt`: frozen main selector;
- `suitesparse_real_paper_le10gib_2026-06-07.txt`: 3,624-entry coverage set;
- `suitesparse_real_testable_cusparse3632_2026-06-07.txt`: parent real-valued
  cuSPARSE-testable inventory.

The le10gib set contains eleven same-basename pairs from different SuiteSparse
groups. Raw result consumers must use `(matrix, matrix_bytes)` or the original
full path as the identity; basename-only aggregation collapses the denominator
from 3,624 to 3,613.

## Sweep and analysis scripts

| Script | Purpose |
|---|---|
| `run_gpu_spmv_family_sweep.py` | position/value operator-family sweep |
| `run_joint_sweep.py` | joint position-plus-value execution sweep |
| `run_csr5_sweep.py` | CSR5 comparison using the port recipe |
| `run_external_baseline_campaign.py` | external-baseline command wrapper |
| `plot_three_axes.py` | speed/storage/accuracy evidence |
| `plot_eval_performance.py` | selector and scale performance views |
| `plot_value_census.py` | value-domain population census |
| `plot_cg_solver.py` | conjugate-gradient behavior |

Every runner prints its complete argument surface with `--help`.

## Result data

`results/studies/` contains the measured CSVs distributed with the paper
artifact. The raw core shards retain operator status, timing, storage, and
verification fields. Derived figures must preserve each script's declared
success set and must not impute unsupported rows.

The H100 and A800 cross-device CSVs contain performance-only timing for three
large matrices. Their stored error fields are not cross-device quality evidence.

## Format documentation

- `docs/data-format.md` defines MatrixMarket normalization and host format
  behavior.
- `docs/gpu-convert.md` defines the device conversion contract.

## Measurement boundary

Paper CSVs were measured under the frozen June 2026 contracts. Re-running on a
new device or software stack is a new measurement. Build success, CPU agreement,
or a narrow GPU smoke test must not be reported as reproduction of the published
end-to-end speed distributions.
