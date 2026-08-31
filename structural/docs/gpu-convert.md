# GPU CSR to BitBSR Conversion

The public GPU conversion surface is intentionally narrow: high-performance
8x4 warp-direct conversion with device-resident output.

```bash
structural/build/struct_gpu_convert MATRIX.mtx
structural/build/struct_gpu_convert MATRIX.mtx --br 8 --bc 4 --mode warp_direct --device-only
```

Both commands use the same path:

```cpp
convert_to_bitbsr_gpu_warp_direct_8x4_device(...)
```

The converter starts after MatrixMarket loading. The input contract is a
normalized CSR matrix:

```text
row_ptr[rows + 1]
col_idx[nnz]
values[nnz]
```

The output remains on GPU:

```text
DeviceBitBsrMatrix<T>
  block_row_ptr[num_block_rows + 1]
  block_row_val_ptr[num_block_rows + 1]
  block_col_idx[num_blocks]
  bitmap_words[num_blocks]
  values[nnz]

GpuBitBsrWorkspace<T>
  reusable device CSR staging and block-count scratch
```

`BitmapWord` is `uint32_t`. The 8x4 BitBSR format has one bitmap word per
block.

## Retired Public Paths

The following conversion paths are no longer part of the public conversion
surface:

- `global_sort`
- `warp_direct_8x8`
- host-returning wrappers such as `convert_to_bitbsr_gpu(...)`
- CLI host materialization flags: `--verify-cpu`, `--host-output`,
  `--materialize`

`DeviceBitBsrMatrix<T>::to_host()` still exists for tests and internal helper
code that must compare against a CPU reference. It is not the default CLI or
benchmark path.

## Warp-Direct 8x4

One warp owns one 8-row block row. Lanes 0-7 advance the CSR rows in parallel,
while the remaining lanes participate in warp collectives.

The path is two-pass because BitBSR has variable-length block-row output:

1. `count_warp_direct_kernel` counts non-empty block columns per block row.
2. A prefix scan builds `block_row_ptr`.
3. `fill_warp_direct_kernel` merges the same 8 rows, writes `block_col_idx`,
   one 32-bit bitmap word, `block_row_val_ptr`, and compact `values`.

For a current block column, each active row lane counts its entries, a warp sum
gets the block nnz, lane-local prefix sums place values in row-major bitmap
order, and a warp OR builds the bitmap.

## CLI Output

The command prints device-resident metadata and timing:

```text
gpu_output_residency: device
gpu_num_blocks: ...
verify_cpu: SKIP_DEVICE_RESIDENT
timing_ms: load=..., cuda_context=..., cpu_convert=0, gpu_convert_host=..., gpu_convert_event=..., total=...
```

`gpu_convert_event` is recorded with CUDA events around the GPU conversion
section and is the comparison number for device-side work. `gpu_convert_host`
is host wall time for the conversion call, including allocation and launch
overhead. Matrix loading and CUDA context initialization are reported
separately.

## Build And Test

```bash
cd <repo-root>
make structural-gpu-convert
make structural/build/struct_gpu_tests
```

Default target architecture is inherited from the top-level `ARCH` variable:

```text
-gencode arch=compute_120a,code=sm_120a
```

Override it for other machines:

```bash
make structural-gpu-convert ARCH="-gencode arch=compute_90a,code=sm_90a"
```

Run the local PK conversion table:

```bash
make structural-gpu-pk
```

This now writes only the retained high-performance path:

```text
structural/results/studies/gpu_convert_pk_8x4.csv
```
