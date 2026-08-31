# COVE: Co-Designing Position Indexing and Error-Bounded Value Compression for High-Performance SpMV on GPUs

Official open-source implementation and reproducibility package for the COVE
short paper accepted at IEEE ICCD 2026.

**Authors:** Kaifan Jia, Yongchun Jiang, Minghui Zhang, Zhihao Ling, Xuran Wang,
and Heng Zhang. All authors are with the Institute of Software, Chinese Academy
of Sciences. Heng Zhang is the corresponding author.

COVE represents a sparse matrix with two independently selected encodings over
one warp-shaped 8x4 block unit:

- a representation-lossless position stream using a 64-bit block word plus a
  CSR residual path for long or irregular rows;
- an error-bounded value stream selected from exact dictionaries, block
  floating point with a global outlier pass, BF16, and the original values;
- a fused load-balanced GPU kernel that decodes the selected plan directly in
  registers.

Lossy value routes are admitted per matrix only after the configured SpMV output
error gate passes. A failed gate falls back to a higher-precision route.

## Repository scope

```text
build.sh                  CUDA build entry; auto-detects the GPU architecture
Makefile                  host build, tests, and GPU convenience targets
structural/
  src/                    CUDA/C++ operators, converters, and benchmarks
  include/                formats, codecs, dispatch, and kernels
  scripts/                sweep, analysis, download, and plotting tools
  tests/                  host and GPU correctness tests
  matrix_lists/           frozen SuiteSparse denominators
  results/studies/        measured CSV evidence used by the paper
  docs/                   format and converter documentation
repro/
  csr5/                   CSR5 port recipe and compatibility patch
  baselines/results/      measured external-baseline CSVs
```

SuiteSparse matrices and third-party baseline source trees are not
redistributed. The repository contains download lists, port instructions, and
the measured CSV evidence needed by the plotting scripts. See
[`repro/THIRD_PARTY.md`](repro/THIRD_PARTY.md) for source and license boundaries.

## Requirements

- CUDA toolkit 12.x or 13.x with `nvcc` (paper measurements used CUDA 13.1)
- a CUDA GPU with `sm_120a`, `sm_90a`, or `sm_80` support
- CMake 3.24 or newer and a C++17 compiler
- Python 3.10+; NumPy, pandas, Matplotlib, and SciPy for analysis and figures

## Build and test

Host-only tools and unit tests can be built without a CUDA GPU:

```bash
make structural-build
make structural-test
```

Build the GPU operators with automatic architecture detection or an explicit
target:

```bash
./build.sh
./build.sh sm_90a
# CUDA_HOME=/path/to/cuda ./build.sh sm_120a
```

Download the small demonstration set and run CPU-verified GPU paths:

```bash
structural/scripts/download_stars.sh data

structural/build/struct_gpu_csr_bitbsr_spmv data/cant.mtx --verify-cpu
structural/build/struct_gpu_spmv data/cant.mtx --precision double --verify-cpu
structural/build/struct_gpu_convert data/cant.mtx --br 8 --bc 4
```

Run a binary without arguments to see its complete command-line interface.

## Reproducing the paper evidence

The frozen denominators are under `structural/matrix_lists/`. To download the
value1000 set while preserving its Group/Entry layout:

```bash
structural/scripts/download_matrices.sh \
  structural/matrix_lists/suitesparse_auto_select_value1000_2026-06-07.txt data
```

The main sweep drivers are:

```bash
python3 structural/scripts/run_joint_sweep.py --help
python3 structural/scripts/run_gpu_spmv_family_sweep.py --help
python3 structural/scripts/run_csr5_sweep.py --help
```

The following scripts regenerate the principal evaluation views from the
shipped CSV files. They use the self-contained `paper_style.py` module in the
same directory.

| Evidence view | Script | Primary inputs |
|---|---|---|
| value-domain census | `plot_value_census.py` | le10gib core shards |
| speed, position metadata, and accuracy | `plot_three_axes.py` | value1000 core/joint/CSR5/baseline CSVs |
| coverage and scale | `plot_eval_performance.py` | le10gib and cross-device CSVs |
| value-codec Pareto frontier | `plot_value_pareto.py` | value1000 core shards |
| conjugate-gradient behavior | `plot_cg_solver.py` | `cg_real_spd_2026-06-09.csv` |

For example:

```bash
python3 -m pip install numpy pandas matplotlib scipy
python3 structural/scripts/plot_three_axes.py . output/three_axes
```

## Evidence and claim boundaries

- The le10gib inventory contains 3,624 real-valued SuiteSparse matrices that
  were testable within the paper's size contract; value1000 contains 1,000
  matrix paths. Eleven le10gib pairs and five value1000 pairs share a basename
  across SuiteSparse groups. Public plotting code reconstructs full identities
  from the frozen list and shard order instead of collapsing them to 3,613 and
  995 names.
- value1000 is the frozen main comparison denominator; success sets and route
  coverage are recorded in the CSVs rather than silently imputed.
- Primary timings were collected on an NVIDIA RTX PRO 6000 Blackwell GPU.
  H100 and A800 results cover only the three large matrices recorded in the
  cross-device CSVs.
- Included CSVs are measured evidence from June 2026. Re-running on a different
  GPU, CUDA version, clock state, or matrix cache may produce different timing.
- COVE/cuSPARSE curves use paired minimum post-warmup times. External baseline
  CSVs retain each implementation's native statistic; CSR5 reports its native
  1,000-run average. Those curves are compatibility context, not a uniform-
  statistic ranking.
- Build success, host tests, and figure regeneration do not by themselves prove
  end-to-end performance on an unmeasured system.

## Reproduction cost

1. Host build and unit test: minutes, no GPU.
2. GPU smoke test: minutes, one supported GPU and downloaded matrices.
3. Figure regeneration from shipped CSVs: minutes, no GPU.
4. value1000 sweeps: hours, depending on matrix download and GPU state.
5. le10gib coverage: days and approximately 1 TB of matrix data.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff). The final proceedings DOI
will be added after IEEE publishes the paper record.

## License

The COVE source code is released under the MIT License. See [`LICENSE`](LICENSE).
