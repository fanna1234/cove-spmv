# CSR5 baseline — port to modern CUDA (sm_80 / sm_90a / sm_120a)

bhSPARSE CSR5 (2015, CUDA 6.5) used as the load-balance baseline. The third-party
source belongs under the git-ignored `repro/csr5/src/` directory; this file,
`patch_csr5.py`, and `helper_cuda.h` are the distributed compatibility delta.

## Recipe
```bash
git clone https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5.git repro/csr5/src/Benchmark_SpMV_using_CSR5
git -C repro/csr5/src/Benchmark_SpMV_using_CSR5 checkout 4b06ce35e45c3c61a51e249e12e1c4954af443fa
cd repro/csr5/src/Benchmark_SpMV_using_CSR5/CSR5_cuda
cp <repo>/repro/csr5/helper_cuda.h .          # 1-symbol stub (checkCudaErrors); replaces CUDA-samples dep
: > helper_functions.h                          # CSR5 includes it but only uses checkCudaErrors
python3 <repo>/repro/csr5/patch_csr5.py         # guard double __shfl_*/atomicAdd polyfills (now native)

F="detail/cuda/csr5_spmv_cuda.h detail/cuda/format_cuda.h detail/cuda/utils_cuda.h"
# non-sync warp shuffles were removed in CUDA 12; modern _sync variants support double
sed -i 's/__shfl(\([^,]*\), \([0-9]*\))/__shfl_sync(0xffffffff, \1, \2)/g' $F
sed -i -E 's/__shfl_down\(/__shfl_down_sync(0xffffffff, /g; s/__shfl_up\(/__shfl_up_sync(0xffffffff, /g; s/__shfl_xor\(/__shfl_xor_sync(0xffffffff, /g' $F
# inline PTX shfl.up.b32 -> shfl.sync.up.b32 (+ full membermask operand), utils_cuda.h
sed -i 's/shfl.up.b32 lo|p, lo, %2, %3;/shfl.sync.up.b32 lo|p, lo, %2, %3, %4;/' detail/cuda/utils_cuda.h
sed -i 's/shfl.up.b32 hi|p, hi, %2, %3;/shfl.sync.up.b32 hi|p, hi, %2, %3, %4;/' detail/cuda/utils_cuda.h
sed -i 's/"r"(SHFL_C));/"r"(SHFL_C), "r"(0xffffffff));/' detail/cuda/utils_cuda.h
sed -i 's/deviceProp.clockRate \* 1e-3f/0.0f/' main.cu       # clockRate removed in CUDA 13
sed -i '69s|A.warmup();|// A.warmup();|' main.cu              # warmup_kernel has a modern-arch bug; NUM_RUN=1000 averages anyway

# build (pick arch): sm_80 (A800/CUDA12.1) | sm_90a (H100) | sm_120a (Blackwell)
ARCH=sm_120a; COMPUTE=compute_${ARCH#sm_}
/usr/local/cuda/bin/nvcc -O3 -w -m64 -Xcompiler -fpermissive -gencode=arch=$COMPUTE,code=$ARCH \
  main.cu -o spmv -I. -lcudart -D VALUE_TYPE=double -D NUM_RUN=1000
```

## Run over a denominator
`structural/scripts/run_csr5_sweep.py --bin <.../CSR5_cuda/spmv> --data-dir <suitesparse store>
--matrix-list <list.txt> --out csr5_<set>.csv` (stride-shardable).

## Recorded result boundary

The frozen value1000 CSV contains 1,000 successful CSR5 calls. Under CSR5's
native 1,000-run aggregate and COVE's separately disclosed timing contract, the
paper reports 2.54x for the lossless hybrid on their shared success set. This is
not a uniform-statistic claim that COVE is the fastest SpMV: value1000 is
small/medium dominated, and CSR5's irregular-giant regime was not tested.
