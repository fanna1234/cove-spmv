// Minimal stub replacing CUDA-samples helper_cuda.h for the CSR5 baseline build.
// CSR5 only uses checkCudaErrors(); the samples tree (helper_functions/timer/image)
// is otherwise unused. Keeps the CSR5 port self-contained on modern CUDA.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename T>
static inline void csr5_check_cuda(T err, const char *expr, const char *file, int line) {
    if (static_cast<int>(err) != 0) {
        fprintf(stderr, "CUDA error %s:%d code=%d \"%s\" in %s\n", file, line,
                static_cast<int>(err), cudaGetErrorString(static_cast<cudaError_t>(err)), expr);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) csr5_check_cuda((val), #val, __FILE__, __LINE__)
#define getLastCudaError(msg) csr5_check_cuda(cudaGetLastError(), (msg), __FILE__, __LINE__)
