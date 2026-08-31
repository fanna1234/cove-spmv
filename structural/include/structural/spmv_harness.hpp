#ifndef STRUCTURAL_SPMV_HARNESS_HPP
#define STRUCTURAL_SPMV_HARNESS_HPP

// Shared GPU-SpMV driver harness for the COVE binaries: host timing, canonical input
// vector, and the CPU-fp64 error CORE used by every verify path. Each binary keeps its
// own CLI Options and pass/fail GATE; this header holds only the pieces that were
// byte-identical across binaries.

#include "structural/csr.hpp"
#include "structural/spmv.hpp"
#include "structural/types.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <chrono>
#include <cstddef>
#include <vector>

namespace structural {

// Run fn(), record wall-clock milliseconds into ms, return fn()'s result.
template <typename F>
auto time_call(F&& fn, double& ms) {
    const auto t0 = std::chrono::steady_clock::now();
    auto result = fn();
    const auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

// Canonical zero-mean input vector: x[i] = (i % 17 - 8) * 0.125, range [-1, 1].
template <typename T>
std::vector<T> make_input_vector(Index cols) {
    std::vector<T> x(static_cast<std::size_t>(cols));
    for (Index i = 0; i < cols; ++i)
        x[static_cast<std::size_t>(i)] = static_cast<T>((i % 17) - 8) * static_cast<T>(0.125);
    return x;
}

// Shared verify CORE: reference y = A*x in fp64 on the CPU, compare to the device
// result. Fills cpu_spmv_ms and error. The pass/fail gate is the caller's decision.
template <typename T>
void compute_cpu_error(const CsrMatrix<T>& csr, const std::vector<T>& x,
                       const thrust::device_vector<T>& d_y, double& cpu_spmv_ms,
                       ErrorStats<T>& error) {
    const auto expected = time_call([&] { return spmv_csr(csr, x); }, cpu_spmv_ms);
    std::vector<T> got(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), got.begin());
    error = compare_vectors(expected, got);
}

}  // namespace structural

#endif  // STRUCTURAL_SPMV_HARNESS_HPP
