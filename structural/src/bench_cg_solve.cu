// CG iterative-solver comparison for the COVE value/position axes.
//
// Question: in an iterative solve (CG), end-to-end time = (iterations to target
// tolerance) x (time per iteration). The matvec precision sets BOTH: a residual
// FLOOR (a fixed low-precision encoding solves the perturbed system (A+dA)x=b, so
// the true residual plateaus at ~||dA||/||A|| = the codec rel-err) and, above that
// floor, ~the same convergence as fp64. This bench MEASURES that, faithfully, by
// dropping the REAL COVE operators + the PRODUCTION value codecs into a textbook CG
// whose WORKING vectors are fp64 (cuBLAS) -- only the matvec (matrix precision)
// changes. Variants: cusparse_double, cusparse_float (float matrix, fp64 working),
// cove_lossless (exact fp64 BitBSR-LB), cove_bf16, cove_bfp8, cove_bfp8_outlier.
//
// For each variant it reports: true-residual FLOOR, iterations to {1e-2,1e-3,1e-6,
// 1e-8} (via the TRUE residual ||b - A_exact x||/||b||, A_exact = exact fp64), the
// per-iteration time, and the derived end-to-end time-to-tolerance. The exact
// matvec for the true residual is the lossless COVE operator (bit-exact fp64).
//
// SPD only (CG). If p^T A p <= 0 the matrix is not SPD for this run -> reported, not
// crashed. RHS b = A_exact * x_true (x_true ~ N(0,1), fixed seed) so the system is
// consistent and the known solution lets us also report the final solution error.
//
// Standalone (not in CMake). Build on a GPU host:
//   export PATH=/usr/local/cuda/bin:$PATH CPATH=/usr/local/cuda/include
//   nvcc -O3 -std=c++17 --extended-lambda \
//     -gencode arch=compute_120a,code=sm_120a -Istructural/include \
//     -o structural/build/bench_cg_solve structural/src/bench_cg_solve.cu \
//     -lcusparse -lcublas
//
// Usage: bench_cg_solve matrix.mtx out.csv [append=0] [maxiter=5000] [seed=12345]

#include "structural/csr.hpp"
#include "structural/gpu_convert.cuh"
#include "structural/gpu_value_codecs.cuh"
#include "structural/matrix_market.hpp"
#include "structural/operators/bitbsr_spmv_8x4.cuh"  // umbrella: raw (lossless LB) + common (codec wrappers)

#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using structural::Index;
using structural::Offset;

#define CUDA_CHECK(e)                                                                     \
    do {                                                                                  \
        cudaError_t st = (e);                                                             \
        if (st != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(st));                                         \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)
#define CUSP_CHECK(e)                                                                     \
    do {                                                                                  \
        cusparseStatus_t st = (e);                                                        \
        if (st != CUSPARSE_STATUS_SUCCESS) {                                              \
            std::fprintf(stderr, "cuSPARSE %s:%d status=%d\n", __FILE__, __LINE__, (int)st); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)
#define CUBLAS_CHECK(e)                                                                   \
    do {                                                                                  \
        cublasStatus_t st = (e);                                                          \
        if (st != CUBLAS_STATUS_SUCCESS) {                                                \
            std::fprintf(stderr, "cuBLAS %s:%d status=%d\n", __FILE__, __LINE__, (int)st);\
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

using DVec = thrust::device_vector<double>;
static double* raw(DVec& v) { return thrust::raw_pointer_cast(v.data()); }
using Matvec = std::function<void(DVec& p, DVec& Ap)>;  // Ap = A * p, both length n

static const std::vector<double> kTols = {1e-2, 1e-3, 1e-6, 1e-8};

struct CgResult {
    std::string variant;
    double bytes_per_nnz = 0.0;
    bool spd_ok = true;
    int total_iters = 0;
    double floor = 1.0;
    double sol_err = 1.0;
    std::vector<int> iters_to_tol;  // -1 if never reached
    double ms_per_iter = 0.0;
};

// CG curve run (UNTIMED): drives CG with `mv`, measures the TRUE residual each iter
// with `exact` (bit-exact fp64). Returns floor + iters-to-tol + final solution error.
static CgResult run_cg_curve(cublasHandle_t blas, int n, const Matvec& mv, const Matvec& exact,
                             DVec& d_b, double bnorm, DVec& d_x_true, double xtrue_norm,
                             int maxiter, const std::string& name, double bpn) {
    CgResult res;
    res.variant = name;
    res.bytes_per_nnz = bpn;
    res.iters_to_tol.assign(kTols.size(), -1);

    DVec d_x(n, 0.0), d_r(d_b), d_p(d_b), d_Ap(n, 0.0), d_Ax(n, 0.0), d_tmp(n, 0.0);
    double rs = 0.0;
    CUBLAS_CHECK(cublasDdot(blas, n, raw(d_r), 1, raw(d_r), 1, &rs));
    double best = 1.0, prev_best = 2.0;
    int since_improve = 0;

    for (int k = 1; k <= maxiter; ++k) {
        mv(d_p, d_Ap);
        double pAp = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(d_p), 1, raw(d_Ap), 1, &pAp));
        if (!(pAp > 0.0)) {  // breakdown -> not SPD for this run
            res.spd_ok = false;
            res.total_iters = k - 1;
            res.floor = best;
            return res;
        }
        const double alpha = rs / pAp, nalpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(blas, n, &alpha, raw(d_p), 1, raw(d_x), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &nalpha, raw(d_Ap), 1, raw(d_r), 1));

        // TRUE residual ||b - A_exact x|| / ||b||.
        exact(d_x, d_Ax);
        CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_b), 1, raw(d_tmp), 1));
        const double m1 = -1.0;
        CUBLAS_CHECK(cublasDaxpy(blas, n, &m1, raw(d_Ax), 1, raw(d_tmp), 1));
        double tnorm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas, n, raw(d_tmp), 1, &tnorm));
        const double true_relres = tnorm / (bnorm + 1e-300);
        best = std::min(best, true_relres);
        for (std::size_t t = 0; t < kTols.size(); ++t)
            if (res.iters_to_tol[t] < 0 && true_relres <= kTols[t]) res.iters_to_tol[t] = k;
        res.total_iters = k;

        if (best < prev_best * (1.0 - 1e-4)) { prev_best = best; since_improve = 0; }
        else ++since_improve;
        if (true_relres <= 1e-12 || res.iters_to_tol.back() > 0) break;  // converged
        if (since_improve > 200) break;                                  // plateaued at the floor

        double rs_new = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(d_r), 1, raw(d_r), 1, &rs_new));
        const double beta = rs_new / rs, one = 1.0;
        CUBLAS_CHECK(cublasDscal(blas, n, &beta, raw(d_p), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &one, raw(d_r), 1, raw(d_p), 1));
        rs = rs_new;
    }
    res.floor = best;

    CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_x), 1, raw(d_tmp), 1));
    const double m1 = -1.0;
    CUBLAS_CHECK(cublasDaxpy(blas, n, &m1, raw(d_x_true), 1, raw(d_tmp), 1));
    double enorm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas, n, raw(d_tmp), 1, &enorm));
    res.sol_err = enorm / (xtrue_norm + 1e-300);
    return res;
}

// Timed CG (no true-residual): pure recurrence with variant matvec for `iters` iters.
static double time_cg(cublasHandle_t blas, int n, const Matvec& mv, DVec& d_b, int iters) {
    if (iters <= 0) return 0.0;
    DVec d_x(n, 0.0), d_r(d_b), d_p(d_b), d_Ap(n, 0.0);
    double rs = 0.0;
    CUBLAS_CHECK(cublasDdot(blas, n, raw(d_r), 1, raw(d_r), 1, &rs));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int k = 0; k < iters; ++k) {
        mv(d_p, d_Ap);
        double pAp = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(d_p), 1, raw(d_Ap), 1, &pAp));
        const double alpha = (pAp != 0.0) ? rs / pAp : 0.0, nalpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(blas, n, &alpha, raw(d_p), 1, raw(d_x), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &nalpha, raw(d_Ap), 1, raw(d_r), 1));
        double rs_new = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(d_r), 1, raw(d_r), 1, &rs_new));
        const double beta = (rs != 0.0) ? rs_new / rs : 0.0, one = 1.0;
        CUBLAS_CHECK(cublasDscal(blas, n, &beta, raw(d_p), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &one, raw(d_r), 1, raw(d_p), 1));
        rs = rs_new;
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return static_cast<double>(ms) / iters;
}

// Synthetic large, well-conditioned, dense-block SPD matrix: a symmetric banded
// matrix (off-diagonals random in (-1,1), mirrored) made strictly diagonally
// dominant (a_ii = sum_{j!=i}|a_ij| + delta) => SPD by Gershgorin, conditioning set
// by delta. The band fills COVE's 8x4 blocks densely (high nnz/block = the regime
// where value quantization actually speeds up the matvec). nnz ~ n*(2*halfbw+1):
// pick n large enough that the SpMV dominates the per-iter cuBLAS-launch overhead.
static structural::CsrMatrix<double> make_synthetic_spd(int n, int halfbw, unsigned seed,
                                                        double delta) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<std::vector<std::pair<int, double>>> rows(n);
    for (int i = 0; i < n; ++i) {
        const int jmax = std::min(n - 1, i + halfbw);
        for (int j = i + 1; j <= jmax; ++j) {
            const double v = u(rng);
            rows[i].emplace_back(j, v);
            rows[j].emplace_back(i, v);  // symmetric
        }
    }
    structural::CsrMatrix<double> csr;
    csr.rows = n;
    csr.cols = n;
    csr.row_ptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        double absum = 0.0;
        for (const auto& p : rows[i]) absum += std::abs(p.second);
        rows[i].emplace_back(i, absum + delta);  // diagonally dominant -> SPD
        std::sort(rows[i].begin(), rows[i].end());
        csr.row_ptr[i + 1] = csr.row_ptr[i] + static_cast<Offset>(rows[i].size());
    }
    csr.col_idx.reserve(csr.row_ptr[n]);
    csr.values.reserve(csr.row_ptr[n]);
    for (int i = 0; i < n; ++i)
        for (const auto& p : rows[i]) {
            csr.col_idx.push_back(p.first);
            csr.values.push_back(p.second);
        }
    return csr;
}

// Synthetic SHIFTED-LAPLACIAN SPD: A = (D - W) + sigma*I, W symmetric banded with
// positive weights, D = diag(row-sum of W). SPD (graph Laplacian is PSD, +sigma*I
// => PD) with a SPREAD (non-clustered) spectrum like a discretized PDE, so CG needs
// ~sqrt(kappa) iterations (kappa ~ max_degree/sigma). Same banded => dense 8x4
// blocks => SpMV-dominated per-iter. This gives the MANY-iteration end-to-end curve
// that the diagonally-dominant `make_synthetic_spd` (clustered spectrum) does not.
static structural::CsrMatrix<double> make_synthetic_laplacian_spd(int n, int halfbw, unsigned seed,
                                                                  double sigma) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.1, 1.0);  // positive edge weights
    std::vector<std::vector<std::pair<int, double>>> rows(n);
    std::vector<double> deg(n, 0.0);
    for (int i = 0; i < n; ++i) {
        const int jmax = std::min(n - 1, i + halfbw);
        for (int j = i + 1; j <= jmax; ++j) {
            const double w = u(rng);
            rows[i].emplace_back(j, -w);
            rows[j].emplace_back(i, -w);
            deg[i] += w;
            deg[j] += w;
        }
    }
    structural::CsrMatrix<double> csr;
    csr.rows = n;
    csr.cols = n;
    csr.row_ptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        rows[i].emplace_back(i, deg[i] + sigma);  // Laplacian diagonal + shift
        std::sort(rows[i].begin(), rows[i].end());
        csr.row_ptr[i + 1] = csr.row_ptr[i] + static_cast<Offset>(rows[i].size());
    }
    csr.col_idx.reserve(csr.row_ptr[n]);
    csr.values.reserve(csr.row_ptr[n]);
    for (int i = 0; i < n; ++i)
        for (const auto& p : rows[i]) {
            csr.col_idx.push_back(p.first);
            csr.values.push_back(p.second);
        }
    return csr;
}

// Mixed-precision iterative refinement (IR): OUTER loop keeps x (fp64) and recomputes
// the residual r = b - A_exact*x with the EXACT lossless COVE matvec (so the final
// accuracy is fp64, no floor); the INNER solve approximates A*d = r with a BOUNDED
// (k_inner-iteration) CG using a LOW-PRECISION matvec (cheap, 2.4x/iter) -- bounded
// so it returns a useful correction before the low-precision CG would break down.
// Each outer step contracts ||r|| by ~the inner relative residual; a few outer steps
// reach fp64 tolerance. Timed end-to-end (cudaEvent) to the TRUE tolerance.
struct IrResult {
    std::string name;
    double time_ms = 0.0;
    int outer = 0;
    int inner_total = 0;
    double final_relres = 1.0;
    bool reached = false;
};
static IrResult run_ir(cublasHandle_t blas, int n, const Matvec& mv_inner, const Matvec& exact,
                       DVec& d_b, double bnorm, double tol, int k_inner, int max_outer,
                       const std::string& name) {
    IrResult R;
    R.name = name;
    DVec d_x(n, 0.0), d_r(d_b), d_d(n, 0.0), d_Ax(n, 0.0), d_ri(n, 0.0), d_p(n, 0.0), d_Ap(n, 0.0);
    const double one = 1.0, m1 = -1.0;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (R.outer = 0; R.outer < max_outer; ++R.outer) {
        double rn = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas, n, raw(d_r), 1, &rn));
        if (rn / bnorm < tol) break;
        // inner CG: A d = r, d0 = 0, low-precision matvec, k_inner iters (bounded).
        thrust::fill(d_d.begin(), d_d.end(), 0.0);                      // d = 0
        CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_r), 1, raw(d_ri), 1));  // ri = r
        CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_r), 1, raw(d_p), 1));   // p = r
        double rs = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(d_ri), 1, raw(d_ri), 1, &rs));
        for (int j = 0; j < k_inner; ++j) {
            mv_inner(d_p, d_Ap);
            double pAp = 0.0;
            CUBLAS_CHECK(cublasDdot(blas, n, raw(d_p), 1, raw(d_Ap), 1, &pAp));
            if (!(pAp > 0.0)) break;  // bounded: stop before breakdown
            const double a = rs / pAp, na = -a;
            CUBLAS_CHECK(cublasDaxpy(blas, n, &a, raw(d_p), 1, raw(d_d), 1));
            CUBLAS_CHECK(cublasDaxpy(blas, n, &na, raw(d_Ap), 1, raw(d_ri), 1));
            double rsn = 0.0;
            CUBLAS_CHECK(cublasDdot(blas, n, raw(d_ri), 1, raw(d_ri), 1, &rsn));
            ++R.inner_total;
            const double be = rsn / rs;
            CUBLAS_CHECK(cublasDscal(blas, n, &be, raw(d_p), 1));
            CUBLAS_CHECK(cublasDaxpy(blas, n, &one, raw(d_ri), 1, raw(d_p), 1));
            rs = rsn;
        }
        CUBLAS_CHECK(cublasDaxpy(blas, n, &one, raw(d_d), 1, raw(d_x), 1));  // x += d
        exact(d_x, d_Ax);                                                   // r = b - A_exact x
        CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_b), 1, raw(d_r), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &m1, raw(d_Ax), 1, raw(d_r), 1));
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    R.time_ms = ms;
    double rn = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas, n, raw(d_r), 1, &rn));
    R.final_relres = rn / bnorm;
    R.reached = R.final_relres <= tol;
    return R;
}

// Bounded inner CG: approximately solve A*d = rhs (d0=0) with `mv` for k_inner iters,
// write the correction into d_out. Returns iterations actually run. Work buffers ri/p/Ap
// are caller-provided (pre-allocated) so nothing reallocates in the timed loop.
static int inner_cg_apply(cublasHandle_t blas, int n, const Matvec& mv, DVec& rhs, DVec& d_out,
                          DVec& ri, DVec& p, DVec& Ap, int k_inner) {
    const double one = 1.0;
    thrust::fill(d_out.begin(), d_out.end(), 0.0);
    CUBLAS_CHECK(cublasDcopy(blas, n, raw(rhs), 1, raw(ri), 1));
    CUBLAS_CHECK(cublasDcopy(blas, n, raw(rhs), 1, raw(p), 1));
    double rs = 0.0;
    CUBLAS_CHECK(cublasDdot(blas, n, raw(ri), 1, raw(ri), 1, &rs));
    int it = 0;
    for (int j = 0; j < k_inner; ++j) {
        mv(p, Ap);
        double pAp = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(p), 1, raw(Ap), 1, &pAp));
        if (!(pAp > 0.0)) break;
        const double a = rs / pAp, na = -a;
        CUBLAS_CHECK(cublasDaxpy(blas, n, &a, raw(p), 1, raw(d_out), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &na, raw(Ap), 1, raw(ri), 1));
        double rsn = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(ri), 1, raw(ri), 1, &rsn));
        ++it;
        const double be = rsn / rs;
        CUBLAS_CHECK(cublasDscal(blas, n, &be, raw(p), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &one, raw(ri), 1, raw(p), 1));
        rs = rsn;
    }
    return it;
}

// Flexible (Polak-Ribiere) preconditioned CG: ONE outer CG with the EXACT lossless
// matvec (so it reaches fp64), preconditioned each step by M^{-1}r = a bounded inner
// low-precision CG. Unlike IR it does NOT restart, so it keeps the global Krylov
// optimality (no restart inflation). Beta uses the flexible Polak-Ribiere form
// (r_{k+1}.(z_{k+1}-z_k))/(r_k.z_k), which tolerates the varying preconditioner.
static IrResult run_fcg(cublasHandle_t blas, int n, const Matvec& exact, const Matvec& mv_inner,
                        DVec& d_b, double bnorm, double tol, int k_inner, int max_outer,
                        const std::string& name) {
    IrResult R;
    R.name = name;
    DVec d_x(n, 0.0), d_r(d_b), d_z(n, 0.0), d_p(n, 0.0), d_Ap(n, 0.0), d_rn(n, 0.0), d_zn(n, 0.0);
    DVec w_ri(n, 0.0), w_p(n, 0.0), w_Ap(n, 0.0);  // inner work
    const double one = 1.0, m1 = -1.0;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    R.inner_total += inner_cg_apply(blas, n, mv_inner, d_r, d_z, w_ri, w_p, w_Ap, k_inner);  // z=M r
    CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_z), 1, raw(d_p), 1));                              // p=z
    double rz = 0.0;
    CUBLAS_CHECK(cublasDdot(blas, n, raw(d_r), 1, raw(d_z), 1, &rz));
    for (R.outer = 0; R.outer < max_outer; ++R.outer) {
        exact(d_p, d_Ap);  // EXACT outer matvec
        double pAp = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(d_p), 1, raw(d_Ap), 1, &pAp));
        if (!(pAp > 0.0)) break;
        const double a = rz / pAp, na = -a;
        CUBLAS_CHECK(cublasDaxpy(blas, n, &a, raw(d_p), 1, raw(d_x), 1));   // x += a p
        CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_r), 1, raw(d_rn), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &na, raw(d_Ap), 1, raw(d_rn), 1));  // r_new = r - a Ap
        double rn = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas, n, raw(d_rn), 1, &rn));
        if (rn / bnorm < tol) {
            ++R.outer;
            break;
        }
        R.inner_total += inner_cg_apply(blas, n, mv_inner, d_rn, d_zn, w_ri, w_p, w_Ap, k_inner);
        double rnzn = 0.0, rnz = 0.0;
        CUBLAS_CHECK(cublasDdot(blas, n, raw(d_rn), 1, raw(d_zn), 1, &rnzn));
        CUBLAS_CHECK(cublasDdot(blas, n, raw(d_rn), 1, raw(d_z), 1, &rnz));
        double beta = (rnzn - rnz) / rz;  // flexible Polak-Ribiere
        if (beta < 0.0) beta = 0.0;       // restart safeguard
        CUBLAS_CHECK(cublasDscal(blas, n, &beta, raw(d_p), 1));
        CUBLAS_CHECK(cublasDaxpy(blas, n, &one, raw(d_zn), 1, raw(d_p), 1));  // p = z_new + beta p
        rz = rnzn;
        CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_rn), 1, raw(d_r), 1));
        CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_zn), 1, raw(d_z), 1));
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    R.time_ms = ms;
    // true final residual via the exact operator
    exact(d_x, d_Ap);
    CUBLAS_CHECK(cublasDcopy(blas, n, raw(d_b), 1, raw(d_rn), 1));
    CUBLAS_CHECK(cublasDaxpy(blas, n, &m1, raw(d_Ap), 1, raw(d_rn), 1));
    double rn = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas, n, raw(d_rn), 1, &rn));
    R.final_relres = rn / bnorm;
    R.reached = R.final_relres <= tol;
    return R;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s matrix.mtx out.csv [append=0] [maxiter=5000] [seed=12345]\n"
                             "  matrix may be 'synth:N:halfbw:delta' (diag-dominant SPD) or\n"
                             "  'lapl:N:halfbw:sigma' (shifted-Laplacian SPD, spread spectrum)\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1], csv_path = argv[2];
    const bool append = (argc > 3) ? (std::atoi(argv[3]) != 0) : false;
    const int maxiter = (argc > 4) ? std::atoi(argv[4]) : 5000;
    const unsigned seed = (argc > 5) ? (unsigned)std::strtoul(argv[5], nullptr, 10) : 12345u;

    std::string mname(path);
    structural::CsrMatrix<double> csr;
    if (path.rfind("synth:", 0) == 0) {
        long N = 400000, BW = 32;
        double DELTA = 1.0;
        std::sscanf(path.c_str() + 6, "%ld:%ld:%lf", &N, &BW, &DELTA);
        csr = make_synthetic_spd((int)N, (int)BW, seed, DELTA);
        char buf[96];
        std::snprintf(buf, sizeof buf, "synth_n%ld_bw%ld_d%.3g", N, BW, DELTA);
        mname = buf;
    } else if (path.rfind("lapl:", 0) == 0) {
        long N = 400000, BW = 32;
        double SIGMA = 0.01;
        std::sscanf(path.c_str() + 5, "%ld:%ld:%lf", &N, &BW, &SIGMA);
        csr = make_synthetic_laplacian_spd((int)N, (int)BW, seed, SIGMA);
        char buf[96];
        std::snprintf(buf, sizeof buf, "lapl_n%ld_bw%ld_s%.4g", N, BW, SIGMA);
        mname = buf;
    } else {
        if (auto s = mname.find_last_of('/'); s != std::string::npos) mname = mname.substr(s + 1);
        if (mname.size() > 4 && mname.substr(mname.size() - 4) == ".mtx") mname.resize(mname.size() - 4);
        csr = structural::read_matrix_market<double>(path);
    }
    if (csr.rows != csr.cols) {
        std::fprintf(stderr, "%s not square -> CG N/A\n", mname.c_str());
        return 0;
    }
    const int n = csr.rows;
    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);
    const auto host_bitbsr = device.to_host();
    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);
    structural::BitBsrLoadBalanceLayout8x4 layout;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(device, layout, 32, 32,
                                                                        packed_meta);
    const double nnz_d = static_cast<double>(device.nnz);

    const auto bfp8 = structural::build_checked_bfp8_blkscale_value_codec(host_bitbsr);
    const auto outlier = structural::build_checked_bfp8_outlier_blkscale_value_codec(host_bitbsr, 0.005);
    const auto bf16 = structural::build_checked_bf16_nnz_value_codec(host_bitbsr);
    thrust::device_vector<signed char> d_int8(bfp8.int8_values.begin(), bfp8.int8_values.end());
    thrust::device_vector<std::uint16_t> d_scale(bfp8.block_scale_bf16.begin(), bfp8.block_scale_bf16.end());
    thrust::device_vector<signed char> d_int8_o(outlier.int8_values.begin(), outlier.int8_values.end());
    thrust::device_vector<std::uint16_t> d_scale_o(outlier.block_scale_bf16.begin(), outlier.block_scale_bf16.end());
    thrust::device_vector<std::int32_t> d_orows(outlier.outlier_rows.begin(), outlier.outlier_rows.end());
    thrust::device_vector<std::int32_t> d_ocols(outlier.outlier_cols.begin(), outlier.outlier_cols.end());
    thrust::device_vector<double> d_ovals(outlier.outlier_vals.begin(), outlier.outlier_vals.end());
    thrust::device_vector<std::uint16_t> d_bf16(bf16.values16.begin(), bf16.values16.end());

    // cuSPARSE double + float CSR.
    thrust::device_vector<Offset> d_row_ptr(csr.row_ptr.begin(), csr.row_ptr.end());
    thrust::device_vector<Index> d_col_idx(csr.col_idx.begin(), csr.col_idx.end());
    thrust::device_vector<double> d_vals_d(csr.values.begin(), csr.values.end());
    std::vector<float> vals_f(csr.values.begin(), csr.values.end());
    thrust::device_vector<float> d_vals_f(vals_f.begin(), vals_f.end());
    thrust::device_vector<float> d_pf(n, 0.0f), d_yf(n, 0.0f);

    cusparseHandle_t cusp = nullptr;
    CUSP_CHECK(cusparseCreate(&cusp));
    cusparseSpMatDescr_t mat_d = nullptr, mat_f = nullptr;
    CUSP_CHECK(cusparseCreateCsr(&mat_d, n, n, device.nnz, thrust::raw_pointer_cast(d_row_ptr.data()),
                                 thrust::raw_pointer_cast(d_col_idx.data()), raw(d_vals_d),
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                 CUDA_R_64F));
    CUSP_CHECK(cusparseCreateCsr(&mat_f, n, n, device.nnz, thrust::raw_pointer_cast(d_row_ptr.data()),
                                 thrust::raw_pointer_cast(d_col_idx.data()),
                                 thrust::raw_pointer_cast(d_vals_f.data()), CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vx_d = nullptr, vy_d = nullptr, vx_f = nullptr, vy_f = nullptr;
    double dummy = 0.0;
    CUSP_CHECK(cusparseCreateDnVec(&vx_d, n, &dummy, CUDA_R_64F));
    CUSP_CHECK(cusparseCreateDnVec(&vy_d, n, &dummy, CUDA_R_64F));
    CUSP_CHECK(cusparseCreateDnVec(&vx_f, n, thrust::raw_pointer_cast(d_pf.data()), CUDA_R_32F));
    CUSP_CHECK(cusparseCreateDnVec(&vy_f, n, thrust::raw_pointer_cast(d_yf.data()), CUDA_R_32F));
    const double one = 1.0, zero = 0.0;
    const float onef = 1.0f, zerof = 0.0f;
    size_t buf_d = 0, buf_f = 0;
    CUSP_CHECK(cusparseSpMV_bufferSize(cusp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_d, vx_d,
                                       &zero, vy_d, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_d));
    CUSP_CHECK(cusparseSpMV_bufferSize(cusp, CUSPARSE_OPERATION_NON_TRANSPOSE, &onef, mat_f, vx_f,
                                       &zerof, vy_f, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_f));
    thrust::device_vector<std::uint8_t> ebuf_d(buf_d ? buf_d : 1), ebuf_f(buf_f ? buf_f : 1);

    cublasHandle_t blas = nullptr;
    CUBLAS_CHECK(cublasCreate(&blas));

    Matvec mv_cusp_d = [&](DVec& p, DVec& Ap) {
        Ap.resize(n);
        CUSP_CHECK(cusparseDnVecSetValues(vx_d, raw(p)));
        CUSP_CHECK(cusparseDnVecSetValues(vy_d, raw(Ap)));
        CUSP_CHECK(cusparseSpMV(cusp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_d, vx_d, &zero,
                                vy_d, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                thrust::raw_pointer_cast(ebuf_d.data())));
    };
    Matvec mv_cusp_f = [&](DVec& p, DVec& Ap) {
        Ap.resize(n);
        thrust::transform(p.begin(), p.end(), d_pf.begin(), [] __device__(double v) { return (float)v; });
        CUSP_CHECK(cusparseSpMV(cusp, CUSPARSE_OPERATION_NON_TRANSPOSE, &onef, mat_f, vx_f, &zerof,
                                vy_f, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                thrust::raw_pointer_cast(ebuf_f.data())));
        thrust::transform(d_yf.begin(), d_yf.end(), Ap.begin(), [] __device__(float v) { return (double)v; });
    };
    Matvec mv_lossless = [&](DVec& p, DVec& Ap) {
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_device(
            device, packed_meta, layout, p, Ap);
    };
    Matvec mv_bf16 = [&](DVec& p, DVec& Ap) {
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_nnz_device<true>(
            device, packed_meta, layout, d_bf16, "bf16_nnz", p, Ap);
    };
    Matvec mv_bfp8 = [&](DVec& p, DVec& Ap) {
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bfp8_blkscale_device(
            device, packed_meta, layout, d_int8, d_scale, p, Ap);
    };
    Matvec mv_outlier = [&](DVec& p, DVec& Ap) {
        structural::spmv_bitbsr_gpu_8x4_bfp8_outlier_flag_load_balance_device(
            device, packed_meta, layout, d_int8_o, d_scale_o, d_orows, d_ocols, d_ovals, p, Ap);
    };
    const Matvec exact = mv_lossless;

    // RHS b = A_exact x_true, x_true ~ N(0,1).
    std::vector<double> xt(n);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < n; ++i) xt[i] = nd(rng);
    DVec d_x_true(xt.begin(), xt.end()), d_b(n, 0.0);
    {
        DVec tmp(n, 0.0);
        exact(d_x_true, tmp);
        d_b = tmp;
    }
    double bnorm = 0.0, xtnorm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas, n, raw(d_b), 1, &bnorm));
    CUBLAS_CHECK(cublasDnrm2(blas, n, raw(d_x_true), 1, &xtnorm));

    struct V { std::string name; Matvec mv; double bpn; };
    std::vector<V> variants = {
        {"cusparse_double", mv_cusp_d, 8.0},
        {"cusparse_float", mv_cusp_f, 4.0},
        {"cove_lossless", mv_lossless, 8.0},
        {"cove_bf16", mv_bf16, static_cast<double>(bf16.payload_bytes()) / nnz_d},
        {"cove_bfp8", mv_bfp8, static_cast<double>(bfp8.payload_bytes()) / nnz_d},
        {"cove_bfp8_outlier", mv_outlier, static_cast<double>(outlier.payload_bytes()) / nnz_d}};

    std::vector<CgResult> results;
    for (const auto& v : variants) {
        CgResult r = run_cg_curve(blas, n, v.mv, exact, d_b, bnorm, d_x_true, xtnorm, maxiter, v.name,
                                  v.bpn);
        const int titer = std::max(50, std::min(r.total_iters, 400));
        r.ms_per_iter = time_cg(blas, n, v.mv, d_b, titer);
        results.push_back(r);
    }

    FILE* f = std::fopen(csv_path.c_str(), append ? "a" : "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", csv_path.c_str()); return 1; }
    if (!append)
        std::fprintf(f, "matrix,n,nnz,variant,bytes_per_nnz,spd_ok,ms_per_iter,floor,sol_err,"
                        "iter_1e-2,iter_1e-3,iter_1e-6,iter_1e-8,total_iters,time_1e-3_ms,time_1e-6_ms\n");
    auto t2t = [&](const CgResult& r, std::size_t i) -> double {
        return (i < r.iters_to_tol.size() && r.iters_to_tol[i] > 0) ? r.iters_to_tol[i] * r.ms_per_iter
                                                                    : -1.0;
    };
    for (const auto& r : results)
        std::fprintf(f, "%s,%d,%lld,%s,%.4f,%d,%.6f,%.3e,%.3e,%d,%d,%d,%d,%d,%.4f,%.4f\n",
                     mname.c_str(), n, (long long)device.nnz, r.variant.c_str(), r.bytes_per_nnz,
                     r.spd_ok ? 1 : 0, r.ms_per_iter, r.floor, r.sol_err, r.iters_to_tol[0],
                     r.iters_to_tol[1], r.iters_to_tol[2], r.iters_to_tol[3], r.total_iters,
                     t2t(r, 1), t2t(r, 2));
    std::fclose(f);

    std::printf("== %-14s n=%d nnz=%lld | b=A*x_true seed=%u maxiter=%d ==\n", mname.c_str(), n,
                (long long)device.nnz, seed, maxiter);
    std::printf("  %-18s %6s %8s %10s | iters@[1e-2 1e-3 1e-6 1e-8] | t@1e-3 t@1e-6(ms)\n", "variant",
                "B/nnz", "ms/iter", "floor");
    for (const auto& r : results) {
        if (!r.spd_ok) {
            std::printf("  %-18s  NOT SPD (p^T A p<=0 at iter %d)\n", r.variant.c_str(), r.total_iters);
            continue;
        }
        std::printf("  %-18s %6.2f %8.4f %10.2e | %6d %6d %6d %6d | %7.2f %7.2f\n", r.variant.c_str(),
                    r.bytes_per_nnz, r.ms_per_iter, r.floor, r.iters_to_tol[0], r.iters_to_tol[1],
                    r.iters_to_tol[2], r.iters_to_tol[3], t2t(r, 1), t2t(r, 2));
    }

    // ---- mixed-precision IR: low-precision inner + exact lossless outer residual ----
    const CgResult* rd = nullptr;
    for (const auto& r : results)
        if (r.variant == "cusparse_double") rd = &r;
    auto base_time = [&](std::size_t ti) -> double {
        return (rd && rd->iters_to_tol[ti] > 0) ? rd->iters_to_tol[ti] * rd->ms_per_iter : -1.0;
    };
    const double tol_ir = 1e-8;
    std::vector<IrResult> solvers;  // IR (restarted) + FCG (flexible, no restart)
    for (int k : {20, 40, 80})
        solvers.push_back(run_ir(blas, n, mv_bf16, exact, d_b, bnorm, tol_ir, k, 600,
                                 "ir_bf16_k" + std::to_string(k)));
    for (int k : {5, 10, 20, 40})
        solvers.push_back(run_fcg(blas, n, exact, mv_bf16, d_b, bnorm, tol_ir, k, 2000,
                                  "fcg_bf16_k" + std::to_string(k)));
    for (int k : {5, 10})
        solvers.push_back(run_fcg(blas, n, exact, mv_bfp8, d_b, bnorm, tol_ir, k, 2000,
                                  "fcg_bfp8_k" + std::to_string(k)));

    const double base8 = base_time(3), base6 = base_time(2);
    std::printf("  --- IR vs FCG (target true relres %.0e) | baseline double: t@1e-6=%.2f t@1e-8=%.2fms ---\n",
                tol_ir, base6, base8);
    for (const auto& I : solvers)
        std::printf("  %-14s outer=%4d inner=%5d final=%9.2e %s | time=%8.2fms  speedup@1e-8 vs double=%.2fx\n",
                    I.name.c_str(), I.outer, I.inner_total, I.final_relres, I.reached ? "OK" : "NO",
                    I.time_ms, (base8 > 0 && I.time_ms > 0) ? base8 / I.time_ms : -1.0);

    std::string ir_csv = csv_path;
    if (ir_csv.size() > 4 && ir_csv.substr(ir_csv.size() - 4) == ".csv")
        ir_csv = ir_csv.substr(0, ir_csv.size() - 4) + "_ir.csv";
    else
        ir_csv += "_ir.csv";
    FILE* fi = std::fopen(ir_csv.c_str(), append ? "a" : "w");
    if (fi) {
        if (!append)
            std::fprintf(fi, "matrix,n,nnz,solver,outer,inner_total,final_relres,reached,"
                             "time_ms,base_double_t1e8_ms,speedup_1e8\n");
        for (const auto& I : solvers)
            std::fprintf(fi, "%s,%d,%lld,%s,%d,%d,%.3e,%d,%.4f,%.4f,%.4f\n", mname.c_str(), n,
                         (long long)device.nnz, I.name.c_str(), I.outer, I.inner_total,
                         I.final_relres, I.reached ? 1 : 0, I.time_ms, base8,
                         (base8 > 0 && I.time_ms > 0) ? base8 / I.time_ms : -1.0);
        std::fclose(fi);
    }
    return 0;
}
