// Standalone acceptance harness for bfp8_outlier_lb.
// Computes the GLOBAL-NORM relative error  max_r|y - y0| / max_r|y0|  (y0 = fp64
// reference) for both BFP8 and BFP8+outlier, sharing the exact same BitBSR
// position/load-balance layout so the comparison is apples-to-apples. Also
// reports value bytes/nnz (full, with explicit row/col side list) and the
// value-only figure, plus the outlier count.
//
// This is a verification tool only (not wired into the Makefile); build with:
//   nvcc -O2 -std=c++17 -Istructural/include check_outlier_globalnorm.cu -o /tmp/check_outlier -lcusparse
#include "structural/gpu_convert.cuh"
#include "structural/gpu_value_codecs.cuh"
#include "structural/matrix_market.hpp"
#include "structural/operators/bitbsr_spmv_8x4_common.cuh"
#include "structural/spmv.hpp"

#include <thrust/device_vector.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using structural::Index;
using structural::Offset;

static std::vector<double> make_input_vector(Index cols) {
    std::vector<double> x(static_cast<size_t>(cols));
    for (Index i = 0; i < cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 17) - 8) * 0.125;
    }
    return x;
}

static double global_norm_rel(const std::vector<double>& y0,
                              const std::vector<double>& y,
                              double* out_max_abs_err,
                              double* out_max_abs_ref) {
    double max_abs_err = 0.0;
    double max_abs_ref = 0.0;
    for (size_t r = 0; r < y0.size(); ++r) {
        max_abs_err = std::max(max_abs_err, std::abs(y[r] - y0[r]));
        max_abs_ref = std::max(max_abs_ref, std::abs(y0[r]));
    }
    *out_max_abs_err = max_abs_err;
    *out_max_abs_ref = max_abs_ref;
    return max_abs_err / (max_abs_ref + 1e-30);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s matrix.mtx [outlier_fraction=0.005]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const double outlier_fraction = (argc > 2) ? std::atof(argv[2]) : 0.005;

    const auto csr = structural::read_matrix_market<double>(path);
    const auto x = make_input_vector(csr.cols);

    // fp64 reference.
    const auto y0 = structural::spmv_csr(csr, x);

    // GPU BitBSR + shared layout.
    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);
    const auto host_bitbsr = device.to_host();

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);
    structural::BitBsrLoadBalanceLayout8x4 layout;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, layout, /*chunk_blocks=*/32, /*split_threshold_blocks=*/32, packed_meta);

    thrust::device_vector<double> d_x(x.begin(), x.end());
    thrust::device_vector<double> d_y;

    // BFP8 bulk.
    const auto bfp8 = structural::build_checked_bfp8_blkscale_value_codec(host_bitbsr);
    thrust::device_vector<signed char> d_int8(bfp8.int8_values.begin(),
                                              bfp8.int8_values.end());
    thrust::device_vector<std::uint16_t> d_scale(bfp8.block_scale_bf16.begin(),
                                                 bfp8.block_scale_bf16.end());
    structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bfp8_blkscale_device(
        device, packed_meta, layout, d_int8, d_scale, d_x, d_y);
    std::vector<double> y_bfp8(static_cast<size_t>(device.rows));
    thrust::copy(d_y.begin(), d_y.end(), y_bfp8.begin());

    // BFP8 + outlier two-pass.
    const auto outlier = structural::build_checked_bfp8_outlier_blkscale_value_codec(
        host_bitbsr, outlier_fraction);
    thrust::device_vector<signed char> d_int8_o(outlier.int8_values.begin(),
                                                outlier.int8_values.end());
    thrust::device_vector<std::uint16_t> d_scale_o(outlier.block_scale_bf16.begin(),
                                                   outlier.block_scale_bf16.end());
    thrust::device_vector<std::int32_t> d_orows(outlier.outlier_rows.begin(),
                                                outlier.outlier_rows.end());
    thrust::device_vector<std::int32_t> d_ocols(outlier.outlier_cols.begin(),
                                                outlier.outlier_cols.end());
    thrust::device_vector<double> d_ovals(outlier.outlier_vals.begin(),
                                          outlier.outlier_vals.end());
    structural::spmv_bitbsr_gpu_8x4_bfp8_outlier_flag_load_balance_device(
        device, packed_meta, layout, d_int8_o, d_scale_o, d_orows, d_ocols, d_ovals,
        d_x, d_y);
    std::vector<double> y_outlier(static_cast<size_t>(device.rows));
    thrust::copy(d_y.begin(), d_y.end(), y_outlier.begin());

    double e_abs = 0, e_ref = 0;
    const double rel_bfp8 = global_norm_rel(y0, y_bfp8, &e_abs, &e_ref);
    double o_abs = 0, o_ref = 0;
    const double rel_outlier = global_norm_rel(y0, y_outlier, &o_abs, &o_ref);

    const double nnz = static_cast<double>(device.nnz);
    const double bytes_full = static_cast<double>(outlier.payload_bytes());
    const double bytes_value_only = static_cast<double>(outlier.value_only_payload_bytes());
    const double bytes_bfp8 = static_cast<double>(bfp8.payload_bytes());

    const char* name = path.c_str();
    const char* slash = std::strrchr(name, '/');
    if (slash) name = slash + 1;

    std::printf(
        "%-12s nnz=%lld outliers=%lld(%.3f%%) | "
        "globalnorm_rel bfp8=%.4e bfp8_outlier=%.4e (ratio %.2fx) | "
        "bytes/nnz bfp8=%.4f outlier_full=%.4f outlier_value_only=%.4f (value_only-bfp8=+%.4f) | "
        "max|y0|=%.3e\n",
        name, (long long)device.nnz, (long long)outlier.num_outliers,
        100.0 * static_cast<double>(outlier.num_outliers) / (nnz + 1e-30),
        rel_bfp8, rel_outlier,
        (rel_outlier > 0 ? rel_bfp8 / rel_outlier : 0.0),
        bytes_bfp8 / nnz, bytes_full / nnz, bytes_value_only / nnz,
        (bytes_value_only - bytes_bfp8) / nnz,
        e_ref);
    return 0;
}
