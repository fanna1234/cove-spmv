#pragma once

#include "structural/operators/bitbsr_spmv_8x4_common.cuh"

namespace structural {

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_fp16_tile512_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_round16_tile_device<false>(
        bitbsr, packed_meta, tile_code_words, "fp16_tile512", x, y, device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_bf16_tile512_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_round16_tile_device<true>(
        bitbsr, packed_meta, tile_code_words, "bf16_tile512", x, y, device_ms);
}

// Per-nonzero 16-bit (2.0 B/nnz) bf16/fp16 device launches. These replace the
// per-cell tile512 launches above for the fp16/bf16 operators: same FP32-compute
// decode, but one 16-bit value per nonzero instead of 64 bytes/block.
template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_fp16_nnz_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint16_t>& values16,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_round16_nnz_device<false>(
        bitbsr, packed_meta, values16, "fp16_nnz", x, y, device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_bf16_nnz_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint16_t>& values16,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_round16_nnz_device<true>(
        bitbsr, packed_meta, values16, "bf16_nnz", x, y, device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_k256_tile256_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_code_tile_device<8>(
        bitbsr, packed_meta, tile_code_words, codebook,
        gpu_detail::kSpmvK256ValueCodebookSize, "k256_tile256", x, y,
        device_ms);
}

template <typename T>
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_k256_tile256(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_k256_tile256_device(
        bitbsr, packed_meta, tile_code_words, codebook, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_logkmeans_tile256_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_code_tile_device<8>(
        bitbsr, packed_meta, tile_code_words, codebook,
        gpu_detail::kSpmvK256ValueCodebookSize, "global_logkmeans_tile256", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_fp16_tile512_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_tile_device<false>(
        bitbsr, packed_meta, layout, tile_code_words, "fp16_tile512", x, y, device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bf16_tile512_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_tile_device<true>(
        bitbsr, packed_meta, layout, tile_code_words, "bf16_tile512", x, y, device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_fp16_nnz_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint16_t>& values16,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_nnz_device<false>(
        bitbsr, packed_meta, layout, values16, "fp16_nnz", x, y, device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bf16_nnz_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint16_t>& values16,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_nnz_device<true>(
        bitbsr, packed_meta, layout, values16, "bf16_nnz", x, y, device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_k256_tile256_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_code_tile_device<8>(
        bitbsr, packed_meta, layout, tile_code_words, codebook,
        gpu_detail::kSpmvK256ValueCodebookSize, "k256_tile256", x, y,
        device_ms);
}

template <typename T>
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_k256_tile256(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_k256_tile256_device(
        bitbsr, packed_meta, layout, tile_code_words, codebook, d_x, d_y,
        device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_logkmeans_tile256_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_code_tile_device<8>(
        bitbsr, packed_meta, layout, tile_code_words, codebook,
        gpu_detail::kSpmvK256ValueCodebookSize, "global_logkmeans_tile256", x, y,
        device_ms);
}

}  // namespace structural
