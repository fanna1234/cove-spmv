#include "structural/bitbsr.hpp"
#include "structural/convert.hpp"
#include "structural/csr.hpp"
#include "structural/gpu_convert.cuh"
#include "structural/gpu_spmv.cuh"
#include "structural/gpu_value_codecs.cuh"
#include "structural/spmv.hpp"
#include "structural/value_compression.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename T>
void assert_equal_bitbsr(const structural::BitBsrMatrix<T>& a,
                         const structural::BitBsrMatrix<T>& b) {
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);
    assert(a.block_rows == b.block_rows);
    assert(a.block_cols == b.block_cols);
    assert(a.num_block_rows == b.num_block_rows);
    assert(a.num_block_cols == b.num_block_cols);
    assert(a.words_per_block == b.words_per_block);
    assert(a.block_row_ptr == b.block_row_ptr);
    assert(a.block_col_idx == b.block_col_idx);
    assert(a.bitmap_words == b.bitmap_words);
    assert(a.block_val_ptr == b.block_val_ptr);
    assert(a.values == b.values);
}

template <typename T>
void assert_vectors_close(const std::vector<T>& expected,
                          const std::vector<T>& got,
                          T tolerance) {
    assert(expected.size() == got.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(expected[i] - got[i]) <= tolerance);
    }
}

std::vector<structural::Offset> copy_offsets_to_host(
    const thrust::device_vector<structural::Offset>& device_offsets) {
    std::vector<structural::Offset> host(device_offsets.size());
    thrust::copy(device_offsets.begin(), device_offsets.end(), host.begin());
    return host;
}

template <typename T>
std::vector<T> copy_vector_to_host(const thrust::device_vector<T>& device_values) {
    std::vector<T> host(device_values.size());
    thrust::copy(device_values.begin(), device_values.end(), host.begin());
    return host;
}

template <typename T>
std::vector<std::uint32_t> build_test_sign_tile32_masks(
    const structural::BitBsrMatrix<T>& bitbsr) {
    bitbsr.validate();
    assert(bitbsr.block_rows == 8);
    assert(bitbsr.block_cols == 4);
    assert(bitbsr.words_per_block == 1);

    std::vector<std::uint32_t> masks(static_cast<size_t>(bitbsr.num_blocks()), 0);
    for (structural::Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const auto bitmap = static_cast<std::uint32_t>(bitbsr.bitmap_words[block]);
        const auto value_base = bitbsr.block_val_ptr[block];
        for (int bit = 0; bit < 32; ++bit) {
            if ((bitmap & (std::uint32_t{1} << bit)) == 0U) {
                continue;
            }
            const std::uint32_t before = bitmap & ((std::uint32_t{1} << bit) - 1U);
            const auto value = bitbsr.values[value_base + structural::popcount_word(before)];
            if (value == T{1}) {
                masks[static_cast<size_t>(block)] |= std::uint32_t{1} << bit;
            } else {
                assert(value == T{-1});
            }
        }
    }
    return masks;
}

template <int BitsPerCode, typename T>
std::vector<std::uint32_t> build_test_tile_code_words(
    const structural::BitBsrMatrix<T>& bitbsr,
    const std::vector<T>& codebook) {
    static_assert(BitsPerCode == 2 || BitsPerCode == 4 || BitsPerCode == 8,
                  "test tile codec supports 2, 4, or 8 bits per cell");
    bitbsr.validate();
    assert(bitbsr.block_rows == 8);
    assert(bitbsr.block_cols == 4);
    assert(bitbsr.words_per_block == 1);

    constexpr int words_per_tile = BitsPerCode;
    constexpr int codes_per_word = 32 / BitsPerCode;
    std::vector<std::uint32_t> words(
        static_cast<size_t>(bitbsr.num_blocks()) * words_per_tile, 0);
    for (structural::Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const auto bitmap = static_cast<std::uint32_t>(bitbsr.bitmap_words[block]);
        const auto value_base = bitbsr.block_val_ptr[block];
        for (int bit = 0; bit < 32; ++bit) {
            if ((bitmap & (std::uint32_t{1} << bit)) == 0U) {
                continue;
            }
            const std::uint32_t before = bitmap & ((std::uint32_t{1} << bit) - 1U);
            const auto value = bitbsr.values[value_base + structural::popcount_word(before)];
            const auto id = static_cast<std::uint32_t>(
                structural::exact_value_id(codebook, value));
            assert(id < (std::uint32_t{1} << BitsPerCode));
            const int word = bit / codes_per_word;
            const int shift = (bit % codes_per_word) * BitsPerCode;
            words[static_cast<size_t>(block) * words_per_tile + word] |= id << shift;
        }
    }
    return words;
}

void test_gpu_warp_direct_8x4_matches_cpu() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 10;
    csr.row_ptr = {0, 3, 5, 6, 8, 11, 13, 14, 17, 19};
    csr.col_idx = {
        0, 3, 8,
        1, 5,
        4,
        2, 9,
        0, 4, 7,
        3, 8,
        6,
        1, 5, 9,
        0, 4,
    };
    csr.values = {
        1.0, 2.0, 3.0,
        4.0, 5.0,
        6.0,
        7.0, 8.0,
        9.0, 10.0, 11.0,
        12.0, 13.0,
        14.0,
        15.0, 16.0, 17.0,
        18.0, 19.0,
    };

    const auto cpu = structural::convert_to_bitbsr(csr, 8, 4);
    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);
    const auto gpu = device.to_host();

    assert_equal_bitbsr(cpu, gpu);
}

void test_gpu_warp_direct_8x4_empty_block_row_matches_cpu() {
    structural::CsrMatrix<double> csr;
    csr.rows = 17;
    csr.cols = 9;
    csr.row_ptr = {
        0, 2, 3, 3, 4, 4, 4, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        7,
    };
    csr.col_idx = {
        0, 4,
        8,
        3,
        5,
        1, 8,
    };
    csr.values = {
        1.0, 2.0,
        3.0,
        4.0,
        5.0,
        6.0, 7.0,
    };

    const auto cpu = structural::convert_to_bitbsr(csr, 8, 4);
    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);
    const auto gpu = device.to_host();

    assert_equal_bitbsr(cpu, gpu);
}

void test_gpu_warp_direct_8x4_device_resident_matches_cpu() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 10;
    csr.row_ptr = {0, 3, 5, 6, 8, 11, 13, 14, 17, 19};
    csr.col_idx = {
        0, 3, 8,
        1, 5,
        4,
        2, 9,
        0, 4, 7,
        3, 8,
        6,
        1, 5, 9,
        0, 4,
    };
    csr.values = {
        1.0, 2.0, 3.0,
        4.0, 5.0,
        6.0,
        7.0, 8.0,
        9.0, 10.0, 11.0,
        12.0, 13.0,
        14.0,
        15.0, 16.0, 17.0,
        18.0, 19.0,
    };

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    assert(copy_offsets_to_host(device.block_row_val_ptr) ==
           (std::vector<structural::Offset>{0, 17, 19}));

    const auto cpu = structural::convert_to_bitbsr(csr, 8, 4);
    const auto gpu = device.to_host();
    assert_equal_bitbsr(cpu, gpu);
}

void test_gpu_warp_direct_8x4_device_resident_empty_matrix_matches_cpu() {
    structural::CsrMatrix<float> csr;
    csr.rows = 3;
    csr.cols = 4;
    csr.row_ptr = {0, 0, 0, 0};

    structural::GpuBitBsrWorkspace<float> workspace;
    structural::DeviceBitBsrMatrix<float> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    const auto cpu = structural::convert_to_bitbsr(csr, 8, 4);
    const auto gpu = device.to_host();
    assert_equal_bitbsr(cpu, gpu);
}

void test_gpu_warp_direct_8x4_workspace_reuse_matches_cpu() {
    structural::CsrMatrix<double> first;
    first.rows = 17;
    first.cols = 9;
    first.row_ptr = {
        0, 2, 3, 3, 4, 4, 4, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        7,
    };
    first.col_idx = {
        0, 4,
        8,
        3,
        5,
        1, 8,
    };
    first.values = {
        1.0, 2.0,
        3.0,
        4.0,
        5.0,
        6.0, 7.0,
    };

    structural::CsrMatrix<double> second;
    second.rows = 8;
    second.cols = 16;
    second.row_ptr = {0, 1, 3, 3, 5, 6, 7, 9, 10};
    second.col_idx = {0, 4, 15, 1, 8, 7, 3, 11, 12, 15};
    second.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    structural::CsrMatrix<double> empty;
    empty.rows = 5;
    empty.cols = 6;
    empty.row_ptr = {0, 0, 0, 0, 0, 0};

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;

    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(first, workspace, device);
    assert_equal_bitbsr(structural::convert_to_bitbsr(first, 8, 4), device.to_host());

    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(second, workspace, device);
    assert_equal_bitbsr(structural::convert_to_bitbsr(second, 8, 4), device.to_host());

    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(empty, workspace, device);
    assert_equal_bitbsr(structural::convert_to_bitbsr(empty, 8, 4), device.to_host());
}

void test_gpu_exact_value_histogram_counts_unique_values() {
    structural::CsrMatrix<double> csr;
    csr.rows = 8;
    csr.cols = 4;
    csr.row_ptr = {0, 4, 8, 12, 16, 20, 24, 28, 32};
    csr.col_idx = {
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
    };
    csr.values = {
        2.0, -3.0, 5.0, 2.0,
        -3.0, 5.0, 2.0, -3.0,
        5.0, 2.0, -3.0, 5.0,
        2.0, -3.0, 5.0, 2.0,
        -3.0, 5.0, 2.0, -3.0,
        5.0, 2.0, -3.0, 5.0,
        2.0, -3.0, 5.0, 2.0,
        -3.0, 5.0, 2.0, -3.0,
    };

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    const auto histogram = structural::build_device_exact_value_histogram(device);
    assert(histogram.nnz == 32);
    assert(histogram.num_blocks == 1);
    assert(histogram.unique_count == 3);
    assert(!histogram.unique_overflow);
    assert(!histogram.all_one);
    assert(!histogram.const_c);
    assert(!histogram.sign2);
    assert(!histogram.sign2_scale);
    assert(histogram.raw_payload_bytes == 32 * sizeof(double));
    assert(histogram.dict4_payload_bytes < histogram.raw_payload_bytes);
}

void test_gpu_exact_value_histogram_detects_special_domains() {
    structural::CsrMatrix<float> all_one;
    all_one.rows = 8;
    all_one.cols = 4;
    all_one.row_ptr = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    all_one.col_idx = {0, 1, 2, 3, 0, 1, 2, 3};
    all_one.values.assign(all_one.col_idx.size(), 1.0f);

    structural::GpuBitBsrWorkspace<float> workspace;
    structural::DeviceBitBsrMatrix<float> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(all_one, workspace, device);

    const auto all_one_histogram = structural::build_device_exact_value_histogram(device);
    assert(all_one_histogram.unique_count == 1);
    assert(all_one_histogram.all_one);
    assert(all_one_histogram.const_c);
    assert(!all_one_histogram.sign2);
    assert(!all_one_histogram.sign2_scale);
    assert(all_one_histogram.const_c_payload_bytes == sizeof(float));

    structural::CsrMatrix<float> sign2;
    sign2.rows = 8;
    sign2.cols = 4;
    sign2.row_ptr = all_one.row_ptr;
    sign2.col_idx = all_one.col_idx;
    sign2.values = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};

    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(sign2, workspace, device);
    const auto sign2_histogram = structural::build_device_exact_value_histogram(device);
    assert(sign2_histogram.unique_count == 2);
    assert(!sign2_histogram.all_one);
    assert(!sign2_histogram.const_c);
    assert(sign2_histogram.sign2);
    assert(sign2_histogram.sign2_scale);
    assert(sign2_histogram.sign2_payload_bytes < sign2_histogram.raw_payload_bytes);
    assert(sign2_histogram.sign2_scale_payload_bytes ==
           sign2_histogram.sign2_payload_bytes + sizeof(float));

    structural::CsrMatrix<float> const_c = all_one;
    const_c.values.assign(const_c.col_idx.size(), 2.5f);

    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(const_c, workspace, device);
    const auto const_c_histogram = structural::build_device_exact_value_histogram(device);
    assert(const_c_histogram.unique_count == 1);
    assert(!const_c_histogram.all_one);
    assert(const_c_histogram.const_c);
    assert(!const_c_histogram.sign2);
    assert(!const_c_histogram.sign2_scale);
    assert(const_c_histogram.const_c_payload_bytes == sizeof(float));
    assert(const_c_histogram.const_c_payload_bytes < const_c_histogram.raw_payload_bytes);

    structural::CsrMatrix<float> sign2_scale = sign2;
    sign2_scale.values = {
        2.5f, -2.5f, 2.5f, -2.5f,
        2.5f, -2.5f, 2.5f, -2.5f,
    };

    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(sign2_scale, workspace, device);
    const auto sign2_scale_histogram =
        structural::build_device_exact_value_histogram(device);
    assert(sign2_scale_histogram.unique_count == 2);
    assert(!sign2_scale_histogram.all_one);
    assert(!sign2_scale_histogram.const_c);
    assert(!sign2_scale_histogram.sign2);
    assert(sign2_scale_histogram.sign2_scale);
    assert(sign2_scale_histogram.sign2_scale_payload_bytes <
           sign2_scale_histogram.raw_payload_bytes);
}

void test_gpu_spmv_8x4_empty_matrix_returns_zero_vector() {
    structural::CsrMatrix<float> csr;
    csr.rows = 5;
    csr.cols = 6;
    csr.row_ptr = {0, 0, 0, 0, 0, 0};
    const std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    structural::GpuBitBsrWorkspace<float> workspace;
    structural::DeviceBitBsrMatrix<float> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);
    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    const auto expected = structural::spmv_csr(csr, x);
    const auto got = structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed(
        device, packed_meta, x);
    assert_vectors_close(expected, got, 0.0f);
}

void test_gpu_spmv_8x4_col_bitmap_packed_allows_wide_block_span() {
    constexpr structural::Index far_block_col = 70000;

    const std::uint64_t flagged_meta =
        structural::gpu_detail::pack_col_bitmap(0x7fffffffU, 0xffffffffU, true);
    assert(structural::gpu_detail::unpack_abs_col(flagged_meta) == 0x7fffffffU);
    assert(structural::gpu_detail::unpack_bitmap(flagged_meta) == 0xffffffffU);
    assert(structural::gpu_detail::unpack_col_bitmap_flag(flagged_meta));

    structural::CsrMatrix<double> csr;
    csr.rows = 8;
    csr.cols = far_block_col * 4 + 4;
    csr.row_ptr = {0, 2, 2, 2, 2, 2, 2, 2, 3};
    csr.col_idx = {0, far_block_col * 4, far_block_col * 4 + 3};
    csr.values = {1.0, 2.0, 3.0};

    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 11) - 5) * 0.125;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    assert((copy_vector_to_host(device.block_col_idx) ==
            std::vector<structural::Index>{0, far_block_col}));

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    const auto pack = [](std::uint32_t bitmap, std::uint32_t col) {
        return (static_cast<std::uint64_t>(bitmap) << 32) |
               static_cast<std::uint64_t>(col);
    };
    assert((copy_vector_to_host(packed_meta) ==
            std::vector<std::uint64_t>{
                pack(1U, 0U),
                pack(0x80000001U, static_cast<std::uint32_t>(far_block_col)),
            }));
    for (const std::uint64_t meta : copy_vector_to_host(packed_meta)) {
        assert(!structural::gpu_detail::unpack_col_bitmap_flag(meta));
    }

    const auto expected = structural::spmv_csr(csr, x);
    const auto got = structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed(
        device, packed_meta, x);
    assert_vectors_close(expected, got, 1e-12);

    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);
    structural::BitBsrLoadBalanceLayout8x4 flag_load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, flag_load_balance, 1, packed_meta);
    const auto flag_lb_meta = copy_vector_to_host(packed_meta);
    assert(flag_lb_meta.size() == 2);
    assert(structural::gpu_detail::unpack_col_bitmap_flag(flag_lb_meta[0]));
    assert(structural::gpu_detail::unpack_col_bitmap_flag(flag_lb_meta[1]));
    const auto flag_work_items = copy_vector_to_host(flag_load_balance.work_items);
    assert(flag_work_items.size() == 2);
    assert(flag_work_items[0].block_count == 1U);
    assert(flag_work_items[1].block_count == 1U);
    const auto got_flag_lb =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance(
            device, packed_meta, flag_load_balance, x);
    assert_vectors_close(expected, got_flag_lb, 1e-12);
}

void test_gpu_spmv_8x4_col_bitmap_flag_load_balance_matches_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values = {
        1.0, 2.0, 3.0,
        4.0, 5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0, 11.0, 12.0,
    };
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, packed_meta);

    const auto work_items = copy_vector_to_host(load_balance.work_items);
    const auto packed_meta_host = copy_vector_to_host(packed_meta);
    assert(work_items.size() == 4);
    assert(work_items[0].br == 0U);
    assert(work_items[0].block_begin == 0U);
    assert(work_items[0].block_count == 2U);
    assert(work_items[0].value_begin == 0U);
    assert(structural::gpu_detail::unpack_col_bitmap_flag(packed_meta_host[0]));
    assert(work_items[1].br == 0U);
    assert(work_items[1].block_begin == 2U);
    assert(work_items[1].block_count == 1U);
    assert(work_items[1].value_begin == 6U);
    assert(structural::gpu_detail::unpack_col_bitmap_flag(packed_meta_host[2]));
    assert(work_items[2].br == 1U);
    assert(work_items[2].block_begin == 3U);
    assert(work_items[2].block_count == 2U);
    assert(work_items[2].value_begin == 9U);
    assert(structural::gpu_detail::unpack_col_bitmap_flag(packed_meta_host[3]));
    assert(work_items[3].br == 1U);
    assert(work_items[3].block_begin == 5U);
    assert(work_items[3].block_count == 1U);
    assert(work_items[3].value_begin == 11U);
    assert(structural::gpu_detail::unpack_col_bitmap_flag(packed_meta_host[5]));

    const auto expected = structural::spmv_csr(csr, x);
    const auto got = structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance(
        device, packed_meta, load_balance, x);
    assert_vectors_close(expected, got, 1e-12);
}

void test_gpu_spmv_8x4_col_bitmap_flag_load_balance_k256_tile256_matches_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values = {
        1.0, 2.0, 3.0,
        4.0, 5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0, 11.0, 12.0,
    };
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, packed_meta);

    const auto host_bitbsr = device.to_host();
    const auto codec = structural::build_k256_uint8_value_codebook(host_bitbsr.values);
    assert(codec.exact);
    const std::vector<double> codebook_host(codec.codebook.begin(), codec.codebook.end());
    const auto packed_ids = build_test_tile_code_words<8>(host_bitbsr, codebook_host);
    thrust::device_vector<std::uint32_t> value_ids(packed_ids.begin(), packed_ids.end());
    thrust::device_vector<double> codebook(codebook_host.begin(), codebook_host.end());

    const auto expected = structural::spmv_csr(csr, x);
    const auto got =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_k256_tile256(
            device, packed_meta, load_balance, value_ids, codebook, x);
    assert_vectors_close(expected, got, 1e-12);
}

void test_gpu_spmv_8x4_col_bitmap_flag_load_balance_implicit_one_matches_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values.assign(csr.col_idx.size(), 1.0);
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, packed_meta);

    const auto expected = structural::spmv_csr(csr, x);
    const auto got =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_implicit_one(
            device, packed_meta, load_balance, x);
    assert_vectors_close(expected, got, 1e-12);
}

void test_gpu_spmv_8x4_col_bitmap_flag_load_balance_const_c_matches_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values.assign(csr.col_idx.size(), 2.5);
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, packed_meta);

    const auto host_bitbsr = device.to_host();
    const double value = structural::build_const_c_value(host_bitbsr);

    const auto expected = structural::spmv_csr(csr, x);
    const auto nolb_got = structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_const_c(
        device, packed_meta, value, x);
    assert_vectors_close(expected, nolb_got, 1e-12);

    const auto lb_got =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_const_c(
            device, packed_meta, load_balance, value, x);
    assert_vectors_close(expected, lb_got, 1e-12);
}

void test_gpu_spmv_8x4_col_bitmap_flag_load_balance_sign2_matches_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values = {
        1.0, -1.0, 1.0,
        -1.0, 1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0, 1.0, -1.0,
    };
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, packed_meta);

    const auto host_bitbsr = device.to_host();
    const auto sign_masks = build_test_sign_tile32_masks(host_bitbsr);
    thrust::device_vector<std::uint32_t> value_sign_masks(
        sign_masks.begin(), sign_masks.end());

    const auto expected = structural::spmv_csr(csr, x);
    const auto got =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign_tile32(
            device, packed_meta, load_balance, value_sign_masks, x);
    assert_vectors_close(expected, got, 1e-12);
}

void test_gpu_spmv_8x4_col_bitmap_flag_load_balance_sign2_scale_matches_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values = {
        2.5, -2.5, 2.5,
        -2.5, 2.5,
        -2.5,
        2.5,
        -2.5,
        2.5,
        -2.5, 2.5, -2.5,
    };
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, packed_meta);

    const auto host_bitbsr = device.to_host();
    const auto codec = structural::build_sign2_scale_tile32_value_codec(host_bitbsr);
    thrust::device_vector<std::uint32_t> value_sign_masks(
        codec.sign_masks.begin(), codec.sign_masks.end());

    const auto expected = structural::spmv_csr(csr, x);
    const auto nolb_got =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign2_scale_tile32(
            device, packed_meta, value_sign_masks, codec.scale, x);
    assert_vectors_close(expected, nolb_got, 1e-12);

    const auto lb_got =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign2_scale_tile32(
            device, packed_meta, load_balance, value_sign_masks, codec.scale, x);
    assert_vectors_close(expected, lb_got, 1e-12);
}

void test_gpu_spmv_8x4_col_bitmap_flag_load_balance_unique16_matches_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values = {
        2.0, -3.0, 5.0,
        -3.0, 2.0,
        5.0,
        2.0,
        -3.0,
        5.0,
        2.0, -3.0, 5.0,
    };
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, packed_meta);

    const auto host_bitbsr = device.to_host();
    const auto codebook_host = structural::sorted_unique_values(host_bitbsr.values, 16);
    const auto packed_ids = build_test_tile_code_words<4>(host_bitbsr, codebook_host);
    thrust::device_vector<std::uint32_t> value_ids(packed_ids.begin(), packed_ids.end());
    thrust::device_vector<double> codebook(codebook_host.begin(), codebook_host.end());

    const auto expected = structural::spmv_csr(csr, x);
    const auto got =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique16_tile128(
            device, packed_meta, load_balance, value_ids, codebook, x);
    assert_vectors_close(expected, got, 1e-12);
}

void test_gpu_spmv_8x4_col_bitmap_flag_load_balance_unique4_matches_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values = {
        2.0, -3.0, 5.0,
        -3.0, 2.0,
        5.0,
        2.0,
        -3.0,
        5.0,
        2.0, -3.0, 5.0,
    };
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, packed_meta);

    const auto host_bitbsr = device.to_host();
    const auto codebook_host = structural::sorted_unique_values(host_bitbsr.values, 4);
    const auto packed_ids = build_test_tile_code_words<2>(host_bitbsr, codebook_host);
    thrust::device_vector<std::uint32_t> value_ids(packed_ids.begin(), packed_ids.end());
    thrust::device_vector<double> codebook(codebook_host.begin(), codebook_host.end());

    const auto expected = structural::spmv_csr(csr, x);
    const auto got =
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique4_tile64(
            device, packed_meta, load_balance, value_ids, codebook, x);
    assert_vectors_close(expected, got, 1e-12);
}

void test_gpu_spmv_8x4_load_balance_split_threshold_controls_chunking() {
    structural::CsrMatrix<double> csr;
    csr.rows = 9;
    csr.cols = 4404;
    csr.row_ptr = {0, 3, 5, 6, 7, 8, 8, 9, 9, 12};
    csr.col_idx = {
        400, 416, 2000,
        401, 417,
        2001,
        418,
        402,
        2002,
        2800, 3600, 4400,
    };
    csr.values = {
        1.0, 2.0, 3.0,
        4.0, 5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0, 11.0, 12.0,
    };
    std::vector<double> x(static_cast<size_t>(csr.cols));
    for (structural::Index i = 0; i < csr.cols; ++i) {
        x[static_cast<size_t>(i)] = static_cast<double>((i % 13) - 6) * 0.25;
    }

    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);

    structural::BitBsrLoadBalanceLayout8x4 load_balance;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, load_balance, 2, 4, packed_meta);

    const auto work_items = copy_vector_to_host(load_balance.work_items);
    const auto packed_meta_host = copy_vector_to_host(packed_meta);
    assert(work_items.size() == 2);
    assert(work_items[0].br == 0U);
    assert(work_items[0].block_begin == 0U);
    assert(work_items[0].block_count == 3U);
    assert(work_items[0].value_begin == 0U);
    assert(!structural::gpu_detail::unpack_col_bitmap_flag(packed_meta_host[0]));
    assert(work_items[1].br == 1U);
    assert(work_items[1].block_begin == 3U);
    assert(work_items[1].block_count == 3U);
    assert(work_items[1].value_begin == 9U);
    assert(!structural::gpu_detail::unpack_col_bitmap_flag(packed_meta_host[3]));

    const auto expected = structural::spmv_csr(csr, x);
    const auto got = structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance(
        device, packed_meta, load_balance, x);
    assert_vectors_close(expected, got, 1e-12);
}

void test_gpu_spmv_8x4_rejects_other_block_shapes() {
    structural::CsrMatrix<double> csr;
    csr.rows = 8;
    csr.cols = 8;
    csr.row_ptr = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    csr.col_idx = {0, 1, 2, 3, 4, 5, 6, 7};
    csr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const std::vector<double> x(8, 1.0);

    const auto host_8x8 = structural::convert_to_bitbsr(csr, 8, 8);
    structural::DeviceBitBsrMatrix<double> device;
    structural::copy_bitbsr_to_device(host_8x8, device);
    thrust::device_vector<std::uint64_t> packed_meta;

    bool threw = false;
    try {
        (void)structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed(
            device, packed_meta, x);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 0;
    }

    test_gpu_warp_direct_8x4_matches_cpu();
    test_gpu_warp_direct_8x4_empty_block_row_matches_cpu();
    test_gpu_warp_direct_8x4_device_resident_matches_cpu();
    test_gpu_warp_direct_8x4_device_resident_empty_matrix_matches_cpu();
    test_gpu_warp_direct_8x4_workspace_reuse_matches_cpu();
    test_gpu_exact_value_histogram_counts_unique_values();
    test_gpu_exact_value_histogram_detects_special_domains();
    test_gpu_spmv_8x4_empty_matrix_returns_zero_vector();
    test_gpu_spmv_8x4_col_bitmap_packed_allows_wide_block_span();
    test_gpu_spmv_8x4_col_bitmap_flag_load_balance_matches_csr();
    test_gpu_spmv_8x4_col_bitmap_flag_load_balance_k256_tile256_matches_csr();
    test_gpu_spmv_8x4_col_bitmap_flag_load_balance_implicit_one_matches_csr();
    test_gpu_spmv_8x4_col_bitmap_flag_load_balance_const_c_matches_csr();
    test_gpu_spmv_8x4_col_bitmap_flag_load_balance_sign2_matches_csr();
    test_gpu_spmv_8x4_col_bitmap_flag_load_balance_sign2_scale_matches_csr();
    test_gpu_spmv_8x4_col_bitmap_flag_load_balance_unique4_matches_csr();
    test_gpu_spmv_8x4_col_bitmap_flag_load_balance_unique16_matches_csr();
    test_gpu_spmv_8x4_load_balance_split_threshold_controls_chunking();
    test_gpu_spmv_8x4_rejects_other_block_shapes();
    return 0;
}
