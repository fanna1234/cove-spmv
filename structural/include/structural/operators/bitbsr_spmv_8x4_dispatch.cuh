#pragma once

#include "structural/operators/bitbsr_spmv_8x4.cuh"

#include <thrust/device_vector.h>

#include <cstdint>
#include <stdexcept>

namespace structural {

enum class BitBsrSpmvLayoutKind8x4 {
    ColBitmap64,
    ColBitmap64FlagLb,
    CusparseCsr,
};

enum class BitBsrValueCodecKind8x4 {
    Raw,
    ImplicitOne,
    ConstC,
    Sign2,
    Sign2Scale,
    Dict2,
    Unique4,
    Unique16,
    Dict256U8,
    Dict65536U16,
    Fp16,
    Bf16,
    Bfp8,
    Bfp8Outlier,
    Bfp4,
    K256U8,
    GlobalLogKmeans16,
    GlobalLogKmeans32,
    GlobalLogKmeans64,
    GlobalLogKmeans128,
};

inline bool bitbsr_spmv_8x4_uses_col_bitmap64_meta(BitBsrSpmvLayoutKind8x4 layout) {
    return layout == BitBsrSpmvLayoutKind8x4::ColBitmap64 ||
           layout == BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb;
}

inline bool bitbsr_spmv_8x4_uses_load_balance(BitBsrSpmvLayoutKind8x4 layout) {
    return layout == BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb;
}

inline bool bitbsr_spmv_8x4_is_non_raw_value_codec(BitBsrValueCodecKind8x4 value_codec) {
    return value_codec != BitBsrValueCodecKind8x4::Raw;
}

inline bool bitbsr_spmv_8x4_is_controlled_lossy_value_codec(
    BitBsrValueCodecKind8x4 value_codec) {
    switch (value_codec) {
        case BitBsrValueCodecKind8x4::Fp16:
        case BitBsrValueCodecKind8x4::Bf16:
        case BitBsrValueCodecKind8x4::Bfp8:
        case BitBsrValueCodecKind8x4::Bfp8Outlier:
        case BitBsrValueCodecKind8x4::Bfp4:
        case BitBsrValueCodecKind8x4::K256U8:
        case BitBsrValueCodecKind8x4::GlobalLogKmeans16:
        case BitBsrValueCodecKind8x4::GlobalLogKmeans32:
        case BitBsrValueCodecKind8x4::GlobalLogKmeans64:
        case BitBsrValueCodecKind8x4::GlobalLogKmeans128:
            return true;
        case BitBsrValueCodecKind8x4::Raw:
        case BitBsrValueCodecKind8x4::ImplicitOne:
        case BitBsrValueCodecKind8x4::ConstC:
        case BitBsrValueCodecKind8x4::Sign2:
        case BitBsrValueCodecKind8x4::Sign2Scale:
        case BitBsrValueCodecKind8x4::Dict2:
        case BitBsrValueCodecKind8x4::Unique4:
        case BitBsrValueCodecKind8x4::Unique16:
        case BitBsrValueCodecKind8x4::Dict256U8:
        case BitBsrValueCodecKind8x4::Dict65536U16:
            return false;
    }
    return false;
}

template <typename T>
struct BitBsrSpmvPlan8x4 {
    BitBsrSpmvLayoutKind8x4 layout = BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb;
    BitBsrValueCodecKind8x4 value_codec = BitBsrValueCodecKind8x4::Raw;
    T value_scale = T{1};
    thrust::device_vector<std::uint32_t> tile_code_words;
    thrust::device_vector<std::uint32_t> sign_masks;
    thrust::device_vector<T> value_codebook;
    // BFP8 (int8 mantissa + per-8x4-block bf16 scale): per-nnz int8 mantissas in
    // block-grouped bit-rank order, plus one bf16-bits scale per block.
    thrust::device_vector<signed char> int8_values;
    thrust::device_vector<std::uint16_t> block_scale_bf16;
    // BFP4 (int4 mantissa + per-8x4-block bf16 scale): two int4 mantissas per
    // byte (linear nnz order); reuses block_scale_bf16 for the per-block scale.
    thrust::device_vector<std::uint8_t> packed_int4;
    // Per-nonzero 16-bit values (bf16/fp16, NO block scale): one 16-bit value per
    // nonzero in block-grouped bit-rank order = 2.0 B/nnz. One vector serves both
    // bf16 and fp16 since only one value codec is active per run.
    thrust::device_vector<std::uint16_t> values16;
    // BFP8 outlier two-pass side list (absolute row, absolute col, exact value).
    // Reuses int8_values/block_scale_bf16 for the bulk pass-1 BFP8 codec.
    thrust::device_vector<std::int32_t> outlier_rows;
    thrust::device_vector<std::int32_t> outlier_cols;
    thrust::device_vector<T> outlier_vals;
    thrust::device_vector<std::uint64_t> packed_col_bitmap_meta;
    BitBsrLoadBalanceLayout8x4 load_balance;
};

namespace gpu_detail {

template <typename T>
void run_bitbsr_spmv_8x4_col_bitmap64_once(const DeviceBitBsrMatrix<T>& device,
                                           const BitBsrSpmvPlan8x4<T>& plan,
                                           bool use_load_balance,
                                           const thrust::device_vector<T>& d_x,
                                           thrust::device_vector<T>& d_y,
                                           float* iter_ms) {
    switch (plan.value_codec) {
        case BitBsrValueCodecKind8x4::Raw:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_device(
                    device, plan.packed_col_bitmap_meta, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::ImplicitOne:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_implicit_one_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_implicit_one_device(
                    device, plan.packed_col_bitmap_meta, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::ConstC:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_const_c_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.value_scale, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_const_c_device(
                    device, plan.packed_col_bitmap_meta, plan.value_scale, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Sign2:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign_tile32_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance, plan.sign_masks,
                    d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign_tile32_device(
                    device, plan.packed_col_bitmap_meta, plan.sign_masks, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Sign2Scale:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign2_scale_tile32_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance, plan.sign_masks,
                    plan.value_scale, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign2_scale_tile32_device(
                    device, plan.packed_col_bitmap_meta, plan.sign_masks, plan.value_scale,
                    d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Dict2:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_dict2_tile32_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.tile_code_words, plan.value_codebook, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_dict2_tile32_device(
                    device, plan.packed_col_bitmap_meta, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Unique4:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique4_tile64_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_unique4_tile64_device(
                    device, plan.packed_col_bitmap_meta, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Unique16:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique16_tile128_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_unique16_tile128_device(
                    device, plan.packed_col_bitmap_meta, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Dict256U8:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_dict256_tile256_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.tile_code_words, plan.value_codebook, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_dict256_tile256_device(
                    device, plan.packed_col_bitmap_meta, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Dict65536U16:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_dict65536_tile512_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.tile_code_words, plan.value_codebook, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_dict65536_tile512_device(
                    device, plan.packed_col_bitmap_meta, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Fp16:
            // Per-nonzero 16-bit fp16 (2.0 B/nnz), replacing the per-cell tile512.
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_fp16_nnz_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.values16, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_fp16_nnz_device(
                    device, plan.packed_col_bitmap_meta, plan.values16, d_x, d_y,
                    iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Bf16:
            // Per-nonzero 16-bit bf16 (2.0 B/nnz), replacing the per-cell tile512.
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bf16_nnz_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.values16, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_bf16_nnz_device(
                    device, plan.packed_col_bitmap_meta, plan.values16, d_x, d_y,
                    iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Bfp8:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bfp8_blkscale_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.int8_values, plan.block_scale_bf16, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_bfp8_blkscale_device(
                    device, plan.packed_col_bitmap_meta, plan.int8_values,
                    plan.block_scale_bf16, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::Bfp8Outlier:
            if (use_load_balance) {
                // Pass 1 (bulk BFP8) + pass 2 (outlier correction), timed together.
                spmv_bitbsr_gpu_8x4_bfp8_outlier_flag_load_balance_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.int8_values, plan.block_scale_bf16,
                    plan.outlier_rows, plan.outlier_cols, plan.outlier_vals,
                    d_x, d_y, iter_ms);
            } else {
                throw std::runtime_error(
                    "bfp8_outlier value codec requires the col_bitmap64_flag_lb layout");
            }
            return;
        case BitBsrValueCodecKind8x4::Bfp4:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bfp4_blkscale_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.packed_int4, plan.block_scale_bf16, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_bfp4_blkscale_device(
                    device, plan.packed_col_bitmap_meta, plan.packed_int4,
                    plan.block_scale_bf16, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::K256U8:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_k256_tile256_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_k256_tile256_device(
                    device, plan.packed_col_bitmap_meta, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            }
            return;
        case BitBsrValueCodecKind8x4::GlobalLogKmeans16:
        case BitBsrValueCodecKind8x4::GlobalLogKmeans32:
        case BitBsrValueCodecKind8x4::GlobalLogKmeans64:
        case BitBsrValueCodecKind8x4::GlobalLogKmeans128:
            if (use_load_balance) {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_logkmeans_tile256_device(
                    device, plan.packed_col_bitmap_meta, plan.load_balance,
                    plan.tile_code_words, plan.value_codebook, d_x, d_y, iter_ms);
            } else {
                spmv_bitbsr_gpu_8x4_col_bitmap_packed_logkmeans_tile256_device(
                    device, plan.packed_col_bitmap_meta, plan.tile_code_words,
                    plan.value_codebook, d_x, d_y, iter_ms);
            }
            return;
    }
    throw std::runtime_error("internal error: unsupported value codec dispatch");
}

}  // namespace gpu_detail

template <typename T>
void run_bitbsr_spmv_once(const DeviceBitBsrMatrix<T>& device,
                          const BitBsrSpmvPlan8x4<T>& plan,
                          const thrust::device_vector<T>& d_x,
    thrust::device_vector<T>& d_y,
    float* iter_ms = nullptr) {
    switch (plan.layout) {
        case BitBsrSpmvLayoutKind8x4::ColBitmap64:
            gpu_detail::run_bitbsr_spmv_8x4_col_bitmap64_once(
                device, plan, false, d_x, d_y, iter_ms);
            return;
        case BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb:
            gpu_detail::run_bitbsr_spmv_8x4_col_bitmap64_once(
                device, plan, true, d_x, d_y, iter_ms);
            return;
        case BitBsrSpmvLayoutKind8x4::CusparseCsr:
            break;
    }
    throw std::runtime_error("internal error: unsupported BitBSR layout dispatch");
}

}  // namespace structural
