#pragma once

#include "structural/bitbsr.hpp"
#include "structural/operators/bitbsr_spmv_8x4_common.cuh"
#include "structural/value_compression.hpp"

#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace structural {

struct DeviceExactValueHistogram {
    Offset nnz = 0;
    Offset num_blocks = 0;
    std::size_t raw_payload_bytes = 0;
    int unique_count = 0;
    bool unique_overflow = false;
    bool all_one = false;
    bool const_c = false;
    bool sign2 = false;
    bool sign2_scale = false;
    std::size_t const_c_payload_bytes = 0;
    std::size_t sign2_payload_bytes = 0;
    std::size_t sign2_scale_payload_bytes = 0;
    std::size_t dict2_payload_bytes = 0;
    std::size_t dict4_payload_bytes = 0;
    std::size_t dict16_payload_bytes = 0;
    std::size_t dict256_u8_payload_bytes = 0;
    std::size_t dict65536_u16_payload_bytes = 0;
};

template <typename T>
DeviceExactValueHistogram build_device_exact_value_histogram(
    const DeviceBitBsrMatrix<T>& bitbsr,
    float* device_ms = nullptr) {
    bitbsr.validate_shape();

    DeviceExactValueHistogram histogram;
    histogram.nnz = bitbsr.nnz;
    histogram.num_blocks = bitbsr.num_blocks;
    histogram.raw_payload_bytes = static_cast<std::size_t>(bitbsr.nnz) * sizeof(T);
    histogram.const_c_payload_bytes = sizeof(T);
    histogram.sign2_payload_bytes =
        static_cast<std::size_t>(bitbsr.num_blocks) * sizeof(std::uint32_t);
    histogram.sign2_scale_payload_bytes = histogram.sign2_payload_bytes + sizeof(T);

    if (bitbsr.nnz == 0) {
        histogram.const_c_payload_bytes = 0;
        histogram.sign2_scale_payload_bytes = 0;
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return histogram;
    }

    thrust::device_vector<T> sorted_values;
    {
        gpu_detail::CudaEventTimer timer(device_ms);
        sorted_values = bitbsr.values;
        thrust::sort(sorted_values.begin(), sorted_values.end());
        const auto unique_end = thrust::unique(sorted_values.begin(), sorted_values.end());
        histogram.unique_count = static_cast<int>(unique_end - sorted_values.begin());
        timer.stop();
    }

    histogram.unique_overflow =
        histogram.unique_count > gpu_detail::kSpmvDict65536ValueCodebookSize;

    if (histogram.unique_count == 1) {
        const T only = sorted_values[0];
        histogram.const_c = true;
        histogram.all_one = only == T{1};
    } else if (histogram.unique_count == 2) {
        const T first = sorted_values[0];
        const T second = sorted_values[1];
        histogram.sign2 = first == T{-1} && second == T{1};
        histogram.sign2_scale = first != T{0} && first == -second;
    }

    const auto dict_payload_bytes = [&](int words_per_tile) {
        return static_cast<std::size_t>(bitbsr.num_blocks) *
                   static_cast<std::size_t>(words_per_tile) * sizeof(std::uint32_t) +
               static_cast<std::size_t>(histogram.unique_count) * sizeof(T);
    };
    histogram.dict2_payload_bytes =
        dict_payload_bytes(gpu_detail::kSpmvCode1Tile32Words);
    histogram.dict4_payload_bytes =
        dict_payload_bytes(gpu_detail::kSpmvCode2Tile64Words);
    histogram.dict16_payload_bytes =
        dict_payload_bytes(gpu_detail::kSpmvCode4Tile128Words);
    histogram.dict256_u8_payload_bytes =
        dict_payload_bytes(gpu_detail::kSpmvCode8Tile256Words);
    histogram.dict65536_u16_payload_bytes =
        dict_payload_bytes(gpu_detail::kSpmvCode16Tile512Words);

    return histogram;
}

template <typename T>
T build_const_c_value(const BitBsrMatrix<T>& bitbsr) {
    bitbsr.validate();
    if (bitbsr.values.empty()) {
        throw std::runtime_error("const_c value codec requires at least one value");
    }
    const T value = bitbsr.values.front();
    for (const T candidate : bitbsr.values) {
        if (candidate != value) {
            throw std::runtime_error("const_c value codec requires every stored value to equal C");
        }
    }
    return value;
}

template <typename T>
void validate_implicit_one_values(const std::vector<T>& values) {
    for (const T value : values) {
        if (value != T{1}) {
            throw std::runtime_error("all_one value codec requires every stored value to be 1");
        }
    }
}

template <typename T>
struct Sign2ScaleTile32ValueCodec {
    T scale = T{1};
    std::vector<std::uint32_t> sign_masks;

    std::size_t payload_bytes() const {
        return sign_masks.size() * sizeof(std::uint32_t) + sizeof(T);
    }
};

template <typename T>
std::vector<std::uint32_t> build_sign2_tile32_masks(
    const BitBsrMatrix<T>& bitbsr) {
    bitbsr.validate();
    std::vector<std::uint32_t> masks(static_cast<std::size_t>(bitbsr.num_blocks()), 0);
    for (Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const auto bitmap = static_cast<std::uint32_t>(bitbsr.bitmap_words[block]);
        const auto value_base = bitbsr.block_val_ptr[block];
        for (int bit = 0; bit < 32; ++bit) {
            if ((bitmap & (std::uint32_t{1} << bit)) == 0U) {
                continue;
            }
            const std::uint32_t before = bitmap & ((std::uint32_t{1} << bit) - 1U);
            const T value = bitbsr.values[value_base + popcount_word(before)];
            if (value == T{1}) {
                masks[static_cast<std::size_t>(block)] |= std::uint32_t{1} << bit;
            } else if (value != T{-1}) {
                throw std::runtime_error("sign2 value codec requires values in {-1, +1}");
            }
        }
    }
    return masks;
}

template <typename T>
Sign2ScaleTile32ValueCodec<T> build_sign2_scale_tile32_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    bitbsr.validate();
    const auto unique_values = sorted_unique_values(bitbsr.values, 2);
    if (unique_values.size() != 2) {
        throw std::runtime_error("sign2_scale value codec requires exactly two values");
    }
    const T negative = unique_values[0];
    const T positive = unique_values[1];
    if (negative == T{0} || negative != -positive) {
        throw std::runtime_error("sign2_scale value codec requires values in {-C, +C}");
    }

    Sign2ScaleTile32ValueCodec<T> codec;
    codec.scale = positive;
    codec.sign_masks.assign(static_cast<std::size_t>(bitbsr.num_blocks()), 0);
    for (Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const auto bitmap = static_cast<std::uint32_t>(bitbsr.bitmap_words[block]);
        const auto value_base = bitbsr.block_val_ptr[block];
        for (int bit = 0; bit < 32; ++bit) {
            if ((bitmap & (std::uint32_t{1} << bit)) == 0U) {
                continue;
            }
            const std::uint32_t before = bitmap & ((std::uint32_t{1} << bit) - 1U);
            const T value = bitbsr.values[value_base + popcount_word(before)];
            if (value == positive) {
                codec.sign_masks[static_cast<std::size_t>(block)] |= std::uint32_t{1} << bit;
            } else if (value != negative) {
                throw std::runtime_error("sign2_scale value codec requires values in {-C, +C}");
            }
        }
    }
    return codec;
}

template <int BitsPerCode, typename T>
std::vector<std::uint32_t> build_tile_code_words_from_value_ids(
    const BitBsrMatrix<T>& bitbsr,
    const std::vector<std::uint32_t>& value_ids) {
    static_assert(BitsPerCode == 1 || BitsPerCode == 2 || BitsPerCode == 4 ||
                      BitsPerCode == 8 || BitsPerCode == 16,
                  "tile codec supports 1, 2, 4, 8, or 16 bits per 8x4 cell");
    bitbsr.validate();
    if (value_ids.size() != bitbsr.values.size()) {
        throw std::runtime_error("tile value codec id count mismatch");
    }

    constexpr int words_per_tile = gpu_detail::TileCodeTraits8x4<BitsPerCode>::words_per_tile;
    constexpr int codes_per_word = 32 / BitsPerCode;
    std::vector<std::uint32_t> words(
        static_cast<std::size_t>(bitbsr.num_blocks()) * words_per_tile, 0);
    for (Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const auto bitmap = static_cast<std::uint32_t>(bitbsr.bitmap_words[block]);
        const auto value_base = bitbsr.block_val_ptr[block];
        for (int bit = 0; bit < 32; ++bit) {
            if ((bitmap & (std::uint32_t{1} << bit)) == 0U) {
                continue;
            }
            const std::uint32_t before = bitmap & ((std::uint32_t{1} << bit) - 1U);
            const auto value_index =
                static_cast<std::size_t>(value_base + popcount_word(before));
            const auto id = value_ids[value_index];
            if (id >= (std::uint32_t{1} << BitsPerCode)) {
                throw std::runtime_error("tile value codec id does not fit in tile field");
            }
            const int word = bit / codes_per_word;
            const int shift = (bit % codes_per_word) * BitsPerCode;
            words[static_cast<std::size_t>(block) * words_per_tile + word] |= id << shift;
        }
    }
    return words;
}

template <int BitsPerCode, typename T>
struct TileCodeValueCodec {
    std::vector<T> codebook;
    std::vector<std::uint32_t> value_ids;
    std::vector<std::uint32_t> tile_code_words;
    bool exact = true;

    std::size_t payload_bytes() const {
        return tile_code_words.size() * sizeof(std::uint32_t) +
               codebook.size() * sizeof(T);
    }
};

struct Round16Tile512ValueCodec {
    std::vector<std::uint32_t> tile_code_words;

    std::size_t payload_bytes() const {
        return tile_code_words.size() * sizeof(std::uint32_t);
    }
};

// Per-nonzero 16-bit value codec (bf16 or fp16, FP32 compute, NO block scale).
// values16: one 16-bit value per NONZERO, in the same block-grouped bit-rank
// (linear nnz) order as bitbsr.values / the bfp8 int8_values. payload = nnz*2,
// i.e. a flat 2.0 bytes/nnz regardless of block fill (unlike the wasteful
// per-cell tile512 packing which spends 64 bytes/block).
struct Round16NnzValueCodec {
    std::vector<std::uint16_t> values16;

    std::size_t payload_bytes() const {
        return values16.size() * sizeof(std::uint16_t);
    }
};

// BFP8 = int8 mantissa + per-8x4-block bf16 scale.
// int8_values: one signed char per nonzero, in the same block-grouped bit-rank
// order as bitbsr.values. block_scale_bf16: one bf16-bits scale per block.
struct Bfp8BlkScaleValueCodec {
    std::vector<signed char> int8_values;
    std::vector<std::uint16_t> block_scale_bf16;

    std::size_t payload_bytes() const {
        return int8_values.size() * sizeof(signed char) +
               block_scale_bf16.size() * sizeof(std::uint16_t);
    }
};

template <typename T>
Bfp8BlkScaleValueCodec build_bfp8_blkscale_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    bitbsr.validate();
    Bfp8BlkScaleValueCodec codec;
    codec.int8_values.assign(static_cast<std::size_t>(bitbsr.nnz()), 0);
    codec.block_scale_bf16.assign(static_cast<std::size_t>(bitbsr.num_blocks()), 0);
    for (Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const Offset begin = bitbsr.block_val_ptr[block];
        const Offset end = bitbsr.block_val_ptr[block + 1];
        double max_abs = 0.0;
        for (Offset k = begin; k < end; ++k) {
            max_abs = std::max(max_abs, std::abs(static_cast<double>(bitbsr.values[k])));
        }
        float scale = static_cast<float>(max_abs / 127.0);
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        codec.block_scale_bf16[static_cast<std::size_t>(block)] =
            float_to_bf16_bits(scale);
        const float decoded_scale = bf16_bits_to_float(
            codec.block_scale_bf16[static_cast<std::size_t>(block)]);
        for (Offset k = begin; k < end; ++k) {
            const long q = std::lround(static_cast<double>(bitbsr.values[k]) /
                                       static_cast<double>(decoded_scale));
            codec.int8_values[static_cast<std::size_t>(k)] = static_cast<signed char>(
                std::max(-127L, std::min(127L, q)));
        }
    }
    return codec;
}

// BFP8 + outlier two-pass codec.
// Identical to BFP8 (int8 mantissa + per-8x4-block bf16 scale) EXCEPT:
//   (a) the GLOBAL top-P% of nonzeros by |value| are marked as outliers;
//   (b) each block's bf16 scale = max|value| over its NON-outlier nonzeros, so
//       the bulk quantizes well and is not dominated by the outlier;
//   (c) the int8 mantissa slot for each outlier nonzero is 0 (contributes 0 in
//       pass 1) -- the bulk kernel runs unchanged on int8_values/block_scale_bf16.
// The outliers are kept exactly in a side list (absolute row, absolute col, exact
// value) for the pass-2 correction kernel, which does
//   y[row_i] += value_i * x[col_i].
// row/col are recovered from the host BitBSR position (block row*8 + local_row,
// block_col_idx*4 + local_col) via the bitmap bit, mirroring the device kernels.
template <typename T>
struct Bfp8OutlierBlkScaleValueCodec {
    Offset nnz = 0;
    Offset num_blocks = 0;
    Offset num_outliers = 0;
    std::vector<signed char> int8_values;        // outlier slots are 0
    std::vector<std::uint16_t> block_scale_bf16; // non-outlier per-block max
    std::vector<std::int32_t> outlier_rows;
    std::vector<std::int32_t> outlier_cols;
    std::vector<T> outlier_vals;

    // Full payload the operator actually stores: int8 mantissas + per-block bf16
    // scale + the explicit side list (rows, cols, values).
    std::size_t payload_bytes() const {
        return int8_values.size() * sizeof(signed char) +
               block_scale_bf16.size() * sizeof(std::uint16_t) +
               static_cast<std::size_t>(num_outliers) *
                   (sizeof(std::int32_t) + sizeof(std::int32_t) + sizeof(T));
    }

    // Value-only accounting: positions come from the bitmap, so only
    // the exact outlier VALUES count as extra storage over plain BFP8.
    std::size_t value_only_payload_bytes() const {
        return int8_values.size() * sizeof(signed char) +
               block_scale_bf16.size() * sizeof(std::uint16_t) +
               static_cast<std::size_t>(num_outliers) * sizeof(T);
    }
};

template <typename T>
Bfp8OutlierBlkScaleValueCodec<T> build_bfp8_outlier_blkscale_value_codec(
    const BitBsrMatrix<T>& bitbsr,
    double outlier_fraction = 0.005) {
    bitbsr.validate();
    Bfp8OutlierBlkScaleValueCodec<T> codec;
    codec.nnz = bitbsr.nnz();
    codec.num_blocks = bitbsr.num_blocks();
    codec.int8_values.assign(static_cast<std::size_t>(bitbsr.nnz()), 0);
    codec.block_scale_bf16.assign(static_cast<std::size_t>(bitbsr.num_blocks()), 0);

    const Offset nnz = bitbsr.nnz();

    // GLOBAL top-P% by |value| (mirror bench_outlier.cu: outlier if |v| > thr,
    // thr = the (1-P)-quantile via nth_element). With nth_element the partition
    // point is fuzzy at ties, which is fine -- it picks ~P% of the largest nnz.
    std::vector<char> is_outlier(static_cast<std::size_t>(nnz), 0);
    if (outlier_fraction > 0.0 && nnz > 0) {
        std::vector<double> abs_values(static_cast<std::size_t>(nnz));
        for (Offset k = 0; k < nnz; ++k) {
            abs_values[static_cast<std::size_t>(k)] =
                std::abs(static_cast<double>(bitbsr.values[k]));
        }
        std::vector<double> scratch(abs_values);
        std::size_t idx =
            static_cast<std::size_t>((1.0 - outlier_fraction) * static_cast<double>(nnz));
        if (idx >= scratch.size()) {
            idx = scratch.size() - 1;
        }
        std::nth_element(scratch.begin(), scratch.begin() + idx, scratch.end());
        const double threshold = scratch[idx];
        for (Offset k = 0; k < nnz; ++k) {
            if (abs_values[static_cast<std::size_t>(k)] > threshold) {
                is_outlier[static_cast<std::size_t>(k)] = 1;
            }
        }
    }

    // Per-block scale from the NON-outlier max; quantize non-outliers, zero the
    // outlier mantissa slots.
    for (Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const Offset begin = bitbsr.block_val_ptr[block];
        const Offset end = bitbsr.block_val_ptr[block + 1];
        double max_abs = 0.0;
        for (Offset k = begin; k < end; ++k) {
            if (is_outlier[static_cast<std::size_t>(k)]) {
                continue;
            }
            max_abs = std::max(max_abs, std::abs(static_cast<double>(bitbsr.values[k])));
        }
        float scale = static_cast<float>(max_abs / 127.0);
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        codec.block_scale_bf16[static_cast<std::size_t>(block)] =
            float_to_bf16_bits(scale);
        const float decoded_scale = bf16_bits_to_float(
            codec.block_scale_bf16[static_cast<std::size_t>(block)]);
        for (Offset k = begin; k < end; ++k) {
            if (is_outlier[static_cast<std::size_t>(k)]) {
                codec.int8_values[static_cast<std::size_t>(k)] = 0;
                continue;
            }
            const long q = std::lround(static_cast<double>(bitbsr.values[k]) /
                                       static_cast<double>(decoded_scale));
            codec.int8_values[static_cast<std::size_t>(k)] = static_cast<signed char>(
                std::max(-127L, std::min(127L, q)));
        }
    }

    // Build the outlier side list. Recover absolute row/col from the BitBSR
    // position: walk per block-row so we know br, then per set bit decode
    // local_row/local_col. Value order within a block is bit-rank, matching the
    // mantissa indexing.
    for (Index br = 0; br < bitbsr.num_block_rows; ++br) {
        const Offset block_begin = bitbsr.block_row_ptr[br];
        const Offset block_end = bitbsr.block_row_ptr[br + 1];
        for (Offset block = block_begin; block < block_end; ++block) {
            const auto bitmap = static_cast<std::uint32_t>(bitbsr.bitmap_words[block]);
            const Offset value_base = bitbsr.block_val_ptr[block];
            for (int bit = 0; bit < 32; ++bit) {
                if ((bitmap & (std::uint32_t{1} << bit)) == 0U) {
                    continue;
                }
                const std::uint32_t before = bitmap & ((std::uint32_t{1} << bit) - 1U);
                const Offset k = value_base + popcount_word(before);
                if (!is_outlier[static_cast<std::size_t>(k)]) {
                    continue;
                }
                const int local_row = bit / bitbsr.block_cols;
                const int local_col = bit % bitbsr.block_cols;
                const Index row = br * bitbsr.block_rows + local_row;
                const Index col = bitbsr.block_col_idx[block] * bitbsr.block_cols + local_col;
                codec.outlier_rows.push_back(static_cast<std::int32_t>(row));
                codec.outlier_cols.push_back(static_cast<std::int32_t>(col));
                codec.outlier_vals.push_back(bitbsr.values[static_cast<std::size_t>(k)]);
            }
        }
    }
    codec.num_outliers = static_cast<Offset>(codec.outlier_vals.size());
    return codec;
}

// BFP4 = int4 mantissa + per-8x4-block bf16 scale.
// packed_int4: two signed 4-bit (two's complement, range [-7, 7]) mantissas per
// byte, in the same block-grouped bit-rank (linear nnz) order as bitbsr.values;
// value k occupies nibble (k & 1) of byte k/2. block_scale_bf16: one bf16-bits
// scale per block (scale = max|v| / 7).
struct Bfp4BlkScaleValueCodec {
    Offset nnz = 0;
    std::vector<std::uint8_t> packed_int4;
    std::vector<std::uint16_t> block_scale_bf16;

    std::size_t payload_bytes() const {
        return packed_int4.size() * sizeof(std::uint8_t) +
               block_scale_bf16.size() * sizeof(std::uint16_t);
    }
};

template <typename T>
Bfp4BlkScaleValueCodec build_bfp4_blkscale_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    bitbsr.validate();
    Bfp4BlkScaleValueCodec codec;
    codec.nnz = bitbsr.nnz();
    codec.packed_int4.assign(
        static_cast<std::size_t>((bitbsr.nnz() + 1) / 2), 0);
    codec.block_scale_bf16.assign(static_cast<std::size_t>(bitbsr.num_blocks()), 0);
    for (Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const Offset begin = bitbsr.block_val_ptr[block];
        const Offset end = bitbsr.block_val_ptr[block + 1];
        double max_abs = 0.0;
        for (Offset k = begin; k < end; ++k) {
            max_abs = std::max(max_abs, std::abs(static_cast<double>(bitbsr.values[k])));
        }
        float scale = static_cast<float>(max_abs / 7.0);
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        codec.block_scale_bf16[static_cast<std::size_t>(block)] =
            float_to_bf16_bits(scale);
        const float decoded_scale = bf16_bits_to_float(
            codec.block_scale_bf16[static_cast<std::size_t>(block)]);
        for (Offset k = begin; k < end; ++k) {
            const long q = std::max(
                -7L, std::min(7L, std::lround(static_cast<double>(bitbsr.values[k]) /
                                              static_cast<double>(decoded_scale))));
            const auto nibble = static_cast<std::uint8_t>(q & 0xF);
            std::uint8_t& byte =
                codec.packed_int4[static_cast<std::size_t>(k >> 1)];
            if ((k & 1) == 0) {
                byte = static_cast<std::uint8_t>((byte & 0xF0) | nibble);
            } else {
                byte = static_cast<std::uint8_t>((byte & 0x0F) |
                                                 (nibble << 4));
            }
        }
    }
    return codec;
}

template <typename T, typename EncodeBitsFn>
Round16Tile512ValueCodec build_round16_tile512_value_codec(
    const BitBsrMatrix<T>& bitbsr,
    EncodeBitsFn encode_bits) {
    bitbsr.validate();
    Round16Tile512ValueCodec codec;
    codec.tile_code_words.assign(
        static_cast<std::size_t>(bitbsr.num_blocks()) *
            gpu_detail::kSpmvRound16Tile512Words,
        0);
    for (Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        const auto bitmap = static_cast<std::uint32_t>(bitbsr.bitmap_words[block]);
        const auto value_base = bitbsr.block_val_ptr[block];
        for (int bit = 0; bit < 32; ++bit) {
            if ((bitmap & (std::uint32_t{1} << bit)) == 0U) {
                continue;
            }
            const std::uint32_t before = bitmap & ((std::uint32_t{1} << bit) - 1U);
            const auto value_index =
                static_cast<std::size_t>(value_base + popcount_word(before));
            const std::uint32_t packed_bits =
                static_cast<std::uint32_t>(encode_bits(bitbsr.values[value_index]));
            const int word = bit / 2;
            const int shift = (bit % 2) * 16;
            codec.tile_code_words[static_cast<std::size_t>(block) *
                                      gpu_detail::kSpmvRound16Tile512Words +
                                  word] |= packed_bits << shift;
        }
    }
    return codec;
}

// Per-nonzero 16-bit builder (bf16/fp16). Stores one 16-bit value per NONZERO in
// the SAME block-grouped bit-rank order the bfp8 builder uses -- which is just
// the linear bitbsr.values order, since value index k already advances in
// bit-rank order within each block. encode_bits(v) -> bf16/fp16 bits.
template <typename T, typename EncodeBitsFn>
Round16NnzValueCodec build_round16_nnz_value_codec(
    const BitBsrMatrix<T>& bitbsr,
    EncodeBitsFn encode_bits) {
    bitbsr.validate();
    Round16NnzValueCodec codec;
    codec.values16.assign(static_cast<std::size_t>(bitbsr.nnz()), 0);
    for (Offset k = 0; k < bitbsr.nnz(); ++k) {
        codec.values16[static_cast<std::size_t>(k)] =
            static_cast<std::uint16_t>(encode_bits(bitbsr.values[static_cast<std::size_t>(k)]));
    }
    return codec;
}

template <int BitsPerCode, typename T>
TileCodeValueCodec<BitsPerCode, T> build_exact_tile_code_value_codec(
    const BitBsrMatrix<T>& bitbsr,
    int max_codebook_size,
    const char* name) {
    TileCodeValueCodec<BitsPerCode, T> codec;
    codec.codebook = sorted_unique_values(bitbsr.values, max_codebook_size);
    if (static_cast<int>(codec.codebook.size()) > max_codebook_size) {
        throw std::runtime_error(std::string(name) +
                                 " value codec requires fewer distinct values");
    }
    codec.value_ids.resize(bitbsr.values.size());
    std::unordered_map<T, std::uint32_t> value_to_id;
    value_to_id.reserve(codec.codebook.size());
    for (std::size_t i = 0; i < codec.codebook.size(); ++i) {
        value_to_id.emplace(codec.codebook[i], static_cast<std::uint32_t>(i));
    }
    for (std::size_t i = 0; i < bitbsr.values.size(); ++i) {
        const auto it = value_to_id.find(bitbsr.values[i]);
        if (it == value_to_id.end()) {
            throw std::runtime_error(std::string(name) +
                                     " exact value was not found in codebook");
        }
        codec.value_ids[i] = it->second;
    }
    codec.tile_code_words =
        build_tile_code_words_from_value_ids<BitsPerCode>(bitbsr, codec.value_ids);
    return codec;
}

template <typename T>
struct Unique16Tile128ValueCodec : TileCodeValueCodec<4, T> {};

template <typename T>
struct Unique4Tile64ValueCodec : TileCodeValueCodec<2, T> {};

template <typename T>
struct Dict2Tile32ValueCodec : TileCodeValueCodec<1, T> {};

template <typename T>
struct Dict256Tile256ValueCodec : TileCodeValueCodec<8, T> {};

template <typename T>
struct Dict65536Tile512ValueCodec : TileCodeValueCodec<16, T> {};

template <typename T>
Dict2Tile32ValueCodec<T> build_dict2_tile32_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    const auto base = build_exact_tile_code_value_codec<1>(
        bitbsr, 2, "dict2_tile32");
    Dict2Tile32ValueCodec<T> out;
    out.codebook = base.codebook;
    out.value_ids = base.value_ids;
    out.tile_code_words = base.tile_code_words;
    out.exact = base.exact;
    return out;
}

template <typename T>
Unique4Tile64ValueCodec<T> build_unique4_tile64_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    const auto base = build_exact_tile_code_value_codec<2>(
        bitbsr, 4, "unique4_tile64");
    Unique4Tile64ValueCodec<T> out;
    out.codebook = base.codebook;
    out.value_ids = base.value_ids;
    out.tile_code_words = base.tile_code_words;
    out.exact = base.exact;
    return out;
}

template <typename T>
Unique16Tile128ValueCodec<T> build_unique16_tile128_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    const auto base = build_exact_tile_code_value_codec<4>(
        bitbsr, 16, "unique16_tile128");
    Unique16Tile128ValueCodec<T> out;
    out.codebook = base.codebook;
    out.value_ids = base.value_ids;
    out.tile_code_words = base.tile_code_words;
    out.exact = base.exact;
    return out;
}

template <typename T>
Dict256Tile256ValueCodec<T> build_dict256_tile256_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    const auto base = build_exact_tile_code_value_codec<8>(
        bitbsr, 256, "dict256_tile256");
    Dict256Tile256ValueCodec<T> out;
    out.codebook = base.codebook;
    out.value_ids = base.value_ids;
    out.tile_code_words = base.tile_code_words;
    out.exact = base.exact;
    return out;
}

template <typename T>
Dict65536Tile512ValueCodec<T> build_dict65536_tile512_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    const auto base = build_exact_tile_code_value_codec<16>(
        bitbsr, 65536, "dict65536_tile512");
    Dict65536Tile512ValueCodec<T> out;
    out.codebook = base.codebook;
    out.value_ids = base.value_ids;
    out.tile_code_words = base.tile_code_words;
    out.exact = base.exact;
    return out;
}

template <typename T>
struct K256Tile256BuildResult {
    TileCodeValueCodec<8, T> tile;
    int used_codebook_size = 0;
};

template <typename T>
struct LogKmeansTile256BuildResult {
    TileCodeValueCodec<8, T> tile;
    int used_codebook_size = 0;
};

template <typename T>
K256Tile256BuildResult<T> build_k256_tile256_build_result(
    const BitBsrMatrix<T>& bitbsr) {
    K256Tile256BuildResult<T> out;
    const auto codec = build_k256_uint8_value_codebook(bitbsr.values);
    out.used_codebook_size = codec.codebook_size;
    out.tile.exact = codec.exact;
    out.tile.value_ids.assign(codec.ids.begin(), codec.ids.end());
    out.tile.codebook.assign(codec.codebook.begin(), codec.codebook.end());
    out.tile.tile_code_words =
        build_tile_code_words_from_value_ids<8>(bitbsr, out.tile.value_ids);
    return out;
}

template <typename T>
LogKmeansTile256BuildResult<T> build_logkmeans_tile256_build_result(
    const BitBsrMatrix<T>& bitbsr,
    int requested_k,
    int histogram_bins = 4096,
    int iterations = 16) {
    bitbsr.validate();
    LogKmeansTile256BuildResult<T> out;
    if (requested_k <= 0 || requested_k > 128) {
        throw std::runtime_error("global_logkmeans value codec requires k in [1, 128]");
    }
    std::vector<double> logs;
    logs.reserve(bitbsr.values.size());
    for (const T raw : bitbsr.values) {
        const double value = static_cast<double>(raw);
        if (!std::isfinite(value)) {
            throw std::runtime_error("global_logkmeans value codec requires finite values");
        }
        if (value == 0.0) {
            throw std::runtime_error("global_logkmeans value codec requires nonzero BitBSR values");
        }
        logs.push_back(std::log2(std::abs(value)));
    }

    const auto centers =
        make_log_kmeans_centers(std::move(logs), requested_k, histogram_bins, iterations);
    const int codebook_size = static_cast<int>(centers.size()) * 2;
    if (codebook_size > gpu_detail::kSpmvK256ValueCodebookSize) {
        throw std::runtime_error(
            "global_logkmeans value codec exceeded the uint8 codebook size");
    }

    out.tile.codebook.reserve(static_cast<std::size_t>(codebook_size));
    for (const double center : centers) {
        const T magnitude = static_cast<T>(std::exp2(center));
        out.tile.codebook.push_back(-magnitude);
        out.tile.codebook.push_back(magnitude);
    }
    out.used_codebook_size = static_cast<int>(out.tile.codebook.size());
    out.tile.exact = false;
    out.tile.value_ids.resize(bitbsr.values.size(), 0);
    for (std::size_t i = 0; i < bitbsr.values.size(); ++i) {
        const double value = static_cast<double>(bitbsr.values[i]);
        const int center = nearest_sorted_center(centers, std::log2(std::abs(value)));
        const int sign_offset = value < 0.0 ? 0 : 1;
        out.tile.value_ids[i] = static_cast<std::uint32_t>(center * 2 + sign_offset);
    }
    out.tile.tile_code_words =
        build_tile_code_words_from_value_ids<8>(bitbsr, out.tile.value_ids);
    return out;
}

template <typename T>
void validate_tile_shape_8x4(const BitBsrMatrix<T>& bitbsr) {
    if (bitbsr.block_rows != 8 || bitbsr.block_cols != 4 || bitbsr.words_per_block != 1) {
        throw std::runtime_error("tile value codecs require 8x4 BitBSR blocks");
    }
}

template <typename T>
std::vector<std::uint32_t> build_checked_sign2_tile32_masks(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_sign2_tile32_masks(bitbsr);
}

template <typename T>
T build_checked_const_c_value(const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_const_c_value(bitbsr);
}

template <typename T>
Sign2ScaleTile32ValueCodec<T> build_checked_sign2_scale_tile32_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_sign2_scale_tile32_value_codec(bitbsr);
}

template <typename T>
Dict2Tile32ValueCodec<T> build_checked_dict2_tile32_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_dict2_tile32_value_codec(bitbsr);
}

template <typename T>
Unique4Tile64ValueCodec<T> build_checked_unique4_tile64_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_unique4_tile64_value_codec(bitbsr);
}

template <typename T>
Unique16Tile128ValueCodec<T> build_checked_unique16_tile128_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_unique16_tile128_value_codec(bitbsr);
}

template <typename T>
Dict256Tile256ValueCodec<T> build_checked_dict256_tile256_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_dict256_tile256_value_codec(bitbsr);
}

template <typename T>
Dict65536Tile512ValueCodec<T> build_checked_dict65536_tile512_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_dict65536_tile512_value_codec(bitbsr);
}

template <typename T>
K256Tile256BuildResult<T> build_checked_k256_tile256_build_result(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_k256_tile256_build_result(bitbsr);
}

template <typename T>
Round16Tile512ValueCodec build_checked_fp16_tile512_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_round16_tile512_value_codec(
        bitbsr, [](T value) { return float_to_fp16_bits(static_cast<float>(value)); });
}

template <typename T>
Round16Tile512ValueCodec build_checked_bf16_tile512_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_round16_tile512_value_codec(
        bitbsr, [](T value) { return float_to_bf16_bits(static_cast<float>(value)); });
}

template <typename T>
Round16NnzValueCodec build_checked_fp16_nnz_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_round16_nnz_value_codec(
        bitbsr, [](T value) { return float_to_fp16_bits(static_cast<float>(value)); });
}

template <typename T>
Round16NnzValueCodec build_checked_bf16_nnz_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_round16_nnz_value_codec(
        bitbsr, [](T value) { return float_to_bf16_bits(static_cast<float>(value)); });
}

template <typename T>
Bfp8BlkScaleValueCodec build_checked_bfp8_blkscale_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_bfp8_blkscale_value_codec(bitbsr);
}

template <typename T>
Bfp8OutlierBlkScaleValueCodec<T> build_checked_bfp8_outlier_blkscale_value_codec(
    const BitBsrMatrix<T>& bitbsr,
    double outlier_fraction = 0.005) {
    validate_tile_shape_8x4(bitbsr);
    return build_bfp8_outlier_blkscale_value_codec(bitbsr, outlier_fraction);
}

template <typename T>
Bfp4BlkScaleValueCodec build_checked_bfp4_blkscale_value_codec(
    const BitBsrMatrix<T>& bitbsr) {
    validate_tile_shape_8x4(bitbsr);
    return build_bfp4_blkscale_value_codec(bitbsr);
}

template <typename T>
LogKmeansTile256BuildResult<T> build_checked_logkmeans_tile256_build_result(
    const BitBsrMatrix<T>& bitbsr,
    int requested_k) {
    validate_tile_shape_8x4(bitbsr);
    return build_logkmeans_tile256_build_result(bitbsr, requested_k);
}

}  // namespace structural
