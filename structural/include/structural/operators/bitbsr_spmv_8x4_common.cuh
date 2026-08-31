#pragma once

#include "structural/gpu_convert.cuh"

#include <cuda_fp16.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace structural {

struct BitBsrWorkItem8x4 {
    std::uint32_t br = 0;
    std::uint32_t block_begin = 0;
    std::uint32_t block_count = 0;
    std::uint32_t value_begin = 0;
};

static_assert(sizeof(BitBsrWorkItem8x4) == 16, "8x4 work item must stay 16 bytes");

struct BitBsrLoadBalanceLayout8x4 {
    int chunk_blocks = 0;
    int split_threshold_blocks = 0;
    thrust::device_vector<Offset> work_item_row_ptr;
    thrust::device_vector<BitBsrWorkItem8x4> work_items;
};

namespace gpu_detail {

constexpr int kSpmv8x4BlockRows = 8;
constexpr int kSpmv8x4BlockCols = 4;
#ifndef COVE_TPB
#define COVE_TPB 128
#endif
constexpr int kSpmv8x4ThreadsPerBlock = COVE_TPB;
constexpr int kSpmv8x4WarpsPerBlock = kSpmv8x4ThreadsPerBlock / 32;
constexpr int kSpmv8x4PackedBitmapShift = 32;
constexpr int kSpmv8x4PackedAbsColShift = 0;
constexpr int kSpmv8x4PackedFlagShift = 31;
constexpr int kSpmvK256ValueCodebookSize = 256;
constexpr int kSpmvDict2ValueCodebookSize = 2;
constexpr int kSpmvUnique4ValueCodebookSize = 4;
constexpr int kSpmvUnique16ValueCodebookSize = 16;
constexpr int kSpmvDict256ValueCodebookSize = 256;
constexpr int kSpmvDict65536ValueCodebookSize = 65536;
constexpr int kSpmv8x4TileCells = kSpmv8x4BlockRows * kSpmv8x4BlockCols;
constexpr int kSpmvSignTile32Words = 1;
constexpr int kSpmvCode1Tile32Words = 1;
constexpr int kSpmvCode2Tile64Words = 2;
constexpr int kSpmvCode4Tile128Words = 4;
constexpr int kSpmvCode8Tile256Words = 8;
constexpr int kSpmvCode16Tile512Words = 16;
constexpr int kSpmvRound16Tile512Words = 16;
constexpr std::uint32_t kSpmv8x4PackedFlagMask = 0x80000000U;
constexpr std::uint64_t kSpmv8x4PackedFlagBit =
    std::uint64_t{1} << kSpmv8x4PackedFlagShift;
constexpr std::uint64_t kSpmv8x4PackedAbsColMask = 0x7fffffffULL;

template <int BitsPerCode>
struct TileCodeTraits8x4;

template <>
struct TileCodeTraits8x4<1> {
    static constexpr int words_per_tile = kSpmvCode1Tile32Words;
    static constexpr std::uint32_t mask = 0x1U;
};

template <>
struct TileCodeTraits8x4<2> {
    static constexpr int words_per_tile = kSpmvCode2Tile64Words;
    static constexpr std::uint32_t mask = 0x3U;
};

template <>
struct TileCodeTraits8x4<4> {
    static constexpr int words_per_tile = kSpmvCode4Tile128Words;
    static constexpr std::uint32_t mask = 0xfU;
};

template <>
struct TileCodeTraits8x4<8> {
    static constexpr int words_per_tile = kSpmvCode8Tile256Words;
    static constexpr std::uint32_t mask = 0xffU;
};

template <>
struct TileCodeTraits8x4<16> {
    static constexpr int words_per_tile = kSpmvCode16Tile512Words;
    static constexpr std::uint32_t mask = 0xffffU;
};

__host__ __device__ inline std::uint64_t pack_col_bitmap(std::uint32_t col,
                                                         BitmapWord bitmap,
                                                         bool flag = false) {
    const std::uint32_t col_and_flag =
        (col & static_cast<std::uint32_t>(kSpmv8x4PackedAbsColMask)) |
        (flag ? kSpmv8x4PackedFlagMask : std::uint32_t{0});
    return (static_cast<std::uint64_t>(bitmap) << kSpmv8x4PackedBitmapShift) |
           (static_cast<std::uint64_t>(col_and_flag) << kSpmv8x4PackedAbsColShift);
}

__host__ __device__ inline std::uint32_t unpack_abs_col(std::uint64_t meta) {
    return static_cast<std::uint32_t>((meta >> kSpmv8x4PackedAbsColShift) &
                                      kSpmv8x4PackedAbsColMask);
}

__host__ __device__ inline bool unpack_col_bitmap_flag(std::uint64_t meta) {
    return (static_cast<std::uint32_t>(meta >> kSpmv8x4PackedAbsColShift) &
            kSpmv8x4PackedFlagMask) != 0U;
}

__host__ __device__ inline BitmapWord unpack_bitmap(std::uint64_t meta) {
    return static_cast<BitmapWord>(meta >> kSpmv8x4PackedBitmapShift);
}

template <typename T>
__device__ inline T decode_sign_tile32_value(const std::uint32_t* __restrict__ sign_masks,
                                             Offset block,
                                             int bit) {
    const std::uint32_t sign_bit = (sign_masks[block] >> bit) & 1U;
    return sign_bit != 0U ? T{1} : T{-1};
}

__device__ inline float decode_fp16_bits_to_float(std::uint16_t bits) {
    // Use the hardware fp16->fp32 convert (single instruction, no warp
    // divergence). The previous pure-software path had a per-lane denormal
    // normalization loop that dominated this controlled-lossy kernel's runtime.
    return __half2float(__ushort_as_half(bits));
}

__device__ inline float decode_bf16_bits_to_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

template <bool Bf16>
__device__ inline float decode_round16_tile_value_f32(
    const std::uint32_t* __restrict__ tile_code_words,
    Offset block,
    int bit) {
    const int word = bit / 2;
    const int shift = (bit % 2) * 16;
    const std::uint32_t packed =
        tile_code_words[block * static_cast<Offset>(kSpmvRound16Tile512Words) + word];
    const auto bits = static_cast<std::uint16_t>((packed >> shift) & 0xffffU);
    return Bf16 ? decode_bf16_bits_to_float(bits) : decode_fp16_bits_to_float(bits);
}

template <bool Bf16, typename T>
__device__ inline T decode_round16_tile_value(
    const std::uint32_t* __restrict__ tile_code_words,
    Offset block,
    int bit) {
    return static_cast<T>(
        decode_round16_tile_value_f32<Bf16>(tile_code_words, block, bit));
}

template <int BitsPerCode, typename T>
__device__ inline T decode_tile_code_value(
    const std::uint32_t* __restrict__ tile_code_words,
    const T* __restrict__ codebook,
    Offset block,
    int bit) {
    constexpr int words_per_tile = TileCodeTraits8x4<BitsPerCode>::words_per_tile;
    constexpr int codes_per_word = 32 / BitsPerCode;
    const int word = bit / codes_per_word;
    const int shift = (bit % codes_per_word) * BitsPerCode;
    const std::uint32_t packed =
        tile_code_words[block * static_cast<Offset>(words_per_tile) + word];
    const std::uint32_t code = (packed >> shift) & TileCodeTraits8x4<BitsPerCode>::mask;
    return codebook[code];
}

__global__ void make_bitbsr_8x4_col_bitmap_packed_kernel(
    Offset num_blocks,
    const Index* __restrict__ block_col_idx,
    const BitmapWord* __restrict__ bitmap_words,
    std::uint64_t* __restrict__ packed_meta) {
    const Offset block = static_cast<Offset>(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= num_blocks) {
        return;
    }
    packed_meta[block] =
        pack_col_bitmap(static_cast<std::uint32_t>(block_col_idx[block]), bitmap_words[block]);
}

__global__ void clear_bitbsr_8x4_col_bitmap_flags_kernel(
    Offset num_blocks,
    std::uint64_t* __restrict__ packed_meta) {
    const Offset block = static_cast<Offset>(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= num_blocks) {
        return;
    }
    packed_meta[block] &= ~kSpmv8x4PackedFlagBit;
}

__global__ void count_bitbsr_8x4_load_balance_work_items_kernel(
    Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    int chunk_blocks,
    int split_threshold_blocks,
    Offset* __restrict__ work_counts) {
    const Index br = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
    if (br >= num_block_rows) {
        return;
    }
    const Offset block_count = block_row_ptr[br + 1] - block_row_ptr[br];
    if (block_count == 0) {
        work_counts[br] = 0;
    } else if (block_count <= static_cast<Offset>(split_threshold_blocks)) {
        work_counts[br] = 1;
    } else {
        work_counts[br] =
            (block_count + static_cast<Offset>(chunk_blocks) - 1) /
            static_cast<Offset>(chunk_blocks);
    }
}

__global__ void fill_bitbsr_8x4_load_balance_work_items_kernel(
    Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const Offset* __restrict__ block_row_val_ptr,
    const BitmapWord* __restrict__ bitmap_words,
    const Offset* __restrict__ work_item_row_ptr,
    int chunk_blocks,
    std::uint64_t* __restrict__ packed_col_bitmap_meta,
    BitBsrWorkItem8x4* __restrict__ work_items) {
    const Index br = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
    if (br >= num_block_rows) {
        return;
    }

    const Offset row_start = block_row_ptr[br];
    const Offset row_end = block_row_ptr[br + 1];
    const Offset num_chunks = work_item_row_ptr[br + 1] - work_item_row_ptr[br];
    if (num_chunks == 0) {
        return;
    }

    const bool atomic_output = num_chunks > 1;
    const Offset effective_chunk =
        atomic_output ? static_cast<Offset>(chunk_blocks)
                      : static_cast<Offset>(row_end - row_start);
    Offset block = row_start;
    Offset value_pos = block_row_val_ptr[br];
    Offset out = work_item_row_ptr[br];
    while (block < row_end) {
        const Offset count = min(effective_chunk, static_cast<Offset>(row_end - block));
        BitBsrWorkItem8x4 item;
        item.br = static_cast<std::uint32_t>(br);
        item.block_begin = static_cast<std::uint32_t>(block);
        item.block_count = static_cast<std::uint32_t>(count);
        item.value_begin = static_cast<std::uint32_t>(value_pos);
        work_items[out] = item;

        if (packed_col_bitmap_meta != nullptr && atomic_output) {
            packed_col_bitmap_meta[block] |= kSpmv8x4PackedFlagBit;
        }
        for (Offset i = 0; i < count; ++i) {
            value_pos += static_cast<Offset>(__popc(bitmap_words[block + i]));
        }
        block += count;
        ++out;
    }
}


template <int BitsPerCode, typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_code_tile_kernel(
    Index rows, Index cols, Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const std::uint32_t* __restrict__ tile_code_words,
    const T* __restrict__ codebook,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Index br = static_cast<Index>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (br >= num_block_rows) {
        return;
    }

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    T sum = T{0};
    const Offset start = block_row_ptr[br];
    const Offset end = block_row_ptr[br + 1];

    for (Offset block = start; block < end; ++block) {
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            sum += decode_tile_code_value<BitsPerCode, T>(
                       tile_code_words, codebook, block, bit) *
                   x[col];
        }
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        y[row] = sum;
    }
}

template <bool Bf16, typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_round16_tile_kernel(
    Index rows, Index cols, Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const std::uint32_t* __restrict__ tile_code_words,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Index br = static_cast<Index>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (br >= num_block_rows) {
        return;
    }

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    // Controlled-lossy codec: accumulate in FP32 regardless of T so that the
    // double baseline's fp64 multiply/add cost is replaced by fp32 (the
    // fp64->fp32 compute win on Blackwell). Output is cast back to T.
    float sum = 0.0f;
    const Offset start = block_row_ptr[br];
    const Offset end = block_row_ptr[br + 1];

    for (Offset block = start; block < end; ++block) {
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            sum += decode_round16_tile_value_f32<Bf16>(tile_code_words, block, bit) *
                   static_cast<float>(x[col]);
        }
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        y[row] = static_cast<T>(sum);
    }
}

template <int BitsPerCode, typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_code_tile_kernel(
    Index rows, Index cols, Offset num_work_items,
    const std::uint64_t* __restrict__ packed_meta,
    const BitBsrWorkItem8x4* __restrict__ work_items,
    const std::uint32_t* __restrict__ tile_code_words,
    const T* __restrict__ codebook,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Offset work_id = static_cast<Offset>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (work_id >= num_work_items) {
        return;
    }

    const BitBsrWorkItem8x4 work = work_items[work_id];
    const Index br = static_cast<Index>(work.br);
    const Offset block_begin = static_cast<Offset>(work.block_begin);
    const Offset block_count = static_cast<Offset>(work.block_count);
    const bool atomic_output = unpack_col_bitmap_flag(packed_meta[block_begin]);

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    T sum = T{0};

    for (Offset i = 0; i < block_count; ++i) {
        const Offset block = block_begin + i;
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            sum += decode_tile_code_value<BitsPerCode, T>(
                       tile_code_words, codebook, block, bit) *
                   x[col];
        }
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        if (atomic_output) {
            atomicAdd(&y[row], sum);
        } else {
            y[row] = sum;
        }
    }
}

template <bool Bf16, typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_round16_tile_kernel(
    Index rows, Index cols, Offset num_work_items,
    const std::uint64_t* __restrict__ packed_meta,
    const BitBsrWorkItem8x4* __restrict__ work_items,
    const std::uint32_t* __restrict__ tile_code_words,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Offset work_id = static_cast<Offset>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (work_id >= num_work_items) {
        return;
    }

    const BitBsrWorkItem8x4 work = work_items[work_id];
    const Index br = static_cast<Index>(work.br);
    const Offset block_begin = static_cast<Offset>(work.block_begin);
    const Offset block_count = static_cast<Offset>(work.block_count);
    const bool atomic_output = unpack_col_bitmap_flag(packed_meta[block_begin]);

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    // Controlled-lossy codec: accumulate in FP32 regardless of T (fp64->fp32
    // compute win). Output is cast back to T.
    float sum = 0.0f;

    for (Offset i = 0; i < block_count; ++i) {
        const Offset block = block_begin + i;
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            sum += decode_round16_tile_value_f32<Bf16>(tile_code_words, block, bit) *
                   static_cast<float>(x[col]);
        }
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        if (atomic_output) {
            atomicAdd(&y[row], static_cast<T>(sum));
        } else {
            y[row] = static_cast<T>(sum);
        }
    }
}

// BFP8 = int8 mantissa + per-8x4-block bf16 scale.
// value ~= (float)int8_values[value_base + rank] * bf16(block_scale_bf16[block]),
// with rank = popc(bitmap & ((1<<bit)-1)) and value_base advancing by popc(bitmap)
// per block -- identical per-nnz block-grouped bit-rank indexing to the raw path.
template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_bfp8_blkscale_kernel(
    Index rows, Index cols, Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const Offset* __restrict__ block_row_val_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const signed char* __restrict__ int8_values,
    const std::uint16_t* __restrict__ block_scale_bf16,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Index br = static_cast<Index>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (br >= num_block_rows) {
        return;
    }

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    const Offset block_begin = block_row_ptr[br];
    const Offset block_end = block_row_ptr[br + 1];

    // BFP8 is a controlled-lossy codec: accumulate in FP32 regardless of T
    // (fp64->fp32 compute win, mirrors bench spmv_lb_int8_blkscale).
    float sum = 0.0f;

    Offset value_base = block_row_val_ptr[br];
    for (Offset block = block_begin; block < block_end; ++block) {
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            const BitmapWord before = bitmap & ((BitmapWord{1} << bit) - 1U);
            const Offset rank = static_cast<Offset>(__popc(before));
            const float value =
                static_cast<float>(int8_values[value_base + rank]) *
                decode_bf16_bits_to_float(block_scale_bf16[block]);
            sum += value * static_cast<float>(x[col]);
        }
        value_base += static_cast<Offset>(__popc(bitmap));
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        y[row] = static_cast<T>(sum);
    }
}

// Flag load-balance BFP8 kernel: copies the proven bench spmv_lb_int8_blkscale
// decode/indexing, using work.value_begin as the per-work-item value base.
template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_bfp8_blkscale_kernel(
    Index rows, Index cols, Offset num_work_items,
    const std::uint64_t* __restrict__ packed_meta,
    const BitBsrWorkItem8x4* __restrict__ work_items,
    const signed char* __restrict__ int8_values,
    const std::uint16_t* __restrict__ block_scale_bf16,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Offset work_id = static_cast<Offset>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (work_id >= num_work_items) {
        return;
    }

    const BitBsrWorkItem8x4 work = work_items[work_id];
    const Index br = static_cast<Index>(work.br);
    const Offset block_begin = static_cast<Offset>(work.block_begin);
    const Offset block_count = static_cast<Offset>(work.block_count);
    const bool atomic_output = unpack_col_bitmap_flag(packed_meta[block_begin]);

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    // BFP8 is a controlled-lossy codec: accumulate in FP32 regardless of T
    // (fp64->fp32 compute win, mirrors bench spmv_lb_int8_blkscale).
    float sum = 0.0f;

    Offset value_base = static_cast<Offset>(work.value_begin);
    for (Offset i = 0; i < block_count; ++i) {
        const Offset block = block_begin + i;
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            const BitmapWord before = bitmap & ((BitmapWord{1} << bit) - 1U);
            const Offset rank = static_cast<Offset>(__popc(before));
            const float value =
                static_cast<float>(int8_values[value_base + rank]) *
                decode_bf16_bits_to_float(block_scale_bf16[block]);
            sum += value * static_cast<float>(x[col]);
        }
        value_base += static_cast<Offset>(__popc(bitmap));
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        if (atomic_output) {
            atomicAdd(&y[row], static_cast<T>(sum));
        } else {
            y[row] = static_cast<T>(sum);
        }
    }
}

// Per-nonzero 16-bit (bf16/fp16) value codec, NO block scale. Mirrors the BFP8
// per-nnz kernel exactly (block-grouped bit-rank indexing: value index =
// value_base + popc(bitmap & ((1<<bit)-1)); value_base += popc(bitmap) per
// block) but reads one 16-bit value per nonzero and decodes it directly:
// bf16 via decode_bf16_bits_to_float, fp16 via the hardware __half2float path.
// Controlled-lossy -> FP32 accumulate regardless of T (matches BFP8).
template <bool Bf16, typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_round16_nnz_kernel(
    Index rows, Index cols, Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const Offset* __restrict__ block_row_val_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const std::uint16_t* __restrict__ values16,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Index br = static_cast<Index>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (br >= num_block_rows) {
        return;
    }

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    const Offset block_begin = block_row_ptr[br];
    const Offset block_end = block_row_ptr[br + 1];

    float sum = 0.0f;

    Offset value_base = block_row_val_ptr[br];
    for (Offset block = block_begin; block < block_end; ++block) {
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            const BitmapWord before = bitmap & ((BitmapWord{1} << bit) - 1U);
            const Offset rank = static_cast<Offset>(__popc(before));
            const std::uint16_t bits = values16[value_base + rank];
            const float value = Bf16 ? decode_bf16_bits_to_float(bits)
                                     : decode_fp16_bits_to_float(bits);
            sum += value * static_cast<float>(x[col]);
        }
        value_base += static_cast<Offset>(__popc(bitmap));
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        y[row] = static_cast<T>(sum);
    }
}

// Flag load-balance per-nonzero 16-bit (bf16/fp16) kernel. Mirrors the flag-LB
// BFP8 kernel (work.value_begin as the per-work-item value base), reading one
// 16-bit value per nonzero with no block scale, FP32 accumulate.
template <bool Bf16, typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_round16_nnz_kernel(
    Index rows, Index cols, Offset num_work_items,
    const std::uint64_t* __restrict__ packed_meta,
    const BitBsrWorkItem8x4* __restrict__ work_items,
    const std::uint16_t* __restrict__ values16,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Offset work_id = static_cast<Offset>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (work_id >= num_work_items) {
        return;
    }

    const BitBsrWorkItem8x4 work = work_items[work_id];
    const Index br = static_cast<Index>(work.br);
    const Offset block_begin = static_cast<Offset>(work.block_begin);
    const Offset block_count = static_cast<Offset>(work.block_count);
    const bool atomic_output = unpack_col_bitmap_flag(packed_meta[block_begin]);

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    float sum = 0.0f;

    Offset value_base = static_cast<Offset>(work.value_begin);
    for (Offset i = 0; i < block_count; ++i) {
        const Offset block = block_begin + i;
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            const BitmapWord before = bitmap & ((BitmapWord{1} << bit) - 1U);
            const Offset rank = static_cast<Offset>(__popc(before));
            const std::uint16_t bits = values16[value_base + rank];
            const float value = Bf16 ? decode_bf16_bits_to_float(bits)
                                     : decode_fp16_bits_to_float(bits);
            sum += value * static_cast<float>(x[col]);
        }
        value_base += static_cast<Offset>(__popc(bitmap));
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        if (atomic_output) {
            atomicAdd(&y[row], static_cast<T>(sum));
        } else {
            y[row] = static_cast<T>(sum);
        }
    }
}

// BFP8 outlier pass-2 correction kernel. Tiny: one thread per outlier nonzero.
// Run AFTER the unchanged BFP8 bulk kernel (which already holds the bulk result
// in y, with outlier mantissa slots == 0). Adds the exact outlier contribution
//   y[row_i] += value_i * x[col_i]
// via atomicAdd, in FP32 to match the bulk kernel's accumulation precision.
template <typename T>
__global__ void spmv_bitbsr_8x4_outlier_correction_kernel(
    Offset num_outliers,
    const std::int32_t* __restrict__ outlier_rows,
    const std::int32_t* __restrict__ outlier_cols,
    const T* __restrict__ outlier_vals,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const Offset i =
        static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_outliers) {
        return;
    }
    const Index row = static_cast<Index>(outlier_rows[i]);
    const Index col = static_cast<Index>(outlier_cols[i]);
    const float contribution =
        static_cast<float>(outlier_vals[i]) * static_cast<float>(x[col]);
    atomicAdd(&y[row], static_cast<T>(contribution));
}

// Decode the signed 4-bit (two's complement, range [-7, 7]) mantissa at linear
// nnz position `pos`: nibble (pos & 1) of byte pos/2, then sign-extend.
__device__ inline int decode_bfp4_nibble(const std::uint8_t* __restrict__ packed_int4,
                                         Offset pos) {
    const std::uint8_t byte = packed_int4[pos >> 1];
    const int nibble = ((pos & 1) == 0) ? (byte & 0xF) : (byte >> 4);
    return (nibble ^ 0x8) - 0x8;
}

// BFP4 = int4 mantissa + per-8x4-block bf16 scale. FP32 compute regardless of T.
// value ~= (float)int4[value_base + rank] * bf16(block_scale_bf16[block]).
template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_bfp4_blkscale_kernel(
    Index rows, Index cols, Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const Offset* __restrict__ block_row_val_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const std::uint8_t* __restrict__ packed_int4,
    const std::uint16_t* __restrict__ block_scale_bf16,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Index br = static_cast<Index>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (br >= num_block_rows) {
        return;
    }

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    const Offset block_begin = block_row_ptr[br];
    const Offset block_end = block_row_ptr[br + 1];

    float sum = 0.0f;

    Offset value_base = block_row_val_ptr[br];
    for (Offset block = block_begin; block < block_end; ++block) {
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            const BitmapWord before = bitmap & ((BitmapWord{1} << bit) - 1U);
            const Offset rank = static_cast<Offset>(__popc(before));
            const float value =
                static_cast<float>(decode_bfp4_nibble(packed_int4, value_base + rank)) *
                decode_bf16_bits_to_float(block_scale_bf16[block]);
            sum += value * static_cast<float>(x[col]);
        }
        value_base += static_cast<Offset>(__popc(bitmap));
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        y[row] = static_cast<T>(sum);
    }
}

template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_bfp4_blkscale_kernel(
    Index rows, Index cols, Offset num_work_items,
    const std::uint64_t* __restrict__ packed_meta,
    const BitBsrWorkItem8x4* __restrict__ work_items,
    const std::uint8_t* __restrict__ packed_int4,
    const std::uint16_t* __restrict__ block_scale_bf16,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const Offset work_id = static_cast<Offset>(blockIdx.x * kSpmv8x4WarpsPerBlock + warp_id);

    if (work_id >= num_work_items) {
        return;
    }

    const BitBsrWorkItem8x4 work = work_items[work_id];
    const Index br = static_cast<Index>(work.br);
    const Offset block_begin = static_cast<Offset>(work.block_begin);
    const Offset block_count = static_cast<Offset>(work.block_count);
    const bool atomic_output = unpack_col_bitmap_flag(packed_meta[block_begin]);

    const int local_row = lane / kSpmv8x4BlockCols;
    const int local_col = lane % kSpmv8x4BlockCols;
    const int bit = local_row * kSpmv8x4BlockCols + local_col;
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    float sum = 0.0f;

    Offset value_base = static_cast<Offset>(work.value_begin);
    for (Offset i = 0; i < block_count; ++i) {
        const Offset block = block_begin + i;
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            const BitmapWord before = bitmap & ((BitmapWord{1} << bit) - 1U);
            const Offset rank = static_cast<Offset>(__popc(before));
            const float value =
                static_cast<float>(decode_bfp4_nibble(packed_int4, value_base + rank)) *
                decode_bf16_bits_to_float(block_scale_bf16[block]);
            sum += value * static_cast<float>(x[col]);
        }
        value_base += static_cast<Offset>(__popc(bitmap));
    }

    const unsigned row_mask = 0xFU << (local_row * kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset, kSpmv8x4BlockCols);
    }

    if (valid_row && local_col == 0) {
        if (atomic_output) {
            atomicAdd(&y[row], static_cast<T>(sum));
        } else {
            y[row] = static_cast<T>(sum);
        }
    }
}

template <typename T>
void validate_bitbsr_8x4_shape(const DeviceBitBsrMatrix<T>& bitbsr) {
    bitbsr.validate_shape();
    if (bitbsr.block_rows != kSpmv8x4BlockRows || bitbsr.block_cols != kSpmv8x4BlockCols ||
        bitbsr.words_per_block != 1) {
        throw std::runtime_error("GPU BitBSR SpMV currently supports only 8x4 blocks");
    }
}

template <typename T>
void validate_spmv_8x4_shape(const DeviceBitBsrMatrix<T>& bitbsr,
                             const thrust::device_vector<T>& x) {
    validate_bitbsr_8x4_shape(bitbsr);
    if (static_cast<Index>(x.size()) != bitbsr.cols) {
        throw std::runtime_error("GPU BitBSR SpMV input vector size mismatch");
    }
}

template <typename T>
void validate_sign_tile32_value_codec(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint32_t>& sign_masks) {
    if (static_cast<Offset>(sign_masks.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR sign_tile32 mask size mismatch");
    }
}

template <int BitsPerCode, typename T>
void validate_tile_code_value_codec(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    int max_codebook_size,
    const char* name) {
    constexpr int words_per_tile = TileCodeTraits8x4<BitsPerCode>::words_per_tile;
    const Offset expected_words =
        bitbsr.num_blocks * static_cast<Offset>(words_per_tile);
    if (static_cast<Offset>(tile_code_words.size()) != expected_words) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " tile code word size mismatch");
    }
    if (codebook.size() > static_cast<size_t>(max_codebook_size)) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " codebook is too large");
    }
    if (bitbsr.nnz > 0 && codebook.empty()) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " codebook is empty");
    }
}

template <typename T>
void validate_round16_tile_value_codec(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const char* name) {
    const Offset expected_words =
        bitbsr.num_blocks * static_cast<Offset>(kSpmvRound16Tile512Words);
    if (static_cast<Offset>(tile_code_words.size()) != expected_words) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " tile code word size mismatch");
    }
}

template <typename T>
void validate_bfp8_blkscale_value_codec(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<signed char>& int8_values,
    const thrust::device_vector<std::uint16_t>& block_scale_bf16,
    const char* name) {
    if (static_cast<Offset>(int8_values.size()) != bitbsr.nnz) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " int8 value size mismatch");
    }
    if (static_cast<Offset>(block_scale_bf16.size()) != bitbsr.num_blocks) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " per-block scale size mismatch");
    }
}

template <typename T>
void validate_round16_nnz_value_codec(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint16_t>& values16,
    const char* name) {
    if (static_cast<Offset>(values16.size()) != bitbsr.nnz) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " per-nnz value size mismatch");
    }
}

template <typename T>
void validate_bfp4_blkscale_value_codec(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint8_t>& packed_int4,
    const thrust::device_vector<std::uint16_t>& block_scale_bf16,
    const char* name) {
    if (static_cast<Offset>(packed_int4.size()) != (bitbsr.nnz + 1) / 2) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " int4 value size mismatch");
    }
    if (static_cast<Offset>(block_scale_bf16.size()) != bitbsr.num_blocks) {
        throw std::runtime_error(std::string("GPU BitBSR ") + name +
                                 " per-block scale size mismatch");
    }
}

}  // namespace gpu_detail

template <typename T>
void make_bitbsr_8x4_col_bitmap_packed(const DeviceBitBsrMatrix<T>& bitbsr,
                                       thrust::device_vector<std::uint64_t>& packed_meta,
                                       float* device_ms = nullptr) {
    gpu_detail::validate_bitbsr_8x4_shape(bitbsr);
    packed_meta.resize(bitbsr.num_blocks);
    if (bitbsr.num_blocks == 0) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    gpu_detail::make_bitbsr_8x4_col_bitmap_packed_kernel<<<
        gpu_detail::grid_for(bitbsr.num_blocks), 256>>>(
        bitbsr.num_blocks,
        thrust::raw_pointer_cast(bitbsr.block_col_idx.data()),
        thrust::raw_pointer_cast(bitbsr.bitmap_words.data()),
        thrust::raw_pointer_cast(packed_meta.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <typename T>
void make_bitbsr_8x4_load_balance_work_items_impl(
    const DeviceBitBsrMatrix<T>& bitbsr,
    BitBsrLoadBalanceLayout8x4& layout,
    int chunk_blocks,
    int split_threshold_blocks,
    thrust::device_vector<std::uint64_t>* packed_meta,
    float* device_ms) {
    gpu_detail::validate_bitbsr_8x4_shape(bitbsr);
    if (packed_meta != nullptr && static_cast<Offset>(packed_meta->size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }
    if (chunk_blocks <= 0) {
        throw std::runtime_error("GPU BitBSR load-balance chunk_blocks must be positive");
    }
    if (split_threshold_blocks <= 0) {
        throw std::runtime_error(
            "GPU BitBSR load-balance split_threshold_blocks must be positive");
    }

    layout.chunk_blocks = chunk_blocks;
    layout.split_threshold_blocks = split_threshold_blocks;
    layout.work_item_row_ptr.resize(bitbsr.num_block_rows + 1);
    thrust::fill(layout.work_item_row_ptr.begin(), layout.work_item_row_ptr.end(), Offset{0});
    layout.work_items.clear();
    if (bitbsr.num_block_rows == 0 || bitbsr.num_blocks == 0) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    thrust::device_vector<Offset> work_counts(bitbsr.num_block_rows);
    gpu_detail::CudaEventTimer event_timer(device_ms);
    gpu_detail::count_bitbsr_8x4_load_balance_work_items_kernel<<<
        gpu_detail::grid_for(bitbsr.num_block_rows), 256>>>(
        bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        chunk_blocks,
        split_threshold_blocks,
        thrust::raw_pointer_cast(work_counts.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    thrust::inclusive_scan(work_counts.begin(), work_counts.end(),
                           layout.work_item_row_ptr.begin() + 1);
    const Offset num_work_items = layout.work_item_row_ptr[bitbsr.num_block_rows];
    if (num_work_items == 0) {
        if (device_ms != nullptr) {
            event_timer.stop();
        } else {
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
        }
        return;
    }

    layout.work_items.resize(num_work_items);
    std::uint64_t* packed_meta_ptr = nullptr;
    if (packed_meta != nullptr) {
        gpu_detail::clear_bitbsr_8x4_col_bitmap_flags_kernel<<<
            gpu_detail::grid_for(bitbsr.num_blocks), 256>>>(
            bitbsr.num_blocks,
            thrust::raw_pointer_cast(packed_meta->data()));
        cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
        packed_meta_ptr = thrust::raw_pointer_cast(packed_meta->data());
    }

    gpu_detail::fill_bitbsr_8x4_load_balance_work_items_kernel<<<
        gpu_detail::grid_for(bitbsr.num_block_rows), 256>>>(
        bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(bitbsr.block_row_val_ptr.data()),
        thrust::raw_pointer_cast(bitbsr.bitmap_words.data()),
        thrust::raw_pointer_cast(layout.work_item_row_ptr.data()),
        chunk_blocks,
        packed_meta_ptr,
        thrust::raw_pointer_cast(layout.work_items.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <typename T>
void make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
    const DeviceBitBsrMatrix<T>& bitbsr,
    BitBsrLoadBalanceLayout8x4& layout,
    int chunk_blocks,
    int split_threshold_blocks,
    thrust::device_vector<std::uint64_t>& packed_meta,
    float* device_ms = nullptr) {
    make_bitbsr_8x4_load_balance_work_items_impl(
        bitbsr, layout, chunk_blocks, split_threshold_blocks, &packed_meta, device_ms);
}

template <typename T>
void make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
    const DeviceBitBsrMatrix<T>& bitbsr,
    BitBsrLoadBalanceLayout8x4& layout,
    int chunk_blocks,
    thrust::device_vector<std::uint64_t>& packed_meta,
    float* device_ms = nullptr) {
    make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        bitbsr, layout, chunk_blocks, chunk_blocks, packed_meta, device_ms);
}

template <typename T>
void make_bitbsr_8x4_load_balance_work_items(
    const DeviceBitBsrMatrix<T>& bitbsr,
    BitBsrLoadBalanceLayout8x4& layout,
    int chunk_blocks,
    int split_threshold_blocks,
    float* device_ms = nullptr) {
    make_bitbsr_8x4_load_balance_work_items_impl(
        bitbsr, layout, chunk_blocks, split_threshold_blocks, nullptr, device_ms);
}

template <typename T>
void make_bitbsr_8x4_load_balance_work_items(
    const DeviceBitBsrMatrix<T>& bitbsr,
    BitBsrLoadBalanceLayout8x4& layout,
    int chunk_blocks,
    float* device_ms = nullptr) {
    make_bitbsr_8x4_load_balance_work_items(
        bitbsr, layout, chunk_blocks, chunk_blocks, device_ms);
}

template <int BitsPerCode, typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_code_tile_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    int max_codebook_size,
    const char* codec_name,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_tile_code_value_codec<BitsPerCode>(
        bitbsr, tile_code_words, codebook, max_codebook_size, codec_name);
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const int grid =
        gpu_detail::grid_for(bitbsr.num_block_rows, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_code_tile_kernel<BitsPerCode, T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(tile_code_words.data()),
        thrust::raw_pointer_cast(codebook.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <bool Bf16, typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_round16_tile_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const char* codec_name,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_round16_tile_value_codec(bitbsr, tile_code_words, codec_name);
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const int grid =
        gpu_detail::grid_for(bitbsr.num_block_rows, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_round16_tile_kernel<Bf16, T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(tile_code_words.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <int BitsPerCode, typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_code_tile_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    int max_codebook_size,
    const char* codec_name,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_tile_code_value_codec<BitsPerCode>(
        bitbsr, tile_code_words, codebook, max_codebook_size, codec_name);
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }
    if (static_cast<Index>(layout.work_item_row_ptr.size()) != bitbsr.num_block_rows + 1) {
        throw std::runtime_error("GPU BitBSR load-balance work_item_row_ptr size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0 || layout.work_items.empty()) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const Offset num_work_items = static_cast<Offset>(layout.work_items.size());
    const int grid =
        gpu_detail::grid_for(num_work_items, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_code_tile_kernel<
        BitsPerCode, T><<<grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(tile_code_words.data()),
        thrust::raw_pointer_cast(codebook.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <bool Bf16, typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_tile_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const char* codec_name,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_round16_tile_value_codec(bitbsr, tile_code_words, codec_name);
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }
    if (static_cast<Index>(layout.work_item_row_ptr.size()) != bitbsr.num_block_rows + 1) {
        throw std::runtime_error("GPU BitBSR load-balance work_item_row_ptr size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0 || layout.work_items.empty()) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const Offset num_work_items = static_cast<Offset>(layout.work_items.size());
    const int grid =
        gpu_detail::grid_for(num_work_items, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_round16_tile_kernel<
        Bf16, T><<<grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(tile_code_words.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

// Per-nonzero 16-bit (bf16/fp16) device launch. Mirrors the BFP8 per-nnz device
// wrapper but uses the per-nnz values16 array (no block scale).
template <bool Bf16, typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_round16_nnz_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint16_t>& values16,
    const char* codec_name,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_round16_nnz_value_codec(bitbsr, values16, codec_name);
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const int grid =
        gpu_detail::grid_for(bitbsr.num_block_rows, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_round16_nnz_kernel<Bf16, T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(bitbsr.block_row_val_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(values16.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <bool Bf16, typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_nnz_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint16_t>& values16,
    const char* codec_name,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_round16_nnz_value_codec(bitbsr, values16, codec_name);
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }
    if (static_cast<Index>(layout.work_item_row_ptr.size()) != bitbsr.num_block_rows + 1) {
        throw std::runtime_error("GPU BitBSR load-balance work_item_row_ptr size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0 || layout.work_items.empty()) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const Offset num_work_items = static_cast<Offset>(layout.work_items.size());
    const int grid =
        gpu_detail::grid_for(num_work_items, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_round16_nnz_kernel<
        Bf16, T><<<grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(values16.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_bfp8_blkscale_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<signed char>& int8_values,
    const thrust::device_vector<std::uint16_t>& block_scale_bf16,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_bfp8_blkscale_value_codec(
        bitbsr, int8_values, block_scale_bf16, "bfp8_blkscale");
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const int grid =
        gpu_detail::grid_for(bitbsr.num_block_rows, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_bfp8_blkscale_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(bitbsr.block_row_val_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(int8_values.data()),
        thrust::raw_pointer_cast(block_scale_bf16.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bfp8_blkscale_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<signed char>& int8_values,
    const thrust::device_vector<std::uint16_t>& block_scale_bf16,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_bfp8_blkscale_value_codec(
        bitbsr, int8_values, block_scale_bf16, "bfp8_blkscale");
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }
    if (static_cast<Index>(layout.work_item_row_ptr.size()) != bitbsr.num_block_rows + 1) {
        throw std::runtime_error("GPU BitBSR load-balance work_item_row_ptr size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0 || layout.work_items.empty()) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const Offset num_work_items = static_cast<Offset>(layout.work_items.size());
    const int grid =
        gpu_detail::grid_for(num_work_items, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_bfp8_blkscale_kernel<
        T><<<grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(int8_values.data()),
        thrust::raw_pointer_cast(block_scale_bf16.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

// BFP8 outlier two-pass, flag load-balance. Pass 1 is the UNCHANGED BFP8 bulk
// kernel (outlier mantissa slots are 0, scale excludes outliers). Pass 2 is the
// tiny correction kernel over the side list. Both launches are inside one timer
// so the reported event time covers pass1 + pass2 together.
template <typename T>
void spmv_bitbsr_gpu_8x4_bfp8_outlier_flag_load_balance_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<signed char>& int8_values,
    const thrust::device_vector<std::uint16_t>& block_scale_bf16,
    const thrust::device_vector<std::int32_t>& outlier_rows,
    const thrust::device_vector<std::int32_t>& outlier_cols,
    const thrust::device_vector<T>& outlier_vals,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_bfp8_blkscale_value_codec(
        bitbsr, int8_values, block_scale_bf16, "bfp8_outlier_blkscale");
    if (outlier_rows.size() != outlier_cols.size() ||
        outlier_rows.size() != outlier_vals.size()) {
        throw std::runtime_error("GPU BitBSR bfp8_outlier side-list size mismatch");
    }
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }
    if (static_cast<Index>(layout.work_item_row_ptr.size()) != bitbsr.num_block_rows + 1) {
        throw std::runtime_error("GPU BitBSR load-balance work_item_row_ptr size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0 || layout.work_items.empty()) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    const Offset num_outliers = static_cast<Offset>(outlier_vals.size());
    gpu_detail::CudaEventTimer event_timer(device_ms);

    // Pass 1: bulk BFP8 (writes y, outlier slots contribute 0).
    const Offset num_work_items = static_cast<Offset>(layout.work_items.size());
    const int grid =
        gpu_detail::grid_for(num_work_items, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_bfp8_blkscale_kernel<
        T><<<grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(int8_values.data()),
        thrust::raw_pointer_cast(block_scale_bf16.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    // Pass 2: exact outlier correction (atomicAdd onto the bulk result).
    if (num_outliers > 0) {
        constexpr int kOutlierThreads = 256;
        const int outlier_grid =
            gpu_detail::grid_for(num_outliers, kOutlierThreads);
        gpu_detail::spmv_bitbsr_8x4_outlier_correction_kernel<T>
            <<<outlier_grid, kOutlierThreads>>>(
                num_outliers,
                thrust::raw_pointer_cast(outlier_rows.data()),
                thrust::raw_pointer_cast(outlier_cols.data()),
                thrust::raw_pointer_cast(outlier_vals.data()),
                thrust::raw_pointer_cast(x.data()),
                thrust::raw_pointer_cast(y.data()));
        cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
    }

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_bfp4_blkscale_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint8_t>& packed_int4,
    const thrust::device_vector<std::uint16_t>& block_scale_bf16,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_bfp4_blkscale_value_codec(
        bitbsr, packed_int4, block_scale_bf16, "bfp4_blkscale");
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const int grid =
        gpu_detail::grid_for(bitbsr.num_block_rows, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_bfp4_blkscale_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(bitbsr.block_row_val_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(packed_int4.data()),
        thrust::raw_pointer_cast(block_scale_bf16.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bfp4_blkscale_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint8_t>& packed_int4,
    const thrust::device_vector<std::uint16_t>& block_scale_bf16,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_bfp4_blkscale_value_codec(
        bitbsr, packed_int4, block_scale_bf16, "bfp4_blkscale");
    if (static_cast<Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }
    if (static_cast<Index>(layout.work_item_row_ptr.size()) != bitbsr.num_block_rows + 1) {
        throw std::runtime_error("GPU BitBSR load-balance work_item_row_ptr size mismatch");
    }

    y.assign(static_cast<size_t>(bitbsr.rows), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0 || layout.work_items.empty()) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);
    const Offset num_work_items = static_cast<Offset>(layout.work_items.size());
    const int grid =
        gpu_detail::grid_for(num_work_items, gpu_detail::kSpmv8x4WarpsPerBlock);
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_bfp4_blkscale_kernel<
        T><<<grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(packed_int4.data()),
        thrust::raw_pointer_cast(block_scale_bf16.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()));
    cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);

    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
    }
}

}  // namespace structural
