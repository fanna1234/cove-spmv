#pragma once

#include "structural/operators/bitbsr_spmv_8x4_common.cuh"

namespace structural {

namespace gpu_detail {

template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_implicit_one_kernel(
    Index rows, Index cols, Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    T value,
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
        const int bit = local_row * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            sum += value * x[col];
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

template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_sign_tile32_kernel(
    Index rows, Index cols, Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const std::uint32_t* __restrict__ sign_masks,
    T scale,
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
            sum += scale * decode_sign_tile32_value<T>(sign_masks, block, bit) * x[col];
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

template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_implicit_one_kernel(
    Index rows, Index cols, Offset num_work_items,
    const std::uint64_t* __restrict__ packed_meta,
    const BitBsrWorkItem8x4* __restrict__ work_items,
    T value,
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
    const Index row = br * kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    T sum = T{0};

    for (Offset i = 0; i < block_count; ++i) {
        const Offset block = block_begin + i;
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;
        const int bit = local_row * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
            sum += value * x[col];
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

template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_sign_tile32_kernel(
    Index rows, Index cols, Offset num_work_items,
    const std::uint64_t* __restrict__ packed_meta,
    const BitBsrWorkItem8x4* __restrict__ work_items,
    const std::uint32_t* __restrict__ sign_masks,
    T scale,
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
            sum += scale * decode_sign_tile32_value<T>(sign_masks, block, bit) * x[col];
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

}  // namespace gpu_detail

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_implicit_one_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_implicit_one_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        T{1},
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
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_implicit_one(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_implicit_one_device(
        bitbsr, packed_meta, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_const_c_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    T value,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_implicit_one_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        value,
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
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_const_c(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    T value,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_const_c_device(
        bitbsr, packed_meta, value, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign_tile32_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& sign_masks,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_sign_tile32_value_codec(bitbsr, sign_masks);
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_sign_tile32_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(sign_masks.data()),
        T{1},
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
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign_tile32(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& sign_masks,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign_tile32_device(
        bitbsr, packed_meta, sign_masks, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign2_scale_tile32_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& sign_masks,
    T scale,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_sign_tile32_value_codec(bitbsr, sign_masks);
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_sign_tile32_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(sign_masks.data()),
        scale,
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
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign2_scale_tile32(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& sign_masks,
    T scale,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_sign2_scale_tile32_device(
        bitbsr, packed_meta, sign_masks, scale, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_dict2_tile32_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_code_tile_device<1>(
        bitbsr, packed_meta, tile_code_words, codebook,
        gpu_detail::kSpmvDict2ValueCodebookSize, "dict2_tile32", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_unique16_tile128_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_code_tile_device<4>(
        bitbsr, packed_meta, tile_code_words, codebook,
        gpu_detail::kSpmvUnique16ValueCodebookSize, "unique16_tile128", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_unique4_tile64_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_code_tile_device<2>(
        bitbsr, packed_meta, tile_code_words, codebook,
        gpu_detail::kSpmvUnique4ValueCodebookSize, "unique4_tile64", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_dict256_tile256_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_code_tile_device<8>(
        bitbsr, packed_meta, tile_code_words, codebook,
        gpu_detail::kSpmvDict256ValueCodebookSize, "dict256_tile256", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_dict65536_tile512_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_code_tile_device<16>(
        bitbsr, packed_meta, tile_code_words, codebook,
        gpu_detail::kSpmvDict65536ValueCodebookSize, "dict65536_tile512", x, y,
        device_ms);
}

template <typename T>
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_unique4_tile64(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_unique4_tile64_device(
        bitbsr, packed_meta, tile_code_words, codebook, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_unique16_tile128(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_unique16_tile128_device(
        bitbsr, packed_meta, tile_code_words, codebook, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_implicit_one_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_implicit_one_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        T{1},
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
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_implicit_one(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_implicit_one_device(
        bitbsr, packed_meta, layout, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_const_c_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    T value,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_implicit_one_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        value,
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
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_const_c(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    T value,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_const_c_device(
        bitbsr, packed_meta, layout, value, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign_tile32_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& sign_masks,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_sign_tile32_value_codec(bitbsr, sign_masks);
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_sign_tile32_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(sign_masks.data()),
        T{1},
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
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign_tile32(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& sign_masks,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign_tile32_device(
        bitbsr, packed_meta, layout, sign_masks, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign2_scale_tile32_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& sign_masks,
    T scale,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    gpu_detail::validate_sign_tile32_value_codec(bitbsr, sign_masks);
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_sign_tile32_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(sign_masks.data()),
        scale,
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
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign2_scale_tile32(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& sign_masks,
    T scale,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_sign2_scale_tile32_device(
        bitbsr, packed_meta, layout, sign_masks, scale, d_x, d_y, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_dict2_tile32_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_code_tile_device<1>(
        bitbsr, packed_meta, layout, tile_code_words, codebook,
        gpu_detail::kSpmvDict2ValueCodebookSize, "dict2_tile32", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique16_tile128_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_code_tile_device<4>(
        bitbsr, packed_meta, layout, tile_code_words, codebook,
        gpu_detail::kSpmvUnique16ValueCodebookSize, "unique16_tile128", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique4_tile64_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_code_tile_device<2>(
        bitbsr, packed_meta, layout, tile_code_words, codebook,
        gpu_detail::kSpmvUnique4ValueCodebookSize, "unique4_tile64", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_dict256_tile256_device(
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
        gpu_detail::kSpmvDict256ValueCodebookSize, "dict256_tile256", x, y,
        device_ms);
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_dict65536_tile512_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* device_ms = nullptr) {
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_code_tile_device<16>(
        bitbsr, packed_meta, layout, tile_code_words, codebook,
        gpu_detail::kSpmvDict65536ValueCodebookSize, "dict65536_tile512", x, y,
        device_ms);
}

template <typename T>
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique4_tile64(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique4_tile64_device(
        bitbsr, packed_meta, layout, tile_code_words, codebook, d_x, d_y,
        device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique16_tile128(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<std::uint32_t>& tile_code_words,
    const thrust::device_vector<T>& codebook,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_unique16_tile128_device(
        bitbsr, packed_meta, layout, tile_code_words, codebook, d_x, d_y,
        device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

}  // namespace structural
