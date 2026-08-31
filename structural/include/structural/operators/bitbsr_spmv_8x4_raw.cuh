#pragma once

#include "structural/operators/bitbsr_spmv_8x4_common.cuh"

// Diagnostic: -DCOVE_NO_GATHER replaces the irregular x[col] gather with x[0]
// (all-hit, cached) to isolate the x-gather cost by timing (result is wrong).
#ifdef COVE_NO_GATHER
#define COVE_XVAL(xptr, col) ((xptr)[0])
#else
#define COVE_XVAL(xptr, col) ((xptr)[col])
#endif

// Diagnostic: -DCOVE_NO_VALUE replaces the value load with a constant to isolate
// the nonzero-value load cost (result is wrong; timing only).
#ifdef COVE_NO_VALUE
#define COVE_VVAL(vptr, idx) (T{1})
#else
#define COVE_VVAL(vptr, idx) ((vptr)[idx])
#endif

namespace structural {

namespace gpu_detail {

template <typename T>
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_kernel(
    Index rows, Index cols, Index num_block_rows,
    const Offset* __restrict__ block_row_ptr,
    const Offset* __restrict__ block_row_val_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const T* __restrict__ values,
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
    Offset value_base = block_row_val_ptr[br];

    for (Offset block = start; block < end; ++block) {
        const std::uint64_t meta = packed_meta[block];
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;
        const int bit = local_row * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
#ifdef COVE_NO_POPCOUNT
            sum += COVE_VVAL(values, value_base) * COVE_XVAL(x, col);
        }
#else
            const BitmapWord before = bitmap & ((BitmapWord{1} << bit) - 1U);
            const Offset rank = static_cast<Offset>(__popc(before));
            sum += COVE_VVAL(values, value_base + rank) * COVE_XVAL(x, col);
        }
        value_base += static_cast<Offset>(__popc(bitmap));
#endif
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
__global__ void spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_kernel(
    Index rows, Index cols, Offset num_work_items,
    const std::uint64_t* __restrict__ packed_meta,
    const BitBsrWorkItem8x4* __restrict__ work_items,
    const T* __restrict__ values,
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
    Offset value_base = static_cast<Offset>(work.value_begin);

#ifdef COVE_META_PREFETCH
    std::uint64_t meta_next = (block_count > 0) ? packed_meta[block_begin] : std::uint64_t{0};
#endif
    for (Offset i = 0; i < block_count; ++i) {
        const Offset block = block_begin + i;
#ifdef COVE_META_PREFETCH
        const std::uint64_t meta = meta_next;
        if (i + 1 < block_count) {
            meta_next = packed_meta[block_begin + i + 1];
        }
#else
        const std::uint64_t meta = packed_meta[block];
#endif
        const Index bc = static_cast<Index>(unpack_abs_col(meta));
        const BitmapWord bitmap = unpack_bitmap(meta);
        const Index col = bc * kSpmv8x4BlockCols + local_col;
        const int bit = local_row * kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols && ((bitmap & (BitmapWord{1} << bit)) != 0U)) {
#ifdef COVE_NO_POPCOUNT
            sum += COVE_VVAL(values, value_base) * COVE_XVAL(x, col);
        }
#else
            const BitmapWord before = bitmap & ((BitmapWord{1} << bit) - 1U);
            const Offset rank = static_cast<Offset>(__popc(before));
            sum += COVE_VVAL(values, value_base + rank) * COVE_XVAL(x, col);
        }
        value_base += static_cast<Offset>(__popc(bitmap));
#endif
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
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_device(
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, bitbsr.num_block_rows,
        thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
        thrust::raw_pointer_cast(bitbsr.block_row_val_ptr.data()),
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(bitbsr.values.data()),
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
thrust::device_vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const thrust::device_vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_device(bitbsr, packed_meta, x, y, device_ms);
    return y;
}

template <typename T>
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    const auto d_y =
        spmv_bitbsr_gpu_8x4_col_bitmap_packed_device(bitbsr, packed_meta, d_x, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

template <typename T>
void spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_device(
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
    gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_kernel<T><<<
        grid, gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
        bitbsr.rows, bitbsr.cols, num_work_items,
        thrust::raw_pointer_cast(packed_meta.data()),
        thrust::raw_pointer_cast(layout.work_items.data()),
        thrust::raw_pointer_cast(bitbsr.values.data()),
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
thrust::device_vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_device(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> y;
    spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_device(
        bitbsr, packed_meta, layout, x, y, device_ms);
    return y;
}

template <typename T>
std::vector<T> spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance(
    const DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const BitBsrLoadBalanceLayout8x4& layout,
    const std::vector<T>& x,
    float* device_ms = nullptr) {
    thrust::device_vector<T> d_x(x.begin(), x.end());
    const auto d_y = spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_device(
        bitbsr, packed_meta, layout, d_x, device_ms);
    std::vector<T> y(d_y.size());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    return y;
}

}  // namespace structural
