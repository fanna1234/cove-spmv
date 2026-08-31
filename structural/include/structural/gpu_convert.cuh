#pragma once

#include "structural/bitbsr.hpp"
#include "structural/csr.hpp"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace structural {

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << file << ':' << line << ": CUDA call failed: " << expr
            << ": " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

#define STRUCTURAL_CUDA_CHECK(expr) ::structural::cuda_check((expr), #expr, __FILE__, __LINE__)

namespace gpu_detail {

constexpr int kWarpDirectBlockRows = 8;
constexpr int kWarpDirect8x4BlockCols = 4;
constexpr int kWarpDirectWarpsPerBlock = 8;
constexpr int kInfBlockCol = 2147483647;

class CudaEventTimer {
   public:
    explicit CudaEventTimer(float* elapsed_ms) : elapsed_ms_(elapsed_ms) {
        if (elapsed_ms_ == nullptr) {
            return;
        }
        *elapsed_ms_ = 0.0f;
        STRUCTURAL_CUDA_CHECK(cudaEventCreate(&start_));
        STRUCTURAL_CUDA_CHECK(cudaEventCreate(&stop_));
        STRUCTURAL_CUDA_CHECK(cudaEventRecord(start_));
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    ~CudaEventTimer() {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    void stop() {
        if (elapsed_ms_ == nullptr || stopped_) {
            return;
        }
        STRUCTURAL_CUDA_CHECK(cudaEventRecord(stop_));
        STRUCTURAL_CUDA_CHECK(cudaEventSynchronize(stop_));
        STRUCTURAL_CUDA_CHECK(cudaEventElapsedTime(elapsed_ms_, start_, stop_));
        stopped_ = true;
    }

   private:
    float* elapsed_ms_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    bool stopped_ = false;
};

__global__ void fill_block_row_val_ptr_kernel(Index rows, Index num_block_rows,
                                              Index block_rows,
                                              const Offset* __restrict__ row_ptr,
                                              Offset* __restrict__ block_row_val_ptr) {
    const int br = blockIdx.x * blockDim.x + threadIdx.x;
    if (br > num_block_rows) {
        return;
    }
    const Index row = min(rows, static_cast<Index>(br) * block_rows);
    block_row_val_ptr[br] = row_ptr[row];
}

__device__ int warp_reduce_min_i32(int value) {
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = min(value, __shfl_down_sync(mask, value, offset));
    }
    return __shfl_sync(mask, value, 0);
}

__device__ int warp_reduce_sum_i32(int value) {
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return __shfl_sync(mask, value, 0);
}

__device__ unsigned warp_reduce_or_u32(unsigned value) {
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value |= __shfl_down_sync(mask, value, offset);
    }
    return __shfl_sync(mask, value, 0);
}

__device__ int warp_prefix_sum_8_i32(int value, int lane) {
    int prefix = 0;
    for (int src = 0; src < kWarpDirectBlockRows; ++src) {
        const int src_value = __shfl_sync(0xffffffffu, value, src);
        if (src < lane) {
            prefix += src_value;
        }
    }
    return prefix;
}

template <int BlockCols>
__global__ void count_warp_direct_kernel(Index num_block_rows, Index rows,
                                         const Offset* row_ptr, const Index* col_idx,
                                         Offset* block_counts) {
    const int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (warp_id >= num_block_rows) {
        return;
    }

    const int row = warp_id * kWarpDirectBlockRows + lane;
    const bool active_row = lane < kWarpDirectBlockRows && row < rows;
    Offset pos = active_row ? row_ptr[row] : 0;
    const Offset end = active_row ? row_ptr[row + 1] : 0;

    int count = 0;
    while (true) {
        const int candidate = (active_row && pos < end) ? (col_idx[pos] / BlockCols)
                                                        : kInfBlockCol;
        const int block_col = warp_reduce_min_i32(candidate);
        if (block_col == kInfBlockCol) {
            break;
        }
        if (lane == 0) {
            ++count;
        }
        if (active_row) {
            while (pos < end && col_idx[pos] / BlockCols == block_col) {
                ++pos;
            }
        }
    }

    if (lane == 0) {
        block_counts[warp_id] = count;
    }
}

template <typename T, int BlockCols, int WordsPerBlock>
__global__ void fill_warp_direct_kernel(Index num_block_rows, Index rows,
                                        const Offset* row_ptr, const Index* col_idx,
                                        const T* values_in, const Offset* block_row_ptr,
                                        Index* block_col_idx,
                                        BitmapWord* bitmap_words, T* values_out) {
    const int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (warp_id >= num_block_rows) {
        return;
    }

    const int row = warp_id * kWarpDirectBlockRows + lane;
    const bool active_row = lane < kWarpDirectBlockRows && row < rows;
    Offset pos = active_row ? row_ptr[row] : 0;
    const Offset end = active_row ? row_ptr[row + 1] : 0;

    Offset block_id = block_row_ptr[warp_id];
    Offset value_pos = row_ptr[warp_id * kWarpDirectBlockRows];
    while (true) {
        const int candidate = (active_row && pos < end) ? (col_idx[pos] / BlockCols)
                                                        : kInfBlockCol;
        const int block_col = warp_reduce_min_i32(candidate);
        if (block_col == kInfBlockCol) {
            break;
        }

        const Offset start = pos;
        int row_count = 0;
        if (active_row) {
            while (pos < end && col_idx[pos] / BlockCols == block_col) {
                ++pos;
                ++row_count;
            }
        }
        const int total_count = warp_reduce_sum_i32(row_count);
        const int row_offset = warp_prefix_sum_8_i32(row_count, lane);

        unsigned local_bitmap0 = 0;
        unsigned local_bitmap1 = 0;
        if (active_row) {
            for (Offset p = start; p < pos; ++p) {
                const int local_col = col_idx[p] - block_col * BlockCols;
                const int bit = lane * BlockCols + local_col;
                if constexpr (WordsPerBlock == 1) {
                    local_bitmap0 |= 1u << bit;
                } else {
                    if (bit < kBitmapWordBits) {
                        local_bitmap0 |= 1u << bit;
                    } else {
                        local_bitmap1 |= 1u << (bit - kBitmapWordBits);
                    }
                }
            }
        }
        const unsigned block_bitmap0 = warp_reduce_or_u32(local_bitmap0);
        unsigned block_bitmap1 = 0;
        if constexpr (WordsPerBlock > 1) {
            block_bitmap1 = warp_reduce_or_u32(local_bitmap1);
        }

        if (active_row) {
            for (int k = 0; k < row_count; ++k) {
                values_out[value_pos + row_offset + k] = values_in[start + k];
            }
        }
        if (lane == 0) {
            block_col_idx[block_id] = block_col;
            bitmap_words[block_id * WordsPerBlock] = static_cast<BitmapWord>(block_bitmap0);
            if constexpr (WordsPerBlock > 1) {
                bitmap_words[block_id * WordsPerBlock + 1] =
                    static_cast<BitmapWord>(block_bitmap1);
            }
            ++block_id;
            value_pos += total_count;
        }
        block_id = __shfl_sync(0xffffffffu, block_id, 0);
        value_pos = __shfl_sync(0xffffffffu, value_pos, 0);
    }
}

inline int grid_for(int n, int block = 256) {
    return (n + block - 1) / block;
}

}  // namespace gpu_detail

template <typename T>
struct DeviceBitBsrMatrix {
    Index rows = 0;
    Index cols = 0;
    Index block_rows = 0;
    Index block_cols = 0;
    Index num_block_rows = 0;
    Index num_block_cols = 0;
    Index words_per_block = 0;
    Offset num_blocks = 0;
    Offset nnz = 0;

    thrust::device_vector<Offset> block_row_ptr;
    thrust::device_vector<Offset> block_row_val_ptr;
    thrust::device_vector<Index> block_col_idx;
    thrust::device_vector<BitmapWord> bitmap_words;
    thrust::device_vector<T> values;

    void reset_shape(const CsrMatrix<T>& csr, int br, int bc) {
        rows = csr.rows;
        cols = csr.cols;
        block_rows = br;
        block_cols = bc;
        num_block_rows = ceil_div(rows, block_rows);
        num_block_cols = ceil_div(cols, block_cols);
        words_per_block = ceil_div(block_rows * block_cols, kBitmapWordBits);
        nnz = csr.nnz();
        num_blocks = 0;
    }

    void validate_shape() const {
        if (rows < 0 || cols < 0 || block_rows <= 0 || block_cols <= 0) {
            throw std::runtime_error("DeviceBitBSR dimensions or block sizes are invalid");
        }
        if (num_block_rows != ceil_div(rows, block_rows) ||
            num_block_cols != ceil_div(cols, block_cols)) {
            throw std::runtime_error("DeviceBitBSR block-grid dimensions are inconsistent");
        }
        if (words_per_block != ceil_div(block_rows * block_cols, kBitmapWordBits)) {
            throw std::runtime_error("DeviceBitBSR words_per_block is inconsistent");
        }
        if (static_cast<Index>(block_row_ptr.size()) != num_block_rows + 1) {
            throw std::runtime_error("DeviceBitBSR block_row_ptr size is invalid");
        }
        if (static_cast<Index>(block_row_val_ptr.size()) != num_block_rows + 1) {
            throw std::runtime_error("DeviceBitBSR block_row_val_ptr size is invalid");
        }
        if (static_cast<Offset>(block_col_idx.size()) != num_blocks) {
            throw std::runtime_error("DeviceBitBSR block_col_idx size is invalid");
        }
        if (bitmap_words.size() != static_cast<size_t>(num_blocks) * words_per_block) {
            throw std::runtime_error("DeviceBitBSR bitmap_words size is invalid");
        }
        if (static_cast<Offset>(values.size()) != nnz) {
            throw std::runtime_error("DeviceBitBSR values size is invalid");
        }
    }

    BitBsrMatrix<T> to_host() const {
        validate_shape();

        BitBsrMatrix<T> out;
        out.rows = rows;
        out.cols = cols;
        out.block_rows = block_rows;
        out.block_cols = block_cols;
        out.num_block_rows = num_block_rows;
        out.num_block_cols = num_block_cols;
        out.words_per_block = words_per_block;

        out.block_row_ptr.resize(block_row_ptr.size());
        out.block_col_idx.resize(block_col_idx.size());
        out.bitmap_words.resize(bitmap_words.size());
        out.block_val_ptr.assign(static_cast<size_t>(num_blocks) + 1, 0);
        out.values.resize(values.size());

        std::vector<Offset> host_block_row_val_ptr(block_row_val_ptr.size());
        thrust::copy(block_row_ptr.begin(), block_row_ptr.end(), out.block_row_ptr.begin());
        thrust::copy(block_row_val_ptr.begin(), block_row_val_ptr.end(),
                     host_block_row_val_ptr.begin());
        thrust::copy(block_col_idx.begin(), block_col_idx.end(), out.block_col_idx.begin());
        thrust::copy(bitmap_words.begin(), bitmap_words.end(), out.bitmap_words.begin());
        thrust::copy(values.begin(), values.end(), out.values.begin());

        for (Index br = 0; br < num_block_rows; ++br) {
            Offset value_pos = host_block_row_val_ptr[br];
            for (Offset block = out.block_row_ptr[br]; block < out.block_row_ptr[br + 1]; ++block) {
                out.block_val_ptr[block] = value_pos;
                for (Index word = 0; word < words_per_block; ++word) {
                    value_pos += popcount_word(out.bitmap_words[block * words_per_block + word]);
                }
            }
        }
        if (!out.block_val_ptr.empty()) {
            out.block_val_ptr.back() = nnz;
        }

        out.validate();
        return out;
    }
};

template <typename T>
void copy_bitbsr_to_device(const BitBsrMatrix<T>& host, DeviceBitBsrMatrix<T>& out) {
    host.validate();

    out.rows = host.rows;
    out.cols = host.cols;
    out.block_rows = host.block_rows;
    out.block_cols = host.block_cols;
    out.num_block_rows = host.num_block_rows;
    out.num_block_cols = host.num_block_cols;
    out.words_per_block = host.words_per_block;
    out.num_blocks = host.num_blocks();
    out.nnz = host.nnz();

    std::vector<Offset> block_row_val_ptr(static_cast<size_t>(host.num_block_rows) + 1, 0);
    for (Index br = 0; br <= host.num_block_rows; ++br) {
        block_row_val_ptr[static_cast<size_t>(br)] =
            host.block_val_ptr[host.block_row_ptr[static_cast<size_t>(br)]];
    }

    out.block_row_ptr = host.block_row_ptr;
    out.block_row_val_ptr = block_row_val_ptr;
    out.block_col_idx = host.block_col_idx;
    out.bitmap_words = host.bitmap_words;
    out.values = host.values;
    out.validate_shape();
}

template <typename T>
DeviceBitBsrMatrix<T> copy_bitbsr_to_device(const BitBsrMatrix<T>& host) {
    DeviceBitBsrMatrix<T> out;
    copy_bitbsr_to_device(host, out);
    return out;
}

template <typename T>
struct GpuBitBsrWorkspace {
    thrust::device_vector<Offset> row_ptr;
    thrust::device_vector<Index> col_idx;
    thrust::device_vector<T> values;
    thrust::device_vector<Offset> block_counts;

    void upload_csr(const CsrMatrix<T>& csr) {
        row_ptr.resize(csr.row_ptr.size());
        col_idx.resize(csr.col_idx.size());
        values.resize(csr.values.size());

        thrust::copy(csr.row_ptr.begin(), csr.row_ptr.end(), row_ptr.begin());
        thrust::copy(csr.col_idx.begin(), csr.col_idx.end(), col_idx.begin());
        thrust::copy(csr.values.begin(), csr.values.end(), values.begin());
    }
};

template <int BlockCols, typename T>
void convert_to_bitbsr_gpu_warp_direct_device_impl(const CsrMatrix<T>& csr,
                                                   GpuBitBsrWorkspace<T>& workspace,
                                                   DeviceBitBsrMatrix<T>& out,
                                                   float* device_ms = nullptr) {
    static_assert(BlockCols == gpu_detail::kWarpDirect8x4BlockCols,
                  "only warp-direct 8x4 conversion is exposed");
    csr.validate();

    int device_count = 0;
    STRUCTURAL_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("no CUDA device available");
    }

    CsrMatrix<T> filtered;
    const CsrMatrix<T>* effective_csr = &csr;
    if (has_stored_zeros(csr)) {
        filtered = remove_stored_zeros(csr);
        effective_csr = &filtered;
    }

    workspace.upload_csr(*effective_csr);
    out.reset_shape(*effective_csr, gpu_detail::kWarpDirectBlockRows, BlockCols);

    workspace.block_counts.resize(out.num_block_rows);
    out.block_row_ptr.resize(out.num_block_rows + 1);
    out.block_row_val_ptr.resize(out.num_block_rows + 1);
    out.values.resize(out.nnz);
    thrust::fill(out.block_row_ptr.begin(), out.block_row_ptr.end(), 0);
    thrust::fill(out.block_row_val_ptr.begin(), out.block_row_val_ptr.end(), 0);

    if (out.nnz == 0) {
        if (device_ms != nullptr) {
            *device_ms = 0.0f;
        }
        out.num_blocks = 0;
        out.block_col_idx.clear();
        out.bitmap_words.clear();
        out.validate_shape();
        return;
    }

    gpu_detail::CudaEventTimer event_timer(device_ms);

    constexpr int threads = 32 * gpu_detail::kWarpDirectWarpsPerBlock;
    const int grid = gpu_detail::grid_for(out.num_block_rows, gpu_detail::kWarpDirectWarpsPerBlock);
    gpu_detail::count_warp_direct_kernel<BlockCols><<<grid, threads>>>(
        out.num_block_rows, effective_csr->rows, thrust::raw_pointer_cast(workspace.row_ptr.data()),
        thrust::raw_pointer_cast(workspace.col_idx.data()),
        thrust::raw_pointer_cast(workspace.block_counts.data()));
    STRUCTURAL_CUDA_CHECK(cudaGetLastError());

    thrust::inclusive_scan(workspace.block_counts.begin(), workspace.block_counts.end(),
                           out.block_row_ptr.begin() + 1);

    out.num_blocks = out.block_row_ptr[out.num_block_rows];
    out.block_col_idx.resize(out.num_blocks);
    out.bitmap_words.resize(static_cast<size_t>(out.num_blocks) * out.words_per_block);
    thrust::fill(out.bitmap_words.begin(), out.bitmap_words.end(), BitmapWord{0});

    gpu_detail::fill_block_row_val_ptr_kernel<<<
        gpu_detail::grid_for(out.num_block_rows + 1), 256>>>(
        effective_csr->rows, out.num_block_rows, out.block_rows,
        thrust::raw_pointer_cast(workspace.row_ptr.data()),
        thrust::raw_pointer_cast(out.block_row_val_ptr.data()));
    STRUCTURAL_CUDA_CHECK(cudaGetLastError());

    gpu_detail::fill_warp_direct_kernel<
        T, BlockCols,
        (gpu_detail::kWarpDirectBlockRows * BlockCols + kBitmapWordBits - 1) /
            kBitmapWordBits><<<grid, threads>>>(
        out.num_block_rows, effective_csr->rows, thrust::raw_pointer_cast(workspace.row_ptr.data()),
        thrust::raw_pointer_cast(workspace.col_idx.data()),
        thrust::raw_pointer_cast(workspace.values.data()),
        thrust::raw_pointer_cast(out.block_row_ptr.data()),
        thrust::raw_pointer_cast(out.block_col_idx.data()),
        thrust::raw_pointer_cast(out.bitmap_words.data()),
        thrust::raw_pointer_cast(out.values.data()));
    STRUCTURAL_CUDA_CHECK(cudaGetLastError());
    if (device_ms != nullptr) {
        event_timer.stop();
    } else {
        STRUCTURAL_CUDA_CHECK(cudaDeviceSynchronize());
    }

    out.validate_shape();
}

template <typename T>
void convert_to_bitbsr_gpu_warp_direct_8x4_device(const CsrMatrix<T>& csr,
                                                  GpuBitBsrWorkspace<T>& workspace,
                                                  DeviceBitBsrMatrix<T>& out,
                                                  float* device_ms = nullptr) {
    convert_to_bitbsr_gpu_warp_direct_device_impl<gpu_detail::kWarpDirect8x4BlockCols>(
        csr, workspace, out, device_ms);
}

template <typename T>
DeviceBitBsrMatrix<T> convert_to_bitbsr_gpu_warp_direct_8x4_device(
    const CsrMatrix<T>& csr,
    GpuBitBsrWorkspace<T>& workspace,
    float* device_ms = nullptr) {
    DeviceBitBsrMatrix<T> out;
    convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, out, device_ms);
    return out;
}

#undef STRUCTURAL_CUDA_CHECK

}  // namespace structural
