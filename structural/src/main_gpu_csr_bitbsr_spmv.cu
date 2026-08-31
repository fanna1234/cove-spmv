#include "structural/csr_bitbsr_split.hpp"
#include "structural/gpu_convert.cuh"
#include "structural/gpu_spmv.cuh"
#include "structural/gpu_value_codecs.cuh"
#include "structural/matrix_market.hpp"
#include "structural/spmv.hpp"
#include "structural/spmv_harness.hpp"

#include <thrust/transform.h>

#include <cstring>
#include <unordered_map>

#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct Options {
    std::string matrix_path;
    std::string precision = "double";
    std::string mode = "fused_lb";
    std::string split_policy = "cost_gated";
    std::string residual_format = "csr";
    structural::CsrBitBsrSplitCostModel cost_model;
    int block_rows = 8;
    int block_cols = 4;
    int residual_threshold = 32;
    int long_row_split = 0;
    int lb_chunk_blocks = 32;
    int lb_split_threshold_blocks = -1;
    std::string cusparse_alg = "default";
    int warmup = 5;
    int iterations = 20;
    bool verify_cpu = false;
    std::string operator_label = "hybrid_fused_lb";
    std::string value_codec = "original";  // "original" (fp64) | "bfp8" (JOINT: BFP8 blocks + bf16 residual)
};

void print_usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " MATRIX.mtx"
              << " [--precision float|double]"
              << " [--mode fused_lb|two_pass]"
              << " [--split-policy threshold|cost_gated]"
              << " [--residual-format csr]"
              << " [--br 8] [--bc 4]"
              << " [--residual-threshold N]"
              << " [--long-row-split N]"
              << " [--cost-bitmap-block X] [--cost-bitmap-nnz X]"
              << " [--cost-csr-block X] [--cost-csr-nnz X]"
              << " [--lb-chunk-blocks N]"
              << " [--lb-split-threshold-blocks N]"
              << " [--cusparse-alg default|csr_alg1|csr_alg2]"
              << " [--operator-label NAME]"
              << " [--warmup N] [--iters N] [--verify-cpu]\n";
}

Options parse_options(int argc, char** argv) {
    Options opts;
    if (argc < 2) {
        print_usage(argv[0]);
        std::exit(2);
    }
    opts.matrix_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--precision" && i + 1 < argc) {
            opts.precision = argv[++i];
            if (opts.precision != "float" && opts.precision != "double") {
                throw std::runtime_error("precision must be float or double");
            }
        } else if (arg == "--mode" && i + 1 < argc) {
            opts.mode = argv[++i];
            if (opts.mode == "fused") {
                opts.mode = "fused_lb";
            }
            if (opts.mode != "fused_lb" && opts.mode != "two_pass") {
                throw std::runtime_error("mode must be fused_lb or two_pass");
            }
        } else if (arg == "--split-policy" && i + 1 < argc) {
            opts.split_policy = argv[++i];
            if (opts.split_policy == "cost" || opts.split_policy == "auto") {
                opts.split_policy = "cost_gated";
            }
            if (opts.split_policy != "threshold" && opts.split_policy != "cost_gated") {
                throw std::runtime_error("split-policy must be threshold or cost_gated");
            }
        } else if (arg == "--residual-format" && i + 1 < argc) {
            opts.residual_format = argv[++i];
            if (opts.residual_format != "csr") {
                throw std::runtime_error("residual-format must be csr");
            }
        } else if (arg == "--br" && i + 1 < argc) {
            opts.block_rows = std::stoi(argv[++i]);
        } else if (arg == "--bc" && i + 1 < argc) {
            opts.block_cols = std::stoi(argv[++i]);
        } else if ((arg == "--residual-threshold" || arg == "--threshold") &&
                   i + 1 < argc) {
            opts.residual_threshold = std::stoi(argv[++i]);
        } else if (arg == "--value-codec" && i + 1 < argc) {
            opts.value_codec = argv[++i];
        } else if (arg == "--long-row-split" && i + 1 < argc) {
            opts.long_row_split = std::stoi(argv[++i]);
        } else if (arg == "--cost-bitmap-block" && i + 1 < argc) {
            opts.cost_model.bitmap_block_cost = std::stod(argv[++i]);
        } else if (arg == "--cost-bitmap-nnz" && i + 1 < argc) {
            opts.cost_model.bitmap_nnz_cost = std::stod(argv[++i]);
        } else if (arg == "--cost-csr-block" && i + 1 < argc) {
            opts.cost_model.csr_block_cost = std::stod(argv[++i]);
        } else if (arg == "--cost-csr-nnz" && i + 1 < argc) {
            opts.cost_model.csr_nnz_cost = std::stod(argv[++i]);
        } else if ((arg == "--lb-chunk-blocks" || arg == "--chunk-blocks") && i + 1 < argc) {
            opts.lb_chunk_blocks = std::stoi(argv[++i]);
        } else if ((arg == "--lb-split-threshold-blocks" ||
                    arg == "--split-threshold-blocks" ||
                    arg == "--lb-split-threshold") &&
                   i + 1 < argc) {
            opts.lb_split_threshold_blocks = std::stoi(argv[++i]);
        } else if (arg == "--cusparse-alg" && i + 1 < argc) {
            opts.cusparse_alg = argv[++i];
            if (opts.cusparse_alg == "alg1") {
                opts.cusparse_alg = "csr_alg1";
            }
            if (opts.cusparse_alg == "alg2") {
                opts.cusparse_alg = "csr_alg2";
            }
            if (opts.cusparse_alg != "default" && opts.cusparse_alg != "csr_alg1" &&
                opts.cusparse_alg != "csr_alg2") {
                throw std::runtime_error("cusparse-alg must be default, csr_alg1, or csr_alg2");
            }
        } else if (arg == "--operator-label" && i + 1 < argc) {
            opts.operator_label = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts.warmup = std::stoi(argv[++i]);
        } else if ((arg == "--iters" || arg == "--iterations") && i + 1 < argc) {
            opts.iterations = std::stoi(argv[++i]);
        } else if (arg == "--verify-cpu") {
            opts.verify_cpu = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (opts.block_rows != 8 || opts.block_cols != 4) {
        throw std::runtime_error("CSR+BitBSR GPU probe currently supports only 8x4 blocks");
    }
    if (opts.residual_threshold < 0) {
        throw std::runtime_error("residual-threshold must be non-negative");
    }
    if (opts.mode == "two_pass" && opts.residual_format != "csr") {
        throw std::runtime_error("two_pass mode supports only CSR residual format");
    }
    opts.cost_model.validate();
    if (opts.warmup < 0 || opts.iterations <= 0) {
        throw std::runtime_error("warmup must be non-negative and iterations must be positive");
    }
    if (opts.lb_chunk_blocks <= 0) {
        throw std::runtime_error("lb-chunk-blocks must be positive");
    }
    if (opts.lb_split_threshold_blocks < 0) {
        opts.lb_split_threshold_blocks = opts.lb_chunk_blocks;
    }
    if (opts.lb_split_threshold_blocks <= 0) {
        throw std::runtime_error("lb-split-threshold-blocks must be positive");
    }
    return opts;
}

inline void cusparse_check(cusparseStatus_t status,
                           const char* expr,
                           const char* file,
                           int line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << file << ':' << line << ": cuSPARSE call failed: " << expr
            << ": status=" << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

#define STRUCTURAL_CUSPARSE_CHECK(expr) cusparse_check((expr), #expr, __FILE__, __LINE__)

template <typename T>
cudaDataType cuda_data_type();

template <>
cudaDataType cuda_data_type<float>() {
    return CUDA_R_32F;
}

template <>
cudaDataType cuda_data_type<double>() {
    return CUDA_R_64F;
}

cusparseSpMVAlg_t parse_cusparse_spmv_alg(const std::string& alg) {
    if (alg == "default") {
        return CUSPARSE_SPMV_ALG_DEFAULT;
    }
    if (alg == "csr_alg1") {
        return CUSPARSE_SPMV_CSR_ALG1;
    }
    if (alg == "csr_alg2") {
        return CUSPARSE_SPMV_CSR_ALG2;
    }
    throw std::runtime_error("unsupported cuSPARSE SpMV algorithm: " + alg);
}

using structural::make_input_vector;  // shared harness
using structural::time_call;

template <typename T>
bool verify_result(const structural::CsrMatrix<T>& csr,
                   const std::vector<T>& x,
                   const thrust::device_vector<T>& d_y,
                   double& cpu_spmv_ms,
                   structural::ErrorStats<T>& error) {
    structural::compute_cpu_error(csr, x, d_y, cpu_spmv_ms, error);
    const T abs_tol = std::is_same_v<T, float> ? static_cast<T>(1e-2) : static_cast<T>(1e-8);
    const T rel_tol = std::is_same_v<T, float> ? static_cast<T>(1e-4) : static_cast<T>(1e-10);
    return error.max_abs <= abs_tol || error.max_rel <= rel_tol;
}

template <typename T>
void run_bitbsr_col_bitmap64_flag_lb_stable_y(
    const structural::DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const structural::BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* iter_ms = nullptr) {
    structural::gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    if (static_cast<structural::Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("GPU BitBSR col_bitmap_packed layout size mismatch");
    }
    if (static_cast<structural::Index>(layout.work_item_row_ptr.size()) !=
        bitbsr.num_block_rows + 1) {
        throw std::runtime_error("GPU BitBSR load-balance work_item_row_ptr size mismatch");
    }
    if (static_cast<structural::Index>(y.size()) != bitbsr.rows) {
        throw std::runtime_error("GPU BitBSR output vector size mismatch");
    }

    thrust::fill(y.begin(), y.end(), T{0});
    if (bitbsr.rows == 0 || bitbsr.num_blocks == 0 || layout.work_items.empty()) {
        if (iter_ms != nullptr) {
            *iter_ms = 0.0f;
        }
        return;
    }

    structural::gpu_detail::CudaEventTimer event_timer(iter_ms);
    const structural::Offset num_work_items =
        static_cast<structural::Offset>(layout.work_items.size());
    const int grid =
        structural::gpu_detail::grid_for(num_work_items,
                                         structural::gpu_detail::kSpmv8x4WarpsPerBlock);
    structural::gpu_detail::spmv_bitbsr_8x4_col_bitmap_packed_flag_load_balance_kernel<T>
        <<<grid, structural::gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
            bitbsr.rows, bitbsr.cols, num_work_items,
            thrust::raw_pointer_cast(packed_meta.data()),
            thrust::raw_pointer_cast(layout.work_items.data()),
            thrust::raw_pointer_cast(bitbsr.values.data()),
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()));
    structural::cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
    if (iter_ms != nullptr) {
        event_timer.stop();
    } else {
        structural::cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__,
                               __LINE__);
    }
}

template <typename T>
__global__ void spmv_csr_bitbsr_fused_8x4_lb_kernel(
    structural::Index rows,
    structural::Index cols,
    structural::Offset num_work_items,
    const structural::Offset* __restrict__ block_row_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const structural::BitBsrWorkItem8x4* __restrict__ work_items,
    const T* __restrict__ bitbsr_values,
    const structural::Offset* __restrict__ residual_row_ptr,
    const structural::Index* __restrict__ residual_col_idx,
    const T* __restrict__ residual_values,
    const T* __restrict__ x,
    T* __restrict__ y,
    int long_row_tau) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const structural::Offset work_id =
        static_cast<structural::Offset>(blockIdx.x *
                                            structural::gpu_detail::kSpmv8x4WarpsPerBlock +
                                        warp_id);
    if (work_id >= num_work_items) {
        return;
    }

    const structural::BitBsrWorkItem8x4 work = work_items[work_id];
    const structural::Index br = static_cast<structural::Index>(work.br);
    const structural::Offset block_begin = static_cast<structural::Offset>(work.block_begin);
    const structural::Offset block_count = static_cast<structural::Offset>(work.block_count);
    const bool has_bitbsr_chunk = block_count > 0;
    const bool atomic_output =
        has_bitbsr_chunk &&
        structural::gpu_detail::unpack_col_bitmap_flag(packed_meta[block_begin]);
    const bool first_work_item = block_begin == block_row_ptr[br];

    const int local_row = lane / structural::gpu_detail::kSpmv8x4BlockCols;
    const int local_col = lane % structural::gpu_detail::kSpmv8x4BlockCols;
    const structural::Index row =
        br * structural::gpu_detail::kSpmv8x4BlockRows + local_row;
    const bool valid_row = row < rows;

    T sum = T{0};
    structural::Offset value_base = static_cast<structural::Offset>(work.value_begin);
    for (structural::Offset i = 0; i < block_count; ++i) {
        const structural::Offset block = block_begin + i;
        const std::uint64_t meta = packed_meta[block];
        const structural::Index bc =
            static_cast<structural::Index>(structural::gpu_detail::unpack_abs_col(meta));
        const structural::BitmapWord bitmap = structural::gpu_detail::unpack_bitmap(meta);
        const structural::Index col =
            bc * structural::gpu_detail::kSpmv8x4BlockCols + local_col;
        const int bit = local_row * structural::gpu_detail::kSpmv8x4BlockCols + local_col;

        if (valid_row && col < cols &&
            ((bitmap & (structural::BitmapWord{1} << bit)) != 0U)) {
            const structural::BitmapWord before =
                bitmap & ((structural::BitmapWord{1} << bit) - 1U);
            const structural::Offset rank = static_cast<structural::Offset>(__popc(before));
            sum += bitbsr_values[value_base + rank] * x[col];
        }
        value_base += static_cast<structural::Offset>(__popc(bitmap));
    }

    if (first_work_item && valid_row) {
        const structural::Offset row_begin = residual_row_ptr[row];
        const structural::Offset row_end = residual_row_ptr[row + 1];
        const bool is_long_row =
            long_row_tau > 0 &&
            (row_end - row_begin) > static_cast<structural::Offset>(long_row_tau);
        if (!is_long_row) {
            for (structural::Offset p = row_begin + local_col;
                 p < row_end;
                 p += structural::gpu_detail::kSpmv8x4BlockCols) {
                sum += residual_values[p] * x[residual_col_idx[p]];
            }
        }
    }

    const unsigned row_mask =
        0xFU << (local_row * structural::gpu_detail::kSpmv8x4BlockCols);
#pragma unroll
    for (int offset = structural::gpu_detail::kSpmv8x4BlockCols / 2;
         offset > 0;
         offset /= 2) {
        sum += __shfl_down_sync(row_mask, sum, offset,
                                structural::gpu_detail::kSpmv8x4BlockCols);
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
void make_fused_8x4_load_balance_layout_host(
    const structural::BitBsrMatrix<T>& bitbsr,
    structural::BitBsrLoadBalanceLayout8x4& layout,
    thrust::device_vector<std::uint64_t>& packed_meta,
    int chunk_blocks,
    int split_threshold_blocks,
    float* device_ms = nullptr) {
    bitbsr.validate();
    if (bitbsr.block_rows != structural::gpu_detail::kSpmv8x4BlockRows ||
        bitbsr.block_cols != structural::gpu_detail::kSpmv8x4BlockCols ||
        bitbsr.words_per_block != 1) {
        throw std::runtime_error("fused CSR+BitBSR GPU probe currently supports only 8x4 blocks");
    }
    if (chunk_blocks <= 0 || split_threshold_blocks <= 0) {
        throw std::runtime_error("fused CSR+BitBSR load-balance parameters must be positive");
    }

    layout.chunk_blocks = chunk_blocks;
    layout.split_threshold_blocks = split_threshold_blocks;
    std::vector<structural::Offset> host_item_row_ptr(
        static_cast<size_t>(bitbsr.num_block_rows) + 1, 0);
    std::vector<structural::BitBsrWorkItem8x4> host_items;
    host_items.reserve(static_cast<size_t>(bitbsr.num_block_rows) +
                       static_cast<size_t>(bitbsr.num_blocks()));
    std::vector<std::uint64_t> host_packed(static_cast<size_t>(bitbsr.num_blocks()));

    for (structural::Offset block = 0; block < bitbsr.num_blocks(); ++block) {
        host_packed[static_cast<size_t>(block)] =
            structural::gpu_detail::pack_col_bitmap(
                static_cast<std::uint32_t>(bitbsr.block_col_idx[block]),
                bitbsr.bitmap_words[block]);
    }

    for (structural::Index br = 0; br < bitbsr.num_block_rows; ++br) {
        host_item_row_ptr[static_cast<size_t>(br)] =
            static_cast<structural::Offset>(host_items.size());
        const structural::Offset row_start = bitbsr.block_row_ptr[br];
        const structural::Offset row_end = bitbsr.block_row_ptr[br + 1];
        const structural::Offset block_count = row_end - row_start;
        const structural::Offset value_begin = bitbsr.block_val_ptr[row_start];

        if (block_count == 0) {
            structural::BitBsrWorkItem8x4 item;
            item.br = static_cast<std::uint32_t>(br);
            item.block_begin = static_cast<std::uint32_t>(row_start);
            item.block_count = 0;
            item.value_begin = static_cast<std::uint32_t>(value_begin);
            host_items.push_back(item);
            continue;
        }

        const bool atomic_output = block_count > static_cast<structural::Offset>(
                                                   split_threshold_blocks);
        const structural::Offset effective_chunk =
            atomic_output ? static_cast<structural::Offset>(chunk_blocks) : block_count;
        structural::Offset block = row_start;
        structural::Offset value_pos = value_begin;
        while (block < row_end) {
            const structural::Offset count =
                std::min(effective_chunk, static_cast<structural::Offset>(row_end - block));
            structural::BitBsrWorkItem8x4 item;
            item.br = static_cast<std::uint32_t>(br);
            item.block_begin = static_cast<std::uint32_t>(block);
            item.block_count = static_cast<std::uint32_t>(count);
            item.value_begin = static_cast<std::uint32_t>(value_pos);
            host_items.push_back(item);

            if (atomic_output) {
                host_packed[static_cast<size_t>(block)] |=
                    structural::gpu_detail::kSpmv8x4PackedFlagBit;
            }
            for (structural::Offset i = 0; i < count; ++i) {
                value_pos += structural::popcount_word(
                    bitbsr.bitmap_words[static_cast<size_t>(block + i)]);
            }
            block += count;
        }
    }
    host_item_row_ptr[static_cast<size_t>(bitbsr.num_block_rows)] =
        static_cast<structural::Offset>(host_items.size());

    layout.work_item_row_ptr = host_item_row_ptr;
    layout.work_items = host_items;
    packed_meta = host_packed;
    if (device_ms != nullptr) {
        *device_ms = 0.0f;
    }
}

// Balanced pass for the few power-law-long residual rows: one warp per fixed-size
// chunk of a long row, warp-strided sum, atomicAdd into y. Removes the long tail
// that the 4-lanes-per-row fused residual path causes on FullChip/circuit5M/etc.
template <typename T>
__global__ void spmv_csr_long_row_chunks_kernel(
    structural::Offset num_chunks,
    const structural::Index* __restrict__ chunk_row,
    const structural::Offset* __restrict__ chunk_pbegin,
    const structural::Index* __restrict__ chunk_pcount,
    const structural::Index* __restrict__ residual_col_idx,
    const T* __restrict__ residual_values,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int lane = threadIdx.x & 31;
    const structural::Offset chunk = static_cast<structural::Offset>(
        blockIdx.x * structural::gpu_detail::kSpmv8x4WarpsPerBlock + threadIdx.x / 32);
    if (chunk >= num_chunks) {
        return;
    }
    const structural::Index row = chunk_row[chunk];
    const structural::Offset pb = chunk_pbegin[chunk];
    const structural::Index pc = chunk_pcount[chunk];
    T sum = T{0};
    for (structural::Index k = lane; k < pc; k += 32) {
        sum += residual_values[pb + k] * x[residual_col_idx[pb + k]];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) {
        atomicAdd(&y[row], sum);
    }
}

// Unified single-launch kernel: blocks [0, bitbsr_blocks) run the BitBSR +
// short-residual path; blocks [bitbsr_blocks, ...) run long residual-row chunks.
// One launch, block-level boundary, both work-domains run concurrently. When
// long chunks exist all y updates are atomic (atomic_all), since the two domains
// can write the same row concurrently.
template <typename T>
__global__ void spmv_csr_bitbsr_unified_8x4_lb_kernel(
    structural::Index rows, structural::Index cols,
    structural::Offset num_work_items, structural::Offset num_chunks,
    int bitbsr_blocks, int long_row_tau,
    const structural::Offset* __restrict__ block_row_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const structural::BitBsrWorkItem8x4* __restrict__ work_items,
    const T* __restrict__ bitbsr_values,
    const structural::Offset* __restrict__ residual_row_ptr,
    const structural::Index* __restrict__ residual_col_idx,
    const T* __restrict__ residual_values,
    const structural::Index* __restrict__ chunk_row,
    const structural::Offset* __restrict__ chunk_pbegin,
    const structural::Index* __restrict__ chunk_pcount,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const bool atomic_all = num_chunks > 0;

    if (blockIdx.x < static_cast<unsigned int>(bitbsr_blocks)) {
        const structural::Offset work_id = static_cast<structural::Offset>(
            blockIdx.x * structural::gpu_detail::kSpmv8x4WarpsPerBlock + warp_id);
        if (work_id >= num_work_items) {
            return;
        }
        const structural::BitBsrWorkItem8x4 work = work_items[work_id];
        const structural::Index br = static_cast<structural::Index>(work.br);
        const structural::Offset block_begin = static_cast<structural::Offset>(work.block_begin);
        const structural::Offset block_count = static_cast<structural::Offset>(work.block_count);
        const bool has_bitbsr_chunk = block_count > 0;
        const bool atomic_output =
            has_bitbsr_chunk &&
            structural::gpu_detail::unpack_col_bitmap_flag(packed_meta[block_begin]);
        const bool first_work_item = block_begin == block_row_ptr[br];

        const int local_row = lane / structural::gpu_detail::kSpmv8x4BlockCols;
        const int local_col = lane % structural::gpu_detail::kSpmv8x4BlockCols;
        const structural::Index row =
            br * structural::gpu_detail::kSpmv8x4BlockRows + local_row;
        const bool valid_row = row < rows;

        T sum = T{0};
        structural::Offset value_base = static_cast<structural::Offset>(work.value_begin);
        for (structural::Offset i = 0; i < block_count; ++i) {
            const structural::Offset block = block_begin + i;
            const std::uint64_t meta = packed_meta[block];
            const structural::Index bc =
                static_cast<structural::Index>(structural::gpu_detail::unpack_abs_col(meta));
            const structural::BitmapWord bitmap = structural::gpu_detail::unpack_bitmap(meta);
            const structural::Index col =
                bc * structural::gpu_detail::kSpmv8x4BlockCols + local_col;
            const int bit = local_row * structural::gpu_detail::kSpmv8x4BlockCols + local_col;
            if (valid_row && col < cols &&
                ((bitmap & (structural::BitmapWord{1} << bit)) != 0U)) {
                const structural::BitmapWord before =
                    bitmap & ((structural::BitmapWord{1} << bit) - 1U);
                const structural::Offset rank = static_cast<structural::Offset>(__popc(before));
                sum += bitbsr_values[value_base + rank] * x[col];
            }
            value_base += static_cast<structural::Offset>(__popc(bitmap));
        }

        if (first_work_item && valid_row) {
            const structural::Offset row_begin = residual_row_ptr[row];
            const structural::Offset row_end = residual_row_ptr[row + 1];
            const bool is_long_row =
                long_row_tau > 0 &&
                (row_end - row_begin) > static_cast<structural::Offset>(long_row_tau);
            if (!is_long_row) {
                for (structural::Offset p = row_begin + local_col;
                     p < row_end;
                     p += structural::gpu_detail::kSpmv8x4BlockCols) {
                    sum += residual_values[p] * x[residual_col_idx[p]];
                }
            }
        }

        const unsigned row_mask =
            0xFU << (local_row * structural::gpu_detail::kSpmv8x4BlockCols);
#pragma unroll
        for (int offset = structural::gpu_detail::kSpmv8x4BlockCols / 2; offset > 0;
             offset /= 2) {
            sum += __shfl_down_sync(row_mask, sum, offset,
                                    structural::gpu_detail::kSpmv8x4BlockCols);
        }
        if (valid_row && local_col == 0) {
            if (atomic_output || atomic_all) {
                atomicAdd(&y[row], sum);
            } else {
                y[row] = sum;
            }
        }
    } else {
        const structural::Offset chunk = static_cast<structural::Offset>(
            (blockIdx.x - static_cast<unsigned int>(bitbsr_blocks)) *
                structural::gpu_detail::kSpmv8x4WarpsPerBlock +
            warp_id);
        if (chunk >= num_chunks) {
            return;
        }
        const structural::Index row = chunk_row[chunk];
        const structural::Offset pb = chunk_pbegin[chunk];
        const structural::Index pc = chunk_pcount[chunk];
        T sum = T{0};
        for (structural::Index k = lane; k < pc; k += 32) {
            sum += residual_values[pb + k] * x[residual_col_idx[pb + k]];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) {
            atomicAdd(&y[row], sum);
        }
    }
}

template <typename T>
void run_csr_bitbsr_fused_8x4_lb(
    const structural::DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const structural::BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<structural::Offset>& residual_row_ptr,
    const thrust::device_vector<structural::Index>& residual_col_idx,
    const thrust::device_vector<T>& residual_values,
    int long_row_tau,
    const thrust::device_vector<structural::Index>& long_chunk_row,
    const thrust::device_vector<structural::Offset>& long_chunk_pbegin,
    const thrust::device_vector<structural::Index>& long_chunk_pcount,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* iter_ms = nullptr) {
    structural::gpu_detail::validate_spmv_8x4_shape(bitbsr, x);
    if (static_cast<structural::Offset>(packed_meta.size()) != bitbsr.num_blocks) {
        throw std::runtime_error("fused CSR+BitBSR packed metadata size mismatch");
    }
    if (static_cast<structural::Index>(layout.work_item_row_ptr.size()) !=
        bitbsr.num_block_rows + 1) {
        throw std::runtime_error("fused CSR+BitBSR work_item_row_ptr size mismatch");
    }
    if (static_cast<structural::Index>(y.size()) != bitbsr.rows) {
        throw std::runtime_error("fused CSR+BitBSR output vector size mismatch");
    }
    if (static_cast<structural::Index>(residual_row_ptr.size()) != bitbsr.rows + 1) {
        throw std::runtime_error("fused CSR+BitBSR residual row_ptr size mismatch");
    }

    thrust::fill(y.begin(), y.end(), T{0});
    if (bitbsr.rows == 0 || layout.work_items.empty()) {
        if (iter_ms != nullptr) {
            *iter_ms = 0.0f;
        }
        return;
    }

    structural::gpu_detail::CudaEventTimer event_timer(iter_ms);
    const structural::Offset num_work_items =
        static_cast<structural::Offset>(layout.work_items.size());
    const structural::Offset num_chunks =
        static_cast<structural::Offset>(long_chunk_row.size());
    const int bitbsr_blocks = structural::gpu_detail::grid_for(
        num_work_items, structural::gpu_detail::kSpmv8x4WarpsPerBlock);
    const int chunk_blocks =
        num_chunks > 0
            ? structural::gpu_detail::grid_for(num_chunks,
                                               structural::gpu_detail::kSpmv8x4WarpsPerBlock)
            : 0;
    const int total_blocks = bitbsr_blocks + chunk_blocks;
    spmv_csr_bitbsr_unified_8x4_lb_kernel<T>
        <<<total_blocks, structural::gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
            bitbsr.rows, bitbsr.cols, num_work_items, num_chunks, bitbsr_blocks,
            long_row_tau,
            thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
            thrust::raw_pointer_cast(packed_meta.data()),
            thrust::raw_pointer_cast(layout.work_items.data()),
            thrust::raw_pointer_cast(bitbsr.values.data()),
            thrust::raw_pointer_cast(residual_row_ptr.data()),
            residual_col_idx.empty() ? nullptr : thrust::raw_pointer_cast(residual_col_idx.data()),
            residual_values.empty() ? nullptr : thrust::raw_pointer_cast(residual_values.data()),
            long_chunk_row.empty() ? nullptr : thrust::raw_pointer_cast(long_chunk_row.data()),
            long_chunk_pbegin.empty() ? nullptr
                                      : thrust::raw_pointer_cast(long_chunk_pbegin.data()),
            long_chunk_pcount.empty() ? nullptr
                                      : thrust::raw_pointer_cast(long_chunk_pcount.data()),
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()));
    structural::cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
    if (iter_ms != nullptr) {
        event_timer.stop();
    } else {
        structural::cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__,
                               __LINE__);
    }
}

// ============================================================================
// JOINT operator: hybrid STRUCTURE (unified fused-LB: dense bitbsr blocks + short
// residual + long-row chunks, single launch) + value QUANTIZATION (bitbsr blocks →
// BFP8 = int8 mantissa × per-block bf16 scale; CSR residual incl. long rows → bf16),
// fp32 accumulate. This is the operator the title promises: position AND precision
// in ONE kernel. Mirrors spmv_csr_bitbsr_unified_8x4_lb_kernel exactly, value loads
// replaced by decode.
// ============================================================================
__device__ __forceinline__ float cove_dec_bf16(std::uint16_t b) {
    return __uint_as_float(static_cast<std::uint32_t>(b) << 16);
}

struct CoveBfp8Decoder {  // lossy arm: int8 mantissa x per-block bf16 scale; residual = bf16
    const signed char* int8;
    const std::uint16_t* scale;
    const std::uint16_t* res_bf16;
    __device__ __forceinline__ float block(structural::Offset vr, structural::Offset blk) const {
        return static_cast<float>(int8[vr]) * cove_dec_bf16(scale[blk]);
    }
    __device__ __forceinline__ float residual(structural::Offset p) const {
        return cove_dec_bf16(res_bf16[p]);
    }
};

template <typename IdxT, typename T>
struct CoveDictDecoder {  // lossless arm: per-nnz codebook index -> global value table
    const IdxT* blk_idx;
    const T* codebook;
    const IdxT* res_idx;
    __device__ __forceinline__ float block(structural::Offset vr, structural::Offset) const {
        return static_cast<float>(codebook[blk_idx[vr]]);
    }
    __device__ __forceinline__ float residual(structural::Offset p) const {
        return static_cast<float>(codebook[res_idx[p]]);
    }
};

// Single quantized JOINT kernel, templated on a value Decoder (Bfp8 / Dict). Hybrid
// fused-LB structure (dense bitbsr blocks + short residual + long-row chunks); value
// loads go through dec.block()/dec.residual(), fp32 accumulate. One kernel for every
// value arm -- the arms differ only in the Decoder.
template <typename T, typename Decoder>
__global__ void spmv_csr_bitbsr_unified_8x4_lb_quant_kernel(
    structural::Index rows, structural::Index cols,
    structural::Offset num_work_items, structural::Offset num_chunks,
    int bitbsr_blocks, int long_row_tau,
    const structural::Offset* __restrict__ block_row_ptr,
    const std::uint64_t* __restrict__ packed_meta,
    const structural::BitBsrWorkItem8x4* __restrict__ work_items,
    Decoder dec,
    const structural::Offset* __restrict__ residual_row_ptr,
    const structural::Index* __restrict__ residual_col_idx,
    const structural::Index* __restrict__ chunk_row,
    const structural::Offset* __restrict__ chunk_pbegin,
    const structural::Index* __restrict__ chunk_pcount,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const bool atomic_all = num_chunks > 0;

    if (blockIdx.x < static_cast<unsigned int>(bitbsr_blocks)) {
        const structural::Offset work_id = static_cast<structural::Offset>(
            blockIdx.x * structural::gpu_detail::kSpmv8x4WarpsPerBlock + warp_id);
        if (work_id >= num_work_items) return;
        const structural::BitBsrWorkItem8x4 work = work_items[work_id];
        const structural::Index br = static_cast<structural::Index>(work.br);
        const structural::Offset block_begin = static_cast<structural::Offset>(work.block_begin);
        const structural::Offset block_count = static_cast<structural::Offset>(work.block_count);
        const bool has_bitbsr_chunk = block_count > 0;
        const bool atomic_output =
            has_bitbsr_chunk &&
            structural::gpu_detail::unpack_col_bitmap_flag(packed_meta[block_begin]);
        const bool first_work_item = block_begin == block_row_ptr[br];

        const int local_row = lane / structural::gpu_detail::kSpmv8x4BlockCols;
        const int local_col = lane % structural::gpu_detail::kSpmv8x4BlockCols;
        const structural::Index row =
            br * structural::gpu_detail::kSpmv8x4BlockRows + local_row;
        const bool valid_row = row < rows;

        float sum = 0.0f;
        structural::Offset value_base = static_cast<structural::Offset>(work.value_begin);
        for (structural::Offset i = 0; i < block_count; ++i) {
            const structural::Offset block = block_begin + i;
            const std::uint64_t meta = packed_meta[block];
            const structural::Index bc =
                static_cast<structural::Index>(structural::gpu_detail::unpack_abs_col(meta));
            const structural::BitmapWord bitmap = structural::gpu_detail::unpack_bitmap(meta);
            const structural::Index col =
                bc * structural::gpu_detail::kSpmv8x4BlockCols + local_col;
            const int bit = local_row * structural::gpu_detail::kSpmv8x4BlockCols + local_col;
            if (valid_row && col < cols &&
                ((bitmap & (structural::BitmapWord{1} << bit)) != 0U)) {
                const structural::BitmapWord before =
                    bitmap & ((structural::BitmapWord{1} << bit) - 1U);
                const structural::Offset rank = static_cast<structural::Offset>(__popc(before));
                sum += dec.block(value_base + rank, block) * static_cast<float>(x[col]);
            }
            value_base += static_cast<structural::Offset>(__popc(bitmap));
        }

        if (first_work_item && valid_row) {
            const structural::Offset row_begin = residual_row_ptr[row];
            const structural::Offset row_end = residual_row_ptr[row + 1];
            const bool is_long_row =
                long_row_tau > 0 &&
                (row_end - row_begin) > static_cast<structural::Offset>(long_row_tau);
            if (!is_long_row) {
                for (structural::Offset p = row_begin + local_col; p < row_end;
                     p += structural::gpu_detail::kSpmv8x4BlockCols) {
                    sum += dec.residual(p) * static_cast<float>(x[residual_col_idx[p]]);
                }
            }
        }

        const unsigned row_mask =
            0xFU << (local_row * structural::gpu_detail::kSpmv8x4BlockCols);
#pragma unroll
        for (int offset = structural::gpu_detail::kSpmv8x4BlockCols / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(row_mask, sum, offset,
                                    structural::gpu_detail::kSpmv8x4BlockCols);
        }
        if (valid_row && local_col == 0) {
            if (atomic_output || atomic_all) atomicAdd(&y[row], static_cast<T>(sum));
            else y[row] = static_cast<T>(sum);
        }
    } else {
        const structural::Offset chunk = static_cast<structural::Offset>(
            (blockIdx.x - static_cast<unsigned int>(bitbsr_blocks)) *
                structural::gpu_detail::kSpmv8x4WarpsPerBlock +
            warp_id);
        if (chunk >= num_chunks) return;
        const structural::Index row = chunk_row[chunk];
        const structural::Offset pb = chunk_pbegin[chunk];
        const structural::Index pc = chunk_pcount[chunk];
        float sum = 0.0f;
        for (structural::Index k = lane; k < pc; k += 32) {
            sum += dec.residual(pb + k) * static_cast<float>(x[residual_col_idx[pb + k]]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) atomicAdd(&y[row], static_cast<T>(sum));
    }
}

// Grid setup + launch for the unified quant kernel with a given Decoder.
template <typename T, typename Decoder>
void launch_quant_kernel(const structural::DeviceBitBsrMatrix<T>& bitbsr,
                         const thrust::device_vector<std::uint64_t>& packed_meta,
                         const structural::BitBsrLoadBalanceLayout8x4& layout, Decoder dec,
                         const thrust::device_vector<structural::Offset>& residual_row_ptr,
                         const thrust::device_vector<structural::Index>& residual_col_idx,
                         int long_row_tau,
                         const thrust::device_vector<structural::Index>& long_chunk_row,
                         const thrust::device_vector<structural::Offset>& long_chunk_pbegin,
                         const thrust::device_vector<structural::Index>& long_chunk_pcount,
                         const thrust::device_vector<T>& x, thrust::device_vector<T>& y) {
    const structural::Offset num_work_items =
        static_cast<structural::Offset>(layout.work_items.size());
    const structural::Offset num_chunks =
        static_cast<structural::Offset>(long_chunk_row.size());
    const int bitbsr_blocks = structural::gpu_detail::grid_for(
        num_work_items, structural::gpu_detail::kSpmv8x4WarpsPerBlock);
    const int chunk_blocks =
        num_chunks > 0 ? structural::gpu_detail::grid_for(
                             num_chunks, structural::gpu_detail::kSpmv8x4WarpsPerBlock)
                       : 0;
    spmv_csr_bitbsr_unified_8x4_lb_quant_kernel<T, Decoder>
        <<<bitbsr_blocks + chunk_blocks, structural::gpu_detail::kSpmv8x4ThreadsPerBlock>>>(
            bitbsr.rows, bitbsr.cols, num_work_items, num_chunks, bitbsr_blocks, long_row_tau,
            thrust::raw_pointer_cast(bitbsr.block_row_ptr.data()),
            thrust::raw_pointer_cast(packed_meta.data()),
            thrust::raw_pointer_cast(layout.work_items.data()), dec,
            thrust::raw_pointer_cast(residual_row_ptr.data()),
            residual_col_idx.empty() ? nullptr : thrust::raw_pointer_cast(residual_col_idx.data()),
            long_chunk_row.empty() ? nullptr : thrust::raw_pointer_cast(long_chunk_row.data()),
            long_chunk_pbegin.empty() ? nullptr : thrust::raw_pointer_cast(long_chunk_pbegin.data()),
            long_chunk_pcount.empty() ? nullptr : thrust::raw_pointer_cast(long_chunk_pcount.data()),
            thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()));
    structural::cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
}

template <typename T>
void run_csr_bitbsr_fused_8x4_lb_bfp8(
    const structural::DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const structural::BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<structural::Offset>& residual_row_ptr,
    const thrust::device_vector<structural::Index>& residual_col_idx,
    const thrust::device_vector<signed char>& blk_int8,
    const thrust::device_vector<std::uint16_t>& blk_scale_bf16,
    const thrust::device_vector<std::uint16_t>& residual_bf16,
    int long_row_tau,
    const thrust::device_vector<structural::Index>& long_chunk_row,
    const thrust::device_vector<structural::Offset>& long_chunk_pbegin,
    const thrust::device_vector<structural::Index>& long_chunk_pcount,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* iter_ms = nullptr) {
    thrust::fill(y.begin(), y.end(), T{0});
    if (bitbsr.rows == 0 || layout.work_items.empty()) {
        if (iter_ms != nullptr) *iter_ms = 0.0f;
        return;
    }
    structural::gpu_detail::CudaEventTimer event_timer(iter_ms);
    CoveBfp8Decoder dec{thrust::raw_pointer_cast(blk_int8.data()),
                        thrust::raw_pointer_cast(blk_scale_bf16.data()),
                        residual_bf16.empty() ? nullptr
                                              : thrust::raw_pointer_cast(residual_bf16.data())};
    launch_quant_kernel<T>(bitbsr, packed_meta, layout, dec, residual_row_ptr, residual_col_idx,
                           long_row_tau, long_chunk_row, long_chunk_pbegin, long_chunk_pcount, x, y);
    if (iter_ms != nullptr) event_timer.stop();
    else structural::cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
}

template <typename T, typename IdxT>
void run_csr_bitbsr_fused_8x4_lb_dict(
    const structural::DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const structural::BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<structural::Offset>& residual_row_ptr,
    const thrust::device_vector<structural::Index>& residual_col_idx,
    const thrust::device_vector<IdxT>& blk_idx,
    const thrust::device_vector<T>& codebook,
    const thrust::device_vector<IdxT>& residual_idx,
    int long_row_tau,
    const thrust::device_vector<structural::Index>& long_chunk_row,
    const thrust::device_vector<structural::Offset>& long_chunk_pbegin,
    const thrust::device_vector<structural::Index>& long_chunk_pcount,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* iter_ms = nullptr) {
    thrust::fill(y.begin(), y.end(), T{0});
    if (bitbsr.rows == 0 || layout.work_items.empty()) {
        if (iter_ms != nullptr) *iter_ms = 0.0f;
        return;
    }
    structural::gpu_detail::CudaEventTimer event_timer(iter_ms);
    CoveDictDecoder<IdxT, T> dec{thrust::raw_pointer_cast(blk_idx.data()),
                                 thrust::raw_pointer_cast(codebook.data()),
                                 residual_idx.empty() ? nullptr
                                                      : thrust::raw_pointer_cast(residual_idx.data())};
    launch_quant_kernel<T>(bitbsr, packed_meta, layout, dec, residual_row_ptr, residual_col_idx,
                           long_row_tau, long_chunk_row, long_chunk_pbegin, long_chunk_pcount, x, y);
    if (iter_ms != nullptr) event_timer.stop();
    else structural::cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
}

// Exact correction for the global top-P% magnitude outliers (extracted from the block
// values and ZEROED in the BFP8 int8): one thread per outlier, exact fp64 atomicAdd.
template <typename T>
__global__ void cove_outlier_correction_kernel(
    structural::Offset num_outliers,
    const std::int32_t* __restrict__ orows,
    const std::int32_t* __restrict__ ocols,
    const T* __restrict__ ovals,
    const T* __restrict__ x,
    T* __restrict__ y) {
    const structural::Offset i =
        static_cast<structural::Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_outliers) return;
    atomicAdd(&y[orows[i]], ovals[i] * x[ocols[i]]);
}

// JOINT + outlier: bulk = bfp8 blocks (outliers zeroed in int8) + bf16 residual, then a
// tiny exact atomicAdd correction over the global top-P% magnitude side-list. Accuracy
// lever for the high-error tail (e.g. F1); +~0.04 B/nnz. Both passes under one timer.
template <typename T>
void run_csr_bitbsr_fused_8x4_lb_bfp8_outlier(
    const structural::DeviceBitBsrMatrix<T>& bitbsr,
    const thrust::device_vector<std::uint64_t>& packed_meta,
    const structural::BitBsrLoadBalanceLayout8x4& layout,
    const thrust::device_vector<structural::Offset>& residual_row_ptr,
    const thrust::device_vector<structural::Index>& residual_col_idx,
    const thrust::device_vector<signed char>& blk_int8,
    const thrust::device_vector<std::uint16_t>& blk_scale_bf16,
    const thrust::device_vector<std::uint16_t>& residual_bf16,
    const thrust::device_vector<std::int32_t>& outlier_rows,
    const thrust::device_vector<std::int32_t>& outlier_cols,
    const thrust::device_vector<T>& outlier_vals,
    int long_row_tau,
    const thrust::device_vector<structural::Index>& long_chunk_row,
    const thrust::device_vector<structural::Offset>& long_chunk_pbegin,
    const thrust::device_vector<structural::Index>& long_chunk_pcount,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& y,
    float* iter_ms = nullptr) {
    thrust::fill(y.begin(), y.end(), T{0});
    if (bitbsr.rows == 0 || layout.work_items.empty()) {
        if (iter_ms != nullptr) *iter_ms = 0.0f;
        return;
    }
    structural::gpu_detail::CudaEventTimer event_timer(iter_ms);
    CoveBfp8Decoder dec{thrust::raw_pointer_cast(blk_int8.data()),
                        thrust::raw_pointer_cast(blk_scale_bf16.data()),
                        residual_bf16.empty() ? nullptr
                                              : thrust::raw_pointer_cast(residual_bf16.data())};
    launch_quant_kernel<T>(bitbsr, packed_meta, layout, dec, residual_row_ptr, residual_col_idx,
                           long_row_tau, long_chunk_row, long_chunk_pbegin, long_chunk_pcount, x, y);
    const structural::Offset num_out = static_cast<structural::Offset>(outlier_vals.size());
    if (num_out > 0) {
        const int tpb = 256;
        const int blocks = static_cast<int>((num_out + tpb - 1) / tpb);
        cove_outlier_correction_kernel<T><<<blocks, tpb>>>(
            num_out, thrust::raw_pointer_cast(outlier_rows.data()),
            thrust::raw_pointer_cast(outlier_cols.data()),
            thrust::raw_pointer_cast(outlier_vals.data()), thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()));
        structural::cuda_check(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__);
    }
    if (iter_ms != nullptr) event_timer.stop();
    else structural::cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, __LINE__);
}


template <typename T>
void run_cusparse_residual_spmv(cusparseHandle_t handle,
                                cusparseConstSpMatDescr_t mat_a,
                                cusparseConstDnVecDescr_t vec_x,
                                cusparseDnVecDescr_t vec_y,
                                cusparseSpMVAlg_t alg,
                                void* external_buffer,
                                float* iter_ms = nullptr) {
    const T alpha = T{1};
    const T beta = T{1};
    structural::gpu_detail::CudaEventTimer timer(iter_ms);
    STRUCTURAL_CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                           mat_a, vec_x, &beta, vec_y, cuda_data_type<T>(),
                                           alg, external_buffer));
    if (iter_ms != nullptr) {
        timer.stop();
    }
}

template <typename T>
int run(const Options& opts) {
    double load_ms = 0.0;
    double cuda_context_ms = 0.0;
    double host_split_ms = 0.0;
    double residual_format_convert_host_ms = 0.0;
    double bitbsr_upload_host_ms = 0.0;
    double residual_upload_host_ms = 0.0;
    double col_bitmap_pack_host_ms = 0.0;
    double load_balance_pack_host_ms = 0.0;
    double cusparse_setup_host_ms = 0.0;
    double cusparse_preprocess_host_ms = 0.0;
    double cpu_spmv_ms = 0.0;
    float col_bitmap_pack_event_ms = 0.0f;
    float load_balance_pack_event_ms = 0.0f;
    float cusparse_preprocess_event_ms = 0.0f;

    const auto loaded = time_call(
        [&] { return structural::read_matrix_market_with_info<T>(opts.matrix_path); },
        load_ms);

    const int cuda_context_status = time_call(
        [&] {
            structural::cuda_check(cudaFree(nullptr), "cudaFree(nullptr)", __FILE__, __LINE__);
            return 0;
        },
        cuda_context_ms);
    (void)cuda_context_status;

    structural::BitBsrMatrix<T> full_bitbsr;
    structural::CsrBitBsrSplitMatrix<T> split;
    structural::CsrBitBsrSplitStorageStats<T> split_stats;
    (void)time_call(
        [&] {
            full_bitbsr = structural::convert_to_bitbsr(
                loaded.matrix, opts.block_rows, opts.block_cols);
            if (opts.split_policy == "cost_gated") {
                split = structural::convert_bitbsr_to_csr_bitbsr_split_cost_gated(
                    full_bitbsr, opts.residual_threshold, opts.cost_model);
            } else {
                split = structural::convert_bitbsr_to_csr_bitbsr_split(
                    full_bitbsr, opts.residual_threshold);
            }
            split_stats = structural::storage_stats(full_bitbsr, split);
            return 0;
        },
        host_split_ms);

    structural::DeviceBitBsrMatrix<T> device_bitbsr;
    (void)time_call(
        [&] {
            structural::copy_bitbsr_to_device(split.bitbsr, device_bitbsr);
            return 0;
        },
        bitbsr_upload_host_ms);

    const auto x = make_input_vector<T>(loaded.matrix.cols);
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y(static_cast<size_t>(loaded.matrix.rows), T{0});
    thrust::device_vector<std::uint64_t> packed_col_bitmap_meta;
    structural::BitBsrLoadBalanceLayout8x4 load_balance;

    if (opts.mode == "fused_lb") {
        (void)time_call(
            [&] {
                make_fused_8x4_load_balance_layout_host(
                    split.bitbsr, load_balance, packed_col_bitmap_meta,
                    opts.lb_chunk_blocks, opts.lb_split_threshold_blocks,
                    &load_balance_pack_event_ms);
                return 0;
            },
            load_balance_pack_host_ms);
    } else {
        (void)time_call(
            [&] {
                structural::make_bitbsr_8x4_col_bitmap_packed(
                    device_bitbsr, packed_col_bitmap_meta, &col_bitmap_pack_event_ms);
                return 0;
            },
            col_bitmap_pack_host_ms);
        (void)time_call(
            [&] {
                structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
                    device_bitbsr, load_balance, opts.lb_chunk_blocks,
                    opts.lb_split_threshold_blocks, packed_col_bitmap_meta,
                    &load_balance_pack_event_ms);
                return 0;
            },
            load_balance_pack_host_ms);
    }

    thrust::device_vector<structural::Offset> d_residual_row_ptr;
    thrust::device_vector<structural::Index> d_residual_col_idx;
    thrust::device_vector<T> d_residual_values;

    (void)time_call(
        [&] {
            d_residual_row_ptr = split.residual.row_ptr;
            d_residual_col_idx = split.residual.col_idx;
            d_residual_values = split.residual.values;
            return 0;
        },
        residual_upload_host_ms);

    thrust::device_vector<structural::Index> d_long_chunk_row;
    thrust::device_vector<structural::Offset> d_long_chunk_pbegin;
    thrust::device_vector<structural::Index> d_long_chunk_pcount;
    if (opts.mode == "fused_lb" && opts.residual_format == "csr" &&
        opts.long_row_split > 0) {
        const structural::Offset chunk_nnz = 1024;
        const auto& rp = split.residual.row_ptr;
        std::vector<structural::Index> h_row;
        std::vector<structural::Offset> h_pbegin;
        std::vector<structural::Index> h_pcount;
        int n_long_rows = 0;
        for (structural::Index r = 0; r < split.residual.rows; ++r) {
            const structural::Offset len = rp[r + 1] - rp[r];
            if (len > static_cast<structural::Offset>(opts.long_row_split)) {
                ++n_long_rows;
                for (structural::Offset off = 0; off < len; off += chunk_nnz) {
                    h_row.push_back(r);
                    h_pbegin.push_back(rp[r] + off);
                    h_pcount.push_back(static_cast<structural::Index>(
                        std::min<structural::Offset>(chunk_nnz, len - off)));
                }
            }
        }
        d_long_chunk_row = h_row;
        d_long_chunk_pbegin = h_pbegin;
        d_long_chunk_pcount = h_pcount;
        std::cout << "long_row_tau: " << opts.long_row_split << "\n";
        std::cout << "long_rows: " << n_long_rows << "\n";
        std::cout << "long_row_chunks: " << h_row.size() << "\n";
    }



    if (opts.mode == "fused_lb") {
        for (int i = 0; i < opts.warmup; ++i) {
            run_csr_bitbsr_fused_8x4_lb(
                device_bitbsr, packed_col_bitmap_meta, load_balance,
                d_residual_row_ptr, d_residual_col_idx, d_residual_values,
                opts.long_row_split, d_long_chunk_row, d_long_chunk_pbegin,
                d_long_chunk_pcount, d_x, d_y);
        }

        double fused_sum_ms = 0.0;
        float fused_min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < opts.iterations; ++i) {
            float fused_ms = 0.0f;
            run_csr_bitbsr_fused_8x4_lb(
                device_bitbsr, packed_col_bitmap_meta, load_balance,
                d_residual_row_ptr, d_residual_col_idx, d_residual_values,
                opts.long_row_split, d_long_chunk_row, d_long_chunk_pbegin,
                d_long_chunk_pcount, d_x, d_y, &fused_ms);
            fused_sum_ms += fused_ms;
            fused_min_ms = std::min(fused_min_ms, fused_ms);
        }

        structural::ErrorStats<T> error;
        bool verified = true;
        if (opts.verify_cpu) {
            verified = verify_result(loaded.matrix, x, d_y, cpu_spmv_ms, error);
        }

        // ---- JOINT operator: UNIFIED value menu on the hybrid fused-LB structure ----
        // Arms, per-matrix: LOSSY (bfp8 = int8 blocks ×bf16 scale + bf16 residual;
        // bfp8_outlier adds an exact global top-P% magnitude correction) and LOSSLESS
        // (dict = per-nnz codebook index → global value table). `auto` picks dict when
        // value-cardinality is low (≤256 → 1B/nnz index, exact) else bfp8. A/B vs the
        // fp64 hybrid (fused_min_ms) isolates the marginal effect of quantization.
        if (opts.value_codec == "bfp8" || opts.value_codec == "bfp8_outlier" ||
            opts.value_codec == "dict" || opts.value_codec == "auto") {
            std::unordered_map<unsigned long long, int> idx_of;
            std::vector<T> codebook, res_host;
            auto vkey = [](T v) {
                unsigned long long k = 0;
                std::memcpy(&k, &v, sizeof(v));  // sizeof(v): T may be 4 bytes (float)
                return k;
            };
            std::size_t uniq = 0;
            std::string chosen = opts.value_codec;
            if (chosen == "auto" || chosen == "dict") {
                res_host.resize(d_residual_values.size());
                thrust::copy(d_residual_values.begin(), d_residual_values.end(), res_host.begin());
                auto intern = [&](T v) {
                    if (idx_of.find(vkey(v)) != idx_of.end()) return;
                    idx_of.emplace(vkey(v), static_cast<int>(codebook.size()));
                    codebook.push_back(v);
                };
                for (const T& v : split.bitbsr.values) intern(v);
                for (const T& v : res_host) intern(v);
                uniq = codebook.size();
                if (chosen == "auto") chosen = (uniq <= 256) ? "dict" : "bfp8";
                if (chosen == "dict" && uniq > 65536) chosen = "bfp8";  // index too wide
            }

            float joint_min_ms = std::numeric_limits<float>::max();
            double bpn = 0.0, jrel = -1.0;
            std::string codec_name;
            const double total_nnz = static_cast<double>(loaded.matrix.nnz());

            if (chosen == "dict") {
                std::vector<int> blk_i(split.bitbsr.values.size()), res_i(res_host.size());
                for (std::size_t i = 0; i < blk_i.size(); ++i)
                    blk_i[i] = idx_of[vkey(split.bitbsr.values[i])];
                for (std::size_t i = 0; i < res_i.size(); ++i)
                    res_i[i] = idx_of[vkey(res_host[i])];
                thrust::device_vector<T> d_cb(codebook.begin(), codebook.end());
                const double idx_b = (uniq <= 256) ? 1.0 : 2.0;
                bpn = (idx_b * (static_cast<double>(blk_i.size()) +
                                static_cast<double>(res_i.size())) +
                       8.0 * static_cast<double>(uniq)) / total_nnz;
                auto run_idx = [&](auto tag) {
                    using IdxT = decltype(tag);
                    thrust::device_vector<IdxT> d_blk(blk_i.begin(), blk_i.end());
                    thrust::device_vector<IdxT> d_res(res_i.begin(), res_i.end());
                    for (int i = 0; i < opts.warmup; ++i)
                        run_csr_bitbsr_fused_8x4_lb_dict<T, IdxT>(
                            device_bitbsr, packed_col_bitmap_meta, load_balance, d_residual_row_ptr,
                            d_residual_col_idx, d_blk, d_cb, d_res, opts.long_row_split,
                            d_long_chunk_row, d_long_chunk_pbegin, d_long_chunk_pcount, d_x, d_y);
                    for (int i = 0; i < opts.iterations; ++i) {
                        float ms = 0.0f;
                        run_csr_bitbsr_fused_8x4_lb_dict<T, IdxT>(
                            device_bitbsr, packed_col_bitmap_meta, load_balance, d_residual_row_ptr,
                            d_residual_col_idx, d_blk, d_cb, d_res, opts.long_row_split,
                            d_long_chunk_row, d_long_chunk_pbegin, d_long_chunk_pcount, d_x, d_y, &ms);
                        joint_min_ms = std::min(joint_min_ms, ms);
                    }
                };
                if (uniq <= 256) {
                    run_idx(std::uint8_t{});
                    codec_name = "dict_u8_lossless";
                } else {
                    run_idx(std::uint16_t{});
                    codec_name = "dict_u16_lossless";
                }
            } else if (chosen == "bfp8_outlier") {
                const auto oc = structural::build_checked_bfp8_outlier_blkscale_value_codec(
                    split.bitbsr, 0.005);
                thrust::device_vector<signed char> d_int8(oc.int8_values.begin(),
                                                          oc.int8_values.end());
                thrust::device_vector<std::uint16_t> d_scale(oc.block_scale_bf16.begin(),
                                                             oc.block_scale_bf16.end());
                thrust::device_vector<std::int32_t> d_or(oc.outlier_rows.begin(),
                                                         oc.outlier_rows.end());
                thrust::device_vector<std::int32_t> d_oc(oc.outlier_cols.begin(),
                                                         oc.outlier_cols.end());
                thrust::device_vector<T> d_ov(oc.outlier_vals.begin(), oc.outlier_vals.end());
                thrust::device_vector<std::uint16_t> d_res(d_residual_values.size());
                thrust::transform(d_residual_values.begin(), d_residual_values.end(), d_res.begin(),
                                  [] __device__(T v) {
                                      unsigned u = __float_as_uint(static_cast<float>(v));
                                      return static_cast<std::uint16_t>((u + 0x8000u) >> 16);
                                  });
                for (int i = 0; i < opts.warmup; ++i)
                    run_csr_bitbsr_fused_8x4_lb_bfp8_outlier(
                        device_bitbsr, packed_col_bitmap_meta, load_balance, d_residual_row_ptr,
                        d_residual_col_idx, d_int8, d_scale, d_res, d_or, d_oc, d_ov,
                        opts.long_row_split, d_long_chunk_row, d_long_chunk_pbegin,
                        d_long_chunk_pcount, d_x, d_y);
                for (int i = 0; i < opts.iterations; ++i) {
                    float ms = 0.0f;
                    run_csr_bitbsr_fused_8x4_lb_bfp8_outlier(
                        device_bitbsr, packed_col_bitmap_meta, load_balance, d_residual_row_ptr,
                        d_residual_col_idx, d_int8, d_scale, d_res, d_or, d_oc, d_ov,
                        opts.long_row_split, d_long_chunk_row, d_long_chunk_pbegin,
                        d_long_chunk_pcount, d_x, d_y, &ms);
                    joint_min_ms = std::min(joint_min_ms, ms);
                }
                bpn = (static_cast<double>(oc.int8_values.size()) +
                       2.0 * static_cast<double>(oc.block_scale_bf16.size()) +
                       2.0 * static_cast<double>(d_res.size()) +
                       16.0 * static_cast<double>(oc.outlier_vals.size())) / total_nnz;
                codec_name = "bfp8_outlier_blocks+bf16_residual";
            } else {  // bfp8 (lossy)
                const auto bfp8 = structural::build_checked_bfp8_blkscale_value_codec(split.bitbsr);
                thrust::device_vector<signed char> d_blk_int8(bfp8.int8_values.begin(),
                                                             bfp8.int8_values.end());
                thrust::device_vector<std::uint16_t> d_blk_scale(bfp8.block_scale_bf16.begin(),
                                                                 bfp8.block_scale_bf16.end());
                thrust::device_vector<std::uint16_t> d_residual_bf16(d_residual_values.size());
                thrust::transform(d_residual_values.begin(), d_residual_values.end(),
                                  d_residual_bf16.begin(), [] __device__(T v) {
                                      unsigned u = __float_as_uint(static_cast<float>(v));
                                      return static_cast<std::uint16_t>((u + 0x8000u) >> 16);
                                  });
                for (int i = 0; i < opts.warmup; ++i)
                    run_csr_bitbsr_fused_8x4_lb_bfp8(
                        device_bitbsr, packed_col_bitmap_meta, load_balance, d_residual_row_ptr,
                        d_residual_col_idx, d_blk_int8, d_blk_scale, d_residual_bf16,
                        opts.long_row_split, d_long_chunk_row, d_long_chunk_pbegin,
                        d_long_chunk_pcount, d_x, d_y);
                for (int i = 0; i < opts.iterations; ++i) {
                    float ms = 0.0f;
                    run_csr_bitbsr_fused_8x4_lb_bfp8(
                        device_bitbsr, packed_col_bitmap_meta, load_balance, d_residual_row_ptr,
                        d_residual_col_idx, d_blk_int8, d_blk_scale, d_residual_bf16,
                        opts.long_row_split, d_long_chunk_row, d_long_chunk_pbegin,
                        d_long_chunk_pcount, d_x, d_y, &ms);
                    joint_min_ms = std::min(joint_min_ms, ms);
                }
                bpn = (static_cast<double>(bfp8.int8_values.size()) +
                       2.0 * static_cast<double>(bfp8.block_scale_bf16.size()) +
                       2.0 * static_cast<double>(d_residual_bf16.size())) / total_nnz;
                codec_name = "bfp8_blocks+bf16_residual";
            }

            if (opts.verify_cpu) {
                structural::ErrorStats<T> jerr;
                double jcpu = 0.0;
                (void)verify_result(loaded.matrix, x, d_y, jcpu, jerr);
                jrel = static_cast<double>(jerr.global_rel);
            }
            std::cout << "joint_value_codec: " << codec_name << "\n";
            std::cout << "joint_requested_codec: " << opts.value_codec << "\n";
            std::cout << "joint_value_cardinality: " << uniq << "\n";
            std::cout << "joint_min_ms: " << joint_min_ms << "\n";
            std::cout << "joint_hybrid_fp64_min_ms: " << fused_min_ms << "\n";
            std::cout << "joint_speedup_vs_hybrid_fp64: "
                      << (joint_min_ms > 0.0f ? fused_min_ms / joint_min_ms : -1.0) << "\n";
            std::cout << "joint_bytes_per_nnz: " << bpn << "\n";
            std::cout << "joint_rel_err: " << jrel << "\n";
            std::cout << "joint_verify_pass_1e-2: " << ((jrel >= 0.0 && jrel <= 1e-2) ? 1 : 0)
                      << "\n";
        }

        std::cout << "matrix: " << opts.matrix_path << "\n";
        std::cout << "matrix_market_field: " << structural::to_string(loaded.info.field) << "\n";
        std::cout << "matrix_market_symmetry: " << structural::to_string(loaded.info.symmetry)
                  << "\n";
        std::cout << "precision: " << opts.precision << "\n";
        std::cout << "shape: " << loaded.matrix.rows << " x " << loaded.matrix.cols << "\n";
        std::cout << "nnz: " << loaded.matrix.nnz() << "\n";
        std::cout << "block: " << opts.block_rows << " x " << opts.block_cols << "\n";
        std::cout << "spmv_layout: csr_bitbsr_split\n";
        std::cout << "spmv_effective_layout: csr_bitbsr_fused_lb\n";
        std::cout << "residual_format: csr\n";
        std::cout << "residual_threshold: " << opts.residual_threshold << "\n";
        std::cout << "split_policy: " << opts.split_policy << "\n";
        std::cout << "cost_model: bitmap_block=" << opts.cost_model.bitmap_block_cost
                  << ", bitmap_nnz=" << opts.cost_model.bitmap_nnz_cost
                  << ", csr_block=" << opts.cost_model.csr_block_cost
                  << ", csr_nnz=" << opts.cost_model.csr_nnz_cost << "\n";
        std::cout << "cusparse_alg: none\n";
        std::cout << "cusparse_buffer_bytes: 0\n";
        std::cout << "load_balance_chunk_blocks: " << load_balance.chunk_blocks << "\n";
        std::cout << "load_balance_split_threshold_blocks: "
                  << load_balance.split_threshold_blocks << "\n";
        std::cout << "gpu_work_items: " << load_balance.work_items.size() << "\n";
        std::cout << "gpu_output_residency: device\n";
        std::cout << "full_bitbsr_blocks: " << split_stats.full_blocks << "\n";
        std::cout << "kept_blocks: " << split.kept_blocks << "\n";
        std::cout << "residual_blocks: " << split.residual_blocks << "\n";
        std::cout << "cost_skipped_blocks: " << split.cost_skipped_blocks << "\n";
        std::cout << "kept_nnz: " << split.kept_nnz << "\n";
        std::cout << "residual_nnz: " << split.residual_nnz << "\n";
        std::cout << "cost_skipped_nnz: " << split.cost_skipped_nnz << "\n";
        std::cout << "full_bitbsr_position_bits: "
                  << split_stats.full_bitbsr_position_bits << "\n";
        std::cout << "kept_bitbsr_position_bits: "
                  << split_stats.kept_bitbsr_position_bits << "\n";
        std::cout << "residual_csr_position_bits: "
                  << split_stats.residual_csr_position_bits << "\n";
        std::cout << "split_position_bits: " << split_stats.split_position_bits << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "split_vs_full_bitbsr_ratio: "
                  << split_stats.split_vs_full_bitbsr_ratio << "\n";
        std::cout << "residual_block_ratio: " << split_stats.residual_block_ratio << "\n";
        std::cout << "residual_nnz_ratio: " << split_stats.residual_nnz_ratio << "\n";
        std::cout << "residual_position_ratio_in_split: "
                  << split_stats.residual_position_ratio_in_split << "\n";
        std::cout << "residual_position_ratio_vs_full: "
                  << split_stats.residual_position_ratio_vs_full << "\n";
        // ---- runner-compatible top-level keys (family-sweep parse_key_values) ----
        std::cout << "operator_name: " << opts.operator_label << "\n";
        std::cout << "operator_family: structural_hybrid\n";
        std::cout << "operator_promotion_status: promoted\n";
        std::cout << "operator_runner: csr_bitbsr_fused_lb\n";
        std::cout << "status: ok\n";
        std::cout << "verify_gate: exact_abs_or_rel\n";
        std::cout << "value_codec: original\n";
        std::cout << "value_codec_effective: raw_double\n";
        std::cout << "value_codec_exact: 1\n";
        std::cout << "value_codec_codebook_size: 0\n";
        std::cout << "value_codec_bytes_per_nnz: " << sizeof(T) << "\n";
        std::cout << "value_codec_payload_bytes: "
                  << (static_cast<std::size_t>(loaded.matrix.nnz()) * sizeof(T)) << "\n";
        std::cout << "gpu_num_blocks: " << split_stats.full_blocks << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "min_ms: " << fused_min_ms << "\n";
        std::cout << "avg_ms: " << (fused_sum_ms / opts.iterations) << "\n";
        std::cout << std::defaultfloat;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "timing_ms: load=" << load_ms
                  << ", cuda_context=" << cuda_context_ms
                  << ", host_split=" << host_split_ms
                  << ", residual_format_convert_host=" << residual_format_convert_host_ms
                  << ", bitbsr_upload_host=" << bitbsr_upload_host_ms
                  << ", residual_upload_host=" << residual_upload_host_ms
                  << ", col_bitmap_pack_host=" << col_bitmap_pack_host_ms
                  << ", col_bitmap_pack_event=" << col_bitmap_pack_event_ms
                  << ", load_balance_pack_host=" << load_balance_pack_host_ms
                  << ", load_balance_pack_event=" << load_balance_pack_event_ms
                  << ", cusparse_setup_host=0.0000"
                  << ", cusparse_preprocess_host=0.0000"
                  << ", cusparse_preprocess_event=0.0000"
                  << ", gpu_bitbsr_avg_event=0.0000"
                  << ", gpu_bitbsr_min_event=0.0000"
                  << ", gpu_residual_avg_event=0.0000"
                  << ", gpu_residual_min_event=0.0000"
                  << ", gpu_spmv_sum_avg_event=" << (fused_sum_ms / opts.iterations)
                  << ", gpu_spmv_sum_min_event=" << fused_min_ms
                  << ", gpu_fused_avg_event=" << (fused_sum_ms / opts.iterations)
                  << ", gpu_fused_min_event=" << fused_min_ms
                  << ", gpu_spmv_avg_event=" << (fused_sum_ms / opts.iterations)
                  << ", gpu_spmv_min_event=" << fused_min_ms
                  << ", cpu_spmv=" << (opts.verify_cpu ? cpu_spmv_ms : 0.0)
                  << "\n";
        if (opts.verify_cpu) {
            std::cout << "verify_cpu: " << (verified ? "PASS" : "FAIL")
                      << ", max_abs=" << error.max_abs
                      << ", max_rel=" << error.max_rel << "\n";
        } else {
            std::cout << "verify_cpu: SKIP\n";
        }

        return verified ? 0 : 1;
    }

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    const auto alg = parse_cusparse_spmv_alg(opts.cusparse_alg);
    const auto value_type = cuda_data_type<T>();
    size_t cusparse_buffer_bytes = 0;
    const bool has_residual = split.residual.nnz() > 0;

    if (has_residual) {
        (void)time_call(
            [&] {
                STRUCTURAL_CUSPARSE_CHECK(cusparseCreate(&handle));
                STRUCTURAL_CUSPARSE_CHECK(cusparseCreateCsr(
                    &mat_a, split.residual.rows, split.residual.cols, split.residual.nnz(),
                    thrust::raw_pointer_cast(d_residual_row_ptr.data()),
                    thrust::raw_pointer_cast(d_residual_col_idx.data()),
                    thrust::raw_pointer_cast(d_residual_values.data()),
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                    value_type));
                STRUCTURAL_CUSPARSE_CHECK(cusparseCreateDnVec(
                    &vec_x, loaded.matrix.cols, thrust::raw_pointer_cast(d_x.data()),
                    value_type));
                STRUCTURAL_CUSPARSE_CHECK(cusparseCreateDnVec(
                    &vec_y, loaded.matrix.rows, thrust::raw_pointer_cast(d_y.data()),
                    value_type));

                const T alpha = T{1};
                const T beta = T{1};
                STRUCTURAL_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, vec_x, &beta,
                    vec_y, value_type, alg, &cusparse_buffer_bytes));
                return 0;
            },
            cusparse_setup_host_ms);

        thrust::device_vector<std::uint8_t> external_buffer(cusparse_buffer_bytes);
        void* external_buffer_ptr =
            cusparse_buffer_bytes == 0 ? nullptr : thrust::raw_pointer_cast(external_buffer.data());

#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12300
        (void)time_call(
            [&] {
                const T alpha = T{1};
                const T beta = T{1};
                structural::gpu_detail::CudaEventTimer timer(&cusparse_preprocess_event_ms);
                STRUCTURAL_CUSPARSE_CHECK(cusparseSpMV_preprocess(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, vec_x, &beta,
                    vec_y, value_type, alg, external_buffer_ptr));
                timer.stop();
                return 0;
            },
            cusparse_preprocess_host_ms);
#else
        // cusparseSpMV_preprocess was added in cuSPARSE 12.3 (CUDA 12.4). On older
        // toolkits (e.g. CUDA 12.1 on A800) skip the pre-analysis: plain cusparseSpMV
        // below still produces correct results, only without the first-call optimization.
        cusparse_preprocess_event_ms = 0.0;
        cusparse_preprocess_host_ms = 0.0;
#endif

        for (int i = 0; i < opts.warmup; ++i) {
            run_bitbsr_col_bitmap64_flag_lb_stable_y(
                device_bitbsr, packed_col_bitmap_meta, load_balance, d_x, d_y);
            run_cusparse_residual_spmv<T>(
                handle, mat_a, vec_x, vec_y, alg, external_buffer_ptr);
        }

        double bitbsr_sum_ms = 0.0;
        double residual_sum_ms = 0.0;
        double total_sum_ms = 0.0;
        float bitbsr_min_ms = std::numeric_limits<float>::max();
        float residual_min_ms = std::numeric_limits<float>::max();
        float total_min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < opts.iterations; ++i) {
            float bitbsr_ms = 0.0f;
            float residual_ms = 0.0f;
            run_bitbsr_col_bitmap64_flag_lb_stable_y(
                device_bitbsr, packed_col_bitmap_meta, load_balance, d_x, d_y, &bitbsr_ms);
            run_cusparse_residual_spmv<T>(
                handle, mat_a, vec_x, vec_y, alg, external_buffer_ptr, &residual_ms);
            const float total_ms = bitbsr_ms + residual_ms;
            bitbsr_sum_ms += bitbsr_ms;
            residual_sum_ms += residual_ms;
            total_sum_ms += total_ms;
            bitbsr_min_ms = std::min(bitbsr_min_ms, bitbsr_ms);
            residual_min_ms = std::min(residual_min_ms, residual_ms);
            total_min_ms = std::min(total_min_ms, total_ms);
        }

        structural::ErrorStats<T> error;
        bool verified = true;
        if (opts.verify_cpu) {
            verified = verify_result(loaded.matrix, x, d_y, cpu_spmv_ms, error);
        }

        std::cout << "matrix: " << opts.matrix_path << "\n";
        std::cout << "matrix_market_field: " << structural::to_string(loaded.info.field) << "\n";
        std::cout << "matrix_market_symmetry: " << structural::to_string(loaded.info.symmetry)
                  << "\n";
        std::cout << "precision: " << opts.precision << "\n";
        std::cout << "shape: " << loaded.matrix.rows << " x " << loaded.matrix.cols << "\n";
        std::cout << "nnz: " << loaded.matrix.nnz() << "\n";
        std::cout << "block: " << opts.block_rows << " x " << opts.block_cols << "\n";
        std::cout << "spmv_layout: csr_bitbsr_split\n";
        std::cout << "spmv_effective_layout: csr_bitbsr_two_pass\n";
        std::cout << "residual_format: csr\n";
        std::cout << "residual_threshold: " << opts.residual_threshold << "\n";
        std::cout << "split_policy: " << opts.split_policy << "\n";
        std::cout << "cost_model: bitmap_block=" << opts.cost_model.bitmap_block_cost
                  << ", bitmap_nnz=" << opts.cost_model.bitmap_nnz_cost
                  << ", csr_block=" << opts.cost_model.csr_block_cost
                  << ", csr_nnz=" << opts.cost_model.csr_nnz_cost << "\n";
        std::cout << "cusparse_alg: " << opts.cusparse_alg << "\n";
        std::cout << "cusparse_buffer_bytes: " << cusparse_buffer_bytes << "\n";
        std::cout << "load_balance_chunk_blocks: " << load_balance.chunk_blocks << "\n";
        std::cout << "load_balance_split_threshold_blocks: "
                  << load_balance.split_threshold_blocks << "\n";
        std::cout << "gpu_work_items: " << load_balance.work_items.size() << "\n";
        std::cout << "gpu_output_residency: device\n";
        std::cout << "full_bitbsr_blocks: " << split_stats.full_blocks << "\n";
        std::cout << "kept_blocks: " << split.kept_blocks << "\n";
        std::cout << "residual_blocks: " << split.residual_blocks << "\n";
        std::cout << "cost_skipped_blocks: " << split.cost_skipped_blocks << "\n";
        std::cout << "kept_nnz: " << split.kept_nnz << "\n";
        std::cout << "residual_nnz: " << split.residual_nnz << "\n";
        std::cout << "cost_skipped_nnz: " << split.cost_skipped_nnz << "\n";
        std::cout << "full_bitbsr_position_bits: "
                  << split_stats.full_bitbsr_position_bits << "\n";
        std::cout << "kept_bitbsr_position_bits: "
                  << split_stats.kept_bitbsr_position_bits << "\n";
        std::cout << "residual_csr_position_bits: "
                  << split_stats.residual_csr_position_bits << "\n";
        std::cout << "split_position_bits: " << split_stats.split_position_bits << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "split_vs_full_bitbsr_ratio: "
                  << split_stats.split_vs_full_bitbsr_ratio << "\n";
        std::cout << "residual_block_ratio: " << split_stats.residual_block_ratio << "\n";
        std::cout << "residual_nnz_ratio: " << split_stats.residual_nnz_ratio << "\n";
        std::cout << "residual_position_ratio_in_split: "
                  << split_stats.residual_position_ratio_in_split << "\n";
        std::cout << "residual_position_ratio_vs_full: "
                  << split_stats.residual_position_ratio_vs_full << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "timing_ms: load=" << load_ms
                  << ", cuda_context=" << cuda_context_ms
                  << ", host_split=" << host_split_ms
                  << ", residual_format_convert_host=" << residual_format_convert_host_ms
                  << ", bitbsr_upload_host=" << bitbsr_upload_host_ms
                  << ", residual_upload_host=" << residual_upload_host_ms
                  << ", col_bitmap_pack_host=" << col_bitmap_pack_host_ms
                  << ", col_bitmap_pack_event=" << col_bitmap_pack_event_ms
                  << ", load_balance_pack_host=" << load_balance_pack_host_ms
                  << ", load_balance_pack_event=" << load_balance_pack_event_ms
                  << ", cusparse_setup_host=" << cusparse_setup_host_ms
                  << ", cusparse_preprocess_host=" << cusparse_preprocess_host_ms
                  << ", cusparse_preprocess_event=" << cusparse_preprocess_event_ms
                  << ", gpu_bitbsr_avg_event=" << (bitbsr_sum_ms / opts.iterations)
                  << ", gpu_bitbsr_min_event=" << bitbsr_min_ms
                  << ", gpu_residual_avg_event=" << (residual_sum_ms / opts.iterations)
                  << ", gpu_residual_min_event=" << residual_min_ms
                  << ", gpu_spmv_sum_avg_event=" << (total_sum_ms / opts.iterations)
                  << ", gpu_spmv_sum_min_event=" << total_min_ms
                  << ", cpu_spmv=" << (opts.verify_cpu ? cpu_spmv_ms : 0.0)
                  << "\n";
        if (opts.verify_cpu) {
            std::cout << "verify_cpu: " << (verified ? "PASS" : "FAIL")
                      << ", max_abs=" << error.max_abs
                      << ", max_rel=" << error.max_rel << "\n";
        } else {
            std::cout << "verify_cpu: SKIP\n";
        }

        if (vec_y != nullptr) {
            (void)cusparseDestroyDnVec(vec_y);
        }
        if (vec_x != nullptr) {
            (void)cusparseDestroyDnVec(vec_x);
        }
        if (mat_a != nullptr) {
            (void)cusparseDestroySpMat(mat_a);
        }
        if (handle != nullptr) {
            (void)cusparseDestroy(handle);
        }

        return verified ? 0 : 1;
    }

    for (int i = 0; i < opts.warmup; ++i) {
        run_bitbsr_col_bitmap64_flag_lb_stable_y(
            device_bitbsr, packed_col_bitmap_meta, load_balance, d_x, d_y);
    }

    double bitbsr_sum_ms = 0.0;
    float bitbsr_min_ms = std::numeric_limits<float>::max();
    for (int i = 0; i < opts.iterations; ++i) {
        float bitbsr_ms = 0.0f;
        run_bitbsr_col_bitmap64_flag_lb_stable_y(
            device_bitbsr, packed_col_bitmap_meta, load_balance, d_x, d_y, &bitbsr_ms);
        bitbsr_sum_ms += bitbsr_ms;
        bitbsr_min_ms = std::min(bitbsr_min_ms, bitbsr_ms);
    }

    structural::ErrorStats<T> error;
    bool verified = true;
    if (opts.verify_cpu) {
        verified = verify_result(loaded.matrix, x, d_y, cpu_spmv_ms, error);
    }

    std::cout << "matrix: " << opts.matrix_path << "\n";
    std::cout << "matrix_market_field: " << structural::to_string(loaded.info.field) << "\n";
    std::cout << "matrix_market_symmetry: " << structural::to_string(loaded.info.symmetry)
              << "\n";
    std::cout << "precision: " << opts.precision << "\n";
    std::cout << "shape: " << loaded.matrix.rows << " x " << loaded.matrix.cols << "\n";
    std::cout << "nnz: " << loaded.matrix.nnz() << "\n";
    std::cout << "block: " << opts.block_rows << " x " << opts.block_cols << "\n";
    std::cout << "spmv_layout: csr_bitbsr_split\n";
    std::cout << "spmv_effective_layout: csr_bitbsr_two_pass\n";
    std::cout << "residual_format: csr\n";
    std::cout << "residual_threshold: " << opts.residual_threshold << "\n";
    std::cout << "split_policy: " << opts.split_policy << "\n";
    std::cout << "cost_model: bitmap_block=" << opts.cost_model.bitmap_block_cost
              << ", bitmap_nnz=" << opts.cost_model.bitmap_nnz_cost
              << ", csr_block=" << opts.cost_model.csr_block_cost
              << ", csr_nnz=" << opts.cost_model.csr_nnz_cost << "\n";
    std::cout << "cusparse_alg: " << opts.cusparse_alg << "\n";
    std::cout << "cusparse_buffer_bytes: 0\n";
    std::cout << "load_balance_chunk_blocks: " << load_balance.chunk_blocks << "\n";
    std::cout << "load_balance_split_threshold_blocks: "
              << load_balance.split_threshold_blocks << "\n";
    std::cout << "gpu_work_items: " << load_balance.work_items.size() << "\n";
    std::cout << "gpu_output_residency: device\n";
    std::cout << "full_bitbsr_blocks: " << split_stats.full_blocks << "\n";
    std::cout << "kept_blocks: " << split.kept_blocks << "\n";
    std::cout << "residual_blocks: " << split.residual_blocks << "\n";
    std::cout << "cost_skipped_blocks: " << split.cost_skipped_blocks << "\n";
    std::cout << "kept_nnz: " << split.kept_nnz << "\n";
    std::cout << "residual_nnz: " << split.residual_nnz << "\n";
    std::cout << "cost_skipped_nnz: " << split.cost_skipped_nnz << "\n";
    std::cout << "full_bitbsr_position_bits: " << split_stats.full_bitbsr_position_bits << "\n";
    std::cout << "kept_bitbsr_position_bits: "
              << split_stats.kept_bitbsr_position_bits << "\n";
    std::cout << "residual_csr_position_bits: "
              << split_stats.residual_csr_position_bits << "\n";
    std::cout << "split_position_bits: " << split_stats.split_position_bits << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "split_vs_full_bitbsr_ratio: " << split_stats.split_vs_full_bitbsr_ratio
              << "\n";
    std::cout << "residual_block_ratio: " << split_stats.residual_block_ratio << "\n";
    std::cout << "residual_nnz_ratio: " << split_stats.residual_nnz_ratio << "\n";
    std::cout << "residual_position_ratio_in_split: "
              << split_stats.residual_position_ratio_in_split << "\n";
    std::cout << "residual_position_ratio_vs_full: "
              << split_stats.residual_position_ratio_vs_full << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "timing_ms: load=" << load_ms
              << ", cuda_context=" << cuda_context_ms
              << ", host_split=" << host_split_ms
              << ", residual_format_convert_host=" << residual_format_convert_host_ms
              << ", bitbsr_upload_host=" << bitbsr_upload_host_ms
              << ", residual_upload_host=" << residual_upload_host_ms
              << ", col_bitmap_pack_host=" << col_bitmap_pack_host_ms
              << ", col_bitmap_pack_event=" << col_bitmap_pack_event_ms
              << ", load_balance_pack_host=" << load_balance_pack_host_ms
              << ", load_balance_pack_event=" << load_balance_pack_event_ms
              << ", cusparse_setup_host=0.0000"
              << ", cusparse_preprocess_host=0.0000"
              << ", cusparse_preprocess_event=0.0000"
              << ", gpu_bitbsr_avg_event=" << (bitbsr_sum_ms / opts.iterations)
              << ", gpu_bitbsr_min_event=" << bitbsr_min_ms
              << ", gpu_residual_avg_event=0.0000"
              << ", gpu_residual_min_event=0.0000"
              << ", gpu_spmv_sum_avg_event=" << (bitbsr_sum_ms / opts.iterations)
              << ", gpu_spmv_sum_min_event=" << bitbsr_min_ms
              << ", cpu_spmv=" << (opts.verify_cpu ? cpu_spmv_ms : 0.0)
              << "\n";
    if (opts.verify_cpu) {
        std::cout << "verify_cpu: " << (verified ? "PASS" : "FAIL")
                  << ", max_abs=" << error.max_abs
                  << ", max_rel=" << error.max_rel << "\n";
    } else {
        std::cout << "verify_cpu: SKIP\n";
    }

    return verified ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_options(argc, argv);
        if (opts.precision == "float") {
            return run<float>(opts);
        }
        return run<double>(opts);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
