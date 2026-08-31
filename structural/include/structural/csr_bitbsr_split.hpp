#pragma once

#include "structural/bitbsr.hpp"
#include "structural/convert.hpp"
#include "structural/csr.hpp"
#include "structural/spmv.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace structural {

template <typename T>
struct CsrBitBsrSplitMatrix {
    BitBsrMatrix<T> bitbsr;
    CsrMatrix<T> residual;
    Index block_rows = 0;
    Index block_cols = 0;
    Index residual_threshold = 0;
    Offset kept_blocks = 0;
    Offset residual_blocks = 0;
    Offset kept_nnz = 0;
    Offset residual_nnz = 0;
    Offset cost_skipped_blocks = 0;
    Offset cost_skipped_nnz = 0;

    void validate() const {
        if (block_rows <= 0 || block_cols <= 0 || residual_threshold < 0) {
            throw std::runtime_error("CSR+BitBSR split metadata is invalid");
        }
        bitbsr.validate();
        residual.validate();
        if (bitbsr.rows != residual.rows || bitbsr.cols != residual.cols) {
            throw std::runtime_error("CSR+BitBSR split dimensions differ");
        }
        if (bitbsr.block_rows != block_rows || bitbsr.block_cols != block_cols) {
            throw std::runtime_error("CSR+BitBSR split block size metadata differs");
        }
        if (kept_blocks != bitbsr.num_blocks()) {
            throw std::runtime_error("CSR+BitBSR split kept block count differs");
        }
        if (kept_nnz != bitbsr.nnz()) {
            throw std::runtime_error("CSR+BitBSR split kept nnz differs");
        }
        if (residual_nnz != residual.nnz()) {
            throw std::runtime_error("CSR+BitBSR split residual nnz differs");
        }
    }
};

struct CsrBitBsrSplitCostModel {
    double bitmap_block_cost = 10.0;
    double bitmap_nnz_cost = 0.0;
    double csr_block_cost = 0.0;
    double csr_nnz_cost = 1.0;

    void validate() const {
        if (bitmap_block_cost < 0.0 || bitmap_nnz_cost < 0.0 ||
            csr_block_cost < 0.0 || csr_nnz_cost < 0.0) {
            throw std::runtime_error("CSR+BitBSR split cost model must be non-negative");
        }
    }

    bool should_move_to_residual(Offset block_nnz) const {
        validate();
        const double bitmap_cost = bitmap_block_cost + bitmap_nnz_cost *
                                                        static_cast<double>(block_nnz);
        const double csr_cost = csr_block_cost + csr_nnz_cost *
                                                 static_cast<double>(block_nnz);
        return csr_cost < bitmap_cost;
    }
};

template <typename T>
struct BlockRowCooResidualMatrix {
    Index rows = 0;
    Index cols = 0;
    Index block_rows = 0;
    Index block_cols = 0;
    Index num_block_rows = 0;
    Index bits_per_block = 0;

    std::vector<Offset> blockrow_ptr;
    std::vector<Index> packed_col_local;
    std::vector<T> values;

    Offset nnz() const {
        return static_cast<Offset>(values.size());
    }

    void validate() const {
        if (rows < 0 || cols < 0 || block_rows <= 0 || block_cols <= 0) {
            throw std::runtime_error("block-row COO residual dimensions are invalid");
        }
        if (num_block_rows != ceil_div(rows, block_rows)) {
            throw std::runtime_error("block-row COO residual block-row count is inconsistent");
        }
        if (bits_per_block != block_rows * block_cols) {
            throw std::runtime_error("block-row COO residual bits_per_block is inconsistent");
        }
        if (static_cast<Index>(blockrow_ptr.size()) != num_block_rows + 1) {
            throw std::runtime_error("block-row COO residual blockrow_ptr size is invalid");
        }
        if (packed_col_local.size() != values.size()) {
            throw std::runtime_error("block-row COO residual position and value sizes differ");
        }
        if (!blockrow_ptr.empty() && blockrow_ptr.front() != 0) {
            throw std::runtime_error("block-row COO residual blockrow_ptr must start at zero");
        }
        for (Index br = 0; br < num_block_rows; ++br) {
            if (blockrow_ptr[br] > blockrow_ptr[br + 1]) {
                throw std::runtime_error("block-row COO residual blockrow_ptr must be monotonic");
            }
        }
        if (!blockrow_ptr.empty() && blockrow_ptr.back() != nnz()) {
            throw std::runtime_error("block-row COO residual blockrow_ptr.back must equal nnz");
        }
        const Index num_block_cols = ceil_div(cols, block_cols);
        for (Index packed : packed_col_local) {
            if (packed < 0) {
                throw std::runtime_error("block-row COO residual packed position is negative");
            }
            const Index block_col = packed / bits_per_block;
            const Index local_id = packed % bits_per_block;
            if (block_col < 0 || block_col >= num_block_cols ||
                local_id < 0 || local_id >= bits_per_block) {
                throw std::runtime_error("block-row COO residual packed position out of range");
            }
        }
    }
};

template <typename T>
BlockRowCooResidualMatrix<T> convert_residual_csr_to_blockrow_coo(
    const CsrMatrix<T>& csr,
    Index block_rows,
    Index block_cols) {
    csr.validate();
    if (block_rows <= 0 || block_cols <= 0) {
        throw std::runtime_error("block-row COO residual block sizes must be positive");
    }

    BlockRowCooResidualMatrix<T> out;
    out.rows = csr.rows;
    out.cols = csr.cols;
    out.block_rows = block_rows;
    out.block_cols = block_cols;
    out.num_block_rows = ceil_div(csr.rows, block_rows);
    out.bits_per_block = block_rows * block_cols;
    out.blockrow_ptr.assign(static_cast<size_t>(out.num_block_rows) + 1, 0);
    out.packed_col_local.reserve(csr.values.size());
    out.values.reserve(csr.values.size());

    for (Index br = 0; br < out.num_block_rows; ++br) {
        out.blockrow_ptr[br] = static_cast<Offset>(out.values.size());
        const Index row_begin = br * block_rows;
        const Index row_end = std::min(csr.rows, row_begin + block_rows);
        for (Index row = row_begin; row < row_end; ++row) {
            const Index local_row = row - row_begin;
            for (Offset p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
                const Index col = csr.col_idx[p];
                const Index block_col = col / block_cols;
                const Index local_col = col - block_col * block_cols;
                const Index local_id = local_row * block_cols + local_col;
                out.packed_col_local.push_back(block_col * out.bits_per_block + local_id);
                out.values.push_back(csr.values[p]);
            }
        }
        out.blockrow_ptr[br + 1] = static_cast<Offset>(out.values.size());
    }

    out.validate();
    return out;
}

template <typename T>
std::vector<T> spmv_blockrow_coo_residual(const BlockRowCooResidualMatrix<T>& residual,
                                          const std::vector<T>& x) {
    residual.validate();
    if (static_cast<Index>(x.size()) != residual.cols) {
        throw std::runtime_error("block-row COO residual SpMV input vector size mismatch");
    }

    std::vector<T> y(residual.rows, T{0});
    for (Index br = 0; br < residual.num_block_rows; ++br) {
        for (Offset p = residual.blockrow_ptr[br]; p < residual.blockrow_ptr[br + 1]; ++p) {
            const Index packed = residual.packed_col_local[p];
            const Index block_col = packed / residual.bits_per_block;
            const Index local_id = packed - block_col * residual.bits_per_block;
            const Index local_row = local_id / residual.block_cols;
            const Index local_col = local_id - local_row * residual.block_cols;
            const Index row = br * residual.block_rows + local_row;
            const Index col = block_col * residual.block_cols + local_col;
            if (row < residual.rows && col < residual.cols) {
                y[row] += residual.values[p] * x[col];
            }
        }
    }
    return y;
}

template <typename T>
struct SlicedSellResidualMatrix {
    Index rows = 0;
    Index cols = 0;
    Index slice_height = 0;
    Index num_slices = 0;
    Offset actual_nnz = 0;

    std::vector<Offset> slice_ptr;
    std::vector<Index> slice_width;
    std::vector<Index> col_idx;
    std::vector<T> values;

    Offset padded_nnz() const {
        return static_cast<Offset>(values.size());
    }

    void validate() const {
        if (rows < 0 || cols < 0 || slice_height <= 0) {
            throw std::runtime_error("sliced SELL residual dimensions are invalid");
        }
        if (num_slices != ceil_div(rows, slice_height)) {
            throw std::runtime_error("sliced SELL residual slice count is inconsistent");
        }
        if (static_cast<Index>(slice_ptr.size()) != num_slices + 1) {
            throw std::runtime_error("sliced SELL residual slice_ptr size is invalid");
        }
        if (static_cast<Index>(slice_width.size()) != num_slices) {
            throw std::runtime_error("sliced SELL residual slice_width size is invalid");
        }
        if (col_idx.size() != values.size()) {
            throw std::runtime_error("sliced SELL residual position/value sizes differ");
        }
        if (!slice_ptr.empty() && slice_ptr.front() != 0) {
            throw std::runtime_error("sliced SELL residual slice_ptr must start at zero");
        }

        Offset counted_nnz = 0;
        for (Index s = 0; s < num_slices; ++s) {
            if (slice_ptr[s] > slice_ptr[s + 1]) {
                throw std::runtime_error("sliced SELL residual slice_ptr must be monotonic");
            }
            const Offset expected_span =
                static_cast<Offset>(slice_width[s]) * static_cast<Offset>(slice_height);
            if (slice_ptr[s + 1] - slice_ptr[s] != expected_span) {
                throw std::runtime_error("sliced SELL residual slice span is inconsistent");
            }
            for (Offset p = slice_ptr[s]; p < slice_ptr[s + 1]; ++p) {
                const Index col = col_idx[p];
                if (col < -1 || col >= cols) {
                    throw std::runtime_error("sliced SELL residual column out of range");
                }
                if (col >= 0) {
                    ++counted_nnz;
                }
            }
        }
        if (!slice_ptr.empty() && slice_ptr.back() != padded_nnz()) {
            throw std::runtime_error("sliced SELL residual slice_ptr.back must equal padded nnz");
        }
        if (actual_nnz != counted_nnz) {
            throw std::runtime_error("sliced SELL residual actual nnz is inconsistent");
        }
    }
};

template <typename T>
SlicedSellResidualMatrix<T> convert_residual_csr_to_sliced_sell(const CsrMatrix<T>& csr,
                                                                Index slice_height) {
    csr.validate();
    if (slice_height <= 0) {
        throw std::runtime_error("sliced SELL residual slice height must be positive");
    }

    SlicedSellResidualMatrix<T> out;
    out.rows = csr.rows;
    out.cols = csr.cols;
    out.slice_height = slice_height;
    out.num_slices = ceil_div(csr.rows, slice_height);
    out.actual_nnz = csr.nnz();
    out.slice_ptr.assign(static_cast<size_t>(out.num_slices) + 1, 0);
    out.slice_width.assign(static_cast<size_t>(out.num_slices), 0);

    for (Index s = 0; s < out.num_slices; ++s) {
        const Index row_begin = s * slice_height;
        const Index row_end = std::min(csr.rows, row_begin + slice_height);
        Index width = 0;
        for (Index row = row_begin; row < row_end; ++row) {
            width = std::max(width, csr.row_ptr[row + 1] - csr.row_ptr[row]);
        }
        out.slice_width[s] = width;
        out.slice_ptr[s + 1] =
            out.slice_ptr[s] + static_cast<Offset>(width) * static_cast<Offset>(slice_height);
    }

    out.col_idx.assign(static_cast<size_t>(out.slice_ptr.back()), Index{-1});
    out.values.assign(static_cast<size_t>(out.slice_ptr.back()), T{0});
    for (Index s = 0; s < out.num_slices; ++s) {
        const Index row_begin = s * slice_height;
        const Index row_end = std::min(csr.rows, row_begin + slice_height);
        for (Index row = row_begin; row < row_end; ++row) {
            const Index local_row = row - row_begin;
            Offset k = 0;
            for (Offset p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p, ++k) {
                const Offset dst =
                    out.slice_ptr[s] + k * static_cast<Offset>(slice_height) + local_row;
                out.col_idx[dst] = csr.col_idx[p];
                out.values[dst] = csr.values[p];
            }
        }
    }

    out.validate();
    return out;
}

template <typename T>
std::vector<T> spmv_sliced_sell_residual(const SlicedSellResidualMatrix<T>& residual,
                                         const std::vector<T>& x) {
    residual.validate();
    if (static_cast<Index>(x.size()) != residual.cols) {
        throw std::runtime_error("sliced SELL residual SpMV input vector size mismatch");
    }

    std::vector<T> y(residual.rows, T{0});
    for (Index s = 0; s < residual.num_slices; ++s) {
        const Index row_begin = s * residual.slice_height;
        const Index row_end = std::min(residual.rows, row_begin + residual.slice_height);
        const Index width = residual.slice_width[s];
        for (Index row = row_begin; row < row_end; ++row) {
            const Index local_row = row - row_begin;
            T sum = T{0};
            for (Index k = 0; k < width; ++k) {
                const Offset p =
                    residual.slice_ptr[s] +
                    static_cast<Offset>(k) * static_cast<Offset>(residual.slice_height) +
                    local_row;
                const Index col = residual.col_idx[p];
                if (col >= 0) {
                    sum += residual.values[p] * x[col];
                }
            }
            y[row] = sum;
        }
    }
    return y;
}

template <typename T>
struct CsrBitBsrResidualEntry {
    Index col = 0;
    T value{};
};

template <typename T>
void append_bitbsr_block_to_residual_rows(
    const BitBsrMatrix<T>& bitbsr,
    Offset block_id,
    Index block_row,
    std::vector<std::vector<CsrBitBsrResidualEntry<T>>>& rows) {
    const Index block_col = bitbsr.block_col_idx[block_id];
    Offset rank = 0;
    for (Index word_idx = 0; word_idx < bitbsr.words_per_block; ++word_idx) {
        BitmapWord word = bitbsr.bitmap_words[block_id * bitbsr.words_per_block + word_idx];
        while (word != 0) {
            const Index bit_in_word = ctz_word(word);
            const Index bit = word_idx * kBitmapWordBits + bit_in_word;
            const Index local_row = bit / bitbsr.block_cols;
            const Index local_col = bit % bitbsr.block_cols;
            const Index row = block_row * bitbsr.block_rows + local_row;
            const Index col = block_col * bitbsr.block_cols + local_col;
            if (row < bitbsr.rows && col < bitbsr.cols) {
                rows[static_cast<size_t>(row)].push_back(
                    CsrBitBsrResidualEntry<T>{col, bitbsr.values[bitbsr.block_val_ptr[block_id] +
                                                                 rank]});
            }
            ++rank;
            word &= word - 1;
        }
    }
}

template <typename T>
void append_bitbsr_block_to_kept(const BitBsrMatrix<T>& full,
                                 Offset block,
                                 CsrBitBsrSplitMatrix<T>& out) {
    out.bitbsr.block_col_idx.push_back(full.block_col_idx[block]);
    for (Index word = 0; word < full.words_per_block; ++word) {
        out.bitbsr.bitmap_words.push_back(full.bitmap_words[block * full.words_per_block + word]);
    }
    const Offset value_begin = full.block_val_ptr[block];
    const Offset value_end = full.block_val_ptr[block + 1];
    out.bitbsr.values.insert(out.bitbsr.values.end(),
                             full.values.begin() + value_begin,
                             full.values.begin() + value_end);
    const Offset block_nnz = value_end - value_begin;
    ++out.kept_blocks;
    out.kept_nnz += block_nnz;
    out.bitbsr.block_val_ptr.push_back(out.kept_nnz);
}

template <typename T, typename ShouldMoveToResidual>
CsrBitBsrSplitMatrix<T> convert_bitbsr_to_csr_bitbsr_split_if(
    const BitBsrMatrix<T>& full,
    Index residual_threshold,
    ShouldMoveToResidual should_move_to_residual) {
    full.validate();
    if (residual_threshold < 0) {
        throw std::runtime_error("CSR+BitBSR residual threshold must be non-negative");
    }

    CsrBitBsrSplitMatrix<T> out;
    out.block_rows = full.block_rows;
    out.block_cols = full.block_cols;
    out.residual_threshold = residual_threshold;

    out.bitbsr.rows = full.rows;
    out.bitbsr.cols = full.cols;
    out.bitbsr.block_rows = full.block_rows;
    out.bitbsr.block_cols = full.block_cols;
    out.bitbsr.num_block_rows = full.num_block_rows;
    out.bitbsr.num_block_cols = full.num_block_cols;
    out.bitbsr.words_per_block = full.words_per_block;
    out.bitbsr.block_row_ptr.assign(static_cast<size_t>(full.num_block_rows) + 1, 0);
    out.bitbsr.block_val_ptr.assign(1, 0);

    std::vector<std::vector<CsrBitBsrResidualEntry<T>>> residual_rows(
        static_cast<size_t>(full.rows));

    for (Index br = 0; br < full.num_block_rows; ++br) {
        out.bitbsr.block_row_ptr[br] = out.kept_blocks;
        for (Offset block = full.block_row_ptr[br]; block < full.block_row_ptr[br + 1]; ++block) {
            const Offset block_nnz = full.block_val_ptr[block + 1] - full.block_val_ptr[block];
            if (block_nnz <= residual_threshold && should_move_to_residual(block_nnz)) {
                append_bitbsr_block_to_residual_rows(full, block, br, residual_rows);
                ++out.residual_blocks;
                out.residual_nnz += block_nnz;
                continue;
            }
            if (block_nnz <= residual_threshold) {
                ++out.cost_skipped_blocks;
                out.cost_skipped_nnz += block_nnz;
            }
            append_bitbsr_block_to_kept(full, block, out);
        }
        out.bitbsr.block_row_ptr[br + 1] = out.kept_blocks;
    }

    out.residual.rows = full.rows;
    out.residual.cols = full.cols;
    out.residual.row_ptr.assign(static_cast<size_t>(full.rows) + 1, 0);
    for (Index row = 0; row < full.rows; ++row) {
        out.residual.row_ptr[row + 1] =
            out.residual.row_ptr[row] +
            static_cast<Offset>(residual_rows[static_cast<size_t>(row)].size());
    }
    out.residual.col_idx.reserve(static_cast<size_t>(out.residual_nnz));
    out.residual.values.reserve(static_cast<size_t>(out.residual_nnz));
    for (Index row = 0; row < full.rows; ++row) {
        for (const auto& entry : residual_rows[static_cast<size_t>(row)]) {
            out.residual.col_idx.push_back(entry.col);
            out.residual.values.push_back(entry.value);
        }
    }

    out.validate();
    return out;
}

template <typename T>
CsrBitBsrSplitMatrix<T> convert_bitbsr_to_csr_bitbsr_split(
    const BitBsrMatrix<T>& full,
    Index residual_threshold) {
    return convert_bitbsr_to_csr_bitbsr_split_if(
        full, residual_threshold, [](Offset) { return true; });
}

template <typename T>
CsrBitBsrSplitMatrix<T> convert_bitbsr_to_csr_bitbsr_split_cost_gated(
    const BitBsrMatrix<T>& full,
    Index residual_threshold,
    const CsrBitBsrSplitCostModel& cost_model) {
    cost_model.validate();
    return convert_bitbsr_to_csr_bitbsr_split_if(
        full, residual_threshold,
        [&](Offset block_nnz) { return cost_model.should_move_to_residual(block_nnz); });
}

template <typename T>
CsrBitBsrSplitMatrix<T> convert_to_csr_bitbsr_split(const CsrMatrix<T>& csr,
                                                    Index block_rows,
                                                    Index block_cols,
                                                    Index residual_threshold) {
    const auto full = convert_to_bitbsr(csr, block_rows, block_cols);
    return convert_bitbsr_to_csr_bitbsr_split(full, residual_threshold);
}

template <typename T>
CsrBitBsrSplitMatrix<T> convert_to_csr_bitbsr_split_cost_gated(
    const CsrMatrix<T>& csr,
    Index block_rows,
    Index block_cols,
    Index residual_threshold,
    const CsrBitBsrSplitCostModel& cost_model) {
    const auto full = convert_to_bitbsr(csr, block_rows, block_cols);
    return convert_bitbsr_to_csr_bitbsr_split_cost_gated(
        full, residual_threshold, cost_model);
}

template <typename T>
std::vector<T> spmv_csr_bitbsr_split(const CsrBitBsrSplitMatrix<T>& split,
                                     const std::vector<T>& x) {
    split.validate();
    auto y = spmv_bitbsr(split.bitbsr, x);
    const auto residual_y = spmv_csr(split.residual, x);
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += residual_y[i];
    }
    return y;
}

template <typename T>
struct CsrBitBsrSplitStorageStats {
    Offset full_blocks = 0;
    Offset kept_blocks = 0;
    Offset residual_blocks = 0;
    Offset kept_nnz = 0;
    Offset residual_nnz = 0;
    Offset cost_skipped_blocks = 0;
    Offset cost_skipped_nnz = 0;
    long long full_bitbsr_position_bits = 0;
    long long kept_bitbsr_position_bits = 0;
    long long residual_csr_position_bits = 0;
    long long split_position_bits = 0;
    double split_vs_full_bitbsr_ratio = 0.0;
    double residual_nnz_ratio = 0.0;
    double residual_block_ratio = 0.0;
    double residual_position_ratio_in_split = 0.0;
    double residual_position_ratio_vs_full = 0.0;
};

template <typename T>
CsrBitBsrSplitStorageStats<T> storage_stats(const BitBsrMatrix<T>& full,
                                            const CsrBitBsrSplitMatrix<T>& split) {
    full.validate();
    split.validate();

    CsrBitBsrSplitStorageStats<T> stats;
    stats.full_blocks = full.num_blocks();
    stats.kept_blocks = split.kept_blocks;
    stats.residual_blocks = split.residual_blocks;
    stats.kept_nnz = split.kept_nnz;
    stats.residual_nnz = split.residual_nnz;
    stats.cost_skipped_blocks = split.cost_skipped_blocks;
    stats.cost_skipped_nnz = split.cost_skipped_nnz;
    stats.full_bitbsr_position_bits =
        static_cast<long long>(full.block_row_ptr.size()) * 32 +
        static_cast<long long>(full.block_col_idx.size()) * 32 +
        static_cast<long long>(full.bitmap_words.size()) * kBitmapWordBits +
        static_cast<long long>(full.block_val_ptr.size()) * 32;
    stats.kept_bitbsr_position_bits =
        static_cast<long long>(split.bitbsr.block_row_ptr.size()) * 32 +
        static_cast<long long>(split.bitbsr.block_col_idx.size()) * 32 +
        static_cast<long long>(split.bitbsr.bitmap_words.size()) * kBitmapWordBits +
        static_cast<long long>(split.bitbsr.block_val_ptr.size()) * 32;
    stats.residual_csr_position_bits =
        static_cast<long long>(split.residual.row_ptr.size()) * 32 +
        static_cast<long long>(split.residual.col_idx.size()) * 32;
    stats.split_position_bits =
        stats.kept_bitbsr_position_bits + stats.residual_csr_position_bits;
    if (stats.full_bitbsr_position_bits > 0) {
        stats.split_vs_full_bitbsr_ratio =
            static_cast<double>(stats.split_position_bits) /
            static_cast<double>(stats.full_bitbsr_position_bits);
        stats.residual_position_ratio_vs_full =
            static_cast<double>(stats.residual_csr_position_bits) /
            static_cast<double>(stats.full_bitbsr_position_bits);
    }
    if (stats.full_blocks > 0) {
        stats.residual_block_ratio =
            static_cast<double>(stats.residual_blocks) / static_cast<double>(stats.full_blocks);
    }
    if (stats.split_position_bits > 0) {
        stats.residual_position_ratio_in_split =
            static_cast<double>(stats.residual_csr_position_bits) /
            static_cast<double>(stats.split_position_bits);
    }
    const Offset full_nnz = split.kept_nnz + split.residual_nnz;
    if (full_nnz > 0) {
        stats.residual_nnz_ratio =
            static_cast<double>(split.residual_nnz) / static_cast<double>(full_nnz);
    }
    return stats;
}

}  // namespace structural
