#pragma once

#include "structural/bitbsr.hpp"
#include "structural/csr.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace structural {

template <typename T>
BitBsrMatrix<T> convert_to_bitbsr(const CsrMatrix<T>& csr, Index block_rows, Index block_cols) {
    csr.validate();
    if (block_rows <= 0 || block_cols <= 0) {
        throw std::runtime_error("BitBSR block sizes must be positive");
    }

    const Index num_block_rows = ceil_div(csr.rows, block_rows);
    const Index num_block_cols = ceil_div(csr.cols, block_cols);
    const Index words_per_block = ceil_div(block_rows * block_cols, kBitmapWordBits);

    BitBsrMatrix<T> out;
    out.rows = csr.rows;
    out.cols = csr.cols;
    out.block_rows = block_rows;
    out.block_cols = block_cols;
    out.num_block_rows = num_block_rows;
    out.num_block_cols = num_block_cols;
    out.words_per_block = words_per_block;

    std::vector<std::vector<Index>> active_cols_by_br(num_block_rows);
    std::vector<std::vector<Offset>> block_counts_by_br(num_block_rows);
    std::vector<Offset> block_counts_prefix(num_block_rows + 1, 0);
    std::vector<Offset> nnz_prefix(num_block_rows + 1, 0);

#pragma omp parallel for schedule(dynamic)
    for (Index br = 0; br < num_block_rows; ++br) {
        const Index row_begin = br * block_rows;
        const Index row_end = std::min(csr.rows, row_begin + block_rows);
        Offset block_row_nnz = 0;
        for (Index row = row_begin; row < row_end; ++row) {
            for (Offset p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
                if (csr.values[p] != T{}) {
                    ++block_row_nnz;
                }
            }
        }

        std::vector<Index> block_cols_seen;
        block_cols_seen.reserve(static_cast<size_t>(block_row_nnz));
        for (Index row = row_begin; row < row_end; ++row) {
            for (Offset p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
                if (csr.values[p] == T{}) {
                    continue;
                }
                block_cols_seen.push_back(csr.col_idx[p] / block_cols);
            }
        }

        std::sort(block_cols_seen.begin(), block_cols_seen.end());

        auto& active_cols = active_cols_by_br[br];
        auto& block_counts = block_counts_by_br[br];
        active_cols.reserve(block_cols_seen.size());
        block_counts.reserve(block_cols_seen.size());

        size_t cursor = 0;
        while (cursor < block_cols_seen.size()) {
            const Index block_col = block_cols_seen[cursor];
            size_t next = cursor + 1;
            while (next < block_cols_seen.size() && block_cols_seen[next] == block_col) {
                ++next;
            }
            active_cols.push_back(block_col);
            block_counts.push_back(static_cast<Offset>(next - cursor));
            cursor = next;
        }

        block_counts_prefix[br + 1] = static_cast<Offset>(active_cols.size());
        nnz_prefix[br + 1] = block_row_nnz;
    }

    for (Index br = 0; br < num_block_rows; ++br) {
        block_counts_prefix[br + 1] += block_counts_prefix[br];
        nnz_prefix[br + 1] += nnz_prefix[br];
    }

    const Offset total_blocks = block_counts_prefix.back();
    out.block_row_ptr = block_counts_prefix;
    out.block_col_idx.resize(total_blocks);
    out.bitmap_words.assign(static_cast<size_t>(total_blocks) * words_per_block, BitmapWord{0});
    out.block_val_ptr.assign(total_blocks + 1, 0);
    out.values.resize(static_cast<size_t>(nnz_prefix.back()));

#pragma omp parallel for schedule(static)
    for (Index br = 0; br < num_block_rows; ++br) {
        const Offset first_block = out.block_row_ptr[br];
        const auto& active_cols = active_cols_by_br[br];
        const auto& block_counts = block_counts_by_br[br];

        Offset value_offset = nnz_prefix[br];
        for (size_t i = 0; i < active_cols.size(); ++i) {
            const Offset block_id = first_block + static_cast<Offset>(i);
            out.block_col_idx[block_id] = active_cols[i];
            out.block_val_ptr[block_id] = value_offset;
            value_offset += block_counts[i];
        }
        if (!active_cols.empty()) {
            out.block_val_ptr[first_block + static_cast<Offset>(active_cols.size())] = value_offset;
        }
    }
    out.block_val_ptr[total_blocks] = static_cast<Offset>(out.values.size());

#pragma omp parallel for schedule(dynamic)
    for (Index br = 0; br < num_block_rows; ++br) {
        const Offset first_block = out.block_row_ptr[br];
        const auto& active_cols = active_cols_by_br[br];
        std::vector<Offset> write_pos(active_cols.size());
        for (size_t i = 0; i < active_cols.size(); ++i) {
            write_pos[i] = out.block_val_ptr[first_block + static_cast<Offset>(i)];
        }

        const Index row_begin = br * block_rows;
        const Index row_end = std::min(csr.rows, row_begin + block_rows);
        for (Index row = row_begin; row < row_end; ++row) {
            const Index local_row = row - row_begin;
            for (Offset p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
                if (csr.values[p] == T{}) {
                    continue;
                }
                const Index col = csr.col_idx[p];
                const Index block_col = col / block_cols;
                const Index local_col = col % block_cols;
                const Index bit = local_row * block_cols + local_col;
                const auto found = std::lower_bound(active_cols.begin(), active_cols.end(), block_col);
                if (found == active_cols.end() || *found != block_col) {
                    throw std::runtime_error("internal BitBSR block lookup failed");
                }
                const Offset local_block = static_cast<Offset>(found - active_cols.begin());
                const Offset block_id = first_block + local_block;
                const Index word = bit / kBitmapWordBits;
                const Index lane = bit % kBitmapWordBits;
                BitmapWord& bitmap_word = out.bitmap_words[block_id * words_per_block + word];
                const BitmapWord mask = BitmapWord{1} << lane;
                if ((bitmap_word & mask) != 0) {
                    throw std::runtime_error("duplicate CSR entry maps to the same BitBSR bit");
                }
                bitmap_word |= mask;
                out.values[write_pos[local_block]++] = csr.values[p];
            }
        }

        for (size_t i = 0; i < active_cols.size(); ++i) {
            const Offset block_id = first_block + static_cast<Offset>(i);
            if (write_pos[i] != out.block_val_ptr[block_id + 1]) {
                throw std::runtime_error("BitBSR block value fill count mismatch");
            }
        }
    }

    out.validate();
    return out;
}

}  // namespace structural
