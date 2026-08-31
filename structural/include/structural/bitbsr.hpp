#pragma once

#include "structural/types.hpp"

#include <stdexcept>
#include <vector>

namespace structural {

inline Index ceil_div(Index value, Index divisor) {
    return (value + divisor - 1) / divisor;
}

inline int popcount_word(BitmapWord word) {
    return __builtin_popcount(static_cast<unsigned>(word));
}

inline int ctz_word(BitmapWord word) {
    return __builtin_ctz(static_cast<unsigned>(word));
}

template <typename T>
struct BitBsrMatrix {
    Index rows = 0;
    Index cols = 0;
    Index block_rows = 0;
    Index block_cols = 0;
    Index num_block_rows = 0;
    Index num_block_cols = 0;
    Index words_per_block = 0;

    std::vector<Offset> block_row_ptr;
    std::vector<Index> block_col_idx;
    std::vector<BitmapWord> bitmap_words;
    std::vector<Offset> block_val_ptr;
    std::vector<T> values;

    Offset num_blocks() const {
        return static_cast<Offset>(block_col_idx.size());
    }

    Offset nnz() const {
        return static_cast<Offset>(values.size());
    }

    void validate() const {
        if (rows < 0 || cols < 0 || block_rows <= 0 || block_cols <= 0) {
            throw std::runtime_error("BitBSR dimensions or block sizes are invalid");
        }
        if (num_block_rows != ceil_div(rows, block_rows) ||
            num_block_cols != ceil_div(cols, block_cols)) {
            throw std::runtime_error("BitBSR block-grid dimensions are inconsistent");
        }
        if (words_per_block != ceil_div(block_rows * block_cols, kBitmapWordBits)) {
            throw std::runtime_error("BitBSR words_per_block is inconsistent");
        }
        if (static_cast<Index>(block_row_ptr.size()) != num_block_rows + 1) {
            throw std::runtime_error("BitBSR block_row_ptr size is invalid");
        }
        if (block_val_ptr.size() != block_col_idx.size() + 1) {
            throw std::runtime_error("BitBSR block_val_ptr size is invalid");
        }
        if (bitmap_words.size() != block_col_idx.size() * static_cast<size_t>(words_per_block)) {
            throw std::runtime_error("BitBSR bitmap_words size is invalid");
        }
        if (!block_row_ptr.empty() && block_row_ptr.front() != 0) {
            throw std::runtime_error("BitBSR block_row_ptr must start at zero");
        }
        for (Index br = 0; br < num_block_rows; ++br) {
            if (block_row_ptr[br] > block_row_ptr[br + 1]) {
                throw std::runtime_error("BitBSR block_row_ptr must be monotonic");
            }
        }
        if (!block_row_ptr.empty() && block_row_ptr.back() != num_blocks()) {
            throw std::runtime_error("BitBSR block_row_ptr.back must equal num_blocks");
        }
        if (!block_val_ptr.empty() && block_val_ptr.front() != 0) {
            throw std::runtime_error("BitBSR block_val_ptr must start at zero");
        }
        for (Offset b = 0; b < num_blocks(); ++b) {
            if (block_col_idx[b] < 0 || block_col_idx[b] >= num_block_cols) {
                throw std::runtime_error("BitBSR block_col_idx out of range");
            }
            if (block_val_ptr[b] > block_val_ptr[b + 1]) {
                throw std::runtime_error("BitBSR block_val_ptr must be monotonic");
            }

            Offset bits = 0;
            for (Index w = 0; w < words_per_block; ++w) {
                bits += popcount_word(bitmap_words[b * words_per_block + w]);
            }
            if (bits != block_val_ptr[b + 1] - block_val_ptr[b]) {
                throw std::runtime_error("BitBSR bitmap popcount and value range differ");
            }
        }
        if (!block_val_ptr.empty() && block_val_ptr.back() != nnz()) {
            throw std::runtime_error("BitBSR block_val_ptr.back must equal nnz");
        }
        for (const T value : values) {
            if (value == T{}) {
                throw std::runtime_error("BitBSR must not store zero values");
            }
        }
    }
};

}  // namespace structural
