#pragma once

#include "structural/bitbsr.hpp"
#include "structural/csr.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace structural {

template <typename T>
std::vector<T> spmv_csr(const CsrMatrix<T>& csr, const std::vector<T>& x) {
    csr.validate();
    if (static_cast<Index>(x.size()) != csr.cols) {
        throw std::runtime_error("CSR SpMV input vector size mismatch");
    }

    std::vector<T> y(csr.rows, T{0});
#pragma omp parallel for schedule(static)
    for (Index row = 0; row < csr.rows; ++row) {
        T sum = T{0};
        for (Offset p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
            sum += csr.values[p] * x[csr.col_idx[p]];
        }
        y[row] = sum;
    }
    return y;
}

template <typename T>
std::vector<T> spmv_bitbsr(const BitBsrMatrix<T>& bitbsr, const std::vector<T>& x) {
    bitbsr.validate();
    if (static_cast<Index>(x.size()) != bitbsr.cols) {
        throw std::runtime_error("BitBSR SpMV input vector size mismatch");
    }

    std::vector<T> y(bitbsr.rows, T{0});
#pragma omp parallel for schedule(dynamic)
    for (Index br = 0; br < bitbsr.num_block_rows; ++br) {
        for (Offset b = bitbsr.block_row_ptr[br]; b < bitbsr.block_row_ptr[br + 1]; ++b) {
            const Index bc = bitbsr.block_col_idx[b];
            Offset rank = 0;
            for (Index w = 0; w < bitbsr.words_per_block; ++w) {
                BitmapWord word = bitbsr.bitmap_words[b * bitbsr.words_per_block + w];
                while (word != 0) {
                    const Index bit_in_word = ctz_word(word);
                    const Index bit = w * kBitmapWordBits + bit_in_word;
                    const Index local_row = bit / bitbsr.block_cols;
                    const Index local_col = bit % bitbsr.block_cols;
                    const Index row = br * bitbsr.block_rows + local_row;
                    const Index col = bc * bitbsr.block_cols + local_col;
                    y[row] += bitbsr.values[bitbsr.block_val_ptr[b] + rank] * x[col];
                    ++rank;
                    word &= word - 1;
                }
            }
        }
    }
    return y;
}

template <typename T>
struct ErrorStats {
    T max_abs = T{0};
    T max_rel = T{0};
    // Global-norm output relative error: max_r|y[r]-y0[r]| / max_r(|y0[r]|, tiny).
    // Paper-facing accuracy metric for controlled-lossy value codecs (does not blow
    // up on near-zero output rows the way the per-element max_rel does).
    T global_rel = T{0};
};

template <typename T>
ErrorStats<T> compare_vectors(const std::vector<T>& ref, const std::vector<T>& got) {
    if (ref.size() != got.size()) {
        throw std::runtime_error("vector size mismatch");
    }
    ErrorStats<T> stats;
    T max_abs_ref = T{0};
    for (size_t i = 0; i < ref.size(); ++i) {
        const T abs_err = std::abs(ref[i] - got[i]);
        const T denom = std::max<T>(std::abs(ref[i]), T{1e-30});
        stats.max_abs = std::max(stats.max_abs, abs_err);
        stats.max_rel = std::max(stats.max_rel, abs_err / denom);
        max_abs_ref = std::max(max_abs_ref, std::abs(ref[i]));
    }
    stats.global_rel = stats.max_abs / std::max<T>(max_abs_ref, T{1e-30});
    return stats;
}

}  // namespace structural
