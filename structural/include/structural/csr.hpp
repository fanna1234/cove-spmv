#pragma once

#include "structural/types.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace structural {

template <typename T>
struct CsrMatrix {
    Index rows = 0;
    Index cols = 0;
    std::vector<Offset> row_ptr;
    std::vector<Index> col_idx;
    std::vector<T> values;

    Offset nnz() const {
        return static_cast<Offset>(values.size());
    }

    void validate() const {
        if (rows < 0 || cols < 0) {
            throw std::runtime_error("CSR dimensions must be non-negative");
        }
        if (static_cast<Index>(row_ptr.size()) != rows + 1) {
            throw std::runtime_error("CSR row_ptr size must be rows + 1");
        }
        if (col_idx.size() != values.size()) {
            throw std::runtime_error("CSR col_idx and values sizes differ");
        }
        if (!row_ptr.empty() && row_ptr.front() != 0) {
            throw std::runtime_error("CSR row_ptr must start at zero");
        }
        for (Index r = 0; r < rows; ++r) {
            if (row_ptr[r] > row_ptr[r + 1]) {
                throw std::runtime_error("CSR row_ptr must be monotonic");
            }
        }
        if (!row_ptr.empty() && row_ptr.back() != nnz()) {
            throw std::runtime_error("CSR row_ptr.back must equal nnz");
        }
        for (Index c : col_idx) {
            if (c < 0 || c >= cols) {
                throw std::runtime_error("CSR column index out of range");
            }
        }
    }
};

template <typename T>
bool has_stored_zeros(const CsrMatrix<T>& csr) {
    for (const T value : csr.values) {
        if (value == T{}) {
            return true;
        }
    }
    return false;
}

template <typename T>
CsrMatrix<T> remove_stored_zeros(const CsrMatrix<T>& csr) {
    csr.validate();

    CsrMatrix<T> out;
    out.rows = csr.rows;
    out.cols = csr.cols;
    out.row_ptr.assign(static_cast<size_t>(csr.rows) + 1, 0);
    out.col_idx.reserve(csr.col_idx.size());
    out.values.reserve(csr.values.size());

    for (Index row = 0; row < csr.rows; ++row) {
        for (Offset p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
            if (csr.values[p] == T{}) {
                continue;
            }
            out.col_idx.push_back(csr.col_idx[p]);
            out.values.push_back(csr.values[p]);
        }
        out.row_ptr[row + 1] = static_cast<Offset>(out.values.size());
    }
    out.validate();
    return out;
}

}  // namespace structural
