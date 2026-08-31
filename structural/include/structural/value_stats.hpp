#pragma once

#include "structural/csr.hpp"

#include <algorithm>
#include <vector>

namespace structural {

template <typename T>
struct ValueStats {
    Offset nnz = 0;
    Offset explicit_zero_count = 0;
    int unique_values_exact = 0;
    bool unique_values_overflow = false;
    bool all_values_one = false;
    bool binary_zero_one = false;
    bool sign_only = false;
    T min_value = T{};
    T max_value = T{};
};

template <typename T>
ValueStats<T> compute_value_stats(const CsrMatrix<T>& csr, int unique_cap = 16) {
    csr.validate();
    if (unique_cap <= 0) {
        unique_cap = 1;
    }

    ValueStats<T> stats;
    stats.nnz = csr.nnz();
    if (csr.values.empty()) {
        return stats;
    }

    stats.min_value = csr.values.front();
    stats.max_value = csr.values.front();
    stats.all_values_one = true;
    stats.binary_zero_one = true;

    bool sign_candidate = true;
    bool has_negative_one = false;
    std::vector<T> unique_values;
    unique_values.reserve(static_cast<size_t>(std::min(unique_cap, stats.nnz)));

    for (const T value : csr.values) {
        stats.min_value = std::min(stats.min_value, value);
        stats.max_value = std::max(stats.max_value, value);

        if (value == T{}) {
            ++stats.explicit_zero_count;
        }
        if (value != T{1}) {
            stats.all_values_one = false;
        }
        if (!(value == T{} || value == T{1})) {
            stats.binary_zero_one = false;
        }
        if (!(value == T{-1} || value == T{} || value == T{1})) {
            sign_candidate = false;
        }
        if (value == T{-1}) {
            has_negative_one = true;
        }

        if (std::find(unique_values.begin(), unique_values.end(), value) == unique_values.end()) {
            if (static_cast<int>(unique_values.size()) < unique_cap) {
                unique_values.push_back(value);
            } else {
                stats.unique_values_overflow = true;
            }
        }
    }

    stats.unique_values_exact = static_cast<int>(unique_values.size());
    stats.sign_only = sign_candidate && has_negative_one;
    return stats;
}

}  // namespace structural
