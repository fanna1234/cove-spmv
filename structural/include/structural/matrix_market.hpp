#pragma once

#include "structural/csr.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace structural {

enum class MatrixMarketField {
    Real,
    Integer,
    Complex,
    Pattern,
};

enum class MatrixMarketSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
};

inline const char* to_string(MatrixMarketField field) {
    switch (field) {
        case MatrixMarketField::Real:
            return "real";
        case MatrixMarketField::Integer:
            return "integer";
        case MatrixMarketField::Complex:
            return "complex";
        case MatrixMarketField::Pattern:
            return "pattern";
    }
    return "unknown";
}

inline const char* to_string(MatrixMarketSymmetry symmetry) {
    switch (symmetry) {
        case MatrixMarketSymmetry::General:
            return "general";
        case MatrixMarketSymmetry::Symmetric:
            return "symmetric";
        case MatrixMarketSymmetry::SkewSymmetric:
            return "skew-symmetric";
        case MatrixMarketSymmetry::Hermitian:
            return "hermitian";
    }
    return "unknown";
}

struct MatrixMarketInfo {
    Index rows = 0;
    Index cols = 0;
    Offset nnz_file = 0;
    MatrixMarketField field = MatrixMarketField::Real;
    MatrixMarketSymmetry symmetry = MatrixMarketSymmetry::General;

    bool is_pattern() const {
        return field == MatrixMarketField::Pattern;
    }

    bool requires_mirror_expansion() const {
        return symmetry == MatrixMarketSymmetry::Symmetric ||
               symmetry == MatrixMarketSymmetry::SkewSymmetric ||
               symmetry == MatrixMarketSymmetry::Hermitian;
    }

    bool mirrors_with_negation() const {
        return symmetry == MatrixMarketSymmetry::SkewSymmetric;
    }
};

template <typename T>
struct MatrixMarketLoadResult {
    MatrixMarketInfo info;
    CsrMatrix<T> matrix;
    Offset stored_entries = 0;
    Offset expanded_entries = 0;
    Offset coalesced_duplicates = 0;
};

namespace detail {

inline std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline bool is_comment_or_empty(const std::string& line) {
    return line.empty() || line[0] == '%';
}

template <typename T>
struct CooEntry {
    Index row = 0;
    Index col = 0;
    T value = T{};
};

template <typename T>
struct ParsedEntry {
    Index row = 0;
    Index col = 0;
    T value = T{1};
};

template <typename T>
ParsedEntry<T> parse_entry_line(const std::string& line, MatrixMarketField field) {
    const char* ptr = line.c_str();
    char* end = nullptr;

    const long row = std::strtol(ptr, &end, 10);
    if (end == ptr) {
        throw std::runtime_error("invalid MatrixMarket row index");
    }
    ptr = end;

    const long col = std::strtol(ptr, &end, 10);
    if (end == ptr) {
        throw std::runtime_error("invalid MatrixMarket column index");
    }
    ptr = end;
    if (row <= 0 || row > std::numeric_limits<Index>::max() ||
        col <= 0 || col > std::numeric_limits<Index>::max()) {
        throw std::runtime_error("MatrixMarket entry index out of Index range");
    }

    T value = T{1};
    if (field == MatrixMarketField::Complex) {
        throw std::runtime_error("MatrixMarket complex field is not supported by this scalar loader");
    }
    if (field != MatrixMarketField::Pattern) {
        const double parsed = std::strtod(ptr, &end);
        if (end == ptr) {
            throw std::runtime_error("non-pattern MatrixMarket entry missing value");
        }
        value = static_cast<T>(parsed);
    }

    return {static_cast<Index>(row - 1), static_cast<Index>(col - 1), value};
}

inline MatrixMarketField parse_field_token(const std::string& token) {
    if (token == "real") {
        return MatrixMarketField::Real;
    }
    if (token == "integer") {
        return MatrixMarketField::Integer;
    }
    if (token == "complex") {
        return MatrixMarketField::Complex;
    }
    if (token == "pattern") {
        return MatrixMarketField::Pattern;
    }
    throw std::runtime_error("unsupported MatrixMarket field: " + token);
}

inline MatrixMarketSymmetry parse_symmetry_token(const std::string& token) {
    if (token == "general") {
        return MatrixMarketSymmetry::General;
    }
    if (token == "symmetric") {
        return MatrixMarketSymmetry::Symmetric;
    }
    if (token == "skew-symmetric") {
        return MatrixMarketSymmetry::SkewSymmetric;
    }
    if (token == "hermitian") {
        return MatrixMarketSymmetry::Hermitian;
    }
    throw std::runtime_error("unsupported MatrixMarket symmetry: " + token);
}

inline Index parse_positive_int(const char*& ptr, const char* what) {
    char* end = nullptr;
    const long value = std::strtol(ptr, &end, 10);
    if (end == ptr) {
        throw std::runtime_error(std::string("invalid MatrixMarket ") + what);
    }
    if (value < 0 || value > std::numeric_limits<Index>::max()) {
        throw std::runtime_error(std::string("MatrixMarket ") + what + " out of Index range");
    }
    ptr = end;
    return static_cast<Index>(value);
}

inline MatrixMarketInfo read_header_and_size(std::ifstream& in, const std::string& path) {
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty MatrixMarket file: " + path);
    }

    std::istringstream header_stream(lower_copy(line));
    std::string banner;
    std::string object;
    std::string format;
    std::string field_token;
    std::string symmetry_token;
    if (!(header_stream >> banner >> object >> format >> field_token >> symmetry_token)) {
        throw std::runtime_error("invalid MatrixMarket header: " + path);
    }
    if (banner != "%%matrixmarket" || object != "matrix" || format != "coordinate") {
        throw std::runtime_error("only MatrixMarket coordinate matrices are supported");
    }

    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("missing MatrixMarket size line");
        }
    } while (is_comment_or_empty(line));

    const char* ptr = line.c_str();
    MatrixMarketInfo info;
    info.rows = parse_positive_int(ptr, "row count");
    info.cols = parse_positive_int(ptr, "column count");
    info.nnz_file = parse_positive_int(ptr, "nnz count");
    info.field = parse_field_token(field_token);
    info.symmetry = parse_symmetry_token(symmetry_token);

    if (info.field == MatrixMarketField::Complex || info.symmetry == MatrixMarketSymmetry::Hermitian) {
        throw std::runtime_error("complex/hermitian MatrixMarket matrices need a complex loader");
    }
    if (info.field == MatrixMarketField::Pattern &&
        info.symmetry == MatrixMarketSymmetry::SkewSymmetric) {
        throw std::runtime_error("pattern skew-symmetric MatrixMarket matrices have no signed value");
    }
    return info;
}

template <typename T>
T mirrored_value(const MatrixMarketInfo& info, T value) {
    if (info.mirrors_with_negation()) {
        return -value;
    }
    return value;
}

template <typename T, typename Fn>
void for_each_matrix_market_entry(std::ifstream& in, const MatrixMarketInfo& info, Fn&& fn) {
    std::string line;
    Offset entries_read = 0;
    while (entries_read < info.nnz_file && std::getline(in, line)) {
        if (is_comment_or_empty(line)) {
            continue;
        }

        const auto entry = parse_entry_line<T>(line, info.field);
        if (entry.row < 0 || entry.row >= info.rows || entry.col < 0 || entry.col >= info.cols) {
            throw std::runtime_error("MatrixMarket entry index out of range");
        }
        fn(entry.row, entry.col, entry.value);
        if (info.requires_mirror_expansion() && entry.row != entry.col) {
            fn(entry.col, entry.row, mirrored_value(info, entry.value));
        }
        ++entries_read;
    }
    if (entries_read != info.nnz_file) {
        throw std::runtime_error("MatrixMarket ended before all entries were read");
    }
}

template <typename T>
void sort_and_coalesce_rows(CsrMatrix<T>& csr) {
    std::vector<Offset> new_row_ptr(csr.rows + 1, 0);
    std::vector<Index> new_col_idx;
    std::vector<T> new_values;
    new_col_idx.reserve(csr.col_idx.size());
    new_values.reserve(csr.values.size());

    std::vector<std::pair<Index, T>> row_entries;
    for (Index row = 0; row < csr.rows; ++row) {
        row_entries.clear();
        const Offset begin = csr.row_ptr[row];
        const Offset end = csr.row_ptr[row + 1];
        row_entries.reserve(static_cast<size_t>(end - begin));
        for (Offset p = begin; p < end; ++p) {
            row_entries.push_back({csr.col_idx[p], csr.values[p]});
        }

        std::sort(row_entries.begin(), row_entries.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        for (const auto& [col, value] : row_entries) {
            if (!new_col_idx.empty() &&
                new_row_ptr[row] < static_cast<Offset>(new_col_idx.size()) &&
                new_col_idx.back() == col) {
                new_values.back() += value;
            } else {
                new_col_idx.push_back(col);
                new_values.push_back(value);
            }
        }
        Offset write = new_row_ptr[row];
        for (Offset read = new_row_ptr[row]; read < static_cast<Offset>(new_values.size()); ++read) {
            if (new_values[read] == T{}) {
                continue;
            }
            new_col_idx[write] = new_col_idx[read];
            new_values[write] = new_values[read];
            ++write;
        }
        new_col_idx.resize(static_cast<size_t>(write));
        new_values.resize(static_cast<size_t>(write));
        new_row_ptr[row + 1] = static_cast<Offset>(new_col_idx.size());
    }

    csr.row_ptr = std::move(new_row_ptr);
    csr.col_idx = std::move(new_col_idx);
    csr.values = std::move(new_values);
}

}  // namespace detail

template <typename T>
MatrixMarketLoadResult<T> read_matrix_market_with_info(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open MatrixMarket file: " + path);
    }

    const auto info = detail::read_header_and_size(in, path);

    CsrMatrix<T> csr;
    csr.rows = info.rows;
    csr.cols = info.cols;
    csr.row_ptr.assign(csr.rows + 1, 0);

    detail::for_each_matrix_market_entry<T>(in, info, [&](Index row, Index, T) {
        ++csr.row_ptr[row + 1];
    });

    for (Index row = 0; row < csr.rows; ++row) {
        csr.row_ptr[row + 1] += csr.row_ptr[row];
    }

    const Offset expanded_entries = csr.row_ptr.back();
    csr.col_idx.resize(csr.row_ptr.back());
    csr.values.resize(csr.row_ptr.back());

    in.close();
    in.open(path);
    if (!in) {
        throw std::runtime_error("cannot reopen MatrixMarket file: " + path);
    }
    const auto fill_info = detail::read_header_and_size(in, path);
    if (fill_info.rows != info.rows || fill_info.cols != info.cols ||
        fill_info.nnz_file != info.nnz_file || fill_info.field != info.field ||
        fill_info.symmetry != info.symmetry) {
        throw std::runtime_error("MatrixMarket header changed while reading: " + path);
    }

    std::vector<Offset> next = csr.row_ptr;
    detail::for_each_matrix_market_entry<T>(in, info, [&](Index row, Index col, T value) {
        const Offset dst = next[row]++;
        csr.col_idx[dst] = col;
        csr.values[dst] = value;
    });

    detail::sort_and_coalesce_rows(csr);

    csr.validate();
    MatrixMarketLoadResult<T> result;
    result.info = info;
    result.matrix = std::move(csr);
    result.stored_entries = info.nnz_file;
    result.expanded_entries = expanded_entries;
    result.coalesced_duplicates = expanded_entries - result.matrix.nnz();
    return result;
}

template <typename T>
CsrMatrix<T> read_matrix_market(const std::string& path) {
    return read_matrix_market_with_info<T>(path).matrix;
}

}  // namespace structural
