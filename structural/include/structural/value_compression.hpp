#pragma once

#include "structural/csr.hpp"
#include "structural/spmv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace structural {

constexpr int kAbsLog2BinCount = 18;
constexpr int kK256ValueCodebookSize = 256;

struct ValueCompressionOptions {
    double spmv_rel_gate = 1e-6;
    std::vector<int> cluster_sizes = {16, 32, 64, 128};
    int cluster_histogram_bins = 4096;
    int cluster_iterations = 16;
};

struct ValueCodecReport {
    std::string name;
    bool applicable = false;
    bool exact = false;
    bool passes_spmv_gate = false;
    std::int64_t payload_bits = 0;
    double bytes_per_nnz = 0.0;
    double compression_ratio_vs_raw = 0.0;
    int codebook_size = 0;
    int index_bits = 0;
    bool uses_sign_bit = false;
    Offset changed_values = 0;
    double changed_ratio = 0.0;
    double max_abs_value_error = 0.0;
    double max_rel_value_error = 0.0;
    double spmv_max_abs = 0.0;
    double spmv_max_rel = 0.0;
};

struct ValueCompressionAnalysis {
    Offset nnz = 0;
    double raw_bytes_per_nnz = 0.0;
    double value_entropy_bits = 0.0;
    double abs_log2_entropy_bits = 0.0;
    std::array<Offset, kAbsLog2BinCount> abs_log2_bins{};
    std::vector<ValueCodecReport> codecs;
};

template <typename T>
struct K256UInt8ValueCodebook {
    std::array<T, kK256ValueCodebookSize> codebook{};
    std::vector<std::uint8_t> ids;
    int codebook_size = 0;
    bool exact = false;

    Offset nnz() const {
        return static_cast<Offset>(ids.size());
    }

    std::size_t payload_bytes() const {
        return ids.size() + codebook.size() * sizeof(T);
    }

    std::vector<T> reconstruct() const {
        std::vector<T> values;
        values.reserve(ids.size());
        for (const std::uint8_t id : ids) {
            values.push_back(codebook[static_cast<size_t>(id)]);
        }
        return values;
    }

    void validate() const {
        if (codebook_size < 0 || codebook_size > kK256ValueCodebookSize) {
            throw std::runtime_error("K256 uint8 codebook size is invalid");
        }
        for (const std::uint8_t id : ids) {
            if (static_cast<int>(id) >= codebook_size) {
                throw std::runtime_error("K256 uint8 value id is outside the codebook");
            }
        }
    }
};

inline std::uint32_t float_to_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float bits_to_float(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline std::uint16_t float_to_bf16_bits(float value) {
    const std::uint32_t bits = float_to_bits(value);
    const std::uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7fffU + lsb) >> 16);
}

inline float bf16_bits_to_float(std::uint16_t bits) {
    return bits_to_float(static_cast<std::uint32_t>(bits) << 16);
}

inline double round_to_bf16(double value) {
    const float f = static_cast<float>(value);
    return static_cast<double>(bf16_bits_to_float(float_to_bf16_bits(f)));
}

inline std::uint16_t float_to_fp16_bits(float value) {
    const std::uint32_t bits = float_to_bits(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
    const int exp = static_cast<int>((bits >> 23) & 0xffU);
    const std::uint32_t mant = bits & 0x7fffffU;

    if (exp == 0xff) {
        if (mant == 0) {
            return static_cast<std::uint16_t>(sign | 0x7c00U);
        }
        return static_cast<std::uint16_t>(sign | 0x7e00U);
    }

    int half_exp = exp - 127 + 15;
    if (half_exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }

    if (half_exp <= 0) {
        if (half_exp < -10) {
            return sign;
        }
        const std::uint32_t mant_with_hidden = mant | 0x800000U;
        const int shift = 14 - half_exp;
        std::uint32_t rounded = mant_with_hidden >> shift;
        const std::uint32_t remainder = mant_with_hidden & ((1U << shift) - 1U);
        const std::uint32_t halfway = 1U << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | rounded);
    }

    std::uint32_t half_mant = mant >> 13;
    const std::uint32_t remainder = mant & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half_mant & 1U))) {
        ++half_mant;
        if (half_mant == 0x400U) {
            half_mant = 0;
            ++half_exp;
            if (half_exp >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00U);
            }
        }
    }

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(half_exp) << 10) |
                                      static_cast<std::uint16_t>(half_mant));
}

inline float fp16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    int exp = static_cast<int>((bits >> 10) & 0x1fU);
    std::uint32_t mant = bits & 0x03ffU;

    if (exp == 0) {
        if (mant == 0) {
            return bits_to_float(sign);
        }
        int exp32 = -14;
        while ((mant & 0x0400U) == 0) {
            mant <<= 1;
            --exp32;
        }
        mant &= 0x03ffU;
        return bits_to_float(sign | (static_cast<std::uint32_t>(exp32 + 127) << 23) |
                             (mant << 13));
    }
    if (exp == 31) {
        return bits_to_float(sign | 0x7f800000U | (mant << 13));
    }

    exp = exp - 15 + 127;
    return bits_to_float(sign | (static_cast<std::uint32_t>(exp) << 23) | (mant << 13));
}

inline double round_to_fp16(double value) {
    const float f = static_cast<float>(value);
    return static_cast<double>(fp16_bits_to_float(float_to_fp16_bits(f)));
}

inline int ceil_log2_int(int value) {
    if (value <= 1) {
        return 0;
    }
    int bits = 0;
    --value;
    while (value > 0) {
        ++bits;
        value >>= 1;
    }
    return bits;
}

inline int nearest_sorted_center(const std::vector<double>& centers, double value) {
    const auto it = std::lower_bound(centers.begin(), centers.end(), value);
    if (it == centers.begin()) {
        return 0;
    }
    if (it == centers.end()) {
        return static_cast<int>(centers.size() - 1);
    }
    const int hi = static_cast<int>(it - centers.begin());
    const int lo = hi - 1;
    return std::abs(value - centers[lo]) <= std::abs(centers[hi] - value) ? lo : hi;
}

template <typename T>
std::vector<T> make_value_probe_vector(Index cols) {
    if (cols < 0) {
        throw std::runtime_error("probe vector dimension must be non-negative");
    }
    std::vector<T> x(static_cast<size_t>(cols));
    for (Index c = 0; c < cols; ++c) {
        std::uint32_t h = static_cast<std::uint32_t>(c) * 2654435761U + 1013904223U;
        h ^= h >> 16;
        const int centered = static_cast<int>(h % 2001U) - 1000;
        const double value = (centered == 0 ? 1.0 : static_cast<double>(centered)) / 257.0;
        x[static_cast<size_t>(c)] = static_cast<T>(value);
    }
    return x;
}

template <typename T>
std::vector<double> collect_nonzero_log_abs_values(const CsrMatrix<T>& csr, int unique_cap,
                                                   bool* all_finite) {
    std::vector<double> logs;
    logs.reserve(static_cast<size_t>(csr.nnz()));
    *all_finite = true;
    for (const T raw : csr.values) {
        const double value = static_cast<double>(raw);
        if (!std::isfinite(value)) {
            *all_finite = false;
            continue;
        }
        if (value != 0.0) {
            logs.push_back(std::log2(std::abs(value)));
        }
    }
    if (static_cast<int>(logs.size()) <= unique_cap) {
        return logs;
    }
    return logs;
}

inline std::vector<double> make_log_kmeans_centers(std::vector<double> logs, int requested_k,
                                                   int histogram_bins, int iterations) {
    if (logs.empty() || requested_k <= 0) {
        return {};
    }

    std::sort(logs.begin(), logs.end());
    std::vector<double> unique_logs;
    unique_logs.reserve(static_cast<size_t>(requested_k));
    bool unique_overflow = false;
    for (const double value : logs) {
        if (unique_logs.empty() || value != unique_logs.back()) {
            unique_logs.push_back(value);
            if (static_cast<int>(unique_logs.size()) > requested_k) {
                unique_overflow = true;
                break;
            }
        }
    }
    if (!unique_overflow) {
        return unique_logs;
    }

    const int k = std::min<int>(requested_k, static_cast<int>(logs.size()));
    if (k <= 1 || logs.front() == logs.back()) {
        return {logs[logs.size() / 2]};
    }

    histogram_bins = std::max(k * 16, std::min(histogram_bins, static_cast<int>(logs.size())));
    const double min_log = logs.front();
    const double max_log = logs.back();
    const double inv_width = static_cast<double>(histogram_bins - 1) / (max_log - min_log);
    std::vector<Offset> bin_counts(static_cast<size_t>(histogram_bins), 0);
    std::vector<double> bin_sums(static_cast<size_t>(histogram_bins), 0.0);
    for (const double value : logs) {
        const int bin = std::clamp(static_cast<int>((value - min_log) * inv_width), 0,
                                   histogram_bins - 1);
        ++bin_counts[static_cast<size_t>(bin)];
        bin_sums[static_cast<size_t>(bin)] += value;
    }

    std::vector<double> centers(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        const size_t idx = static_cast<size_t>(
            (static_cast<long long>(i) * static_cast<long long>(logs.size() - 1)) /
            std::max(1, k - 1));
        centers[static_cast<size_t>(i)] = logs[idx];
    }
    std::sort(centers.begin(), centers.end());

    std::vector<double> new_sums(static_cast<size_t>(k));
    std::vector<Offset> new_counts(static_cast<size_t>(k));
    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(new_sums.begin(), new_sums.end(), 0.0);
        std::fill(new_counts.begin(), new_counts.end(), 0);
        for (int bin = 0; bin < histogram_bins; ++bin) {
            const Offset count = bin_counts[static_cast<size_t>(bin)];
            if (count == 0) {
                continue;
            }
            const double location = bin_sums[static_cast<size_t>(bin)] / static_cast<double>(count);
            const int center = nearest_sorted_center(centers, location);
            new_sums[static_cast<size_t>(center)] += bin_sums[static_cast<size_t>(bin)];
            new_counts[static_cast<size_t>(center)] += count;
        }
        for (int i = 0; i < k; ++i) {
            if (new_counts[static_cast<size_t>(i)] != 0) {
                centers[static_cast<size_t>(i)] =
                    new_sums[static_cast<size_t>(i)] /
                    static_cast<double>(new_counts[static_cast<size_t>(i)]);
            }
        }
        std::sort(centers.begin(), centers.end());
    }

    return centers;
}

inline std::vector<double> make_scalar_kmeans_centers(std::vector<double> values,
                                                      int requested_k,
                                                      int histogram_bins,
                                                      int iterations) {
    if (values.empty() || requested_k <= 0) {
        return {};
    }

    std::sort(values.begin(), values.end());
    std::vector<double> unique_values;
    unique_values.reserve(static_cast<size_t>(requested_k));
    bool unique_overflow = false;
    for (const double value : values) {
        if (unique_values.empty() || value != unique_values.back()) {
            unique_values.push_back(value);
            if (static_cast<int>(unique_values.size()) > requested_k) {
                unique_overflow = true;
                break;
            }
        }
    }
    if (!unique_overflow) {
        return unique_values;
    }

    const int k = std::min<int>(requested_k, static_cast<int>(values.size()));
    if (k <= 1 || values.front() == values.back()) {
        return {values[values.size() / 2]};
    }

    histogram_bins = std::max(k * 16, std::min(histogram_bins, static_cast<int>(values.size())));
    const double min_value = values.front();
    const double max_value = values.back();
    const double inv_width = static_cast<double>(histogram_bins - 1) / (max_value - min_value);
    std::vector<Offset> bin_counts(static_cast<size_t>(histogram_bins), 0);
    std::vector<double> bin_sums(static_cast<size_t>(histogram_bins), 0.0);
    for (const double value : values) {
        const int bin = std::clamp(static_cast<int>((value - min_value) * inv_width), 0,
                                   histogram_bins - 1);
        ++bin_counts[static_cast<size_t>(bin)];
        bin_sums[static_cast<size_t>(bin)] += value;
    }

    std::vector<double> centers(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        const size_t idx = static_cast<size_t>(
            (static_cast<long long>(i) * static_cast<long long>(values.size() - 1)) /
            std::max(1, k - 1));
        centers[static_cast<size_t>(i)] = values[idx];
    }
    std::sort(centers.begin(), centers.end());

    std::vector<double> new_sums(static_cast<size_t>(k));
    std::vector<Offset> new_counts(static_cast<size_t>(k));
    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(new_sums.begin(), new_sums.end(), 0.0);
        std::fill(new_counts.begin(), new_counts.end(), 0);
        for (int bin = 0; bin < histogram_bins; ++bin) {
            const Offset count = bin_counts[static_cast<size_t>(bin)];
            if (count == 0) {
                continue;
            }
            const double location = bin_sums[static_cast<size_t>(bin)] / static_cast<double>(count);
            const int center = nearest_sorted_center(centers, location);
            new_sums[static_cast<size_t>(center)] += bin_sums[static_cast<size_t>(bin)];
            new_counts[static_cast<size_t>(center)] += count;
        }
        for (int i = 0; i < k; ++i) {
            if (new_counts[static_cast<size_t>(i)] != 0) {
                centers[static_cast<size_t>(i)] =
                    new_sums[static_cast<size_t>(i)] /
                    static_cast<double>(new_counts[static_cast<size_t>(i)]);
            }
        }
        std::sort(centers.begin(), centers.end());
    }

    return centers;
}

template <typename T>
std::vector<T> sorted_unique_values(const std::vector<T>& values, int unique_cap) {
    std::vector<T> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    std::vector<T> unique_values;
    unique_values.reserve(static_cast<size_t>(std::min<int>(unique_cap, sorted.size())));
    for (const T value : sorted) {
        if (unique_values.empty() || value != unique_values.back()) {
            unique_values.push_back(value);
            if (static_cast<int>(unique_values.size()) > unique_cap) {
                break;
            }
        }
    }
    return unique_values;
}

template <typename T>
int exact_value_id(const std::vector<T>& codebook, T value) {
    for (size_t i = 0; i < codebook.size(); ++i) {
        if (codebook[i] == value) {
            return static_cast<int>(i);
        }
    }
    throw std::runtime_error("K256 uint8 exact value was not found in codebook");
}

template <typename T>
K256UInt8ValueCodebook<T> build_k256_uint8_value_codebook(
    const std::vector<T>& values,
    int histogram_bins = 4096,
    int iterations = 16) {
    K256UInt8ValueCodebook<T> codec;
    codec.ids.resize(values.size());
    if (values.empty()) {
        return codec;
    }

    std::vector<double> finite_values;
    finite_values.reserve(values.size());
    for (const T raw : values) {
        const double value = static_cast<double>(raw);
        if (!std::isfinite(value)) {
            throw std::runtime_error("K256 uint8 value codebook requires finite values");
        }
        finite_values.push_back(value);
    }

    const auto unique_values = sorted_unique_values(values, kK256ValueCodebookSize);
    if (static_cast<int>(unique_values.size()) <= kK256ValueCodebookSize) {
        codec.exact = true;
        codec.codebook_size = static_cast<int>(unique_values.size());
        for (size_t i = 0; i < unique_values.size(); ++i) {
            codec.codebook[i] = unique_values[i];
        }
        for (size_t i = 0; i < values.size(); ++i) {
            codec.ids[i] = static_cast<std::uint8_t>(exact_value_id(unique_values, values[i]));
        }
        codec.validate();
        return codec;
    }

    const auto centers = make_scalar_kmeans_centers(
        std::move(finite_values), kK256ValueCodebookSize, histogram_bins, iterations);
    codec.exact = false;
    codec.codebook_size = static_cast<int>(centers.size());
    for (size_t i = 0; i < centers.size(); ++i) {
        codec.codebook[i] = static_cast<T>(centers[i]);
    }
    for (size_t i = 0; i < values.size(); ++i) {
        const int id = nearest_sorted_center(centers, static_cast<double>(values[i]));
        codec.ids[i] = static_cast<std::uint8_t>(id);
    }
    codec.validate();
    return codec;
}

inline int abs_log2_bin(double value) {
    if (value == 0.0) {
        return 0;
    }
    if (!std::isfinite(value)) {
        return kAbsLog2BinCount - 1;
    }
    const int exponent = static_cast<int>(std::floor(std::log2(std::abs(value))));
    const int bucket = 1 + std::clamp((exponent + 64) / 8, 0, 15);
    return bucket;
}

template <typename T>
double entropy_bits_from_counts(const std::vector<Offset>& counts, Offset total) {
    if (total <= 0) {
        return 0.0;
    }
    double entropy = 0.0;
    for (const Offset count : counts) {
        if (count == 0) {
            continue;
        }
        const double p = static_cast<double>(count) / static_cast<double>(total);
        entropy -= p * std::log2(p);
    }
    return entropy;
}

template <typename T>
double exact_value_entropy_bits(const std::vector<T>& values) {
    if (values.empty()) {
        return 0.0;
    }
    std::unordered_map<T, Offset> counts;
    counts.reserve(values.size());
    for (const T value : values) {
        ++counts[value];
    }
    double entropy = 0.0;
    const double total = static_cast<double>(values.size());
    for (const auto& item : counts) {
        const double p = static_cast<double>(item.second) / total;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

template <typename T>
ValueCodecReport make_exact_codec_report(const std::string& name, bool applicable,
                                         std::int64_t payload_bits, Offset nnz,
                                         double raw_bytes_per_nnz) {
    ValueCodecReport report;
    report.name = name;
    report.applicable = applicable;
    report.exact = applicable;
    report.passes_spmv_gate = applicable;
    report.payload_bits = applicable ? payload_bits : 0;
    if (applicable && nnz > 0) {
        report.bytes_per_nnz = static_cast<double>(payload_bits) / 8.0 / static_cast<double>(nnz);
        report.compression_ratio_vs_raw =
            report.bytes_per_nnz == 0.0 ? std::numeric_limits<double>::infinity()
                                         : raw_bytes_per_nnz / report.bytes_per_nnz;
    }
    return report;
}

template <typename T>
ValueCodecReport make_exact_dict_codec_report(const std::string& name,
                                              int max_codebook_size,
                                              int index_bits,
                                              int unique_value_count,
                                              bool unique_overflow,
                                              Offset nnz,
                                              double raw_bytes_per_nnz) {
    const bool applicable = !unique_overflow && unique_value_count <= max_codebook_size;
    const std::int64_t payload_bits =
        static_cast<std::int64_t>(nnz) * index_bits +
        static_cast<std::int64_t>(unique_value_count) * sizeof(T) * 8;
    auto report = make_exact_codec_report<T>(name, applicable, payload_bits, nnz,
                                             raw_bytes_per_nnz);
    if (applicable) {
        report.codebook_size = unique_value_count;
        report.index_bits = index_bits;
    }
    return report;
}

template <typename T, typename RoundFn>
ValueCodecReport make_rounding_codec_report(const CsrMatrix<T>& csr, const std::string& name,
                                            int bits_per_value, double raw_bytes_per_nnz,
                                            double spmv_rel_gate, RoundFn round_value) {
    ValueCodecReport report;
    report.name = name;
    report.applicable = true;
    report.exact = false;
    report.payload_bits = static_cast<std::int64_t>(csr.nnz()) * bits_per_value;
    if (csr.nnz() > 0) {
        report.bytes_per_nnz = static_cast<double>(bits_per_value) / 8.0;
        report.compression_ratio_vs_raw = raw_bytes_per_nnz / report.bytes_per_nnz;
    }

    CsrMatrix<T> rounded = csr;
    for (size_t i = 0; i < rounded.values.size(); ++i) {
        const T original = csr.values[i];
        const T value = static_cast<T>(round_value(static_cast<double>(original)));
        rounded.values[i] = value;
        const double abs_err = std::abs(static_cast<double>(original) - static_cast<double>(value));
        const double denom = std::max(std::abs(static_cast<double>(original)), 1e-30);
        if (abs_err != 0.0) {
            ++report.changed_values;
        }
        report.max_abs_value_error = std::max(report.max_abs_value_error, abs_err);
        report.max_rel_value_error = std::max(report.max_rel_value_error, abs_err / denom);
    }
    if (csr.nnz() > 0) {
        report.changed_ratio =
            static_cast<double>(report.changed_values) / static_cast<double>(csr.nnz());
    }

    const auto x = make_value_probe_vector<T>(csr.cols);
    const auto ref = spmv_csr(csr, x);
    const auto got = spmv_csr(rounded, x);
    const auto err = compare_vectors(ref, got);
    report.spmv_max_abs = static_cast<double>(err.max_abs);
    report.spmv_max_rel = static_cast<double>(err.max_rel);
    report.passes_spmv_gate = report.spmv_max_rel <= spmv_rel_gate;
    return report;
}

template <typename T>
ValueCodecReport make_global_log_kmeans_codec_report(const CsrMatrix<T>& csr, int requested_k,
                                                     const ValueCompressionOptions& options,
                                                     double raw_bytes_per_nnz) {
    ValueCodecReport report;
    report.name = "global_logkmeans" + std::to_string(requested_k);
    report.applicable = true;
    report.exact = false;

    bool all_finite = true;
    auto logs = collect_nonzero_log_abs_values(csr, requested_k, &all_finite);
    if (!all_finite) {
        report.applicable = false;
        return report;
    }

    auto centers = make_log_kmeans_centers(std::move(logs), requested_k,
                                           options.cluster_histogram_bins,
                                           options.cluster_iterations);
    report.codebook_size = static_cast<int>(centers.size());
    report.index_bits = ceil_log2_int(report.codebook_size + 1);
    report.uses_sign_bit = report.codebook_size > 0;
    const int bits_per_nnz = report.index_bits + (report.uses_sign_bit ? 1 : 0);
    report.payload_bits = static_cast<std::int64_t>(csr.nnz()) * bits_per_nnz +
                          static_cast<std::int64_t>(report.codebook_size) * 64;
    if (csr.nnz() > 0) {
        report.bytes_per_nnz =
            static_cast<double>(report.payload_bits) / 8.0 / static_cast<double>(csr.nnz());
        report.compression_ratio_vs_raw =
            report.bytes_per_nnz == 0.0 ? std::numeric_limits<double>::infinity()
                                         : raw_bytes_per_nnz / report.bytes_per_nnz;
    }

    CsrMatrix<T> clustered = csr;
    for (size_t i = 0; i < clustered.values.size(); ++i) {
        const double original = static_cast<double>(csr.values[i]);
        double reconstructed = 0.0;
        if (original != 0.0 && !centers.empty()) {
            const int center = nearest_sorted_center(centers, std::log2(std::abs(original)));
            reconstructed = std::copysign(std::exp2(centers[static_cast<size_t>(center)]),
                                          original);
        }
        clustered.values[i] = static_cast<T>(reconstructed);

        const double abs_err = std::abs(original - reconstructed);
        const double denom = std::max(std::abs(original), 1e-30);
        if (abs_err != 0.0) {
            ++report.changed_values;
        }
        report.max_abs_value_error = std::max(report.max_abs_value_error, abs_err);
        report.max_rel_value_error = std::max(report.max_rel_value_error, abs_err / denom);
    }
    if (csr.nnz() > 0) {
        report.changed_ratio =
            static_cast<double>(report.changed_values) / static_cast<double>(csr.nnz());
    }

    const auto x = make_value_probe_vector<T>(csr.cols);
    const auto ref = spmv_csr(csr, x);
    const auto got = spmv_csr(clustered, x);
    const auto err = compare_vectors(ref, got);
    report.spmv_max_abs = static_cast<double>(err.max_abs);
    report.spmv_max_rel = static_cast<double>(err.max_rel);
    report.passes_spmv_gate = report.spmv_max_rel <= options.spmv_rel_gate;
    return report;
}

template <typename T>
ValueCodecReport make_k256_uint8_codec_report(const CsrMatrix<T>& csr,
                                              const ValueCompressionOptions& options,
                                              double raw_bytes_per_nnz) {
    ValueCodecReport report;
    report.name = "codebook256_u8";
    report.applicable = true;
    report.index_bits = 8;
    report.uses_sign_bit = false;

    K256UInt8ValueCodebook<T> codec;
    try {
        codec = build_k256_uint8_value_codebook(
            csr.values, options.cluster_histogram_bins, options.cluster_iterations);
    } catch (const std::runtime_error&) {
        report.applicable = false;
        return report;
    }

    report.exact = codec.exact;
    report.codebook_size = codec.codebook_size;
    report.payload_bits = static_cast<std::int64_t>(codec.payload_bytes()) * 8;
    if (csr.nnz() > 0) {
        report.bytes_per_nnz =
            static_cast<double>(report.payload_bits) / 8.0 / static_cast<double>(csr.nnz());
        report.compression_ratio_vs_raw =
            report.bytes_per_nnz == 0.0 ? std::numeric_limits<double>::infinity()
                                         : raw_bytes_per_nnz / report.bytes_per_nnz;
    }

    CsrMatrix<T> clustered = csr;
    clustered.values = codec.reconstruct();
    for (size_t i = 0; i < csr.values.size(); ++i) {
        const double original = static_cast<double>(csr.values[i]);
        const double reconstructed = static_cast<double>(clustered.values[i]);
        const double abs_err = std::abs(original - reconstructed);
        const double denom = std::max(std::abs(original), 1e-30);
        if (abs_err != 0.0) {
            ++report.changed_values;
        }
        report.max_abs_value_error = std::max(report.max_abs_value_error, abs_err);
        report.max_rel_value_error = std::max(report.max_rel_value_error, abs_err / denom);
    }
    if (csr.nnz() > 0) {
        report.changed_ratio =
            static_cast<double>(report.changed_values) / static_cast<double>(csr.nnz());
    }

    const auto x = make_value_probe_vector<T>(csr.cols);
    const auto ref = spmv_csr(csr, x);
    const auto got = spmv_csr(clustered, x);
    const auto err = compare_vectors(ref, got);
    report.spmv_max_abs = static_cast<double>(err.max_abs);
    report.spmv_max_rel = static_cast<double>(err.max_rel);
    report.passes_spmv_gate = report.spmv_max_rel <= options.spmv_rel_gate;
    return report;
}

template <typename T>
ValueCompressionAnalysis analyze_value_compression(
    const CsrMatrix<T>& csr, const ValueCompressionOptions& options = ValueCompressionOptions{}) {
    csr.validate();

    ValueCompressionAnalysis analysis;
    analysis.nnz = csr.nnz();
    analysis.raw_bytes_per_nnz = static_cast<double>(sizeof(T));
    if (analysis.nnz == 0) {
        return analysis;
    }

    bool all_one = true;
    bool binary = true;
    bool sign = false;
    bool sign_candidate = true;

    for (const T value : csr.values) {
        if (value != T{1}) {
            all_one = false;
        }
        if (!(value == T{} || value == T{1})) {
            binary = false;
        }
        if (!(value == T{-1} || value == T{} || value == T{1})) {
            sign_candidate = false;
        }
        if (value == T{-1}) {
            sign = true;
        }
        ++analysis.abs_log2_bins[abs_log2_bin(static_cast<double>(value))];
    }
    sign = sign_candidate && sign;
    const auto unique_values = sorted_unique_values(csr.values, 65536);
    const bool unique_overflow = static_cast<int>(unique_values.size()) > 65536;
    const int unique_count = static_cast<int>(unique_values.size());
    const bool const_c = unique_count == 1;
    const bool sign2_scale =
        unique_count == 2 && unique_values[0] != T{0} && unique_values[0] == -unique_values[1];

    analysis.value_entropy_bits = exact_value_entropy_bits(csr.values);
    analysis.abs_log2_entropy_bits =
        entropy_bits_from_counts<T>(std::vector<Offset>(analysis.abs_log2_bins.begin(),
                                                        analysis.abs_log2_bins.end()),
                                    analysis.nnz);

    analysis.codecs.push_back(
        make_exact_codec_report<T>("original", true,
                                   static_cast<std::int64_t>(analysis.nnz) * sizeof(T) * 8,
                                   analysis.nnz, analysis.raw_bytes_per_nnz));
    analysis.codecs.push_back(
        make_exact_codec_report<T>("all_one", all_one, 0, analysis.nnz,
                                   analysis.raw_bytes_per_nnz));
    {
        auto report = make_exact_codec_report<T>(
            "const_c", const_c, static_cast<std::int64_t>(sizeof(T)) * 8,
            analysis.nnz, analysis.raw_bytes_per_nnz);
        if (report.applicable) {
            report.codebook_size = 1;
            report.index_bits = 0;
        }
        analysis.codecs.push_back(report);
    }
    analysis.codecs.push_back(
        make_exact_codec_report<T>("binary01", binary && !all_one,
                                   static_cast<std::int64_t>(analysis.nnz), analysis.nnz,
                                   analysis.raw_bytes_per_nnz));
    analysis.codecs.push_back(
        make_exact_codec_report<T>("sign2", sign, static_cast<std::int64_t>(analysis.nnz) * 2,
                                   analysis.nnz, analysis.raw_bytes_per_nnz));
    {
        auto report = make_exact_codec_report<T>(
            "sign2_scale", sign2_scale,
            static_cast<std::int64_t>(analysis.nnz) +
                static_cast<std::int64_t>(sizeof(T)) * 8,
            analysis.nnz, analysis.raw_bytes_per_nnz);
        if (report.applicable) {
            report.codebook_size = 1;
            report.index_bits = 1;
        }
        analysis.codecs.push_back(report);
    }
    analysis.codecs.push_back(
        make_exact_dict_codec_report<T>("dict2", 2, 1, unique_count, unique_overflow,
                                        analysis.nnz, analysis.raw_bytes_per_nnz));
    analysis.codecs.push_back(
        make_exact_dict_codec_report<T>("dict4", 4, 2, unique_count, unique_overflow,
                                        analysis.nnz, analysis.raw_bytes_per_nnz));
    analysis.codecs.push_back(
        make_exact_dict_codec_report<T>("dict16", 16, 4, unique_count, unique_overflow,
                                        analysis.nnz, analysis.raw_bytes_per_nnz));
    analysis.codecs.push_back(
        make_exact_dict_codec_report<T>("dict256_u8", 256, 8, unique_count,
                                        unique_overflow, analysis.nnz,
                                        analysis.raw_bytes_per_nnz));
    analysis.codecs.push_back(
        make_exact_dict_codec_report<T>("dict65536_u16", 65536, 16, unique_count,
                                        unique_overflow, analysis.nnz,
                                        analysis.raw_bytes_per_nnz));
    analysis.codecs.push_back(
        make_rounding_codec_report<T>(csr, "fp16", 16, analysis.raw_bytes_per_nnz,
                                      options.spmv_rel_gate, round_to_fp16));
    analysis.codecs.push_back(
        make_rounding_codec_report<T>(csr, "bf16", 16, analysis.raw_bytes_per_nnz,
                                      options.spmv_rel_gate, round_to_bf16));
    analysis.codecs.push_back(
        make_k256_uint8_codec_report<T>(csr, options, analysis.raw_bytes_per_nnz));
    for (const int k : options.cluster_sizes) {
        analysis.codecs.push_back(
            make_global_log_kmeans_codec_report<T>(csr, k, options,
                                                   analysis.raw_bytes_per_nnz));
    }

    return analysis;
}

inline std::string normalize_value_compression_codec_name(const std::string& name) {
    if (name == "raw" || name == "orig" || name == "origin") {
        return "original";
    }
    if (name == "implicit_one" || name == "implicit1" || name == "one" ||
        name == "pattern") {
        return "all_one";
    }
    if (name == "dict1" || name == "const" || name == "constant" ||
        name == "constant_c") {
        return "const_c";
    }
    if (name == "binary1" || name == "binary_zero_one" || name == "zero_one") {
        return "binary01";
    }
    if (name == "scaled_sign" || name == "scaled_sign2" ||
        name == "sign_scale" || name == "sign2_c") {
        return "sign2_scale";
    }
    if (name == "unique2" || name == "unique2_tile32" || name == "tile1" ||
        name == "u1") {
        return "dict2";
    }
    if (name == "unique4" || name == "unique4_tile64" || name == "ternary3" ||
        name == "tile2") {
        return "dict4";
    }
    if (name == "unique16" || name == "unique16_tile128" || name == "unique_u8" ||
        name == "tile4") {
        return "dict16";
    }
    if (name == "dict256" || name == "unique256" || name == "unique256_tile256" ||
        name == "dict256_tile256" || name == "exact_u8") {
        return "dict256_u8";
    }
    if (name == "dict65536" || name == "dict_u16" || name == "unique65536" ||
        name == "unique65536_tile512" || name == "dict65536_tile512" ||
        name == "tile16" || name == "exact_u16") {
        return "dict65536_u16";
    }
    if (name == "k256" || name == "k256_uint8" || name == "k256_u8" ||
        name == "k256_tile256" || name == "uint8" || name == "u8") {
        return "codebook256_u8";
    }
    return name;
}

inline const ValueCodecReport* find_value_codec(const ValueCompressionAnalysis& analysis,
                                                const std::string& name) {
    const std::string normalized_name = normalize_value_compression_codec_name(name);
    for (const auto& codec : analysis.codecs) {
        if (codec.name == normalized_name) {
            return &codec;
        }
    }
    return nullptr;
}

}  // namespace structural
