#include "structural/convert.hpp"
#include "structural/matrix_market.hpp"
#include "structural/profile.hpp"
#include "structural/stats.hpp"
#include "structural/value_compression.hpp"
#include "structural/value_stats.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string matrix_path;
    int block_rows = 8;
    int block_cols = 4;
    bool csv = false;
    bool timing = false;
};

void print_usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " MATRIX.mtx [--br N] [--bc N] [--csv] [--timing]\n";
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
        if (arg == "--br" && i + 1 < argc) {
            opts.block_rows = std::stoi(argv[++i]);
        } else if (arg == "--bc" && i + 1 < argc) {
            opts.block_cols = std::stoi(argv[++i]);
        } else if (arg == "--csv") {
            opts.csv = true;
        } else if (arg == "--timing") {
            opts.timing = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    return opts;
}

const char* tf(bool value) {
    return value ? "true" : "false";
}

const std::vector<std::string>& value_codec_csv_names() {
    static const std::vector<std::string> names = {
        "original", "all_one", "const_c", "binary01", "sign2", "sign2_scale",
        "dict2", "dict4", "dict16",
        "dict256_u8", "dict65536_u16", "fp16", "bf16", "codebook256_u8",
        "global_logkmeans16", "global_logkmeans32",
        "global_logkmeans64", "global_logkmeans128",
    };
    return names;
}

template <typename T>
void print_load_and_value_stats(std::ostream& out, const structural::MatrixMarketLoadResult<T>& loaded,
                                const structural::ValueStats<T>& value_stats) {
    out << "matrix_market_field: " << structural::to_string(loaded.info.field) << "\n";
    out << "matrix_market_symmetry: " << structural::to_string(loaded.info.symmetry) << "\n";
    out << "stored_entries_file: " << loaded.stored_entries << "\n";
    out << "expanded_entries_before_coalesce: " << loaded.expanded_entries << "\n";
    out << "coalesced_duplicates: " << loaded.coalesced_duplicates << "\n";
    out << "value_explicit_zero_count: " << value_stats.explicit_zero_count << "\n";
    out << "value_all_values_one: " << tf(value_stats.all_values_one) << "\n";
    out << "value_binary_zero_one: " << tf(value_stats.binary_zero_one) << "\n";
    out << "value_sign_only: " << tf(value_stats.sign_only) << "\n";
    out << "value_unique_values_exact_cap16: " << value_stats.unique_values_exact << "\n";
    out << "value_unique_values_overflow_cap16: " << tf(value_stats.unique_values_overflow) << "\n";
    out << "value_min: " << value_stats.min_value << "\n";
    out << "value_max: " << value_stats.max_value << "\n";
}

std::string format_abs_log2_bins(const structural::ValueCompressionAnalysis& analysis) {
    std::ostringstream ss;
    for (int i = 0; i < structural::kAbsLog2BinCount; ++i) {
        if (i != 0) {
            ss << '|';
        }
        if (i == 0) {
            ss << "zero";
        } else if (i == structural::kAbsLog2BinCount - 1) {
            ss << "nonfinite";
        } else {
            const int lo = -64 + (i - 1) * 8;
            const int hi = lo + 7;
            ss << "e" << lo << ".." << hi;
        }
        ss << ':' << analysis.abs_log2_bins[static_cast<size_t>(i)];
    }
    return ss.str();
}

void print_value_compression_human(std::ostream& out,
                                   const structural::ValueCompressionAnalysis& analysis) {
    out << std::fixed << std::setprecision(6);
    out << "value_entropy_bits: " << analysis.value_entropy_bits << "\n";
    out << "value_abs_log2_entropy_bits: " << analysis.abs_log2_entropy_bits << "\n";
    out << "value_abs_log2_histogram: " << format_abs_log2_bins(analysis) << "\n";
    for (const auto& codec : analysis.codecs) {
        const std::string prefix = "value_codec_" + codec.name + "_";
        out << prefix << "applicable: " << tf(codec.applicable) << "\n";
        out << prefix << "exact: " << tf(codec.exact) << "\n";
        out << prefix << "payload_bits: " << codec.payload_bits << "\n";
        out << prefix << "bytes_per_nnz: " << codec.bytes_per_nnz << "\n";
        out << prefix << "compression_ratio_vs_raw: " << codec.compression_ratio_vs_raw << "\n";
        out << prefix << "codebook_size: " << codec.codebook_size << "\n";
        out << prefix << "index_bits: " << codec.index_bits << "\n";
        out << prefix << "uses_sign_bit: " << tf(codec.uses_sign_bit) << "\n";
        out << prefix << "changed_ratio: " << codec.changed_ratio << "\n";
        out << prefix << "max_rel_value_error: " << codec.max_rel_value_error << "\n";
        out << prefix << "spmv_max_rel: " << codec.spmv_max_rel << "\n";
        out << prefix << "passes_spmv_gate: " << tf(codec.passes_spmv_gate) << "\n";
    }
}

void print_value_compression_csv_header(std::ostream& out) {
    out << "value_entropy_bits,value_abs_log2_entropy_bits,value_abs_log2_histogram";
    for (const std::string& name : value_codec_csv_names()) {
        out << ",value_codec_" << name << "_applicable"
            << ",value_codec_" << name << "_exact"
            << ",value_codec_" << name << "_payload_bits"
            << ",value_codec_" << name << "_bytes_per_nnz"
            << ",value_codec_" << name << "_compression_ratio_vs_raw"
            << ",value_codec_" << name << "_codebook_size"
            << ",value_codec_" << name << "_index_bits"
            << ",value_codec_" << name << "_uses_sign_bit"
            << ",value_codec_" << name << "_changed_ratio"
            << ",value_codec_" << name << "_max_rel_value_error"
            << ",value_codec_" << name << "_spmv_max_rel"
            << ",value_codec_" << name << "_passes_spmv_gate";
    }
}

void print_value_compression_csv_row(std::ostream& out,
                                     const structural::ValueCompressionAnalysis& analysis) {
    out << std::fixed << std::setprecision(6)
        << analysis.value_entropy_bits << ','
        << analysis.abs_log2_entropy_bits << ','
        << format_abs_log2_bins(analysis);
    for (const std::string& name : value_codec_csv_names()) {
        const auto* codec = structural::find_value_codec(analysis, name);
        const structural::ValueCodecReport empty;
        const auto& c = codec == nullptr ? empty : *codec;
        out << ','
            << tf(c.applicable) << ','
            << tf(c.exact) << ','
            << c.payload_bits << ','
            << c.bytes_per_nnz << ','
            << c.compression_ratio_vs_raw << ','
            << c.codebook_size << ','
            << c.index_bits << ','
            << tf(c.uses_sign_bit) << ','
            << c.changed_ratio << ','
            << c.max_rel_value_error << ','
            << c.spmv_max_rel << ','
            << tf(c.passes_spmv_gate);
    }
}

void print_format_csv_header(std::ostream& out) {
    out << "matrix,mm_field,mm_symmetry,stored_entries_file,expanded_entries_before_coalesce,"
        << "coalesced_duplicates,value_explicit_zero_count,value_all_values_one,"
        << "value_binary_zero_one,value_sign_only,value_unique_values_exact_cap16,"
        << "value_unique_values_overflow_cap16,value_min,value_max,";
    print_value_compression_csv_header(out);
    out << ','
        << "rows,cols,nnz,block_rows,block_cols,num_blocks,"
        << "avg_nnz_per_block,csr_col_index_bits_per_nnz,csr_full_position_bits_per_nnz,"
        << "bitbsr_position_bits_per_nnz,bitbsr_vs_csr_col_index_ratio,"
        << "bitbsr_vs_csr_full_position_ratio,csr_row_ptr_bits,csr_col_idx_bits,"
        << "block_row_ptr_bits,block_col_idx_bits,bitmap_bits,block_val_ptr_bits,"
        << "total_position_bits";
    structural::print_profile_csv_header_suffix(out);
    out << "\n";
}

template <typename T>
void print_format_csv_row(std::ostream& out, const std::string& matrix_path,
                          const structural::MatrixMarketLoadResult<T>& loaded,
                          const structural::ValueStats<T>& value_stats,
                          const structural::ValueCompressionAnalysis& value_compression,
                          const structural::BitBsrStorageStats<T>& stats,
                          const structural::BitBsrProfile& profile) {
    out << matrix_path << ','
        << structural::to_string(loaded.info.field) << ','
        << structural::to_string(loaded.info.symmetry) << ','
        << loaded.stored_entries << ','
        << loaded.expanded_entries << ','
        << loaded.coalesced_duplicates << ','
        << value_stats.explicit_zero_count << ','
        << tf(value_stats.all_values_one) << ','
        << tf(value_stats.binary_zero_one) << ','
        << tf(value_stats.sign_only) << ','
        << value_stats.unique_values_exact << ','
        << tf(value_stats.unique_values_overflow) << ','
        << value_stats.min_value << ','
        << value_stats.max_value << ',';
    print_value_compression_csv_row(out, value_compression);
    out << ','
        << stats.rows << ','
        << stats.cols << ','
        << stats.nnz << ','
        << stats.block_rows << ','
        << stats.block_cols << ','
        << stats.num_blocks << ','
        << std::fixed << std::setprecision(6)
        << stats.avg_nnz_per_block << ','
        << stats.csr_bits_per_nnz << ','
        << stats.csr_position_bits_per_nnz << ','
        << stats.bitbsr_bits_per_nnz << ','
        << stats.compression_ratio << ','
        << stats.position_compression_ratio << ','
        << stats.csr_row_ptr_bits << ','
        << stats.csr_index_bits << ','
        << stats.block_row_ptr_bits << ','
        << stats.block_col_idx_bits << ','
        << stats.bitmap_bits << ','
        << stats.block_val_ptr_bits << ','
        << stats.total_position_bits;
    structural::print_profile_csv_row_suffix(out, profile);
    out << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_options(argc, argv);
        const auto t0 = std::chrono::steady_clock::now();
        const auto loaded = structural::read_matrix_market_with_info<double>(opts.matrix_path);
        const auto t1 = std::chrono::steady_clock::now();
        const auto bitbsr = structural::convert_to_bitbsr(loaded.matrix, opts.block_rows, opts.block_cols);
        const auto t2 = std::chrono::steady_clock::now();
        const auto stats = structural::storage_stats(loaded.matrix, bitbsr);
        const auto profile = structural::profile_bitbsr(bitbsr);
        const auto value_stats = structural::compute_value_stats(loaded.matrix);
        const auto value_compression = structural::analyze_value_compression(loaded.matrix);

        if (opts.csv) {
            print_format_csv_header(std::cout);
            print_format_csv_row(std::cout, opts.matrix_path, loaded, value_stats,
                                 value_compression, stats, profile);
        } else {
            print_load_and_value_stats(std::cout, loaded, value_stats);
            print_value_compression_human(std::cout, value_compression);
            structural::print_human_stats(std::cout, opts.matrix_path, stats);
            structural::print_human_profile(std::cout, profile);
        }
        if (opts.timing) {
            const double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double convert_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            std::cout << "timing_ms: load=" << load_ms << ", convert=" << convert_ms
                      << ", total=" << (load_ms + convert_ms) << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
