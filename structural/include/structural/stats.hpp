#pragma once

#include "structural/bitbsr.hpp"
#include "structural/csr.hpp"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <string>

namespace structural {

template <typename T>
struct BitBsrStorageStats {
    Index rows = 0;
    Index cols = 0;
    Offset nnz = 0;
    Index block_rows = 0;
    Index block_cols = 0;
    Offset num_blocks = 0;

    long long csr_index_bits = 0;
    long long csr_row_ptr_bits = 0;
    long long csr_position_bits = 0;
    long long block_row_ptr_bits = 0;
    long long block_col_idx_bits = 0;
    long long bitmap_bits = 0;
    long long block_val_ptr_bits = 0;
    long long total_position_bits = 0;

    double avg_nnz_per_block = 0.0;
    double csr_bits_per_nnz = 0.0;
    double csr_position_bits_per_nnz = 0.0;
    double bitbsr_bits_per_nnz = 0.0;
    double compression_ratio = 0.0;
    double position_compression_ratio = 0.0;
};

template <typename T>
BitBsrStorageStats<T> storage_stats(const CsrMatrix<T>& csr, const BitBsrMatrix<T>& bitbsr) {
    BitBsrStorageStats<T> stats;
    stats.rows = csr.rows;
    stats.cols = csr.cols;
    stats.nnz = csr.nnz();
    stats.block_rows = bitbsr.block_rows;
    stats.block_cols = bitbsr.block_cols;
    stats.num_blocks = bitbsr.num_blocks();

    stats.csr_index_bits = static_cast<long long>(csr.nnz()) * 32;
    stats.csr_row_ptr_bits = static_cast<long long>(csr.row_ptr.size()) * 32;
    stats.csr_position_bits = stats.csr_index_bits + stats.csr_row_ptr_bits;
    stats.block_row_ptr_bits = static_cast<long long>(bitbsr.block_row_ptr.size()) * 32;
    stats.block_col_idx_bits = static_cast<long long>(bitbsr.block_col_idx.size()) * 32;
    stats.bitmap_bits = static_cast<long long>(bitbsr.bitmap_words.size()) * kBitmapWordBits;
    stats.block_val_ptr_bits = static_cast<long long>(bitbsr.block_val_ptr.size()) * 32;
    stats.total_position_bits = stats.block_row_ptr_bits + stats.block_col_idx_bits +
                                stats.bitmap_bits + stats.block_val_ptr_bits;

    if (stats.nnz > 0) {
        stats.avg_nnz_per_block = static_cast<double>(stats.nnz) /
                                  static_cast<double>(std::max(Offset{1}, stats.num_blocks));
        stats.csr_bits_per_nnz = static_cast<double>(stats.csr_index_bits) /
                                 static_cast<double>(stats.nnz);
        stats.csr_position_bits_per_nnz = static_cast<double>(stats.csr_position_bits) /
                                          static_cast<double>(stats.nnz);
        stats.bitbsr_bits_per_nnz = static_cast<double>(stats.total_position_bits) /
                                    static_cast<double>(stats.nnz);
        stats.compression_ratio = static_cast<double>(stats.total_position_bits) /
                                  static_cast<double>(stats.csr_index_bits);
        stats.position_compression_ratio = static_cast<double>(stats.total_position_bits) /
                                           static_cast<double>(stats.csr_position_bits);
    }
    return stats;
}

template <typename T>
void print_human_stats(std::ostream& out, const std::string& matrix_path,
                       const BitBsrStorageStats<T>& stats) {
    out << "matrix: " << matrix_path << "\n";
    out << "shape: " << stats.rows << " x " << stats.cols << "\n";
    out << "nnz: " << stats.nnz << "\n";
    out << "block: " << stats.block_rows << " x " << stats.block_cols << "\n";
    out << "num_blocks: " << stats.num_blocks << "\n";
    out << std::fixed << std::setprecision(4);
    out << "avg_nnz_per_block: " << stats.avg_nnz_per_block << "\n";
    out << "csr_col_index_bits_per_nnz: " << stats.csr_bits_per_nnz << "\n";
    out << "csr_full_position_bits_per_nnz: " << stats.csr_position_bits_per_nnz << "\n";
    out << "bitbsr_position_bits_per_nnz: " << stats.bitbsr_bits_per_nnz << "\n";
    out << "bitbsr_vs_csr_col_index_ratio: " << stats.compression_ratio << "\n";
    out << "bitbsr_vs_csr_full_position_ratio: " << stats.position_compression_ratio << "\n";
    out << "csr_breakdown_bits: row_ptr=" << stats.csr_row_ptr_bits
        << ", col_idx=" << stats.csr_index_bits << "\n";
    out << "bitbsr_breakdown_bits: block_row_ptr=" << stats.block_row_ptr_bits
        << ", block_col_idx=" << stats.block_col_idx_bits
        << ", bitmap=" << stats.bitmap_bits
        << ", block_val_ptr=" << stats.block_val_ptr_bits << "\n";
}

inline void print_csv_header(std::ostream& out) {
    out << "matrix,rows,cols,nnz,block_rows,block_cols,num_blocks,"
        << "avg_nnz_per_block,csr_col_index_bits_per_nnz,csr_full_position_bits_per_nnz,"
        << "bitbsr_position_bits_per_nnz,bitbsr_vs_csr_col_index_ratio,"
        << "bitbsr_vs_csr_full_position_ratio,csr_row_ptr_bits,csr_col_idx_bits,"
        << "block_row_ptr_bits,block_col_idx_bits,bitmap_bits,block_val_ptr_bits,"
        << "total_position_bits\n";
}

template <typename T>
void print_csv_row(std::ostream& out, const std::string& matrix_path,
                   const BitBsrStorageStats<T>& stats) {
    out << matrix_path << ','
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
        << stats.total_position_bits << '\n';
}

}  // namespace structural
