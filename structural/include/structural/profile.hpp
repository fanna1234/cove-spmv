#pragma once

#include "structural/bitbsr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

namespace structural {

struct BitBsrProfile {
    Index rows = 0;
    Index cols = 0;
    Index block_rows = 0;
    Index block_cols = 0;
    Index num_block_rows = 0;
    Offset num_blocks = 0;
    Offset nnz = 0;

    double block_fill_ratio = 0.0;
    double avg_nnz_per_block = 0.0;
    Offset empty_blocks = 0;
    Offset blocks_le_25pct_full = 0;
    Offset blocks_le_50pct_full = 0;
    Offset full_blocks = 0;

    Index empty_block_rows = 0;
    double empty_block_row_ratio = 0.0;
    double avg_blocks_per_block_row = 0.0;
    Offset p50_blocks_per_block_row = 0;
    Offset p90_blocks_per_block_row = 0;
    Offset p99_blocks_per_block_row = 0;
    Offset max_blocks_per_block_row = 0;
    double p99_blocks_per_block_row_over_avg = 0.0;
    double max_blocks_per_block_row_over_avg = 0.0;

    Index active_block_rows = 0;
    Index col_delta_u8_eligible_block_rows = 0;
    Index col_delta_u16_eligible_block_rows = 0;
    Offset col_delta_u8_covered_blocks = 0;
    Offset col_delta_u16_covered_blocks = 0;
    Index max_col_delta_in_block_row = 0;
    int local_coord_bits_per_nnz = 0;
    int local_coord_count_bits_per_block = 0;

    long long runtime_abs_col_position_bits = 0;
    long long runtime_delta_u8_position_bits = 0;
    long long runtime_delta_u16_position_bits = 0;
    long long runtime_local_coord_abs_position_bits = 0;
    long long runtime_local_coord_delta_u8_position_bits = 0;
    long long runtime_local_coord_delta_u16_position_bits = 0;
    long long active_row_abs_meta_bits = 0;
    long long active_row_delta_u8_meta_bits = 0;
    long long active_row_delta_u16_meta_bits = 0;

    double runtime_abs_col_bits_per_nnz = 0.0;
    double runtime_delta_u8_bits_per_nnz = 0.0;
    double runtime_delta_u16_bits_per_nnz = 0.0;
    double runtime_local_coord_abs_bits_per_nnz = 0.0;
    double runtime_local_coord_delta_u8_bits_per_nnz = 0.0;
    double runtime_local_coord_delta_u16_bits_per_nnz = 0.0;
    double active_row_abs_meta_bits_per_nnz = 0.0;
    double active_row_delta_u8_meta_bits_per_nnz = 0.0;
    double active_row_delta_u16_meta_bits_per_nnz = 0.0;
};

inline Offset percentile_nearest_rank(std::vector<Offset> values, double percentile) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = std::min(100.0, std::max(0.0, percentile));
    const auto rank = static_cast<size_t>(std::ceil(clamped / 100.0 * values.size()));
    const size_t idx = rank == 0 ? 0 : std::min(rank - 1, values.size() - 1);
    return values[idx];
}

inline int ceil_log2_cardinality(std::uint64_t cardinality) {
    if (cardinality <= 1) {
        return 0;
    }
    int bits = 0;
    std::uint64_t max_value = cardinality - 1;
    while (max_value > 0) {
        ++bits;
        max_value >>= 1U;
    }
    return bits;
}

inline double bits_per_nnz(long long bits, Offset nnz) {
    if (nnz <= 0) {
        return 0.0;
    }
    return static_cast<double>(bits) / static_cast<double>(nnz);
}

template <typename T>
BitBsrProfile profile_bitbsr(const BitBsrMatrix<T>& bitbsr) {
    bitbsr.validate();

    BitBsrProfile profile;
    profile.rows = bitbsr.rows;
    profile.cols = bitbsr.cols;
    profile.block_rows = bitbsr.block_rows;
    profile.block_cols = bitbsr.block_cols;
    profile.num_block_rows = bitbsr.num_block_rows;
    profile.num_blocks = bitbsr.num_blocks();
    profile.nnz = bitbsr.nnz();

    const Offset block_capacity =
        static_cast<Offset>(bitbsr.block_rows) * static_cast<Offset>(bitbsr.block_cols);
    if (profile.num_blocks > 0 && block_capacity > 0) {
        profile.block_fill_ratio =
            static_cast<double>(profile.nnz) /
            static_cast<double>(profile.num_blocks) /
            static_cast<double>(block_capacity);
        profile.avg_nnz_per_block =
            static_cast<double>(profile.nnz) / static_cast<double>(profile.num_blocks);
    }

    for (Offset block = 0; block < profile.num_blocks; ++block) {
        const Offset block_nnz = bitbsr.block_val_ptr[block + 1] - bitbsr.block_val_ptr[block];
        if (block_nnz == 0) {
            ++profile.empty_blocks;
        }
        if (block_nnz * 4 <= block_capacity) {
            ++profile.blocks_le_25pct_full;
        }
        if (block_nnz * 2 <= block_capacity) {
            ++profile.blocks_le_50pct_full;
        }
        if (block_nnz == block_capacity) {
            ++profile.full_blocks;
        }
    }

    const long long row_ptr_bits = static_cast<long long>(profile.num_block_rows + 1) * 32;
    const long long row_val_ptr_bits = static_cast<long long>(profile.num_block_rows + 1) * 32;
    const long long row_meta_bits = row_ptr_bits + row_val_ptr_bits;
    const long long abs_col_bits = static_cast<long long>(profile.num_blocks) * 32;
    const long long bitmap_bits =
        static_cast<long long>(profile.num_blocks) *
        static_cast<long long>(bitbsr.words_per_block) * kBitmapWordBits;

    long long delta_u8_col_bits = 0;
    long long delta_u16_col_bits = 0;

    std::vector<Offset> blocks_per_row;
    blocks_per_row.reserve(static_cast<size_t>(std::max<Index>(profile.num_block_rows, 0)));
    for (Index br = 0; br < profile.num_block_rows; ++br) {
        const Offset begin = bitbsr.block_row_ptr[br];
        const Offset end = bitbsr.block_row_ptr[br + 1];
        const Offset blocks = end - begin;
        blocks_per_row.push_back(blocks);
        if (blocks == 0) {
            ++profile.empty_block_rows;
            continue;
        }
        ++profile.active_block_rows;
        profile.max_blocks_per_block_row = std::max(profile.max_blocks_per_block_row, blocks);

        Index min_col = bitbsr.block_col_idx[begin];
        Index max_col = bitbsr.block_col_idx[begin];
        for (Offset block = begin + 1; block < end; ++block) {
            min_col = std::min(min_col, bitbsr.block_col_idx[block]);
            max_col = std::max(max_col, bitbsr.block_col_idx[block]);
        }

        const Index col_delta = max_col - min_col;
        profile.max_col_delta_in_block_row =
            std::max(profile.max_col_delta_in_block_row, col_delta);

        const long long abs_row_col_bits = static_cast<long long>(blocks) * 32;
        if (col_delta <= 255) {
            ++profile.col_delta_u8_eligible_block_rows;
            profile.col_delta_u8_covered_blocks += blocks;
            const long long delta_bits = 32 + static_cast<long long>(blocks) * 8;
            delta_u8_col_bits += std::min(abs_row_col_bits, delta_bits);
        } else {
            delta_u8_col_bits += abs_row_col_bits;
        }

        if (col_delta <= 65535) {
            ++profile.col_delta_u16_eligible_block_rows;
            profile.col_delta_u16_covered_blocks += blocks;
            const long long delta_bits = 32 + static_cast<long long>(blocks) * 16;
            delta_u16_col_bits += std::min(abs_row_col_bits, delta_bits);
        } else {
            delta_u16_col_bits += abs_row_col_bits;
        }
    }

    if (!blocks_per_row.empty()) {
        const Offset total_blocks =
            std::accumulate(blocks_per_row.begin(), blocks_per_row.end(), Offset{0});
        profile.avg_blocks_per_block_row =
            static_cast<double>(total_blocks) / static_cast<double>(blocks_per_row.size());
        profile.empty_block_row_ratio =
            static_cast<double>(profile.empty_block_rows) / static_cast<double>(blocks_per_row.size());
        profile.p50_blocks_per_block_row = percentile_nearest_rank(blocks_per_row, 50.0);
        profile.p90_blocks_per_block_row = percentile_nearest_rank(blocks_per_row, 90.0);
        profile.p99_blocks_per_block_row = percentile_nearest_rank(blocks_per_row, 99.0);
    }

    if (profile.avg_blocks_per_block_row > 0.0) {
        profile.p99_blocks_per_block_row_over_avg =
            static_cast<double>(profile.p99_blocks_per_block_row) /
            profile.avg_blocks_per_block_row;
        profile.max_blocks_per_block_row_over_avg =
            static_cast<double>(profile.max_blocks_per_block_row) /
            profile.avg_blocks_per_block_row;
    }

    const auto block_capacity_cardinality =
        static_cast<std::uint64_t>(bitbsr.block_rows) * static_cast<std::uint64_t>(bitbsr.block_cols);
    profile.local_coord_bits_per_nnz = ceil_log2_cardinality(block_capacity_cardinality);
    profile.local_coord_count_bits_per_block = ceil_log2_cardinality(block_capacity_cardinality + 1);

    const long long local_coord_payload_bits =
        static_cast<long long>(profile.nnz) * profile.local_coord_bits_per_nnz +
        static_cast<long long>(profile.num_blocks) * profile.local_coord_count_bits_per_block;

    profile.runtime_abs_col_position_bits = row_meta_bits + abs_col_bits + bitmap_bits;
    profile.runtime_delta_u8_position_bits = row_meta_bits + delta_u8_col_bits + bitmap_bits;
    profile.runtime_delta_u16_position_bits = row_meta_bits + delta_u16_col_bits + bitmap_bits;
    profile.runtime_local_coord_abs_position_bits =
        row_meta_bits + abs_col_bits + local_coord_payload_bits;
    profile.runtime_local_coord_delta_u8_position_bits =
        row_meta_bits + delta_u8_col_bits + local_coord_payload_bits;
    profile.runtime_local_coord_delta_u16_position_bits =
        row_meta_bits + delta_u16_col_bits + local_coord_payload_bits;

    if (profile.active_block_rows > 0) {
        const long long active_row_ptr_bits =
            static_cast<long long>(profile.active_block_rows + 1) * 32;
        const long long active_row_val_ptr_bits =
            static_cast<long long>(profile.active_block_rows + 1) * 32;
        const long long active_row_base_bits = 32;
        profile.active_row_abs_meta_bits =
            active_row_ptr_bits + active_row_val_ptr_bits +
            static_cast<long long>(profile.active_block_rows) * 32;
        profile.active_row_delta_u8_meta_bits =
            active_row_ptr_bits + active_row_val_ptr_bits + active_row_base_bits +
            static_cast<long long>(profile.active_block_rows) * 8;
        profile.active_row_delta_u16_meta_bits =
            active_row_ptr_bits + active_row_val_ptr_bits + active_row_base_bits +
            static_cast<long long>(profile.active_block_rows) * 16;
    }

    profile.runtime_abs_col_bits_per_nnz =
        bits_per_nnz(profile.runtime_abs_col_position_bits, profile.nnz);
    profile.runtime_delta_u8_bits_per_nnz =
        bits_per_nnz(profile.runtime_delta_u8_position_bits, profile.nnz);
    profile.runtime_delta_u16_bits_per_nnz =
        bits_per_nnz(profile.runtime_delta_u16_position_bits, profile.nnz);
    profile.runtime_local_coord_abs_bits_per_nnz =
        bits_per_nnz(profile.runtime_local_coord_abs_position_bits, profile.nnz);
    profile.runtime_local_coord_delta_u8_bits_per_nnz =
        bits_per_nnz(profile.runtime_local_coord_delta_u8_position_bits, profile.nnz);
    profile.runtime_local_coord_delta_u16_bits_per_nnz =
        bits_per_nnz(profile.runtime_local_coord_delta_u16_position_bits, profile.nnz);
    profile.active_row_abs_meta_bits_per_nnz =
        bits_per_nnz(profile.active_row_abs_meta_bits, profile.nnz);
    profile.active_row_delta_u8_meta_bits_per_nnz =
        bits_per_nnz(profile.active_row_delta_u8_meta_bits, profile.nnz);
    profile.active_row_delta_u16_meta_bits_per_nnz =
        bits_per_nnz(profile.active_row_delta_u16_meta_bits, profile.nnz);

    return profile;
}

inline void print_human_profile(std::ostream& out, const BitBsrProfile& profile) {
    out << std::fixed << std::setprecision(4);
    out << "block_fill_ratio: " << profile.block_fill_ratio << "\n";
    out << "empty_blocks: " << profile.empty_blocks << "\n";
    out << "blocks_le_25pct_full: " << profile.blocks_le_25pct_full << "\n";
    out << "blocks_le_50pct_full: " << profile.blocks_le_50pct_full << "\n";
    out << "full_blocks: " << profile.full_blocks << "\n";
    out << "empty_block_rows: " << profile.empty_block_rows << "\n";
    out << "empty_block_row_ratio: " << profile.empty_block_row_ratio << "\n";
    out << "avg_blocks_per_block_row: " << profile.avg_blocks_per_block_row << "\n";
    out << "p50_blocks_per_block_row: " << profile.p50_blocks_per_block_row << "\n";
    out << "p90_blocks_per_block_row: " << profile.p90_blocks_per_block_row << "\n";
    out << "p99_blocks_per_block_row: " << profile.p99_blocks_per_block_row << "\n";
    out << "max_blocks_per_block_row: " << profile.max_blocks_per_block_row << "\n";
    out << "p99_blocks_per_block_row_over_avg: "
        << profile.p99_blocks_per_block_row_over_avg << "\n";
    out << "max_blocks_per_block_row_over_avg: "
        << profile.max_blocks_per_block_row_over_avg << "\n";
    out << "active_block_rows: " << profile.active_block_rows << "\n";
    out << "col_delta_u8_eligible_block_rows: "
        << profile.col_delta_u8_eligible_block_rows << "\n";
    out << "col_delta_u16_eligible_block_rows: "
        << profile.col_delta_u16_eligible_block_rows << "\n";
    out << "col_delta_u8_covered_blocks: "
        << profile.col_delta_u8_covered_blocks << "\n";
    out << "col_delta_u16_covered_blocks: "
        << profile.col_delta_u16_covered_blocks << "\n";
    out << "max_col_delta_in_block_row: "
        << profile.max_col_delta_in_block_row << "\n";
    out << "local_coord_bits_per_nnz: "
        << profile.local_coord_bits_per_nnz << "\n";
    out << "local_coord_count_bits_per_block: "
        << profile.local_coord_count_bits_per_block << "\n";
    out << "runtime_abs_col_position_bits_per_nnz: "
        << profile.runtime_abs_col_bits_per_nnz << "\n";
    out << "runtime_delta_u8_position_bits_per_nnz: "
        << profile.runtime_delta_u8_bits_per_nnz << "\n";
    out << "runtime_delta_u16_position_bits_per_nnz: "
        << profile.runtime_delta_u16_bits_per_nnz << "\n";
    out << "runtime_local_coord_abs_position_bits_per_nnz: "
        << profile.runtime_local_coord_abs_bits_per_nnz << "\n";
    out << "runtime_local_coord_delta_u8_position_bits_per_nnz: "
        << profile.runtime_local_coord_delta_u8_bits_per_nnz << "\n";
    out << "runtime_local_coord_delta_u16_position_bits_per_nnz: "
        << profile.runtime_local_coord_delta_u16_bits_per_nnz << "\n";
    out << "active_row_abs_meta_bits_per_nnz: "
        << profile.active_row_abs_meta_bits_per_nnz << "\n";
    out << "active_row_delta_u8_meta_bits_per_nnz: "
        << profile.active_row_delta_u8_meta_bits_per_nnz << "\n";
    out << "active_row_delta_u16_meta_bits_per_nnz: "
        << profile.active_row_delta_u16_meta_bits_per_nnz << "\n";
}

inline void print_profile_csv_header_suffix(std::ostream& out) {
    out << ",block_fill_ratio,empty_blocks,blocks_le_25pct_full,blocks_le_50pct_full,"
        << "full_blocks,empty_block_rows,empty_block_row_ratio,avg_blocks_per_block_row,"
        << "p50_blocks_per_block_row,p90_blocks_per_block_row,p99_blocks_per_block_row,"
        << "max_blocks_per_block_row,p99_blocks_per_block_row_over_avg,"
        << "max_blocks_per_block_row_over_avg,active_block_rows,"
        << "col_delta_u8_eligible_block_rows,col_delta_u16_eligible_block_rows,"
        << "col_delta_u8_covered_blocks,col_delta_u16_covered_blocks,"
        << "max_col_delta_in_block_row,local_coord_bits_per_nnz,"
        << "local_coord_count_bits_per_block,runtime_abs_col_position_bits,"
        << "runtime_delta_u8_position_bits,runtime_delta_u16_position_bits,"
        << "runtime_local_coord_abs_position_bits,"
        << "runtime_local_coord_delta_u8_position_bits,"
        << "runtime_local_coord_delta_u16_position_bits,"
        << "active_row_abs_meta_bits,active_row_delta_u8_meta_bits,"
        << "active_row_delta_u16_meta_bits,runtime_abs_col_position_bits_per_nnz,"
        << "runtime_delta_u8_position_bits_per_nnz,"
        << "runtime_delta_u16_position_bits_per_nnz,"
        << "runtime_local_coord_abs_position_bits_per_nnz,"
        << "runtime_local_coord_delta_u8_position_bits_per_nnz,"
        << "runtime_local_coord_delta_u16_position_bits_per_nnz,"
        << "active_row_abs_meta_bits_per_nnz,"
        << "active_row_delta_u8_meta_bits_per_nnz,"
        << "active_row_delta_u16_meta_bits_per_nnz";
}

inline void print_profile_csv_row_suffix(std::ostream& out, const BitBsrProfile& profile) {
    out << std::fixed << std::setprecision(6)
        << ',' << profile.block_fill_ratio
        << ',' << profile.empty_blocks
        << ',' << profile.blocks_le_25pct_full
        << ',' << profile.blocks_le_50pct_full
        << ',' << profile.full_blocks
        << ',' << profile.empty_block_rows
        << ',' << profile.empty_block_row_ratio
        << ',' << profile.avg_blocks_per_block_row
        << ',' << profile.p50_blocks_per_block_row
        << ',' << profile.p90_blocks_per_block_row
        << ',' << profile.p99_blocks_per_block_row
        << ',' << profile.max_blocks_per_block_row
        << ',' << profile.p99_blocks_per_block_row_over_avg
        << ',' << profile.max_blocks_per_block_row_over_avg
        << ',' << profile.active_block_rows
        << ',' << profile.col_delta_u8_eligible_block_rows
        << ',' << profile.col_delta_u16_eligible_block_rows
        << ',' << profile.col_delta_u8_covered_blocks
        << ',' << profile.col_delta_u16_covered_blocks
        << ',' << profile.max_col_delta_in_block_row
        << ',' << profile.local_coord_bits_per_nnz
        << ',' << profile.local_coord_count_bits_per_block
        << ',' << profile.runtime_abs_col_position_bits
        << ',' << profile.runtime_delta_u8_position_bits
        << ',' << profile.runtime_delta_u16_position_bits
        << ',' << profile.runtime_local_coord_abs_position_bits
        << ',' << profile.runtime_local_coord_delta_u8_position_bits
        << ',' << profile.runtime_local_coord_delta_u16_position_bits
        << ',' << profile.active_row_abs_meta_bits
        << ',' << profile.active_row_delta_u8_meta_bits
        << ',' << profile.active_row_delta_u16_meta_bits
        << ',' << profile.runtime_abs_col_bits_per_nnz
        << ',' << profile.runtime_delta_u8_bits_per_nnz
        << ',' << profile.runtime_delta_u16_bits_per_nnz
        << ',' << profile.runtime_local_coord_abs_bits_per_nnz
        << ',' << profile.runtime_local_coord_delta_u8_bits_per_nnz
        << ',' << profile.runtime_local_coord_delta_u16_bits_per_nnz
        << ',' << profile.active_row_abs_meta_bits_per_nnz
        << ',' << profile.active_row_delta_u8_meta_bits_per_nnz
        << ',' << profile.active_row_delta_u16_meta_bits_per_nnz;
}

}  // namespace structural
