#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "structural/bitbsr.hpp"
#include "structural/convert.hpp"
#include "structural/csr.hpp"
#include "structural/csr_bitbsr_split.hpp"
#include "structural/load_balance_policy.hpp"
#include "structural/matrix_market.hpp"
#include "structural/profile.hpp"
#include "structural/spmv.hpp"
#include "structural/value_compression.hpp"
#include "structural/value_stats.hpp"

static_assert(std::is_same_v<structural::Index, std::int32_t>,
              "structural indices are signed 32-bit values");
static_assert(std::is_same_v<structural::Offset, std::int32_t>,
              "structural offsets are signed 32-bit values");
static_assert(sizeof(structural::BitmapWord) == sizeof(std::uint32_t),
              "8x4 BitBSR bitmap word should match the 32-bit block bitmap");

static void test_bitbsr_conversion_preserves_block_offsets() {
    structural::CsrMatrix<double> csr;
    csr.rows = 4;
    csr.cols = 4;
    csr.row_ptr = {0, 2, 3, 5, 6};
    csr.col_idx = {0, 3, 1, 0, 2, 3};
    csr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    const auto bitbsr = structural::convert_to_bitbsr(csr, 2, 2);

    assert(bitbsr.block_rows == 2);
    assert(bitbsr.block_cols == 2);
    assert(bitbsr.num_blocks() == 4);
    assert((bitbsr.block_row_ptr == std::vector<structural::Offset>{0, 2, 4}));
    assert((bitbsr.block_col_idx == std::vector<structural::Index>{0, 1, 0, 1}));
    assert((bitbsr.block_val_ptr == std::vector<structural::Offset>{0, 2, 3, 4, 6}));
    assert((bitbsr.bitmap_words == std::vector<structural::BitmapWord>{9U, 2U, 1U, 9U}));
    assert((bitbsr.values == std::vector<double>{1.0, 3.0, 2.0, 4.0, 5.0, 6.0}));
}

static void test_bitbsr_spmv_matches_csr_reference() {
    structural::CsrMatrix<double> csr;
    csr.rows = 4;
    csr.cols = 4;
    csr.row_ptr = {0, 2, 3, 5, 6};
    csr.col_idx = {0, 3, 1, 0, 2, 3};
    csr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    const auto bitbsr = structural::convert_to_bitbsr(csr, 2, 2);
    const auto y_csr = structural::spmv_csr(csr, x);
    const auto y_bitbsr = structural::spmv_bitbsr(bitbsr, x);

    assert((y_csr == std::vector<double>{9.0, 6.0, 19.0, 24.0}));
    assert(y_bitbsr.size() == y_csr.size());
    for (size_t i = 0; i < y_csr.size(); ++i) {
        assert(std::abs(y_csr[i] - y_bitbsr[i]) < 1e-12);
    }
}

static void test_bitbsr_conversion_drops_stored_zeros() {
    structural::CsrMatrix<double> csr;
    csr.rows = 2;
    csr.cols = 4;
    csr.row_ptr = {0, 3, 5};
    csr.col_idx = {0, 1, 2, 1, 3};
    csr.values = {0.0, 2.0, 0.0, 3.0, 0.0};

    const auto bitbsr = structural::convert_to_bitbsr(csr, 2, 2);
    const auto filtered = structural::remove_stored_zeros(csr);
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    const auto y_bitbsr = structural::spmv_bitbsr(bitbsr, x);
    const auto y_filtered = structural::spmv_csr(filtered, x);

    assert(bitbsr.nnz() == 2);
    assert((bitbsr.values == std::vector<double>{2.0, 3.0}));
    assert((y_bitbsr == y_filtered));
}

static void test_bitbsr_validate_rejects_stored_zero() {
    structural::BitBsrMatrix<double> bitbsr;
    bitbsr.rows = 1;
    bitbsr.cols = 1;
    bitbsr.block_rows = 1;
    bitbsr.block_cols = 1;
    bitbsr.num_block_rows = 1;
    bitbsr.num_block_cols = 1;
    bitbsr.words_per_block = 1;
    bitbsr.block_row_ptr = {0, 1};
    bitbsr.block_col_idx = {0};
    bitbsr.bitmap_words = {1U};
    bitbsr.block_val_ptr = {0, 1};
    bitbsr.values = {0.0};

    bool threw = false;
    try {
        bitbsr.validate();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

static void test_csr_bitbsr_split_routes_sparse_blocks_to_residual_csr() {
    structural::CsrMatrix<double> csr;
    csr.rows = 4;
    csr.cols = 8;
    csr.row_ptr = {0, 3, 4, 6, 7};
    csr.col_idx = {0, 1, 4, 0, 2, 3, 7};
    csr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    const auto split = structural::convert_to_csr_bitbsr_split(csr, 2, 2, 1);

    assert(split.block_rows == 2);
    assert(split.block_cols == 2);
    assert(split.residual_threshold == 1);
    assert(split.kept_blocks == 2);
    assert(split.residual_blocks == 2);
    assert(split.kept_nnz == 5);
    assert(split.residual_nnz == 2);
    assert((split.bitbsr.block_row_ptr == std::vector<structural::Offset>{0, 1, 2}));
    assert((split.bitbsr.block_col_idx == std::vector<structural::Index>{0, 1}));
    assert((split.residual.row_ptr == std::vector<structural::Offset>{0, 1, 1, 1, 2}));
    assert((split.residual.col_idx == std::vector<structural::Index>{4, 7}));
    assert((split.residual.values == std::vector<double>{3.0, 7.0}));

    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto y_csr = structural::spmv_csr(csr, x);
    const auto y_split = structural::spmv_csr_bitbsr_split(split, x);

    assert(y_split.size() == y_csr.size());
    for (size_t i = 0; i < y_csr.size(); ++i) {
        assert(std::abs(y_csr[i] - y_split[i]) < 1e-12);
    }
}

static void test_csr_bitbsr_split_cost_gate_keeps_expensive_residual_blocks() {
    structural::CsrMatrix<double> csr;
    csr.rows = 4;
    csr.cols = 8;
    csr.row_ptr = {0, 3, 4, 6, 7};
    csr.col_idx = {0, 1, 4, 0, 2, 3, 7};
    csr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    structural::CsrBitBsrSplitCostModel cost;
    cost.bitmap_block_cost = 2.0;
    cost.csr_nnz_cost = 1.0;
    const auto split = structural::convert_to_csr_bitbsr_split_cost_gated(
        csr, 2, 2, 3, cost);

    assert(split.residual_threshold == 3);
    assert(split.kept_blocks == 2);
    assert(split.residual_blocks == 2);
    assert(split.cost_skipped_blocks == 2);
    assert(split.kept_nnz == 5);
    assert(split.residual_nnz == 2);
    assert(split.cost_skipped_nnz == 5);
    assert((split.bitbsr.block_col_idx == std::vector<structural::Index>{0, 1}));
    assert((split.residual.col_idx == std::vector<structural::Index>{4, 7}));

    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto y_csr = structural::spmv_csr(csr, x);
    const auto y_split = structural::spmv_csr_bitbsr_split(split, x);

    assert(y_split.size() == y_csr.size());
    for (size_t i = 0; i < y_csr.size(); ++i) {
        assert(std::abs(y_csr[i] - y_split[i]) < 1e-12);
    }
}

static void test_blockrow_coo_residual_packs_block_row_local_positions() {
    structural::CsrMatrix<double> csr;
    csr.rows = 4;
    csr.cols = 8;
    csr.row_ptr = {0, 3, 4, 6, 7};
    csr.col_idx = {0, 1, 4, 0, 2, 3, 7};
    csr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    const auto split = structural::convert_to_csr_bitbsr_split(csr, 2, 2, 1);
    const auto residual = structural::convert_residual_csr_to_blockrow_coo(
        split.residual, 2, 2);

    assert(residual.block_rows == 2);
    assert(residual.block_cols == 2);
    assert(residual.bits_per_block == 4);
    assert((residual.blockrow_ptr == std::vector<structural::Offset>{0, 1, 2}));
    assert((residual.packed_col_local == std::vector<structural::Index>{8, 15}));
    assert((residual.values == std::vector<double>{3.0, 7.0}));

    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto y = structural::spmv_bitbsr(split.bitbsr, x);
    const auto residual_y = structural::spmv_blockrow_coo_residual(residual, x);
    const auto y_csr = structural::spmv_csr(csr, x);
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += residual_y[i];
        assert(std::abs(y[i] - y_csr[i]) < 1e-12);
    }
}

static void test_sliced_sell_residual_pads_rows_without_losing_row_ownership() {
    structural::CsrMatrix<double> csr;
    csr.rows = 4;
    csr.cols = 8;
    csr.row_ptr = {0, 3, 4, 6, 7};
    csr.col_idx = {0, 1, 4, 0, 2, 3, 7};
    csr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    const auto split = structural::convert_to_csr_bitbsr_split(csr, 2, 2, 1);
    const auto residual = structural::convert_residual_csr_to_sliced_sell(
        split.residual, 2);

    assert(residual.slice_height == 2);
    assert(residual.num_slices == 2);
    assert((residual.slice_ptr == std::vector<structural::Offset>{0, 2, 4}));
    assert((residual.slice_width == std::vector<structural::Index>{1, 1}));
    assert((residual.col_idx == std::vector<structural::Index>{4, -1, -1, 7}));
    assert((residual.values == std::vector<double>{3.0, 0.0, 0.0, 7.0}));
    assert(residual.padded_nnz() == 4);
    assert(residual.actual_nnz == 2);

    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto y = structural::spmv_bitbsr(split.bitbsr, x);
    const auto residual_y = structural::spmv_sliced_sell_residual(residual, x);
    const auto y_csr = structural::spmv_csr(csr, x);
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += residual_y[i];
        assert(std::abs(y[i] - y_csr[i]) < 1e-12);
    }
}

static void test_matrix_market_loader_handles_symmetric_pattern() {
    const std::string path = "/tmp/structural_symmetric_pattern_test.mtx";
    {
        std::ofstream out(path);
        out << "%%MatrixMarket matrix coordinate pattern symmetric\n";
        out << "% generated by struct_tests\n";
        out << "3 3 2\n";
        out << "1 1\n";
        out << "2 3\n";
    }

    const auto loaded = structural::read_matrix_market_with_info<double>(path);
    const auto& csr = loaded.matrix;

    assert(loaded.info.field == structural::MatrixMarketField::Pattern);
    assert(loaded.info.symmetry == structural::MatrixMarketSymmetry::Symmetric);
    assert(loaded.stored_entries == 2);
    assert(loaded.expanded_entries == 3);
    assert(loaded.coalesced_duplicates == 0);
    assert(csr.rows == 3);
    assert(csr.cols == 3);
    assert(csr.nnz() == 3);
    assert((csr.row_ptr == std::vector<structural::Offset>{0, 1, 2, 3}));
    assert((csr.col_idx == std::vector<structural::Index>{0, 2, 1}));
    assert((csr.values == std::vector<double>{1.0, 1.0, 1.0}));
}

static void test_matrix_market_loader_handles_skew_symmetric_without_substring_bug() {
    const std::string path = "/tmp/structural_skew_symmetric_test.mtx";
    {
        std::ofstream out(path);
        out << "%%MatrixMarket matrix coordinate real skew-symmetric\n";
        out << "3 3 1\n";
        out << "1 3 2.5\n";
    }

    const auto loaded = structural::read_matrix_market_with_info<double>(path);
    const auto& csr = loaded.matrix;

    assert(loaded.info.field == structural::MatrixMarketField::Real);
    assert(loaded.info.symmetry == structural::MatrixMarketSymmetry::SkewSymmetric);
    assert(loaded.expanded_entries == 2);
    assert(csr.rows == 3);
    assert(csr.cols == 3);
    assert(csr.nnz() == 2);
    assert((csr.row_ptr == std::vector<structural::Offset>{0, 1, 1, 2}));
    assert((csr.col_idx == std::vector<structural::Index>{2, 0}));
    assert((csr.values == std::vector<double>{2.5, -2.5}));
}

static void test_matrix_market_loader_reports_integer_general_metadata() {
    const std::string path = "/tmp/structural_integer_general_test.mtx";
    {
        std::ofstream out(path);
        out << "%%MatrixMarket matrix coordinate integer general\n";
        out << "2 3 2\n";
        out << "1 1 7\n";
        out << "2 3 -4\n";
    }

    const auto loaded = structural::read_matrix_market_with_info<double>(path);
    const auto& csr = loaded.matrix;

    assert(loaded.info.field == structural::MatrixMarketField::Integer);
    assert(loaded.info.symmetry == structural::MatrixMarketSymmetry::General);
    assert(!loaded.info.is_pattern());
    assert(!loaded.info.requires_mirror_expansion());
    assert((csr.values == std::vector<double>{7.0, -4.0}));
}

static void test_matrix_market_loader_drops_stored_zeros() {
    const std::string path = "/tmp/structural_drop_stored_zeros_test.mtx";
    {
        std::ofstream out(path);
        out << "%%MatrixMarket matrix coordinate real general\n";
        out << "3 4 5\n";
        out << "1 1 0\n";
        out << "1 2 2\n";
        out << "1 2 -2\n";
        out << "2 4 5\n";
        out << "3 1 -1\n";
    }

    const auto loaded = structural::read_matrix_market_with_info<double>(path);
    const auto& csr = loaded.matrix;

    assert(loaded.stored_entries == 5);
    assert(loaded.expanded_entries == 5);
    assert(csr.nnz() == 2);
    assert((csr.row_ptr == std::vector<structural::Offset>{0, 0, 1, 2}));
    assert((csr.col_idx == std::vector<structural::Index>{3, 0}));
    assert((csr.values == std::vector<double>{5.0, -1.0}));
}

static void test_value_stats_detects_explicit_binary_values() {
    structural::CsrMatrix<double> csr;
    csr.rows = 3;
    csr.cols = 3;
    csr.row_ptr = {0, 1, 2, 4};
    csr.col_idx = {0, 1, 0, 2};
    csr.values = {1.0, 1.0, 0.0, 1.0};

    const auto stats = structural::compute_value_stats(csr);

    assert(stats.explicit_zero_count == 1);
    assert(stats.all_values_one == false);
    assert(stats.binary_zero_one == true);
    assert(stats.sign_only == false);
    assert(stats.unique_values_exact == 2);
    assert(stats.unique_values_overflow == false);
}

static void test_value_compression_reports_exact_codecs() {
    structural::CsrMatrix<double> csr;
    csr.rows = 3;
    csr.cols = 3;
    csr.row_ptr = {0, 1, 3, 5};
    csr.col_idx = {0, 0, 1, 1, 2};
    csr.values = {1.0, -1.0, 0.0, 1.0, -1.0};

    const auto analysis = structural::analyze_value_compression(csr);
    const auto* original = structural::find_value_codec(analysis, "original");
    const auto* sign = structural::find_value_codec(analysis, "sign2");
    const auto* dict4 = structural::find_value_codec(analysis, "dict4");
    const auto* unique = structural::find_value_codec(analysis, "dict16");
    const auto* legacy_unique = structural::find_value_codec(analysis, "unique16");

    assert(original != nullptr);
    assert(original->applicable);
    assert(original->exact);
    assert(original->payload_bits == 5 * 64);
    assert(sign != nullptr);
    assert(sign->applicable);
    assert(sign->exact);
    assert(sign->payload_bits == 10);
    assert(std::abs(sign->bytes_per_nnz - 0.25) < 1e-12);
    assert(sign->spmv_max_rel == 0.0);
    assert(sign->passes_spmv_gate);

    assert(dict4 != nullptr);
    assert(dict4->applicable);
    assert(dict4->exact);
    assert(dict4->payload_bits == 10 + 3 * 64);

    assert(unique != nullptr);
    assert(unique == legacy_unique);
    assert(unique->applicable);
    assert(unique->exact);
    assert(unique->payload_bits == 20 + 3 * 64);
    assert(std::abs(analysis.value_entropy_bits - 1.5219280948873621) < 1e-12);
}

static void test_value_compression_reports_constant_and_scaled_sign_codecs() {
    structural::CsrMatrix<double> constant;
    constant.rows = 2;
    constant.cols = 4;
    constant.row_ptr = {0, 4, 8};
    constant.col_idx = {0, 1, 2, 3, 0, 1, 2, 3};
    constant.values.assign(8, 2.5);

    const auto constant_analysis = structural::analyze_value_compression(constant);
    const auto* const_c = structural::find_value_codec(constant_analysis, "const_c");
    const auto* dict1 = structural::find_value_codec(constant_analysis, "dict1");

    assert(const_c != nullptr);
    assert(const_c == dict1);
    assert(const_c->applicable);
    assert(const_c->exact);
    assert(const_c->codebook_size == 1);
    assert(const_c->index_bits == 0);
    assert(const_c->payload_bits == 64);

    structural::CsrMatrix<double> scaled_sign;
    scaled_sign.rows = 2;
    scaled_sign.cols = 4;
    scaled_sign.row_ptr = {0, 4, 8};
    scaled_sign.col_idx = {0, 1, 2, 3, 0, 1, 2, 3};
    scaled_sign.values = {2.5, -2.5, 2.5, -2.5, -2.5, 2.5, -2.5, 2.5};

    const auto scaled_sign_analysis = structural::analyze_value_compression(scaled_sign);
    const auto* sign2 = structural::find_value_codec(scaled_sign_analysis, "sign2");
    const auto* sign2_scale =
        structural::find_value_codec(scaled_sign_analysis, "sign2_scale");

    assert(sign2 != nullptr);
    assert(!sign2->applicable);
    assert(sign2_scale != nullptr);
    assert(sign2_scale->applicable);
    assert(sign2_scale->exact);
    assert(sign2_scale->codebook_size == 1);
    assert(sign2_scale->index_bits == 1);
    assert(sign2_scale->payload_bits == 8 + 64);
    assert(std::abs(sign2_scale->bytes_per_nnz - 1.125) < 1e-12);
}

static void test_value_compression_reports_exact_dictionary_ladder() {
    structural::CsrMatrix<double> csr;
    csr.rows = 2;
    csr.cols = 4;
    csr.row_ptr = {0, 4, 8};
    csr.col_idx = {0, 1, 2, 3, 0, 1, 2, 3};
    csr.values = {2.0, 7.0, 2.0, 7.0, 7.0, 2.0, 7.0, 2.0};

    const auto analysis = structural::analyze_value_compression(csr);
    const auto* dict2 = structural::find_value_codec(analysis, "dict2");
    const auto* dict4 = structural::find_value_codec(analysis, "dict4");
    const auto* dict16 = structural::find_value_codec(analysis, "dict16");
    const auto* dict256 = structural::find_value_codec(analysis, "dict256_u8");
    const auto* dict65536 = structural::find_value_codec(analysis, "dict65536_u16");
    const auto* codebook256 = structural::find_value_codec(analysis, "codebook256_u8");

    assert(dict2 != nullptr);
    assert(dict2->applicable);
    assert(dict2->exact);
    assert(dict2->codebook_size == 2);
    assert(dict2->index_bits == 1);
    assert(dict2->payload_bits == 8 + 2 * 64);

    assert(dict4 != nullptr);
    assert(dict4->applicable);
    assert(dict4->exact);
    assert(dict4->codebook_size == 2);
    assert(dict4->index_bits == 2);
    assert(dict4->payload_bits == 8 * 2 + 2 * 64);

    assert(dict16 != nullptr);
    assert(dict16->applicable);
    assert(dict16->exact);
    assert(dict16->codebook_size == 2);
    assert(dict16->index_bits == 4);
    assert(dict16->payload_bits == 8 * 4 + 2 * 64);

    assert(dict256 != nullptr);
    assert(dict256->applicable);
    assert(dict256->exact);
    assert(dict256->codebook_size == 2);
    assert(dict256->index_bits == 8);
    assert(dict256->payload_bits == 8 * 8 + 2 * 64);

    assert(dict65536 != nullptr);
    assert(dict65536->applicable);
    assert(dict65536->exact);
    assert(dict65536->codebook_size == 2);
    assert(dict65536->index_bits == 16);
    assert(dict65536->payload_bits == 8 * 16 + 2 * 64);

    assert(codebook256 != nullptr);
    assert(codebook256->applicable);
    assert(codebook256->exact);
    assert(codebook256->codebook_size == 2);
    assert(codebook256->index_bits == 8);
    assert(codebook256->payload_bits == 8 * (8 + 256 * 8));
}

static void test_value_compression_reports_roundtrip_spmv_error() {
    structural::CsrMatrix<double> csr;
    csr.rows = 2;
    csr.cols = 2;
    csr.row_ptr = {0, 2, 4};
    csr.col_idx = {0, 1, 0, 1};
    csr.values = {1.0, 1.0 / 3.0, -2.0, 0.125};

    const auto analysis = structural::analyze_value_compression(csr);
    const auto* fp16 = structural::find_value_codec(analysis, "fp16");
    const auto* bf16 = structural::find_value_codec(analysis, "bf16");

    assert(fp16 != nullptr);
    assert(fp16->applicable);
    assert(fp16->payload_bits == 64);
    assert(std::abs(fp16->bytes_per_nnz - 2.0) < 1e-12);
    assert(fp16->changed_values == 1);
    assert(fp16->spmv_max_rel > 0.0);

    assert(bf16 != nullptr);
    assert(bf16->applicable);
    assert(bf16->payload_bits == 64);
    assert(bf16->changed_values == 1);
    assert(bf16->spmv_max_rel >= fp16->spmv_max_rel);
}

static void test_value_compression_reports_global_log_kmeans_codec() {
    structural::CsrMatrix<double> csr;
    csr.rows = 2;
    csr.cols = 2;
    csr.row_ptr = {0, 2, 4};
    csr.col_idx = {0, 1, 0, 1};
    csr.values = {1.0, 2.0, -1.0, -2.0};

    const auto analysis = structural::analyze_value_compression(csr);
    const auto* codec = structural::find_value_codec(analysis, "global_logkmeans16");

    assert(codec != nullptr);
    assert(codec->applicable);
    assert(!codec->exact);
    assert(codec->codebook_size == 2);
    assert(codec->index_bits == 2);
    assert(codec->uses_sign_bit);
    assert(codec->payload_bits == 4 * 3 + 2 * 64);
    assert(codec->changed_values == 0);
    assert(codec->max_rel_value_error == 0.0);
    assert(codec->spmv_max_rel == 0.0);
    assert(codec->passes_spmv_gate);
}

static void test_k256_uint8_value_codebook_exactly_reconstructs_small_unique_values() {
    const std::vector<double> values = {3.0, -1.0, 0.0, 3.0, 7.0, -1.0};

    const auto codec = structural::build_k256_uint8_value_codebook(values);
    const auto reconstructed = codec.reconstruct();

    assert(codec.exact);
    assert(codec.codebook_size == 4);
    assert(codec.ids.size() == values.size());
    assert(codec.payload_bytes() == values.size() + 256 * sizeof(double));
    assert(reconstructed == values);

    structural::CsrMatrix<double> csr;
    csr.rows = 2;
    csr.cols = 3;
    csr.row_ptr = {0, 3, 6};
    csr.col_idx = {0, 1, 2, 0, 1, 2};
    csr.values = values;

    const auto analysis = structural::analyze_value_compression(csr);
    const auto* report = structural::find_value_codec(analysis, "codebook256_u8");
    const auto* legacy_report = structural::find_value_codec(analysis, "k256_uint8");

    assert(report != nullptr);
    assert(report == legacy_report);
    assert(report->applicable);
    assert(report->exact);
    assert(report->codebook_size == 4);
    assert(report->payload_bits == static_cast<std::int64_t>(codec.payload_bytes()) * 8);
}

static void test_bitbsr_profile_reports_fill_and_block_row_imbalance() {
    structural::BitBsrMatrix<double> bitbsr;
    bitbsr.rows = 6;
    bitbsr.cols = 8;
    bitbsr.block_rows = 2;
    bitbsr.block_cols = 2;
    bitbsr.num_block_rows = 3;
    bitbsr.num_block_cols = 4;
    bitbsr.words_per_block = 1;
    bitbsr.block_row_ptr = {0, 3, 4, 4};
    bitbsr.block_col_idx = {0, 1, 3, 2};
    bitbsr.bitmap_words = {15U, 1U, 3U, 5U};
    bitbsr.block_val_ptr = {0, 4, 5, 7, 9};
    bitbsr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    const auto profile = structural::profile_bitbsr(bitbsr);

    assert(profile.num_block_rows == 3);
    assert(profile.num_blocks == 4);
    assert(profile.empty_block_rows == 1);
    assert(profile.max_blocks_per_block_row == 3);
    assert(profile.p50_blocks_per_block_row == 1);
    assert(profile.p90_blocks_per_block_row == 3);
    assert(profile.p99_blocks_per_block_row == 3);
    assert(std::abs(profile.avg_blocks_per_block_row - (4.0 / 3.0)) < 1e-12);
    assert(std::abs(profile.max_blocks_per_block_row_over_avg - 2.25) < 1e-12);
    assert(std::abs(profile.block_fill_ratio - 0.5625) < 1e-12);
    assert(profile.blocks_le_25pct_full == 1);
    assert(profile.blocks_le_50pct_full == 3);
    assert(profile.full_blocks == 1);
}

static void test_bitbsr_profile_reports_delta_position_candidates() {
    structural::BitBsrMatrix<double> bitbsr;
    bitbsr.rows = 6;
    bitbsr.cols = 4096;
    bitbsr.block_rows = 2;
    bitbsr.block_cols = 2;
    bitbsr.num_block_rows = 3;
    bitbsr.num_block_cols = 2048;
    bitbsr.words_per_block = 1;
    bitbsr.block_row_ptr = {0, 3, 6, 6};
    bitbsr.block_col_idx = {100, 104, 120, 700, 900, 1100};
    bitbsr.bitmap_words = {15U, 1U, 3U, 5U, 1U, 2U};
    bitbsr.block_val_ptr = {0, 4, 5, 7, 9, 10, 11};
    bitbsr.values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                     7.0, 8.0, 9.0, 10.0, 11.0};

    const auto profile = structural::profile_bitbsr(bitbsr);

    assert(profile.active_block_rows == 2);
    assert(profile.col_delta_u8_eligible_block_rows == 1);
    assert(profile.col_delta_u16_eligible_block_rows == 2);
    assert(profile.col_delta_u8_covered_blocks == 3);
    assert(profile.col_delta_u16_covered_blocks == 6);
    assert(profile.max_col_delta_in_block_row == 400);
    assert(profile.runtime_abs_col_position_bits == 640);
    assert(profile.runtime_delta_u8_position_bits == 600);
    assert(profile.runtime_delta_u16_position_bits == 608);
    assert(profile.local_coord_bits_per_nnz == 2);
    assert(profile.local_coord_count_bits_per_block == 3);
    assert(profile.runtime_local_coord_abs_position_bits == 488);
    assert(profile.runtime_local_coord_delta_u8_position_bits == 448);
    assert(profile.runtime_local_coord_delta_u16_position_bits == 456);
    assert(profile.active_row_abs_meta_bits == 256);
    assert(profile.active_row_delta_u16_meta_bits == 256);
    assert(profile.active_row_delta_u8_meta_bits == 240);
}

static void test_binary_implicit_one_load_balance_policy_keeps_small_cases_light() {
    const auto tiny = structural::select_binary_implicit_one_load_balance_policy(5, 4);
    assert(tiny.policy == structural::LoadBalancePolicy::BinaryImplicitOne);
    assert(tiny.chunk_blocks == 8);
    assert(tiny.split_threshold_blocks == 8);

    const auto small_bibd = structural::select_binary_implicit_one_load_balance_policy(14948, 105);
    assert(small_bibd.chunk_blocks == 8);
    assert(small_bibd.split_threshold_blocks == 8);

    const auto medium_bibd =
        structural::select_binary_implicit_one_load_balance_policy(66480, 136);
    assert(medium_bibd.chunk_blocks == 16);
    assert(medium_bibd.split_threshold_blocks == 16);
}

static void test_binary_implicit_one_load_balance_policy_widens_large_rows() {
    const auto bibd_19 = structural::select_binary_implicit_one_load_balance_policy(319126, 171);
    assert(bibd_19.chunk_blocks == 64);
    assert(bibd_19.split_threshold_blocks == 64);

    const auto connectus =
        structural::select_binary_implicit_one_load_balance_policy(422802, 512);
    assert(connectus.chunk_blocks == 128);
    assert(connectus.split_threshold_blocks == 128);

    const auto sparse_graph =
        structural::select_binary_implicit_one_load_balance_policy(2516849, 1791489);
    assert(sparse_graph.chunk_blocks == 32);
    assert(sparse_graph.split_threshold_blocks == 32);
}

int main() {
    test_bitbsr_conversion_preserves_block_offsets();
    test_bitbsr_spmv_matches_csr_reference();
    test_bitbsr_conversion_drops_stored_zeros();
    test_bitbsr_validate_rejects_stored_zero();
    test_csr_bitbsr_split_routes_sparse_blocks_to_residual_csr();
    test_csr_bitbsr_split_cost_gate_keeps_expensive_residual_blocks();
    test_blockrow_coo_residual_packs_block_row_local_positions();
    test_sliced_sell_residual_pads_rows_without_losing_row_ownership();
    test_matrix_market_loader_handles_symmetric_pattern();
    test_matrix_market_loader_handles_skew_symmetric_without_substring_bug();
    test_matrix_market_loader_reports_integer_general_metadata();
    test_matrix_market_loader_drops_stored_zeros();
    test_value_stats_detects_explicit_binary_values();
    test_value_compression_reports_exact_codecs();
    test_value_compression_reports_constant_and_scaled_sign_codecs();
    test_value_compression_reports_exact_dictionary_ladder();
    test_value_compression_reports_roundtrip_spmv_error();
    test_value_compression_reports_global_log_kmeans_codec();
    test_k256_uint8_value_codebook_exactly_reconstructs_small_unique_values();
    test_bitbsr_profile_reports_fill_and_block_row_imbalance();
    test_bitbsr_profile_reports_delta_position_candidates();
    test_binary_implicit_one_load_balance_policy_keeps_small_cases_light();
    test_binary_implicit_one_load_balance_policy_widens_large_rows();
    return 0;
}
