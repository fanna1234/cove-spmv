#pragma once

#include "structural/load_balance_policy.hpp"
#include "structural/operators/bitbsr_spmv_8x4_dispatch.cuh"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace structural {

struct SpmvCliOptions {
    std::string matrix_path;
    std::string precision = "double";
    std::string layout = "col_bitmap64_flag_lb";
    std::string value_codec = "original";
    int lb_chunk_blocks = 32;
    int lb_split_threshold_blocks = -1;
    std::string lb_policy = "manual";
    bool lb_chunk_blocks_explicit = false;
    bool lb_split_threshold_blocks_explicit = false;
    std::string cusparse_alg = "default";
    int warmup = 5;
    int iterations = 20;
    bool verify_cpu = false;
};

inline void print_spmv_cli_usage(const char* argv0, std::ostream& os = std::cerr) {
    os << "usage: " << argv0
       << " MATRIX.mtx"
       << " [--precision float|double]"
       << " [--layout col_bitmap64|col_bitmap64_flag_lb|cusparse_csr]"
       << " [--value-codec original|all_one|const_c|sign2|sign2_scale|dict2|dict4|dict16|"
          "dict256_u8|dict65536_u16|auto_lossless|fp16|bf16|codebook256_u8|"
          "global_logkmeans16|global_logkmeans32|"
          "global_logkmeans64|global_logkmeans128]"
       << " [--lb-policy manual|binary_implicit_one]"
       << " [--lb-chunk-blocks N]"
       << " [--lb-split-threshold-blocks N]"
       << " [--cusparse-alg default|csr_alg1|csr_alg2]"
       << " [--warmup N] [--iters N] [--verify-cpu]\n";
}

inline std::string normalize_spmv_layout_name(const std::string& layout,
                                              std::string& cusparse_alg) {
    if (layout == "col_bitmap" || layout == "packed_abs" ||
        layout == "abs_packed" || layout == "col_bitmap64" ||
        layout == "col_bitmap64_packed") {
        return "col_bitmap64";
    }
    if (layout == "col_bitmap_flag_lb" || layout == "packed_abs_flag_lb" ||
        layout == "abs_packed_flag_lb" || layout == "col_bitmap64_flag_lb" ||
        layout == "col_bitmap64_packed_flag_lb") {
        return "col_bitmap64_flag_lb";
    }
    if (layout == "cusparse" || layout == "cusparse_spmv" ||
        layout == "cusparse_csr") {
        return "cusparse_csr";
    }
    if (layout == "cusparse_csr_alg1" || layout == "cusparse_alg1") {
        cusparse_alg = "csr_alg1";
        return "cusparse_csr";
    }
    if (layout == "cusparse_csr_alg2" || layout == "cusparse_alg2") {
        cusparse_alg = "csr_alg2";
        return "cusparse_csr";
    }
    throw std::runtime_error(
        "layout must be col_bitmap64, col_bitmap64_flag_lb, or cusparse_csr");
}

inline std::string normalize_spmv_value_codec_name(const std::string& value_codec) {
    if (value_codec == "original" || value_codec == "all_one" ||
        value_codec == "const_c" ||
        value_codec == "sign2" || value_codec == "sign2_scale" ||
        value_codec == "dict4" ||
        value_codec == "dict2" || value_codec == "dict16" ||
        value_codec == "dict256_u8" || value_codec == "dict65536_u16" ||
        value_codec == "auto_lossless" ||
        value_codec == "fp16" || value_codec == "bf16" ||
        value_codec == "bfp8" || value_codec == "bfp8_outlier" ||
        value_codec == "bfp4" ||
        value_codec == "codebook256_u8" ||
        value_codec == "global_logkmeans16" ||
        value_codec == "global_logkmeans32" ||
        value_codec == "global_logkmeans64" ||
        value_codec == "global_logkmeans128") {
        return value_codec;
    }
    if (value_codec == "raw" || value_codec == "orig" || value_codec == "origin") {
        return "original";
    }
    if (value_codec == "one" || value_codec == "pattern" || value_codec == "implicit1" ||
        value_codec == "implicit_one") {
        return "all_one";
    }
    if (value_codec == "dict1" || value_codec == "const" ||
        value_codec == "constant" || value_codec == "constant_c") {
        return "const_c";
    }
    if (value_codec == "sign" || value_codec == "sign_i8" ||
        value_codec == "sign_tile32" || value_codec == "tile_sign") {
        return "sign2";
    }
    if (value_codec == "scaled_sign" || value_codec == "scaled_sign2" ||
        value_codec == "sign_scale" || value_codec == "sign2_c") {
        return "sign2_scale";
    }
    if (value_codec == "code1" || value_codec == "u1" ||
        value_codec == "unique2_tile32" || value_codec == "tile1" ||
        value_codec == "unique2") {
        return "dict2";
    }
    if (value_codec == "code2" || value_codec == "u2" ||
        value_codec == "unique4_tile64" || value_codec == "ternary" ||
        value_codec == "ternary3" || value_codec == "tile2" ||
        value_codec == "unique4") {
        return "dict4";
    }
    if (value_codec == "dict16" || value_codec == "u4" ||
        value_codec == "unique_u8" || value_codec == "unique16_tile128" ||
        value_codec == "tile4" || value_codec == "unique16") {
        return "dict16";
    }
    if (value_codec == "dict256" || value_codec == "unique256" ||
        value_codec == "unique256_tile256" || value_codec == "dict256_tile256" ||
        value_codec == "exact_u8") {
        return "dict256_u8";
    }
    if (value_codec == "dict65536" || value_codec == "dict_u16" ||
        value_codec == "unique65536" || value_codec == "unique65536_tile512" ||
        value_codec == "dict65536_tile512" || value_codec == "tile16" ||
        value_codec == "exact_u16") {
        return "dict65536_u16";
    }
    if (value_codec == "k256" || value_codec == "k256_uint8" ||
        value_codec == "uint8" || value_codec == "u8" ||
        value_codec == "k256_tile256" || value_codec == "tile8" ||
        value_codec == "k256_u8") {
        return "codebook256_u8";
    }
    if (value_codec == "half" || value_codec == "fp16_tile512") {
        return "fp16";
    }
    if (value_codec == "bfloat16" || value_codec == "bf16_tile512") {
        return "bf16";
    }
    if (value_codec == "bfp8_blkscale" || value_codec == "int8_blkscale" ||
        value_codec == "int8_8x4") {
        return "bfp8";
    }
    if (value_codec == "bfp8_outlier_blkscale" || value_codec == "bfp8_outlier_lb" ||
        value_codec == "int8_outlier" || value_codec == "bfp8_2pass") {
        return "bfp8_outlier";
    }
    if (value_codec == "bfp4_blkscale" || value_codec == "int4_blkscale" ||
        value_codec == "int4_8x4") {
        return "bfp4";
    }
    if (value_codec == "logkmeans16" || value_codec == "log_kmeans16") {
        return "global_logkmeans16";
    }
    if (value_codec == "logkmeans32" || value_codec == "log_kmeans32") {
        return "global_logkmeans32";
    }
    if (value_codec == "logkmeans64" || value_codec == "log_kmeans64") {
        return "global_logkmeans64";
    }
    if (value_codec == "logkmeans128" || value_codec == "log_kmeans128") {
        return "global_logkmeans128";
    }
    throw std::runtime_error(
        "value-codec must be original, all_one, const_c, sign2, sign2_scale, "
        "dict2, dict4, dict16, dict256_u8, dict65536_u16, auto_lossless, "
        "fp16, bf16, codebook256_u8, or "
        "global_logkmeans{16,32,64,128}");
}

inline std::string normalize_spmv_load_balance_policy_name(const std::string& policy) {
    if (policy == "manual" || policy == "binary_implicit_one") {
        return policy;
    }
    if (policy == "binary" || policy == "implicit_one" || policy == "all_one" ||
        policy == "binary_one") {
        return "binary_implicit_one";
    }
    throw std::runtime_error("lb-policy must be manual or binary_implicit_one");
}

inline std::string normalize_cusparse_spmv_alg_name(const std::string& alg) {
    if (alg == "default" || alg == "csr_alg1" || alg == "csr_alg2") {
        return alg;
    }
    if (alg == "alg1") {
        return "csr_alg1";
    }
    if (alg == "alg2") {
        return "csr_alg2";
    }
    throw std::runtime_error("cusparse-alg must be default, csr_alg1, or csr_alg2");
}

inline bool spmv_has_manual_load_balance_override(const SpmvCliOptions& opts) {
    return opts.lb_chunk_blocks_explicit || opts.lb_split_threshold_blocks_explicit;
}

inline BitBsrSpmvLayoutKind8x4 spmv_layout_kind_from_name(const std::string& layout) {
    if (layout == "col_bitmap64") {
        return BitBsrSpmvLayoutKind8x4::ColBitmap64;
    }
    if (layout == "col_bitmap64_flag_lb") {
        return BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb;
    }
    if (layout == "cusparse_csr") {
        return BitBsrSpmvLayoutKind8x4::CusparseCsr;
    }
    throw std::runtime_error("internal error: unknown normalized layout");
}

inline BitBsrValueCodecKind8x4 spmv_value_codec_kind_from_name(
    const std::string& value_codec) {
    if (value_codec == "original") {
        return BitBsrValueCodecKind8x4::Raw;
    }
    if (value_codec == "all_one") {
        return BitBsrValueCodecKind8x4::ImplicitOne;
    }
    if (value_codec == "const_c") {
        return BitBsrValueCodecKind8x4::ConstC;
    }
    if (value_codec == "sign2") {
        return BitBsrValueCodecKind8x4::Sign2;
    }
    if (value_codec == "sign2_scale") {
        return BitBsrValueCodecKind8x4::Sign2Scale;
    }
    if (value_codec == "dict2") {
        return BitBsrValueCodecKind8x4::Dict2;
    }
    if (value_codec == "dict4") {
        return BitBsrValueCodecKind8x4::Unique4;
    }
    if (value_codec == "dict16") {
        return BitBsrValueCodecKind8x4::Unique16;
    }
    if (value_codec == "dict256_u8") {
        return BitBsrValueCodecKind8x4::Dict256U8;
    }
    if (value_codec == "dict65536_u16") {
        return BitBsrValueCodecKind8x4::Dict65536U16;
    }
    if (value_codec == "fp16") {
        return BitBsrValueCodecKind8x4::Fp16;
    }
    if (value_codec == "bf16") {
        return BitBsrValueCodecKind8x4::Bf16;
    }
    if (value_codec == "bfp8") {
        return BitBsrValueCodecKind8x4::Bfp8;
    }
    if (value_codec == "bfp8_outlier") {
        return BitBsrValueCodecKind8x4::Bfp8Outlier;
    }
    if (value_codec == "bfp4") {
        return BitBsrValueCodecKind8x4::Bfp4;
    }
    if (value_codec == "codebook256_u8") {
        return BitBsrValueCodecKind8x4::K256U8;
    }
    if (value_codec == "global_logkmeans16") {
        return BitBsrValueCodecKind8x4::GlobalLogKmeans16;
    }
    if (value_codec == "global_logkmeans32") {
        return BitBsrValueCodecKind8x4::GlobalLogKmeans32;
    }
    if (value_codec == "global_logkmeans64") {
        return BitBsrValueCodecKind8x4::GlobalLogKmeans64;
    }
    if (value_codec == "global_logkmeans128") {
        return BitBsrValueCodecKind8x4::GlobalLogKmeans128;
    }
    throw std::runtime_error("internal error: unknown normalized value codec");
}

inline LoadBalancePolicy spmv_load_balance_policy_from_name(const std::string& policy) {
    if (policy == "manual") {
        return LoadBalancePolicy::Manual;
    }
    if (policy == "binary_implicit_one") {
        return LoadBalancePolicy::BinaryImplicitOne;
    }
    throw std::runtime_error("internal error: unknown normalized lb policy");
}

inline SpmvCliOptions parse_spmv_cli_options(int argc, char** argv) {
    SpmvCliOptions opts;
    if (argc < 2) {
        print_spmv_cli_usage(argv[0]);
        std::exit(2);
    }
    opts.matrix_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--precision" && i + 1 < argc) {
            opts.precision = argv[++i];
            if (opts.precision != "float" && opts.precision != "double") {
                throw std::runtime_error("precision must be float or double");
            }
        } else if (arg == "--layout" && i + 1 < argc) {
            opts.layout = normalize_spmv_layout_name(argv[++i], opts.cusparse_alg);
        } else if (arg == "--value-codec" && i + 1 < argc) {
            opts.value_codec = normalize_spmv_value_codec_name(argv[++i]);
        } else if ((arg == "--lb-chunk-blocks" || arg == "--chunk-blocks") &&
                   i + 1 < argc) {
            opts.lb_chunk_blocks = std::stoi(argv[++i]);
            opts.lb_chunk_blocks_explicit = true;
        } else if ((arg == "--lb-split-threshold-blocks" ||
                    arg == "--split-threshold-blocks" ||
                    arg == "--lb-split-threshold") &&
                   i + 1 < argc) {
            opts.lb_split_threshold_blocks = std::stoi(argv[++i]);
            opts.lb_split_threshold_blocks_explicit = true;
        } else if (arg == "--lb-policy" && i + 1 < argc) {
            opts.lb_policy = normalize_spmv_load_balance_policy_name(argv[++i]);
        } else if (arg == "--cusparse-alg" && i + 1 < argc) {
            opts.cusparse_alg = normalize_cusparse_spmv_alg_name(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts.warmup = std::stoi(argv[++i]);
        } else if ((arg == "--iters" || arg == "--iterations") && i + 1 < argc) {
            opts.iterations = std::stoi(argv[++i]);
        } else if (arg == "--verify-cpu") {
            opts.verify_cpu = true;
        } else if (arg == "--help" || arg == "-h") {
            print_spmv_cli_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (opts.warmup < 0 || opts.iterations <= 0) {
        throw std::runtime_error("warmup must be non-negative and iterations must be positive");
    }
    if (opts.lb_chunk_blocks <= 0) {
        throw std::runtime_error("lb-chunk-blocks must be positive");
    }
    if (opts.lb_split_threshold_blocks < 0) {
        opts.lb_split_threshold_blocks = opts.lb_chunk_blocks;
    }
    if (opts.lb_split_threshold_blocks <= 0) {
        throw std::runtime_error("lb-split-threshold-blocks must be positive");
    }
    return opts;
}

}  // namespace structural
