#include "structural/spmv_cli_options.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

structural::SpmvCliOptions parse_argv(const std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (const std::string& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    return structural::parse_spmv_cli_options(static_cast<int>(argv.size()), argv.data());
}

void assert_throws(const std::function<void()>& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_default_options() {
    const auto opts = parse_argv({"struct_gpu_spmv", "matrix.mtx"});

    assert(opts.matrix_path == "matrix.mtx");
    assert(opts.precision == "double");
    assert(opts.layout == "col_bitmap64_flag_lb");
    assert(opts.value_codec == "original");
    assert(opts.lb_policy == "manual");
    assert(opts.lb_chunk_blocks == 32);
    assert(opts.lb_split_threshold_blocks == 32);
    assert(!opts.lb_chunk_blocks_explicit);
    assert(!opts.lb_split_threshold_blocks_explicit);
    assert(opts.cusparse_alg == "default");
    assert(opts.warmup == 5);
    assert(opts.iterations == 20);
    assert(!opts.verify_cpu);
    assert(!structural::spmv_has_manual_load_balance_override(opts));
}

void test_aliases_and_explicit_flags() {
    const auto cusparse_alg2 =
        parse_argv({"struct_gpu_spmv", "m.mtx", "--layout", "cusparse_alg2"});
    assert(cusparse_alg2.layout == "cusparse_csr");
    assert(cusparse_alg2.cusparse_alg == "csr_alg2");

    const auto k256 = parse_argv({"struct_gpu_spmv", "m.mtx",
                                  "--value-codec", "codebook256_u8",
                                  "--lb-policy", "binary",
                                  "--lb-chunk-blocks", "64",
                                  "--precision", "float",
                                  "--warmup", "1",
                                  "--iters", "2",
                                  "--verify-cpu"});
    assert(k256.value_codec == "codebook256_u8");
    assert(k256.lb_policy == "binary_implicit_one");
    assert(k256.lb_chunk_blocks == 64);
    assert(k256.lb_split_threshold_blocks == 64);
    assert(k256.lb_chunk_blocks_explicit);
    assert(!k256.lb_split_threshold_blocks_explicit);
    assert(structural::spmv_has_manual_load_balance_override(k256));
    assert(k256.precision == "float");
    assert(k256.warmup == 1);
    assert(k256.iterations == 2);
    assert(k256.verify_cpu);

    const auto fp16 = parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "fp16"});
    assert(fp16.value_codec == "fp16");
    const auto bf16 = parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "bf16"});
    assert(bf16.value_codec == "bf16");
    const auto const_c =
        parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "const_c"});
    assert(const_c.value_codec == "const_c");
    const auto sign2_scale =
        parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "sign2_scale"});
    assert(sign2_scale.value_codec == "sign2_scale");
    const auto logk = parse_argv({"struct_gpu_spmv", "m.mtx",
                                  "--value-codec", "global_logkmeans64"});
    assert(logk.value_codec == "global_logkmeans64");
    const auto auto_lossless =
        parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "auto_lossless"});
    assert(auto_lossless.value_codec == "auto_lossless");

    const auto split = parse_argv({"struct_gpu_spmv", "m.mtx",
                                   "--lb-split-threshold-blocks", "16"});
    assert(split.lb_split_threshold_blocks == 16);
    assert(!split.lb_chunk_blocks_explicit);
    assert(split.lb_split_threshold_blocks_explicit);
    assert(structural::spmv_has_manual_load_balance_override(split));

    const auto alg1 = parse_argv({"struct_gpu_spmv", "m.mtx", "--cusparse-alg", "alg1"});
    assert(alg1.cusparse_alg == "csr_alg1");
}

void test_legacy_value_codec_aliases() {
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "raw"}).value_codec ==
           "original");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "implicit_one"}).value_codec ==
           "all_one");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "dict1"}).value_codec ==
           "const_c");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "const"}).value_codec ==
           "const_c");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "scaled_sign2"}).value_codec ==
           "sign2_scale");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "unique4"}).value_codec ==
           "dict4");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "unique16"}).value_codec ==
           "dict16");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "unique2"}).value_codec ==
           "dict2");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "unique256"}).value_codec ==
           "dict256_u8");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "unique65536"}).value_codec ==
           "dict65536_u16");
    assert(parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "k256_u8"}).value_codec ==
           "codebook256_u8");
}

void test_enum_mapping() {
    using Layout = structural::BitBsrSpmvLayoutKind8x4;
    using ValueCodec = structural::BitBsrValueCodecKind8x4;
    using Policy = structural::LoadBalancePolicy;

    assert(structural::spmv_layout_kind_from_name("col_bitmap64_flag_lb") ==
           Layout::ColBitmap64FlagLb);
    assert(structural::spmv_value_codec_kind_from_name("codebook256_u8") ==
           ValueCodec::K256U8);
    assert(structural::spmv_value_codec_kind_from_name("const_c") == ValueCodec::ConstC);
    assert(structural::spmv_value_codec_kind_from_name("sign2_scale") ==
           ValueCodec::Sign2Scale);
    assert(structural::spmv_value_codec_kind_from_name("dict2") == ValueCodec::Dict2);
    assert(structural::spmv_value_codec_kind_from_name("dict256_u8") ==
           ValueCodec::Dict256U8);
    assert(structural::spmv_value_codec_kind_from_name("dict65536_u16") ==
           ValueCodec::Dict65536U16);
    assert_throws([] { (void)structural::spmv_value_codec_kind_from_name("auto_lossless"); });
    assert(structural::spmv_value_codec_kind_from_name("fp16") == ValueCodec::Fp16);
    assert(structural::spmv_value_codec_kind_from_name("bf16") == ValueCodec::Bf16);
    assert(structural::spmv_value_codec_kind_from_name("global_logkmeans16") ==
           ValueCodec::GlobalLogKmeans16);
    assert(structural::spmv_value_codec_kind_from_name("global_logkmeans32") ==
           ValueCodec::GlobalLogKmeans32);
    assert(structural::spmv_value_codec_kind_from_name("global_logkmeans64") ==
           ValueCodec::GlobalLogKmeans64);
    assert(structural::spmv_value_codec_kind_from_name("global_logkmeans128") ==
           ValueCodec::GlobalLogKmeans128);
    assert(structural::spmv_load_balance_policy_from_name("binary_implicit_one") ==
           Policy::BinaryImplicitOne);
}

void test_invalid_values_throw() {
    assert_throws([] { (void)parse_argv({"struct_gpu_spmv", "m.mtx", "--layout", "bad"}); });
    assert_throws([] {
        (void)parse_argv({"struct_gpu_spmv", "m.mtx", "--layout", "meta"});
    });
    assert_throws([] {
        (void)parse_argv({"struct_gpu_spmv", "m.mtx", "--layout", "block_meta"});
    });
    assert_throws([] {
        (void)parse_argv({"struct_gpu_spmv", "m.mtx", "--layout", "col_delta_u16_raw_lb"});
    });
    assert_throws([] {
        (void)parse_argv({"struct_gpu_spmv", "m.mtx", "--value-codec", "bad"});
    });
    assert_throws([] {
        (void)parse_argv({"struct_gpu_spmv", "m.mtx", "--lb-policy", "bad"});
    });
    assert_throws([] { (void)structural::spmv_layout_kind_from_name("bad"); });
    assert_throws([] { (void)structural::spmv_layout_kind_from_name("block_meta"); });
    assert_throws([] {
        (void)structural::spmv_layout_kind_from_name("col_delta_u16_raw_lb");
    });
    assert_throws([] { (void)structural::spmv_value_codec_kind_from_name("bad"); });
    assert_throws([] { (void)structural::spmv_load_balance_policy_from_name("bad"); });
}

}  // namespace

int main() {
    test_default_options();
    test_aliases_and_explicit_flags();
    test_legacy_value_codec_aliases();
    test_enum_mapping();
    test_invalid_values_throw();
    std::cout << "spmv cli options tests passed\n";
    return 0;
}
