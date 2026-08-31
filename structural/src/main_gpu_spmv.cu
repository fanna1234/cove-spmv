#include "structural/gpu_convert.cuh"
#include "structural/gpu_spmv.cuh"
#include "structural/gpu_value_codecs.cuh"
#include "structural/load_balance_policy.hpp"
#include "structural/matrix_market.hpp"
#include "structural/spmv.hpp"
#include "structural/spmv_cli_options.hpp"
#include "structural/spmv_harness.hpp"
#include "structural/spmv_operator_plan.hpp"

#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Options = structural::SpmvCliOptions;
using LayoutKind = structural::BitBsrSpmvLayoutKind8x4;
using ValueCodecKind = structural::BitBsrValueCodecKind8x4;

struct EffectiveLoadBalanceParams {
    int chunk_blocks = 32;
    int split_threshold_blocks = 32;
    structural::LoadBalancePolicy policy = structural::LoadBalancePolicy::Manual;
};

inline bool is_auto_lossless_value_codec(const Options& opts) {
    return opts.value_codec == "auto_lossless";
}

inline const char* value_codec_name(ValueCodecKind value_codec) {
    switch (value_codec) {
        case ValueCodecKind::Raw:
            return "original";
        case ValueCodecKind::ImplicitOne:
            return "all_one";
        case ValueCodecKind::ConstC:
            return "const_c";
        case ValueCodecKind::Sign2:
            return "sign2";
        case ValueCodecKind::Sign2Scale:
            return "sign2_scale";
        case ValueCodecKind::Dict2:
            return "dict2";
        case ValueCodecKind::Unique4:
            return "dict4";
        case ValueCodecKind::Unique16:
            return "dict16";
        case ValueCodecKind::Dict256U8:
            return "dict256_u8";
        case ValueCodecKind::Dict65536U16:
            return "dict65536_u16";
        case ValueCodecKind::Fp16:
            return "fp16";
        case ValueCodecKind::Bf16:
            return "bf16";
        case ValueCodecKind::Bfp8:
            return "bfp8";
        case ValueCodecKind::Bfp8Outlier:
            return "bfp8_outlier";
        case ValueCodecKind::Bfp4:
            return "bfp4";
        case ValueCodecKind::K256U8:
            return "codebook256_u8";
        case ValueCodecKind::GlobalLogKmeans16:
            return "global_logkmeans16";
        case ValueCodecKind::GlobalLogKmeans32:
            return "global_logkmeans32";
        case ValueCodecKind::GlobalLogKmeans64:
            return "global_logkmeans64";
        case ValueCodecKind::GlobalLogKmeans128:
            return "global_logkmeans128";
    }
    return "unknown";
}

std::size_t auto_lossless_selected_payload_bytes(
    const structural::DeviceExactValueHistogram& histogram,
    ValueCodecKind value_codec) {
    switch (value_codec) {
        case ValueCodecKind::Raw:
            return histogram.raw_payload_bytes;
        case ValueCodecKind::ImplicitOne:
            return 0;
        case ValueCodecKind::ConstC:
            return histogram.const_c_payload_bytes;
        case ValueCodecKind::Sign2:
            return histogram.sign2_payload_bytes;
        case ValueCodecKind::Sign2Scale:
            return histogram.sign2_scale_payload_bytes;
        case ValueCodecKind::Dict2:
            return histogram.dict2_payload_bytes;
        case ValueCodecKind::Unique4:
            return histogram.dict4_payload_bytes;
        case ValueCodecKind::Unique16:
            return histogram.dict16_payload_bytes;
        case ValueCodecKind::Dict256U8:
            return histogram.dict256_u8_payload_bytes;
        case ValueCodecKind::Dict65536U16:
            return histogram.dict65536_u16_payload_bytes;
        case ValueCodecKind::Fp16:
        case ValueCodecKind::Bf16:
        case ValueCodecKind::Bfp8:
        case ValueCodecKind::Bfp8Outlier:
        case ValueCodecKind::Bfp4:
        case ValueCodecKind::K256U8:
        case ValueCodecKind::GlobalLogKmeans16:
        case ValueCodecKind::GlobalLogKmeans32:
        case ValueCodecKind::GlobalLogKmeans64:
        case ValueCodecKind::GlobalLogKmeans128:
            break;
    }
    return histogram.raw_payload_bytes;
}

ValueCodecKind select_auto_lossless_value_codec(
    const structural::DeviceExactValueHistogram& histogram) {
    if (histogram.nnz == 0) {
        return ValueCodecKind::Raw;
    }

    const auto saves_bytes = [&](std::size_t payload_bytes) {
        return payload_bytes < histogram.raw_payload_bytes;
    };

    if (histogram.all_one && saves_bytes(0)) {
        return ValueCodecKind::ImplicitOne;
    }

    if (histogram.const_c && saves_bytes(histogram.const_c_payload_bytes)) {
        return ValueCodecKind::ConstC;
    }

    if (histogram.sign2 && saves_bytes(histogram.sign2_payload_bytes)) {
        return ValueCodecKind::Sign2;
    }

    if (histogram.sign2_scale && saves_bytes(histogram.sign2_scale_payload_bytes)) {
        return ValueCodecKind::Sign2Scale;
    }

    if (histogram.unique_count <= 2 &&
        saves_bytes(histogram.dict2_payload_bytes)) {
        return ValueCodecKind::Dict2;
    }
    if (histogram.unique_count <= 4 &&
        saves_bytes(histogram.dict4_payload_bytes)) {
        return ValueCodecKind::Unique4;
    }
    if (histogram.unique_count <= 16 &&
        saves_bytes(histogram.dict16_payload_bytes)) {
        return ValueCodecKind::Unique16;
    }
    if (histogram.unique_count <= 256 &&
        saves_bytes(histogram.dict256_u8_payload_bytes)) {
        return ValueCodecKind::Dict256U8;
    }
    if (!histogram.unique_overflow && histogram.unique_count <= 65536 &&
        saves_bytes(histogram.dict65536_u16_payload_bytes)) {
        return ValueCodecKind::Dict65536U16;
    }
    return ValueCodecKind::Raw;
}

template <typename T>
EffectiveLoadBalanceParams select_effective_load_balance_params(
    const Options& opts,
    LayoutKind layout,
    ValueCodecKind value_codec,
    const structural::DeviceBitBsrMatrix<T>& device) {
    const bool manual_override = structural::spmv_has_manual_load_balance_override(opts);
    if (opts.lb_policy == "manual" || manual_override) {
        const auto params =
            structural::manual_load_balance_params(opts.lb_chunk_blocks,
                                                   opts.lb_split_threshold_blocks);
        return {params.chunk_blocks, params.split_threshold_blocks, params.policy};
    }
    if (opts.lb_policy == "binary_implicit_one") {
        if (layout != LayoutKind::ColBitmap64FlagLb ||
            value_codec != ValueCodecKind::ImplicitOne) {
            throw std::runtime_error(
                "lb-policy binary_implicit_one requires col_bitmap64_flag_lb layout "
                "and all_one value codec");
        }
        const auto params = structural::select_binary_implicit_one_load_balance_policy(
            device.num_blocks, device.rows);
        return {params.chunk_blocks, params.split_threshold_blocks, params.policy};
    }
    throw std::runtime_error("internal error: unknown normalized lb policy");
}

using VerifyGate = structural::SpmvVerifyGate;

template <typename T>
T exact_abs_tol() {
    return std::is_same_v<T, float> ? static_cast<T>(1e-2) : static_cast<T>(1e-9);
}

template <typename T>
T exact_rel_tol() {
    return std::is_same_v<T, float> ? static_cast<T>(1e-4) : static_cast<T>(1e-10);
}

template <typename T>
T spmv_max_rel_tol() {
    return static_cast<T>(1e-6);
}

// Default tolerance for the controlled-lossy verify gate, applied to the
// global-norm output relative error (max_r|y-y0| / max_r|y0|).
constexpr double kSpmvOutputRelErrTol = 1e-2;

template <typename T>
T spmv_output_rel_err_tol() {
    return static_cast<T>(kSpmvOutputRelErrTol);
}

template <typename T>
bool passes_verify_gate(const structural::ErrorStats<T>& error, VerifyGate gate) {
    if (gate == VerifyGate::SpmvMaxRel) {
        // Controlled-lossy family: gate on the global-norm output rel-err so that
        // near-zero output rows do not spuriously fail an otherwise-accurate codec.
        return error.global_rel <= spmv_output_rel_err_tol<T>();
    }
    return error.max_abs <= exact_abs_tol<T>() || error.max_rel <= exact_rel_tol<T>();
}

inline void cusparse_check(cusparseStatus_t status,
                           const char* expr,
                           const char* file,
                           int line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << file << ':' << line << ": cuSPARSE call failed: " << expr
            << ": status=" << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

#define STRUCTURAL_CUSPARSE_CHECK(expr) cusparse_check((expr), #expr, __FILE__, __LINE__)

template <typename T>
cudaDataType cuda_data_type();

template <>
cudaDataType cuda_data_type<float>() {
    return CUDA_R_32F;
}

template <>
cudaDataType cuda_data_type<double>() {
    return CUDA_R_64F;
}

cusparseSpMVAlg_t parse_cusparse_spmv_alg(const std::string& alg) {
    if (alg == "default") {
        return CUSPARSE_SPMV_ALG_DEFAULT;
    }
    if (alg == "csr_alg1") {
        return CUSPARSE_SPMV_CSR_ALG1;
    }
    if (alg == "csr_alg2") {
        return CUSPARSE_SPMV_CSR_ALG2;
    }
    throw std::runtime_error("unsupported cuSPARSE SpMV algorithm: " + alg);
}

using structural::make_input_vector;  // shared harness
using structural::time_call;

template <typename T>
bool verify_result(const structural::CsrMatrix<T>& csr,
                   const std::vector<T>& x,
                   const thrust::device_vector<T>& d_y,
                   VerifyGate gate,
                   double& cpu_spmv_ms,
                   structural::ErrorStats<T>& error) {
    structural::compute_cpu_error(csr, x, d_y, cpu_spmv_ms, error);
    return passes_verify_gate(error, gate);
}

template <typename T>
void run_cusparse_spmv(cusparseHandle_t handle,
                       cusparseConstSpMatDescr_t mat_a,
                       cusparseConstDnVecDescr_t vec_x,
                       cusparseDnVecDescr_t vec_y,
                       cusparseSpMVAlg_t alg,
                       void* external_buffer,
                       float* iter_ms = nullptr) {
    const T alpha = T{1};
    const T beta = T{0};
    structural::gpu_detail::CudaEventTimer timer(iter_ms);
    STRUCTURAL_CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                           mat_a, vec_x, &beta, vec_y, cuda_data_type<T>(),
                                           alg, external_buffer));
    timer.stop();
}

constexpr const char* kSpmvCliReportSchema = "structural_spmv_cli_v3";

template <typename T>
void print_loaded_matrix_report(const Options& opts,
                                const structural::MatrixMarketLoadResult<T>& loaded,
                                const char* block) {
    std::cout << "report_schema: " << kSpmvCliReportSchema << "\n";
    std::cout << "matrix: " << opts.matrix_path << "\n";
    std::cout << "matrix_market_field: " << structural::to_string(loaded.info.field) << "\n";
    std::cout << "matrix_market_symmetry: " << structural::to_string(loaded.info.symmetry)
              << "\n";
    std::cout << "precision: " << opts.precision << "\n";
    std::cout << "shape: " << loaded.matrix.rows << " x " << loaded.matrix.cols << "\n";
    std::cout << "nnz: " << loaded.matrix.nnz() << "\n";
    std::cout << "block: " << block << "\n";
}

void print_operator_contract_report(const structural::SpmvOperatorSpec& spec) {
    std::cout << "operator_name: " << spec.operator_name << "\n";
    std::cout << "operator_family: " << structural::spmv_operator_family_name(spec.family)
              << "\n";
    std::cout << "operator_promotion_status: "
              << structural::spmv_promotion_status_name(spec.promotion_status) << "\n";
    std::cout << "operator_runner: " << structural::spmv_operator_runner_name(spec.runner)
              << "\n";
}

template <typename T>
void print_verify_cpu_report(bool verify_cpu,
                             bool verified,
                             const structural::ErrorStats<T>& error) {
    if (verify_cpu) {
        std::cout << "verify_cpu: " << (verified ? "PASS" : "FAIL")
                  << ", max_abs=" << error.max_abs
                  << ", max_rel=" << error.max_rel << "\n";
        std::cout << "spmv_output_rel_err: " << error.global_rel << "\n";
    } else {
        std::cout << "verify_cpu: SKIP\n";
    }
}

template <typename T>
void print_cusparse_csr_report(const Options& opts,
                               const structural::MatrixMarketLoadResult<T>& loaded,
                               const structural::SpmvOperatorSpec& spec,
                               std::size_t cusparse_buffer_bytes,
                               double load_ms,
                               double cuda_context_ms,
                               double csr_upload_host_ms,
                               double cusparse_setup_host_ms,
                               double cusparse_preprocess_host_ms,
                               float cusparse_preprocess_event_ms,
                               double spmv_sum_ms,
                               float spmv_min_ms,
                               double cpu_spmv_ms,
                               bool verified,
                               const structural::ErrorStats<T>& error) {
    print_loaded_matrix_report(opts, loaded, "csr");
    print_operator_contract_report(spec);
    std::cout << "spmv_layout: " << opts.layout << "\n";
    std::cout << "spmv_effective_layout: cusparse_csr\n";
    std::cout << "position_codec: csr\n";
    std::cout << "position_codec_exact: true\n";
    std::cout << "position_codec_payload_bytes: "
              << (static_cast<std::size_t>(loaded.matrix.rows + 1) * sizeof(structural::Offset) +
                  static_cast<std::size_t>(loaded.matrix.nnz()) * sizeof(structural::Index))
              << "\n";
    std::cout << "value_codec: original\n";
    std::cout << "value_codec_exact: true\n";
    std::cout << "value_codec_codebook_size: 0\n";
    std::cout << "value_codec_payload_bytes: "
              << static_cast<std::size_t>(loaded.matrix.nnz()) * sizeof(T) << "\n";
    std::cout << "value_codec_bytes_per_nnz: " << sizeof(T) << "\n";
    std::cout << "cusparse_alg: " << opts.cusparse_alg << "\n";
    std::cout << "cusparse_buffer_bytes: " << cusparse_buffer_bytes << "\n";
    std::cout << "load_balance_policy: manual\n";
    std::cout << "load_balance_chunk_blocks: 0\n";
    std::cout << "load_balance_split_threshold_blocks: 0\n";
    std::cout << "gpu_work_items: 0\n";
    std::cout << "gpu_output_residency: device\n";
    std::cout << "gpu_num_blocks: 0\n";
    std::cout << "verify_gate: " << structural::spmv_verify_gate_name(spec.verify_gate) << "\n";
    std::cout << "timing_scope: cusparse_event_includes_beta_zero\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "timing_ms: load=" << load_ms
              << ", cuda_context=" << cuda_context_ms
              << ", gpu_convert_host=0.0000"
              << ", gpu_convert_event=0.0000"
              << ", csr_upload_host=" << csr_upload_host_ms
              << ", cusparse_setup_host=" << cusparse_setup_host_ms
              << ", cusparse_preprocess_host=" << cusparse_preprocess_host_ms
              << ", cusparse_preprocess_event=" << cusparse_preprocess_event_ms
              << ", col_bitmap_pack_host=0.0000"
              << ", col_bitmap_pack_event=0.0000"
              << ", load_balance_pack_host=0.0000"
              << ", load_balance_pack_event=0.0000"
              << ", value_codec_pack_host=0.0000"
              << ", gpu_spmv_avg_event=" << (spmv_sum_ms / opts.iterations)
              << ", gpu_spmv_min_event=" << spmv_min_ms
              << ", cpu_spmv=" << (opts.verify_cpu ? cpu_spmv_ms : 0.0)
              << "\n";
    print_verify_cpu_report(opts.verify_cpu, verified, error);
}

template <typename T>
int run_cusparse_csr(const Options& opts, const structural::SpmvOperatorSpec& spec) {
    if (opts.value_codec != "original") {
        throw std::runtime_error("compressed value codecs are only supported for BitBSR layouts");
    }

    double load_ms = 0.0;
    double cuda_context_ms = 0.0;
    double csr_upload_host_ms = 0.0;
    double cusparse_setup_host_ms = 0.0;
    double cusparse_preprocess_host_ms = 0.0;
    double cpu_spmv_ms = 0.0;
    float cusparse_preprocess_event_ms = 0.0f;

    const auto loaded = time_call(
        [&] { return structural::read_matrix_market_with_info<T>(opts.matrix_path); },
        load_ms);

    const int cuda_context_status = time_call(
        [&] {
            structural::cuda_check(cudaFree(nullptr), "cudaFree(nullptr)", __FILE__, __LINE__);
            return 0;
        },
        cuda_context_ms);
    (void)cuda_context_status;

    const auto x = make_input_vector<T>(loaded.matrix.cols);
    thrust::device_vector<structural::Offset> d_row_ptr;
    thrust::device_vector<structural::Index> d_col_idx;
    thrust::device_vector<T> d_values;
    thrust::device_vector<T> d_x;
    thrust::device_vector<T> d_y;
    (void)time_call(
        [&] {
            d_row_ptr = loaded.matrix.row_ptr;
            d_col_idx = loaded.matrix.col_idx;
            d_values = loaded.matrix.values;
            d_x = x;
            d_y.assign(static_cast<size_t>(loaded.matrix.rows), T{0});
            return 0;
        },
        csr_upload_host_ms);

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    const auto alg = parse_cusparse_spmv_alg(opts.cusparse_alg);
    const auto value_type = cuda_data_type<T>();
    size_t cusparse_buffer_bytes = 0;

    (void)time_call(
        [&] {
            STRUCTURAL_CUSPARSE_CHECK(cusparseCreate(&handle));
            STRUCTURAL_CUSPARSE_CHECK(cusparseCreateCsr(
                &mat_a, loaded.matrix.rows, loaded.matrix.cols, loaded.matrix.nnz(),
                thrust::raw_pointer_cast(d_row_ptr.data()),
                thrust::raw_pointer_cast(d_col_idx.data()),
                thrust::raw_pointer_cast(d_values.data()),
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                value_type));
            STRUCTURAL_CUSPARSE_CHECK(cusparseCreateDnVec(
                &vec_x, loaded.matrix.cols, thrust::raw_pointer_cast(d_x.data()), value_type));
            STRUCTURAL_CUSPARSE_CHECK(cusparseCreateDnVec(
                &vec_y, loaded.matrix.rows, thrust::raw_pointer_cast(d_y.data()), value_type));

            const T alpha = T{1};
            const T beta = T{0};
            STRUCTURAL_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, vec_x, &beta, vec_y,
                value_type, alg, &cusparse_buffer_bytes));
            return 0;
        },
        cusparse_setup_host_ms);

    thrust::device_vector<std::uint8_t> external_buffer(cusparse_buffer_bytes);
    void* external_buffer_ptr =
        cusparse_buffer_bytes == 0 ? nullptr : thrust::raw_pointer_cast(external_buffer.data());

    (void)time_call(
        [&] {
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12700
            const T alpha = T{1};
            const T beta = T{0};
            structural::gpu_detail::CudaEventTimer timer(&cusparse_preprocess_event_ms);
            STRUCTURAL_CUSPARSE_CHECK(cusparseSpMV_preprocess(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, vec_x, &beta, vec_y,
                value_type, alg, external_buffer_ptr));
            timer.stop();
#endif
            return 0;
        },
        cusparse_preprocess_host_ms);

    for (int i = 0; i < opts.warmup; ++i) {
        run_cusparse_spmv<T>(handle, mat_a, vec_x, vec_y, alg, external_buffer_ptr);
    }

    double spmv_sum_ms = 0.0;
    float spmv_min_ms = std::numeric_limits<float>::max();
    for (int i = 0; i < opts.iterations; ++i) {
        float iter_ms = 0.0f;
        run_cusparse_spmv<T>(handle, mat_a, vec_x, vec_y, alg, external_buffer_ptr, &iter_ms);
        spmv_sum_ms += iter_ms;
        spmv_min_ms = std::min(spmv_min_ms, iter_ms);
    }

    structural::ErrorStats<T> error;
    bool verified = true;
    const VerifyGate verify_gate = spec.verify_gate;
    if (opts.verify_cpu) {
        verified = verify_result(loaded.matrix, x, d_y, verify_gate, cpu_spmv_ms, error);
    }

    print_cusparse_csr_report(opts, loaded, spec, cusparse_buffer_bytes, load_ms,
                              cuda_context_ms, csr_upload_host_ms, cusparse_setup_host_ms,
                              cusparse_preprocess_host_ms, cusparse_preprocess_event_ms,
                              spmv_sum_ms, spmv_min_ms, cpu_spmv_ms, verified, error);

    if (vec_y != nullptr) {
        (void)cusparseDestroyDnVec(vec_y);
    }
    if (vec_x != nullptr) {
        (void)cusparseDestroyDnVec(vec_x);
    }
    if (mat_a != nullptr) {
        (void)cusparseDestroySpMat(mat_a);
    }
    if (handle != nullptr) {
        (void)cusparseDestroy(handle);
    }

    return verified ? 0 : 1;
}

struct BitBsrSpmvTiming {
    double load_ms = 0.0;
    double cuda_context_ms = 0.0;
    double gpu_convert_host_ms = 0.0;
    double value_codec_auto_select_host_ms = 0.0;
    double col_bitmap_pack_host_ms = 0.0;
    double load_balance_pack_host_ms = 0.0;
    double value_codec_pack_host_ms = 0.0;
    double cpu_spmv_ms = 0.0;
    float gpu_convert_event_ms = 0.0f;
    float value_codec_auto_select_event_ms = 0.0f;
    float col_bitmap_pack_event_ms = 0.0f;
    float load_balance_pack_event_ms = 0.0f;
};

struct PositionCodecReport {
    std::string name = "col_bitmap64";
    std::size_t payload_bytes = 0;
    bool exact = true;
};

struct BitBsrSpmvReport {
    const char* operator_name = "unknown";
    structural::SpmvOperatorFamily operator_family =
        structural::SpmvOperatorFamily::LoadBalancedTraversal;
    structural::SpmvPromotionStatus operator_promotion_status =
        structural::SpmvPromotionStatus::Promoted;
    structural::SpmvOperatorRunner operator_runner =
        structural::SpmvOperatorRunner::BitBsrRawTraversal;
    std::string effective_layout;
    std::string effective_value_codec = "original";
    bool value_codec_auto_selected = false;
    int value_codec_auto_unique_count = 0;
    bool value_codec_auto_unique_overflow = false;
    std::size_t value_codec_auto_raw_payload_bytes = 0;
    std::size_t value_codec_auto_candidate_payload_bytes = 0;
    PositionCodecReport position;
    bool value_codec_exact = false;
    int value_codec_codebook_size = 0;
    std::size_t value_codec_payload_bytes = 0;
    // For bfp8_outlier: the §6 "value-only" storage figure (positions assumed to
    // come free from the bitmap). Defaults to the full payload for every other
    // codec so the reported value-only bytes match the stored bytes there.
    std::size_t value_codec_value_only_payload_bytes = 0;
    bool value_codec_value_only_valid = false;
    long long value_codec_num_outliers = 0;
    VerifyGate verify_gate = VerifyGate::ExactAbsOrRel;
    structural::LoadBalancePolicy load_balance_policy = structural::LoadBalancePolicy::Manual;
};

template <typename T>
void prepare_bitbsr_col_bitmap64_position(
    const structural::DeviceBitBsrMatrix<T>& device,
    const EffectiveLoadBalanceParams& effective_load_balance,
    BitBsrSpmvTiming& timing,
    structural::BitBsrSpmvPlan8x4<T>& plan,
    BitBsrSpmvReport& report) {
    (void)time_call(
        [&] {
            structural::make_bitbsr_8x4_col_bitmap_packed(
                device, plan.packed_col_bitmap_meta, &timing.col_bitmap_pack_event_ms);
            return 0;
        },
        timing.col_bitmap_pack_host_ms);
    report.position.name = "col_bitmap64";
    report.position.payload_bytes =
        plan.packed_col_bitmap_meta.size() * sizeof(std::uint64_t);

    if (!structural::bitbsr_spmv_8x4_uses_load_balance(plan.layout)) {
        return;
    }
    (void)time_call(
        [&] {
            structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
                device, plan.load_balance, effective_load_balance.chunk_blocks,
                effective_load_balance.split_threshold_blocks, plan.packed_col_bitmap_meta,
                &timing.load_balance_pack_event_ms);
            return 0;
        },
        timing.load_balance_pack_host_ms);
}

template <typename T>
void prepare_bitbsr_value_codec(const structural::DeviceBitBsrMatrix<T>& device,
                                BitBsrSpmvTiming& timing,
                                structural::BitBsrSpmvPlan8x4<T>& plan,
                                BitBsrSpmvReport& report) {
    (void)time_call(
        [&] {
            const auto host_bitbsr = device.to_host();
            switch (plan.value_codec) {
                case ValueCodecKind::ImplicitOne:
                    structural::validate_implicit_one_values(host_bitbsr.values);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = 0;
                    report.value_codec_payload_bytes = 0;
                    break;
                case ValueCodecKind::ConstC:
                    plan.value_scale = structural::build_checked_const_c_value(host_bitbsr);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = 1;
                    report.value_codec_payload_bytes = sizeof(T);
                    break;
                case ValueCodecKind::Sign2: {
                    const auto masks =
                        structural::build_checked_sign2_tile32_masks(host_bitbsr);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = 0;
                    report.value_codec_payload_bytes = masks.size() * sizeof(std::uint32_t);
                    plan.sign_masks.assign(masks.begin(), masks.end());
                    break;
                }
                case ValueCodecKind::Sign2Scale: {
                    const auto codec =
                        structural::build_checked_sign2_scale_tile32_value_codec(host_bitbsr);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = 1;
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.value_scale = codec.scale;
                    plan.sign_masks.assign(codec.sign_masks.begin(), codec.sign_masks.end());
                    break;
                }
                case ValueCodecKind::Dict2: {
                    const auto codec =
                        structural::build_checked_dict2_tile32_value_codec(host_bitbsr);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = static_cast<int>(codec.codebook.size());
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.tile_code_words.assign(codec.tile_code_words.begin(),
                                                codec.tile_code_words.end());
                    plan.value_codebook.assign(codec.codebook.begin(), codec.codebook.end());
                    break;
                }
                case ValueCodecKind::Unique4: {
                    const auto codec =
                        structural::build_checked_unique4_tile64_value_codec(host_bitbsr);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = static_cast<int>(codec.codebook.size());
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.tile_code_words.assign(codec.tile_code_words.begin(),
                                                codec.tile_code_words.end());
                    plan.value_codebook.assign(codec.codebook.begin(), codec.codebook.end());
                    break;
                }
                case ValueCodecKind::Unique16: {
                    const auto codec =
                        structural::build_checked_unique16_tile128_value_codec(host_bitbsr);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = static_cast<int>(codec.codebook.size());
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.tile_code_words.assign(codec.tile_code_words.begin(),
                                                codec.tile_code_words.end());
                    plan.value_codebook.assign(codec.codebook.begin(), codec.codebook.end());
                    break;
                }
                case ValueCodecKind::Dict256U8: {
                    const auto codec =
                        structural::build_checked_dict256_tile256_value_codec(host_bitbsr);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = static_cast<int>(codec.codebook.size());
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.tile_code_words.assign(codec.tile_code_words.begin(),
                                                codec.tile_code_words.end());
                    plan.value_codebook.assign(codec.codebook.begin(), codec.codebook.end());
                    break;
                }
                case ValueCodecKind::Dict65536U16: {
                    const auto codec =
                        structural::build_checked_dict65536_tile512_value_codec(host_bitbsr);
                    report.value_codec_exact = true;
                    report.value_codec_codebook_size = static_cast<int>(codec.codebook.size());
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.tile_code_words.assign(codec.tile_code_words.begin(),
                                                codec.tile_code_words.end());
                    plan.value_codebook.assign(codec.codebook.begin(), codec.codebook.end());
                    break;
                }
                case ValueCodecKind::Fp16: {
                    // Per-nonzero 16-bit fp16: 2.0 B/nnz (payload_bytes == nnz*2),
                    // mirroring bfp8's per-nnz layout with no block scale.
                    const auto codec =
                        structural::build_checked_fp16_nnz_value_codec(host_bitbsr);
                    report.value_codec_exact = false;
                    report.value_codec_codebook_size = 0;
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.values16.assign(codec.values16.begin(), codec.values16.end());
                    break;
                }
                case ValueCodecKind::Bf16: {
                    // Per-nonzero 16-bit bf16: 2.0 B/nnz (payload_bytes == nnz*2),
                    // mirroring bfp8's per-nnz layout with no block scale.
                    const auto codec =
                        structural::build_checked_bf16_nnz_value_codec(host_bitbsr);
                    report.value_codec_exact = false;
                    report.value_codec_codebook_size = 0;
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.values16.assign(codec.values16.begin(), codec.values16.end());
                    break;
                }
                case ValueCodecKind::Bfp8: {
                    const auto codec =
                        structural::build_checked_bfp8_blkscale_value_codec(host_bitbsr);
                    report.value_codec_exact = false;
                    report.value_codec_codebook_size = 0;
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.int8_values.assign(codec.int8_values.begin(),
                                            codec.int8_values.end());
                    plan.block_scale_bf16.assign(codec.block_scale_bf16.begin(),
                                                 codec.block_scale_bf16.end());
                    break;
                }
                case ValueCodecKind::Bfp8Outlier: {
                    const auto codec =
                        structural::build_checked_bfp8_outlier_blkscale_value_codec(
                            host_bitbsr);
                    report.value_codec_exact = false;
                    report.value_codec_codebook_size = 0;
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    report.value_codec_value_only_payload_bytes =
                        codec.value_only_payload_bytes();
                    report.value_codec_value_only_valid = true;
                    report.value_codec_num_outliers =
                        static_cast<long long>(codec.num_outliers);
                    plan.int8_values.assign(codec.int8_values.begin(),
                                            codec.int8_values.end());
                    plan.block_scale_bf16.assign(codec.block_scale_bf16.begin(),
                                                 codec.block_scale_bf16.end());
                    plan.outlier_rows.assign(codec.outlier_rows.begin(),
                                             codec.outlier_rows.end());
                    plan.outlier_cols.assign(codec.outlier_cols.begin(),
                                             codec.outlier_cols.end());
                    plan.outlier_vals.assign(codec.outlier_vals.begin(),
                                             codec.outlier_vals.end());
                    break;
                }
                case ValueCodecKind::Bfp4: {
                    const auto codec =
                        structural::build_checked_bfp4_blkscale_value_codec(host_bitbsr);
                    report.value_codec_exact = false;
                    report.value_codec_codebook_size = 0;
                    report.value_codec_payload_bytes = codec.payload_bytes();
                    plan.packed_int4.assign(codec.packed_int4.begin(),
                                            codec.packed_int4.end());
                    plan.block_scale_bf16.assign(codec.block_scale_bf16.begin(),
                                                 codec.block_scale_bf16.end());
                    break;
                }
                case ValueCodecKind::K256U8: {
                    const auto codec =
                        structural::build_checked_k256_tile256_build_result(host_bitbsr);
                    report.value_codec_exact = codec.tile.exact;
                    report.value_codec_codebook_size = codec.used_codebook_size;
                    report.value_codec_payload_bytes = codec.tile.payload_bytes();
                    plan.tile_code_words.assign(codec.tile.tile_code_words.begin(),
                                                codec.tile.tile_code_words.end());
                    plan.value_codebook.assign(codec.tile.codebook.begin(),
                                               codec.tile.codebook.end());
                    break;
                }
                case ValueCodecKind::GlobalLogKmeans16:
                case ValueCodecKind::GlobalLogKmeans32:
                case ValueCodecKind::GlobalLogKmeans64:
                case ValueCodecKind::GlobalLogKmeans128: {
                    int requested_k = 0;
                    switch (plan.value_codec) {
                        case ValueCodecKind::GlobalLogKmeans16:
                            requested_k = 16;
                            break;
                        case ValueCodecKind::GlobalLogKmeans32:
                            requested_k = 32;
                            break;
                        case ValueCodecKind::GlobalLogKmeans64:
                            requested_k = 64;
                            break;
                        case ValueCodecKind::GlobalLogKmeans128:
                            requested_k = 128;
                            break;
                        default:
                            break;
                    }
                    const auto codec =
                        structural::build_checked_logkmeans_tile256_build_result(
                            host_bitbsr, requested_k);
                    report.value_codec_exact = false;
                    report.value_codec_codebook_size = codec.used_codebook_size;
                    report.value_codec_payload_bytes = codec.tile.payload_bytes();
                    plan.tile_code_words.assign(codec.tile.tile_code_words.begin(),
                                                codec.tile.tile_code_words.end());
                    plan.value_codebook.assign(codec.tile.codebook.begin(),
                                               codec.tile.codebook.end());
                    break;
                }
                case ValueCodecKind::Raw:
                    throw std::runtime_error(
                        "original value codec must use the original traversal runner");
            }
            return 0;
        },
        timing.value_codec_pack_host_ms);
}

template <typename T>
void prepare_bitbsr_raw_traversal_operator(
    const structural::DeviceBitBsrMatrix<T>& device,
    const EffectiveLoadBalanceParams& effective_load_balance,
    BitBsrSpmvTiming& timing,
    structural::BitBsrSpmvPlan8x4<T>& plan,
    BitBsrSpmvReport& report) {
    if (plan.value_codec != ValueCodecKind::Raw) {
        throw std::runtime_error(
            "internal error: original traversal runner received compressed codec");
    }
    report.value_codec_exact = true;
    report.value_codec_codebook_size = 0;
    report.value_codec_payload_bytes = static_cast<std::size_t>(device.nnz) * sizeof(T);
    prepare_bitbsr_col_bitmap64_position(device, effective_load_balance, timing, plan, report);
}

template <typename T>
void prepare_bitbsr_value_specialized_operator(
    const structural::DeviceBitBsrMatrix<T>& device,
    const EffectiveLoadBalanceParams& effective_load_balance,
    BitBsrSpmvTiming& timing,
    structural::BitBsrSpmvPlan8x4<T>& plan,
    BitBsrSpmvReport& report) {
    if (!structural::bitbsr_spmv_8x4_is_non_raw_value_codec(plan.value_codec)) {
        throw std::runtime_error("internal error: value-specialized runner received original codec");
    }
    prepare_bitbsr_value_codec(device, timing, plan, report);
    prepare_bitbsr_col_bitmap64_position(device, effective_load_balance, timing, plan, report);
}

template <typename T>
void prepare_bitbsr_operator(const structural::DeviceBitBsrMatrix<T>& device,
                             const EffectiveLoadBalanceParams& effective_load_balance,
                             BitBsrSpmvTiming& timing,
                             structural::BitBsrSpmvPlan8x4<T>& plan,
                             BitBsrSpmvReport& report) {
    switch (report.operator_runner) {
        case structural::SpmvOperatorRunner::BitBsrRawTraversal:
            prepare_bitbsr_raw_traversal_operator(
                device, effective_load_balance, timing, plan, report);
            return;
        case structural::SpmvOperatorRunner::BitBsrValueSpecialized:
            prepare_bitbsr_value_specialized_operator(
                device, effective_load_balance, timing, plan, report);
            return;
        case structural::SpmvOperatorRunner::CusparseCsr:
            break;
    }
    throw std::runtime_error("internal error: cuSPARSE runner reached BitBSR prepare path");
}

void print_bitbsr_operator_contract_report(const BitBsrSpmvReport& report) {
    std::cout << "operator_name: " << report.operator_name << "\n";
    std::cout << "operator_family: "
              << structural::spmv_operator_family_name(report.operator_family) << "\n";
    std::cout << "operator_promotion_status: "
              << structural::spmv_promotion_status_name(report.operator_promotion_status)
              << "\n";
    std::cout << "operator_runner: "
              << structural::spmv_operator_runner_name(report.operator_runner) << "\n";
}

template <typename T>
void print_bitbsr_spmv_report(const Options& opts,
                              const structural::MatrixMarketLoadResult<T>& loaded,
                              const structural::DeviceBitBsrMatrix<T>& device,
                              const structural::BitBsrSpmvPlan8x4<T>& plan,
                              const BitBsrSpmvReport& report,
                              const BitBsrSpmvTiming& timing,
                              double spmv_sum_ms,
                              float spmv_min_ms,
                              bool verified,
                              const structural::ErrorStats<T>& error) {
    print_loaded_matrix_report(opts, loaded, "8 x 4");
    print_bitbsr_operator_contract_report(report);
    std::cout << "spmv_layout: " << opts.layout << "\n";
    std::cout << "spmv_effective_layout: " << report.effective_layout << "\n";
    std::cout << "position_codec: " << report.position.name << "\n";
    std::cout << "position_codec_exact: " << (report.position.exact ? "true" : "false") << "\n";
    std::cout << "position_codec_payload_bytes: " << report.position.payload_bytes << "\n";
    std::cout << "value_codec: " << opts.value_codec << "\n";
    std::cout << "value_codec_effective: " << report.effective_value_codec << "\n";
    std::cout << "value_codec_auto_selected: "
              << (report.value_codec_auto_selected ? "true" : "false") << "\n";
    std::cout << "value_codec_auto_unique_count: "
              << report.value_codec_auto_unique_count << "\n";
    std::cout << "value_codec_auto_unique_overflow: "
              << (report.value_codec_auto_unique_overflow ? "true" : "false") << "\n";
    std::cout << "value_codec_auto_raw_payload_bytes: "
              << report.value_codec_auto_raw_payload_bytes << "\n";
    std::cout << "value_codec_auto_candidate_payload_bytes: "
              << report.value_codec_auto_candidate_payload_bytes << "\n";
    std::cout << "value_codec_exact: " << (report.value_codec_exact ? "true" : "false") << "\n";
    std::cout << "value_codec_codebook_size: " << report.value_codec_codebook_size << "\n";
    std::cout << "value_codec_payload_bytes: " << report.value_codec_payload_bytes << "\n";
    std::cout << "value_codec_bytes_per_nnz: "
              << (device.nnz == 0 ? 0.0
                                  : static_cast<double>(report.value_codec_payload_bytes) /
                                        static_cast<double>(device.nnz))
              << "\n";
    {
        // value-only figure: equals the full payload for non-outlier codecs;
        // for bfp8_outlier it drops the explicit row/col side-list bytes (§6
        // assumes positions come free from the bitmap).
        const std::size_t value_only_bytes =
            report.value_codec_value_only_valid
                ? report.value_codec_value_only_payload_bytes
                : report.value_codec_payload_bytes;
        std::cout << "value_codec_value_only_payload_bytes: " << value_only_bytes << "\n";
        std::cout << "value_codec_value_only_bytes_per_nnz: "
                  << (device.nnz == 0 ? 0.0
                                      : static_cast<double>(value_only_bytes) /
                                            static_cast<double>(device.nnz))
                  << "\n";
        std::cout << "value_codec_num_outliers: " << report.value_codec_num_outliers << "\n";
    }
    std::cout << "load_balance_policy: "
              << structural::load_balance_policy_name(report.load_balance_policy) << "\n";
    std::cout << "load_balance_chunk_blocks: " << plan.load_balance.chunk_blocks << "\n";
    std::cout << "load_balance_split_threshold_blocks: "
              << plan.load_balance.split_threshold_blocks << "\n";
    std::cout << "gpu_work_items: " << plan.load_balance.work_items.size() << "\n";
    std::cout << "gpu_output_residency: device\n";
    std::cout << "gpu_num_blocks: " << device.num_blocks << "\n";
    std::cout << "verify_gate: " << structural::spmv_verify_gate_name(report.verify_gate)
              << "\n";
    std::cout << "timing_scope: event_excludes_output_init\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "timing_ms: load=" << timing.load_ms
              << ", cuda_context=" << timing.cuda_context_ms
              << ", gpu_convert_host=" << timing.gpu_convert_host_ms
              << ", gpu_convert_event=" << timing.gpu_convert_event_ms
              << ", value_codec_auto_select_host="
              << timing.value_codec_auto_select_host_ms
              << ", value_codec_auto_select_event="
              << timing.value_codec_auto_select_event_ms
              << ", col_bitmap_pack_host=" << timing.col_bitmap_pack_host_ms
              << ", col_bitmap_pack_event=" << timing.col_bitmap_pack_event_ms
              << ", load_balance_pack_host=" << timing.load_balance_pack_host_ms
              << ", load_balance_pack_event=" << timing.load_balance_pack_event_ms
              << ", value_codec_pack_host=" << timing.value_codec_pack_host_ms
              << ", gpu_spmv_avg_event=" << (spmv_sum_ms / opts.iterations)
              << ", gpu_spmv_min_event=" << spmv_min_ms
              << ", cpu_spmv=" << (opts.verify_cpu ? timing.cpu_spmv_ms : 0.0)
              << "\n";
    print_verify_cpu_report(opts.verify_cpu, verified, error);
}

template <typename T>
int run(const Options& opts) {
    const LayoutKind layout = structural::spmv_layout_kind_from_name(opts.layout);
    const bool auto_lossless = is_auto_lossless_value_codec(opts);
    ValueCodecKind value_codec =
        auto_lossless ? ValueCodecKind::Raw
                      : structural::spmv_value_codec_kind_from_name(opts.value_codec);
    const auto requested_load_balance_policy =
        structural::spmv_load_balance_policy_from_name(opts.lb_policy);
    const bool load_balance_policy_manual_override =
        structural::spmv_has_manual_load_balance_override(opts);
    const auto initial_load_balance_policy =
        auto_lossless ? structural::LoadBalancePolicy::Manual : requested_load_balance_policy;
    auto operator_spec = structural::make_spmv_operator_spec(
        layout, value_codec, initial_load_balance_policy, load_balance_policy_manual_override);
    if (auto_lossless && layout == LayoutKind::CusparseCsr) {
        throw std::runtime_error("auto_lossless value codec is only supported for BitBSR layouts");
    }
    if (layout == LayoutKind::CusparseCsr) {
        return run_cusparse_csr<T>(opts, operator_spec);
    }

    BitBsrSpmvTiming timing;
    structural::BitBsrSpmvPlan8x4<T> plan;
    BitBsrSpmvReport report;
    plan.layout = operator_spec.layout;
    plan.value_codec = operator_spec.value_codec;
    report.operator_name = operator_spec.operator_name;
    report.operator_family = operator_spec.family;
    report.operator_promotion_status = operator_spec.promotion_status;
    report.operator_runner = operator_spec.runner;
    report.effective_layout = opts.layout;
    report.effective_value_codec = value_codec_name(operator_spec.value_codec);
    report.value_codec_auto_selected = auto_lossless;
    report.verify_gate = operator_spec.verify_gate;

    const auto loaded = time_call(
        [&] { return structural::read_matrix_market_with_info<T>(opts.matrix_path); },
        timing.load_ms);

    const int cuda_context_status = time_call(
        [&] {
            structural::cuda_check(cudaFree(nullptr), "cudaFree(nullptr)", __FILE__, __LINE__);
            return 0;
        },
        timing.cuda_context_ms);
    (void)cuda_context_status;

    structural::GpuBitBsrWorkspace<T> workspace;
    structural::DeviceBitBsrMatrix<T> device;
    (void)time_call(
        [&] {
            structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(
                loaded.matrix, workspace, device, &timing.gpu_convert_event_ms);
            return 0;
        },
        timing.gpu_convert_host_ms);

    if (auto_lossless) {
        const auto histogram = time_call(
            [&] {
                return structural::build_device_exact_value_histogram(
                    device, &timing.value_codec_auto_select_event_ms);
            },
            timing.value_codec_auto_select_host_ms);
        value_codec = select_auto_lossless_value_codec(histogram);
        operator_spec = structural::make_spmv_operator_spec(
            layout, value_codec, requested_load_balance_policy,
            load_balance_policy_manual_override);
        plan.layout = operator_spec.layout;
        plan.value_codec = operator_spec.value_codec;
        report.operator_name = operator_spec.operator_name;
        report.operator_family = operator_spec.family;
        report.operator_promotion_status = operator_spec.promotion_status;
        report.operator_runner = operator_spec.runner;
        report.effective_value_codec = value_codec_name(operator_spec.value_codec);
        report.verify_gate = operator_spec.verify_gate;
        report.value_codec_auto_unique_count = histogram.unique_count;
        report.value_codec_auto_unique_overflow = histogram.unique_overflow;
        report.value_codec_auto_raw_payload_bytes = histogram.raw_payload_bytes;
        report.value_codec_auto_candidate_payload_bytes =
            auto_lossless_selected_payload_bytes(histogram, value_codec);
    }

    const auto x = make_input_vector<T>(loaded.matrix.cols);
    thrust::device_vector<T> d_x(x.begin(), x.end());
    thrust::device_vector<T> d_y;
    const auto effective_load_balance =
        select_effective_load_balance_params(opts, layout, value_codec, device);
    report.load_balance_policy = effective_load_balance.policy;
    prepare_bitbsr_operator(device, effective_load_balance, timing, plan, report);

    for (int i = 0; i < opts.warmup; ++i) {
        structural::run_bitbsr_spmv_once(device, plan, d_x, d_y);
    }

    double spmv_sum_ms = 0.0;
    float spmv_min_ms = std::numeric_limits<float>::max();
    for (int i = 0; i < opts.iterations; ++i) {
        float iter_ms = 0.0f;
        structural::run_bitbsr_spmv_once(device, plan, d_x, d_y, &iter_ms);
        spmv_sum_ms += iter_ms;
        spmv_min_ms = std::min(spmv_min_ms, iter_ms);
    }

    structural::ErrorStats<T> error;
    bool verified = true;
    if (opts.verify_cpu) {
        verified = verify_result(
            loaded.matrix, x, d_y, report.verify_gate, timing.cpu_spmv_ms, error);
    }

    print_bitbsr_spmv_report(opts, loaded, device, plan, report, timing, spmv_sum_ms,
                             spmv_min_ms, verified, error);

    return verified ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = structural::parse_spmv_cli_options(argc, argv);
        if (opts.precision == "float") {
            return run<float>(opts);
        }
        return run<double>(opts);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
