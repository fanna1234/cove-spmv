#pragma once

#include "structural/load_balance_policy.hpp"
#include "structural/operators/bitbsr_spmv_8x4_dispatch.cuh"

#include <stdexcept>

namespace structural {

enum class SpmvOperatorFamily {
    LoadBalancedTraversal,
    NonLoadBalancedTraversal,
    LosslessValueSpecialized,
    ControlledLossyValueSpecialized,
    ExternalBaseline,
};

enum class SpmvVerifyGate {
    ExactAbsOrRel,
    SpmvMaxRel,
};

enum class SpmvPromotionStatus {
    Promoted,
    BuildableNeedsReport,
    ExternalBaseline,
};

enum class SpmvOperatorRunner {
    BitBsrRawTraversal,
    BitBsrValueSpecialized,
    CusparseCsr,
};

struct SpmvOperatorSpec {
    BitBsrSpmvLayoutKind8x4 layout = BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb;
    BitBsrValueCodecKind8x4 value_codec = BitBsrValueCodecKind8x4::Raw;
    LoadBalancePolicy requested_load_balance_policy = LoadBalancePolicy::Manual;
    bool load_balance_policy_manual_override = false;
    const char* operator_name = "original_lb";
    SpmvOperatorFamily family = SpmvOperatorFamily::LoadBalancedTraversal;
    SpmvVerifyGate verify_gate = SpmvVerifyGate::ExactAbsOrRel;
    SpmvPromotionStatus promotion_status = SpmvPromotionStatus::Promoted;
    SpmvOperatorRunner runner = SpmvOperatorRunner::BitBsrRawTraversal;
};

inline const char* spmv_operator_family_name(SpmvOperatorFamily family) {
    switch (family) {
        case SpmvOperatorFamily::LoadBalancedTraversal:
            return "load_balanced_traversal";
        case SpmvOperatorFamily::NonLoadBalancedTraversal:
            return "non_load_balanced_traversal";
        case SpmvOperatorFamily::LosslessValueSpecialized:
            return "lossless_value_specialized";
        case SpmvOperatorFamily::ControlledLossyValueSpecialized:
            return "controlled_lossy_value_specialized";
        case SpmvOperatorFamily::ExternalBaseline:
            return "external_baseline";
    }
    return "unknown";
}

inline const char* spmv_verify_gate_name(SpmvVerifyGate gate) {
    switch (gate) {
        case SpmvVerifyGate::ExactAbsOrRel:
            return "exact_abs_or_rel";
        case SpmvVerifyGate::SpmvMaxRel:
            return "spmv_max_rel";
    }
    return "unknown";
}

inline const char* spmv_promotion_status_name(SpmvPromotionStatus status) {
    switch (status) {
        case SpmvPromotionStatus::Promoted:
            return "promoted";
        case SpmvPromotionStatus::BuildableNeedsReport:
            return "buildable_needs_report";
        case SpmvPromotionStatus::ExternalBaseline:
            return "external_baseline";
    }
    return "unknown";
}

inline const char* spmv_operator_runner_name(SpmvOperatorRunner runner) {
    switch (runner) {
        case SpmvOperatorRunner::BitBsrRawTraversal:
            return "bitbsr_original_traversal";
        case SpmvOperatorRunner::BitBsrValueSpecialized:
            return "bitbsr_value_specialized";
        case SpmvOperatorRunner::CusparseCsr:
            return "cusparse_csr";
    }
    return "unknown";
}

inline LoadBalancePolicy effective_requested_load_balance_policy(LoadBalancePolicy requested,
                                                                 bool manual_override) {
    return manual_override ? LoadBalancePolicy::Manual : requested;
}

inline bool spmv_operator_uses_col_bitmap64_layout(BitBsrSpmvLayoutKind8x4 layout) {
    return layout == BitBsrSpmvLayoutKind8x4::ColBitmap64 ||
           layout == BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb;
}

inline const char* spmv_operator_name(BitBsrSpmvLayoutKind8x4 layout,
                                      BitBsrValueCodecKind8x4 value_codec) {
    const bool uses_lb = layout == BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb;
    switch (value_codec) {
        case BitBsrValueCodecKind8x4::Raw:
            switch (layout) {
                case BitBsrSpmvLayoutKind8x4::ColBitmap64:
                    return "original_nolb";
                case BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb:
                    return "original_lb";
                case BitBsrSpmvLayoutKind8x4::CusparseCsr:
                    return "cusparse_csr";
            }
            break;
        case BitBsrValueCodecKind8x4::ImplicitOne:
            return uses_lb ? "all_one_lb" : "all_one_nolb";
        case BitBsrValueCodecKind8x4::ConstC:
            return uses_lb ? "const_c_lb" : "const_c_nolb";
        case BitBsrValueCodecKind8x4::Sign2:
            return uses_lb ? "sign2_lb" : "sign2_nolb";
        case BitBsrValueCodecKind8x4::Sign2Scale:
            return uses_lb ? "sign2_scale_lb" : "sign2_scale_nolb";
        case BitBsrValueCodecKind8x4::Dict2:
            return uses_lb ? "dict2_lb" : "dict2_nolb";
        case BitBsrValueCodecKind8x4::Unique4:
            return uses_lb ? "dict4_lb" : "dict4_nolb";
        case BitBsrValueCodecKind8x4::Unique16:
            return uses_lb ? "dict16_lb" : "dict16_nolb";
        case BitBsrValueCodecKind8x4::Dict256U8:
            return uses_lb ? "dict256_u8_lb" : "dict256_u8_nolb";
        case BitBsrValueCodecKind8x4::Dict65536U16:
            return uses_lb ? "dict65536_u16_lb" : "dict65536_u16_nolb";
        case BitBsrValueCodecKind8x4::Fp16:
            return uses_lb ? "fp16_lb" : "fp16_nolb";
        case BitBsrValueCodecKind8x4::Bf16:
            return uses_lb ? "bf16_lb" : "bf16_nolb";
        case BitBsrValueCodecKind8x4::Bfp8:
            return uses_lb ? "bfp8_lb" : "bfp8_nolb";
        case BitBsrValueCodecKind8x4::Bfp8Outlier:
            return uses_lb ? "bfp8_outlier_lb" : "bfp8_outlier_nolb";
        case BitBsrValueCodecKind8x4::Bfp4:
            return uses_lb ? "bfp4_lb" : "bfp4_nolb";
        case BitBsrValueCodecKind8x4::K256U8:
            return uses_lb ? "codebook256_u8_lb" : "codebook256_u8_nolb";
        case BitBsrValueCodecKind8x4::GlobalLogKmeans16:
            return uses_lb ? "global_logkmeans16_lb" : "global_logkmeans16_nolb";
        case BitBsrValueCodecKind8x4::GlobalLogKmeans32:
            return uses_lb ? "global_logkmeans32_lb" : "global_logkmeans32_nolb";
        case BitBsrValueCodecKind8x4::GlobalLogKmeans64:
            return uses_lb ? "global_logkmeans64_lb" : "global_logkmeans64_nolb";
        case BitBsrValueCodecKind8x4::GlobalLogKmeans128:
            return uses_lb ? "global_logkmeans128_lb" : "global_logkmeans128_nolb";
    }
    return "unknown";
}

inline SpmvOperatorSpec make_spmv_operator_spec(
    BitBsrSpmvLayoutKind8x4 layout,
    BitBsrValueCodecKind8x4 value_codec,
    LoadBalancePolicy requested_load_balance_policy,
    bool load_balance_policy_manual_override) {
    const LoadBalancePolicy effective_policy = effective_requested_load_balance_policy(
        requested_load_balance_policy, load_balance_policy_manual_override);
    if (effective_policy == LoadBalancePolicy::BinaryImplicitOne &&
        (layout != BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb ||
         value_codec != BitBsrValueCodecKind8x4::ImplicitOne)) {
        throw std::runtime_error(
            "lb-policy binary_implicit_one requires col_bitmap64_flag_lb layout "
            "and all_one value codec");
    }

    const bool non_raw = bitbsr_spmv_8x4_is_non_raw_value_codec(value_codec);
    if (layout == BitBsrSpmvLayoutKind8x4::CusparseCsr && non_raw) {
        throw std::runtime_error("compressed value codecs are only supported for BitBSR layouts");
    }
    if (non_raw && !spmv_operator_uses_col_bitmap64_layout(layout)) {
        throw std::runtime_error("compressed value codecs currently require col_bitmap64 layout");
    }

    SpmvOperatorSpec spec;
    spec.layout = layout;
    spec.value_codec = value_codec;
    spec.requested_load_balance_policy = requested_load_balance_policy;
    spec.load_balance_policy_manual_override = load_balance_policy_manual_override;
    spec.operator_name = spmv_operator_name(layout, value_codec);
    const bool controlled_lossy =
        bitbsr_spmv_8x4_is_controlled_lossy_value_codec(value_codec);
    spec.verify_gate = controlled_lossy
                           ? SpmvVerifyGate::SpmvMaxRel
                           : SpmvVerifyGate::ExactAbsOrRel;

    if (layout == BitBsrSpmvLayoutKind8x4::CusparseCsr) {
        spec.family = SpmvOperatorFamily::ExternalBaseline;
        spec.promotion_status = SpmvPromotionStatus::ExternalBaseline;
        spec.runner = SpmvOperatorRunner::CusparseCsr;
        return spec;
    }
    if (controlled_lossy) {
        spec.family = SpmvOperatorFamily::ControlledLossyValueSpecialized;
        spec.promotion_status = SpmvPromotionStatus::BuildableNeedsReport;
        spec.runner = SpmvOperatorRunner::BitBsrValueSpecialized;
        return spec;
    }
    if (non_raw) {
        spec.family = SpmvOperatorFamily::LosslessValueSpecialized;
        spec.promotion_status =
            effective_policy == LoadBalancePolicy::BinaryImplicitOne
                ? SpmvPromotionStatus::Promoted
                : SpmvPromotionStatus::BuildableNeedsReport;
        spec.runner = SpmvOperatorRunner::BitBsrValueSpecialized;
        return spec;
    }
    if (layout == BitBsrSpmvLayoutKind8x4::ColBitmap64FlagLb) {
        spec.family = SpmvOperatorFamily::LoadBalancedTraversal;
        spec.promotion_status = SpmvPromotionStatus::Promoted;
        spec.runner = SpmvOperatorRunner::BitBsrRawTraversal;
        return spec;
    }

    spec.family = SpmvOperatorFamily::NonLoadBalancedTraversal;
    spec.promotion_status = SpmvPromotionStatus::BuildableNeedsReport;
    spec.runner = SpmvOperatorRunner::BitBsrRawTraversal;
    return spec;
}

}  // namespace structural
