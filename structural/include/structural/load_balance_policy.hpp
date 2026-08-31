#pragma once

#include "structural/types.hpp"

#include <algorithm>

namespace structural {

enum class LoadBalancePolicy {
    Manual,
    BinaryImplicitOne,
};

struct LoadBalanceParams {
    int chunk_blocks = 32;
    int split_threshold_blocks = 32;
    LoadBalancePolicy policy = LoadBalancePolicy::Manual;
};

inline const char* load_balance_policy_name(LoadBalancePolicy policy) {
    switch (policy) {
        case LoadBalancePolicy::Manual:
            return "manual";
        case LoadBalancePolicy::BinaryImplicitOne:
            return "binary_implicit_one";
    }
    return "unknown";
}

inline LoadBalanceParams manual_load_balance_params(int chunk_blocks,
                                                    int split_threshold_blocks) {
    LoadBalanceParams params;
    params.chunk_blocks = chunk_blocks;
    params.split_threshold_blocks = split_threshold_blocks;
    params.policy = LoadBalancePolicy::Manual;
    return params;
}

inline LoadBalanceParams select_binary_implicit_one_load_balance_policy(Offset num_blocks,
                                                                        Index rows) {
    LoadBalanceParams params;
    params.policy = LoadBalancePolicy::BinaryImplicitOne;
    if (num_blocks <= 0 || rows <= 0) {
        return params;
    }

    constexpr Index kBlockRows = 8;
    const Index num_block_rows = (rows + kBlockRows - 1) / kBlockRows;
    const double avg_blocks_per_block_row =
        static_cast<double>(num_blocks) /
        static_cast<double>(std::max<Index>(num_block_rows, 1));

    // Calibrated against the 2026-06-04 binary all-one LB sweep.
    if (num_blocks < 30000) {
        params.chunk_blocks = 8;
        params.split_threshold_blocks = 8;
        return params;
    }
    if (avg_blocks_per_block_row >= 20000.0 && num_blocks >= 700000) {
        params.chunk_blocks = 128;
        params.split_threshold_blocks = 128;
        return params;
    }
    if (avg_blocks_per_block_row >= 6000.0) {
        if (num_blocks >= 400000) {
            params.chunk_blocks = 128;
            params.split_threshold_blocks = 128;
            return params;
        }
        if (num_blocks >= 250000) {
            params.chunk_blocks = 64;
            params.split_threshold_blocks = 64;
            return params;
        }
    }
    if (avg_blocks_per_block_row >= 500.0) {
        if (num_blocks < 120000) {
            params.chunk_blocks = 16;
            params.split_threshold_blocks = 16;
            return params;
        }
        if (num_blocks >= 3000000) {
            params.chunk_blocks = 64;
            params.split_threshold_blocks = 64;
            return params;
        }
        return params;
    }
    if (num_blocks >= 8000000 && avg_blocks_per_block_row >= 128.0) {
        params.chunk_blocks = 128;
        params.split_threshold_blocks = 128;
        return params;
    }
    if (num_blocks >= 3000000) {
        params.chunk_blocks = 64;
        params.split_threshold_blocks = 64;
        return params;
    }
    return params;
}

}  // namespace structural
