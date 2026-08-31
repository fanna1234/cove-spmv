#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def assert_no_tokens(path, tokens):
    text = (ROOT / path).read_text()
    found = [token for token in tokens if token in text]
    assert not found, f"{path} still contains deprecated tokens: {found}"


def test_deprecated_gpu_spmv_layouts_are_removed_from_production_code():
    assert_no_tokens(
        "include/structural/operators/bitbsr_spmv_8x4_dispatch.cuh",
        ["BlockMeta", "ColDeltaU16RawLb", "block_meta", "col_delta_u16"],
    )
    assert_no_tokens(
        "include/structural/operators/bitbsr_spmv_8x4_common.cuh",
        [
            "BitBsrBlockMeta8x4",
            "make_bitbsr_8x4_block_meta",
            "make_bitbsr_8x4_col_delta_u16",
        ],
    )
    assert_no_tokens(
        "include/structural/operators/bitbsr_spmv_8x4_raw.cuh",
        ["block_meta", "col_delta_u16"],
    )


if __name__ == "__main__":
    test_deprecated_gpu_spmv_layouts_are_removed_from_production_code()
