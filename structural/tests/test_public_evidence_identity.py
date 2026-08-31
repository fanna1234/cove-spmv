#!/usr/bin/env python3
"""Regression tests for same-basename matrix identity in public evidence."""

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "structural/scripts/plot_eval_performance.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("plot_eval_performance", SCRIPT)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def matrix_ids(result_pattern, matrix_list):
    return {
        matrix_id
        for matrix_id, _ in PLOT.iter_sharded_rows(
            str(ROOT / result_pattern), str(ROOT / matrix_list)
        )
    }


def test_value1000_identity_recovers_all_paths():
    matrix_list = "structural/matrix_lists/suitesparse_auto_select_value1000_2026-06-07.txt"
    assert len(matrix_ids(
        "structural/results/studies/cove_core_value1000_2026-06-09/shard*.csv",
        matrix_list,
    )) == 1000
    assert len(matrix_ids(
        "structural/results/studies/cove_joint_value1000_2026-06-09/shard*.csv",
        matrix_list,
    )) == 1000
    assert len(matrix_ids(
        "structural/results/studies/csr5_value1000_2026-06-09/shard*.csv",
        matrix_list,
    )) == 1000


def test_le10gib_identity_recovers_all_paths():
    matrix_list = "structural/matrix_lists/suitesparse_real_paper_le10gib_2026-06-07.txt"
    assert len(matrix_ids(
        "structural/results/studies/cove_core_le10gib_2026-06-09/shard*.csv",
        matrix_list,
    )) == 3624
    assert len(matrix_ids(
        "structural/results/studies/cove_joint_le10gib_2026-06-09/shard*.csv",
        matrix_list,
    )) == 3624
