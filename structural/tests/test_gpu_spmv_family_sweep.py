#!/usr/bin/env python3
import csv
import importlib.util
import tempfile
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu_spmv_family_sweep.py"
spec = importlib.util.spec_from_file_location("run_gpu_spmv_family_sweep", SCRIPT)
family_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(family_sweep)


def test_default_operator_catalog():
    names = [op.name for op in family_sweep.DEFAULT_OPERATORS]

    assert names == [
        "original_lb",
        "original_nolb",
        "all_one_lb",
        "all_one_nolb",
        "const_c_lb",
        "const_c_nolb",
        "sign2_lb",
        "sign2_nolb",
        "sign2_scale_lb",
        "sign2_scale_nolb",
        "dict2_lb",
        "dict2_nolb",
        "dict4_lb",
        "dict4_nolb",
        "dict16_lb",
        "dict16_nolb",
        "dict256_u8_lb",
        "dict256_u8_nolb",
        "dict65536_u16_lb",
        "dict65536_u16_nolb",
        "auto_lossless_lb",
        "auto_lossless_nolb",
        "fp16_lb",
        "fp16_nolb",
        "bf16_lb",
        "bf16_nolb",
        "bfp8_lb",
        "bfp8_nolb",
        "bfp8_outlier_lb",
        "bfp4_lb",
        "bfp4_nolb",
        "codebook256_u8_lb",
        "codebook256_u8_nolb",
        "global_logkmeans16_lb",
        "global_logkmeans16_nolb",
        "global_logkmeans32_lb",
        "global_logkmeans32_nolb",
        "global_logkmeans64_lb",
        "global_logkmeans64_nolb",
        "global_logkmeans128_lb",
        "global_logkmeans128_nolb",
        "hybrid_lb",
        "cusparse_default",
    ]


def test_operator_metadata_is_frozen_dataclass():
    operator = family_sweep.DEFAULT_OPERATORS[0]

    assert is_dataclass(operator)
    try:
        operator.name = "changed"
    except FrozenInstanceError:
        pass
    else:
        assert False, "operator metadata must be immutable"


def test_operator_filter_preserves_catalog_order():
    selected = family_sweep.select_operators(
        "dict65536_u16_nolb,dict16_nolb,original_lb,cusparse_default"
    )

    assert [op.name for op in selected] == [
        "original_lb",
        "dict16_nolb",
        "dict65536_u16_nolb",
        "cusparse_default",
    ]


def test_default_operator_preset_is_paper_perf_subset():
    selected = family_sweep.resolve_operators("paper_perf", None)

    assert [op.name for op in selected] == [
        "original_lb",
        "original_nolb",
        "dict65536_u16_lb",
        "auto_lossless_lb",
        "fp16_lb",
        "bf16_lb",
        "bfp8_lb",
        "bfp8_outlier_lb",
        "bfp4_lb",
        "hybrid_lb",
        "cusparse_default",
    ]


def test_full_operator_preset_keeps_complete_catalog():
    selected = family_sweep.resolve_operators("full", None)

    assert selected == list(family_sweep.DEFAULT_OPERATORS)


def test_operator_filter_overrides_preset():
    selected = family_sweep.resolve_operators(
        "paper_perf", "dict16_nolb,original_lb,cusparse_default"
    )

    assert [op.name for op in selected] == [
        "original_lb",
        "dict16_nolb",
        "cusparse_default",
    ]


def test_matrix_filter_and_limit_are_applied_after_matrix_list_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for name in ["a.mtx", "b_keep.mtx", "c_keep.mtx", "d_keep.mtx"]:
            (root / name).write_text("%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 1\n")
        matrix_list = root / "matrices.txt"
        matrix_list.write_text("a.mtx\nb_keep.mtx\nc_keep.mtx\nd_keep.mtx\n")

        matrices = family_sweep.load_matrices(
            root, matrix_list, matrix_filter="keep", matrix_limit=2
        )

    assert [m.name for m in matrices] == ["b_keep.mtx", "c_keep.mtx"]


def test_timing_preset_fills_missing_counts():
    args = SimpleNamespace(timing_preset="dev", warmup=None, iters=None)

    family_sweep.apply_timing_defaults(args)

    assert args.warmup == 2
    assert args.iters == 10


def test_timing_preset_preserves_explicit_counts():
    args = SimpleNamespace(timing_preset="paper", warmup=1, iters=3)

    family_sweep.apply_timing_defaults(args)

    assert args.warmup == 1
    assert args.iters == 3


def make_plan_args(root, data_dir, matrix_list):
    return SimpleNamespace(
        bin="structural/build/struct_gpu_spmv",
        data_dir=str(data_dir),
        matrix_list=str(matrix_list),
        out_prefix=str(root / "results" / "sweep"),
        precision="double",
        operator_preset="paper_perf",
        operators="preset",
        matrix_filter="keep",
        matrix_limit=1,
        cusparse_alg="default",
        timing_preset="dev",
        warmup=None,
        iters=None,
        verify_cpu=False,
    )


def test_build_sweep_plan_resolves_inputs_without_mutating_args():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        data_dir.mkdir()
        for name in ["drop.mtx", "keep_a.mtx", "keep_b.mtx"]:
            (data_dir / name).write_text("%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 1\n")
        matrix_list = root / "matrices.txt"
        matrix_list.write_text("drop.mtx\nkeep_a.mtx\nkeep_b.mtx\n")
        args = make_plan_args(root, data_dir, matrix_list)

        plan = family_sweep.build_sweep_plan(args)

    assert args.warmup is None
    assert args.iters is None
    assert plan.warmup == 2
    assert plan.iters == 10
    assert [matrix.name for matrix in plan.matrices] == ["keep_a.mtx"]
    assert [operator.name for operator in plan.operators] == [
        "original_lb",
        "original_nolb",
        "dict65536_u16_lb",
        "auto_lossless_lb",
        "fp16_lb",
        "bf16_lb",
        "bfp8_lb",
        "bfp8_outlier_lb",
        "bfp4_lb",
        "hybrid_lb",
        "cusparse_default",
    ]
    assert plan.csv_path == root / "results" / "sweep.csv"
    assert plan.summary_path == root / "results" / "sweep.summary.csv"
    assert plan.log_path == root / "results" / "sweep.log"


def test_format_plan_summary_makes_fast_default_visible():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        data_dir.mkdir()
        (data_dir / "keep.mtx").write_text("%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 1\n")
        matrix_list = root / "matrices.txt"
        matrix_list.write_text("keep.mtx\n")

        plan = family_sweep.build_sweep_plan(make_plan_args(root, data_dir, matrix_list))

    assert family_sweep.format_plan_summary(plan) == (
        "PLAN matrices=1 operators=11 rows=11 operator_preset=paper_perf "
        "operators=preset timing_preset=dev warmup=2 iters=10 verify_cpu=false"
    )


def test_command_uses_canonical_codec_names():
    args = SimpleNamespace(
        bin="structural/build/struct_gpu_spmv",
        precision="double",
        warmup=0,
        iters=1,
        cusparse_alg="default",
        verify_cpu=True,
    )

    cmd = family_sweep.make_command(
        args, Path("data_all/example.mtx"), family_sweep.OPERATORS_BY_NAME["codebook256_u8_nolb"]
    )

    assert "--layout" in cmd
    assert cmd[cmd.index("--layout") + 1] == "col_bitmap64"
    assert "--value-codec" in cmd
    assert cmd[cmd.index("--value-codec") + 1] == "codebook256_u8"
    assert "--verify-cpu" in cmd


def test_cusparse_command_has_no_value_codec():
    args = SimpleNamespace(
        bin="structural/build/struct_gpu_spmv",
        precision="double",
        warmup=0,
        iters=1,
        cusparse_alg="csr_alg1",
        verify_cpu=False,
    )

    cmd = family_sweep.make_command(
        args, Path("data_all/example.mtx"), family_sweep.OPERATORS_BY_NAME["cusparse_default"]
    )

    assert cmd[cmd.index("--layout") + 1] == "cusparse_csr"
    assert "--value-codec" not in cmd
    assert cmd[cmd.index("--cusparse-alg") + 1] == "csr_alg1"


def test_hybrid_command_invokes_csr_bitbsr_fused_lb_binary():
    args = SimpleNamespace(
        bin="structural/build/struct_gpu_spmv",
        hybrid_bin=None,
        precision="double",
        warmup=5,
        iters=30,
        cusparse_alg="default",
        verify_cpu=False,
    )

    cmd = family_sweep.make_command(
        args, Path("data_canonical25/cant.mtx"), family_sweep.OPERATORS_BY_NAME["hybrid_lb"]
    )

    assert cmd[0].endswith("struct_gpu_csr_bitbsr_spmv")
    assert cmd[1] == "data_canonical25/cant.mtx"
    assert cmd[cmd.index("--mode") + 1] == "fused_lb"
    assert cmd[cmd.index("--long-row-split") + 1] == "256"
    assert cmd[cmd.index("--precision") + 1] == "double"
    assert cmd[cmd.index("--warmup") + 1] == "5"
    assert cmd[cmd.index("--iters") + 1] == "30"
    assert "--layout" not in cmd
    assert "--value-codec" not in cmd


def test_hybrid_bin_override_is_respected():
    args = SimpleNamespace(
        bin="structural/build/struct_gpu_spmv",
        hybrid_bin="/custom/path/hybrid_bin",
        precision="double",
        warmup=5,
        iters=30,
        cusparse_alg="default",
        verify_cpu=False,
    )

    cmd = family_sweep.make_command(
        args, Path("data_canonical25/cant.mtx"), family_sweep.OPERATORS_BY_NAME["hybrid_lb"]
    )

    assert cmd[0] == "/custom/path/hybrid_bin"


def test_logkmeans_command_uses_canonical_codec_name():
    args = SimpleNamespace(
        bin="structural/build/struct_gpu_spmv",
        precision="double",
        warmup=0,
        iters=1,
        cusparse_alg="default",
        verify_cpu=True,
    )

    cmd = family_sweep.make_command(
        args,
        Path("data_all/example.mtx"),
        family_sweep.OPERATORS_BY_NAME["global_logkmeans64_lb"],
    )

    assert cmd[cmd.index("--layout") + 1] == "col_bitmap64_flag_lb"
    assert cmd[cmd.index("--value-codec") + 1] == "global_logkmeans64"


def test_auto_lossless_command_uses_auto_codec_name():
    args = SimpleNamespace(
        bin="structural/build/struct_gpu_spmv",
        precision="double",
        warmup=0,
        iters=1,
        cusparse_alg="default",
        verify_cpu=True,
    )

    cmd = family_sweep.make_command(
        args,
        Path("data_all/example.mtx"),
        family_sweep.OPERATORS_BY_NAME["auto_lossless_lb"],
    )

    assert cmd[cmd.index("--layout") + 1] == "col_bitmap64_flag_lb"
    assert cmd[cmd.index("--value-codec") + 1] == "auto_lossless"


def test_exact_dict_ladder_is_lossless_not_controlled_lossy():
    dict256 = family_sweep.OPERATORS_BY_NAME["dict256_u8_lb"]
    dict65536 = family_sweep.OPERATORS_BY_NAME["dict65536_u16_lb"]
    codebook256 = family_sweep.OPERATORS_BY_NAME["codebook256_u8_lb"]

    assert dict256.value_codec == "dict256_u8"
    assert dict256.family == "lossless_value_specialized"
    assert dict256.verify_gate == "exact_abs_or_rel"
    assert dict65536.value_codec == "dict65536_u16"
    assert dict65536.family == "lossless_value_specialized"
    assert dict65536.verify_gate == "exact_abs_or_rel"
    assert codebook256.value_codec == "codebook256_u8"
    assert codebook256.family == "controlled_lossy_value_specialized"
    assert codebook256.verify_gate == "spmv_max_rel"


def test_parse_auto_lossless_histogram_fields():
    parsed = family_sweep.parse_key_values(
        "\n".join(
            [
                "value_codec_auto_selected: true",
                "value_codec_auto_unique_count: 3",
                "value_codec_auto_candidate_payload_bytes: 24",
                "timing_ms: value_codec_auto_select_host=0.1234, value_codec_auto_select_event=0.0500",
            ]
        )
    )

    assert parsed["value_codec_auto_selected"] == "true"
    assert parsed["value_codec_auto_unique_count"] == "3"
    assert parsed["value_codec_auto_candidate_payload_bytes"] == "24"
    assert parsed["timing.value_codec_auto_select_host"] == "0.1234"
    assert parsed["timing.value_codec_auto_select_event"] == "0.0500"


def row(**overrides):
    base = {
        "matrix": "example.mtx",
        "operator_name": "fp16_lb",
        "requested_operator": "fp16_lb",
        "operator_family": "controlled_lossy_value_specialized",
        "status": "ok",
        "avg_ms": "1.0",
        "min_ms": "0.9",
        "value_payload_bytes": "128",
        "position_payload_bytes": "64",
        "value_exact": "0",
        "value_codebook_size": "0",
        "verify_gate": "spmv_max_rel",
        "verify_cpu": "PASS",
    }
    base.update(overrides)
    return base


def test_classifies_lossless_domain_reject_separately():
    assert (
        family_sweep.classify_failure(
            row(
                operator_name="dict16_lb",
                requested_operator="dict16_lb",
                operator_family="lossless_value_specialized",
                status="fail:1",
                avg_ms="",
                min_ms="",
                value_exact="",
                verify_gate="exact_abs_or_rel",
                verify_cpu="",
            )
        )
        == "unsupported_value_domain"
    )


def test_classifies_lossy_accuracy_failure_separately():
    assert (
        family_sweep.classify_failure(
            row(
                status="fail:1",
                avg_ms="0.321",
                min_ms="0.300",
                verify_cpu="FAIL max_rel=0.25",
            )
        )
        == "accuracy_fail"
    )


def test_summary_claim_status_distinguishes_runtime_wins_and_losses():
    rows = [
        row(
            operator_name="original_lb",
            requested_operator="original_lb",
            operator_family="load_balanced_traversal",
            avg_ms="1.0",
            verify_gate="exact_abs_or_rel",
            value_exact="1",
        ),
        row(
            operator_name="cusparse_csr",
            requested_operator="cusparse_default",
            operator_family="baseline",
            avg_ms="1.1",
            verify_gate="exact_abs_or_rel",
            value_exact="1",
        ),
        row(operator_name="fp16_lb", requested_operator="fp16_lb", avg_ms="0.8"),
        row(operator_name="bf16_lb", requested_operator="bf16_lb", avg_ms="1.2"),
        row(
            operator_name="codebook256_u8_lb",
            requested_operator="codebook256_u8_lb",
            status="fail:1",
            avg_ms="0.7",
            verify_cpu="FAIL max_rel=0.10",
        ),
    ]
    for item in rows:
        item["failure_class"] = family_sweep.classify_failure(item)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "summary.csv"
        family_sweep.write_summary(rows, out)
        with out.open(newline="") as f:
            summary = {row["requested_operator"]: row for row in csv.DictReader(f)}

    assert summary["original_lb"]["claim_status"] == "promoted_fallback"
    assert summary["cusparse_default"]["claim_status"] == "external_baseline"
    assert summary["fp16_lb"]["claim_status"] == "accuracy_pass_runtime_win"
    assert summary["bf16_lb"]["claim_status"] == "accuracy_pass_runtime_loss"
    assert summary["codebook256_u8_lb"]["claim_status"] == "accuracy_fail"


def test_summary_keeps_only_measured_operator_rows():
    rows = [
        row(
            operator_name="original_lb",
            requested_operator="original_lb",
            operator_family="load_balanced_traversal",
            avg_ms="1.0",
            min_ms="0.9",
            verify_gate="exact_abs_or_rel",
            value_exact="1",
            value_codec="original",
            value_codec_effective="original",
        ),
        row(
            operator_name="cusparse_csr",
            requested_operator="cusparse_default",
            operator_family="baseline",
            avg_ms="1.1",
            min_ms="1.0",
            verify_gate="exact_abs_or_rel",
            value_exact="1",
            value_codec="original",
            value_codec_effective="original",
        ),
        row(
            operator_name="fp16_lb",
            requested_operator="fp16_lb",
            avg_ms="0.8",
            min_ms="0.7",
            value_codec="fp16",
            value_codec_effective="fp16",
        ),
        row(
            operator_name="bf16_lb",
            requested_operator="bf16_lb",
            avg_ms="1.2",
            min_ms="1.1",
            value_codec="bf16",
            value_codec_effective="bf16",
        ),
        row(
            operator_name="codebook256_u8_lb",
            requested_operator="codebook256_u8_lb",
            status="fail:1",
            avg_ms="0.6",
            min_ms="0.5",
            verify_cpu="FAIL max_rel=0.10",
            value_codec="codebook256_u8",
            value_codec_effective="codebook256_u8",
        ),
    ]
    for item in rows:
        item["failure_class"] = family_sweep.classify_failure(item)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "summary.csv"
        family_sweep.write_summary(rows, out)
        with out.open(newline="") as f:
            reader = csv.DictReader(f)
            assert "selector_selected_operator" not in reader.fieldnames
            assert "selector_reason" not in reader.fieldnames
            summary = {row["requested_operator"]: row for row in reader}

    assert "guarded_lossy_lb" not in summary
    assert set(summary) == {
        "original_lb",
        "cusparse_default",
        "fp16_lb",
        "bf16_lb",
        "codebook256_u8_lb",
    }


if __name__ == "__main__":
    test_default_operator_catalog()
    test_operator_metadata_is_frozen_dataclass()
    test_operator_filter_preserves_catalog_order()
    test_default_operator_preset_is_paper_perf_subset()
    test_full_operator_preset_keeps_complete_catalog()
    test_operator_filter_overrides_preset()
    test_matrix_filter_and_limit_are_applied_after_matrix_list_order()
    test_timing_preset_fills_missing_counts()
    test_timing_preset_preserves_explicit_counts()
    test_build_sweep_plan_resolves_inputs_without_mutating_args()
    test_format_plan_summary_makes_fast_default_visible()
    test_command_uses_canonical_codec_names()
    test_cusparse_command_has_no_value_codec()
    test_hybrid_command_invokes_csr_bitbsr_fused_lb_binary()
    test_hybrid_bin_override_is_respected()
    test_logkmeans_command_uses_canonical_codec_name()
    test_auto_lossless_command_uses_auto_codec_name()
    test_exact_dict_ladder_is_lossless_not_controlled_lossy()
    test_parse_auto_lossless_histogram_fields()
    test_classifies_lossless_domain_reject_separately()
    test_classifies_lossy_accuracy_failure_separately()
    test_summary_claim_status_distinguishes_runtime_wins_and_losses()
    test_summary_keeps_only_measured_operator_rows()
