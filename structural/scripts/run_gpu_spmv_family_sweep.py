#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Operator:
    name: str
    layout: str
    value_codec: str
    family: str
    verify_gate: str
    promotion_status: str
    binary: str = None


@dataclass(frozen=True)
class SweepPlan:
    bin: str
    data_dir: Path
    matrix_list: str
    out_prefix: Path
    csv_path: Path
    summary_path: Path
    log_path: Path
    precision: str
    operator_preset: str
    operator_filter: str
    matrix_filter: str
    matrix_limit: int
    cusparse_alg: str
    timing_preset: str
    warmup: int
    iters: int
    verify_cpu: bool
    matrices: tuple
    operators: tuple

    @property
    def expected_rows(self):
        return len(self.matrices) * len(self.operators)


DEFAULT_OPERATORS = (
    Operator("original_lb", "col_bitmap64_flag_lb", "original", "load_balanced_traversal", "exact_abs_or_rel", "promoted"),
    Operator("original_nolb", "col_bitmap64", "original", "non_load_balanced_traversal", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("all_one_lb", "col_bitmap64_flag_lb", "all_one", "lossless_value_specialized", "exact_abs_or_rel", "promoted"),
    Operator("all_one_nolb", "col_bitmap64", "all_one", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("const_c_lb", "col_bitmap64_flag_lb", "const_c", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("const_c_nolb", "col_bitmap64", "const_c", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("sign2_lb", "col_bitmap64_flag_lb", "sign2", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("sign2_nolb", "col_bitmap64", "sign2", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("sign2_scale_lb", "col_bitmap64_flag_lb", "sign2_scale", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("sign2_scale_nolb", "col_bitmap64", "sign2_scale", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict2_lb", "col_bitmap64_flag_lb", "dict2", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict2_nolb", "col_bitmap64", "dict2", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict4_lb", "col_bitmap64_flag_lb", "dict4", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict4_nolb", "col_bitmap64", "dict4", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict16_lb", "col_bitmap64_flag_lb", "dict16", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict16_nolb", "col_bitmap64", "dict16", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict256_u8_lb", "col_bitmap64_flag_lb", "dict256_u8", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict256_u8_nolb", "col_bitmap64", "dict256_u8", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict65536_u16_lb", "col_bitmap64_flag_lb", "dict65536_u16", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("dict65536_u16_nolb", "col_bitmap64", "dict65536_u16", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("auto_lossless_lb", "col_bitmap64_flag_lb", "auto_lossless", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("auto_lossless_nolb", "col_bitmap64", "auto_lossless", "lossless_value_specialized", "exact_abs_or_rel", "buildable_needs_report"),
    Operator("fp16_lb", "col_bitmap64_flag_lb", "fp16", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("fp16_nolb", "col_bitmap64", "fp16", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("bf16_lb", "col_bitmap64_flag_lb", "bf16", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("bf16_nolb", "col_bitmap64", "bf16", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("bfp8_lb", "col_bitmap64_flag_lb", "bfp8", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("bfp8_nolb", "col_bitmap64", "bfp8", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("bfp8_outlier_lb", "col_bitmap64_flag_lb", "bfp8_outlier", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("bfp4_lb", "col_bitmap64_flag_lb", "bfp4", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("bfp4_nolb", "col_bitmap64", "bfp4", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("codebook256_u8_lb", "col_bitmap64_flag_lb", "codebook256_u8", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("codebook256_u8_nolb", "col_bitmap64", "codebook256_u8", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("global_logkmeans16_lb", "col_bitmap64_flag_lb", "global_logkmeans16", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("global_logkmeans16_nolb", "col_bitmap64", "global_logkmeans16", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("global_logkmeans32_lb", "col_bitmap64_flag_lb", "global_logkmeans32", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("global_logkmeans32_nolb", "col_bitmap64", "global_logkmeans32", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("global_logkmeans64_lb", "col_bitmap64_flag_lb", "global_logkmeans64", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("global_logkmeans64_nolb", "col_bitmap64", "global_logkmeans64", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("global_logkmeans128_lb", "col_bitmap64_flag_lb", "global_logkmeans128", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("global_logkmeans128_nolb", "col_bitmap64", "global_logkmeans128", "controlled_lossy_value_specialized", "spmv_max_rel", "buildable_needs_report"),
    Operator("hybrid_lb", "csr_bitbsr_fused_lb", "", "structural_hybrid", "exact_abs_or_rel", "promoted"),
    Operator("cusparse_default", "cusparse_csr", "", "baseline", "exact_abs_or_rel", "external_baseline"),
)


OPERATORS_BY_NAME = {op.name: op for op in DEFAULT_OPERATORS}
OPERATOR_PRESETS = {
    "full": tuple(op.name for op in DEFAULT_OPERATORS),
    "paper_perf": (
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
    ),
    "lb_perf": tuple(
        op.name
        for op in DEFAULT_OPERATORS
        if op.name == "cusparse_default" or op.name.endswith("_lb")
    ),
    "lossless_lb": (
        "original_lb",
        "all_one_lb",
        "const_c_lb",
        "sign2_lb",
        "sign2_scale_lb",
        "dict2_lb",
        "dict4_lb",
        "dict16_lb",
        "dict256_u8_lb",
        "dict65536_u16_lb",
        "auto_lossless_lb",
        "cusparse_default",
    ),
    "lossy_lb": (
        "original_lb",
        "fp16_lb",
        "bf16_lb",
        "bfp8_lb",
        "bfp8_outlier_lb",
        "bfp4_lb",
        "cusparse_default",
    ),
}
TIMING_PRESETS = {
    "smoke": (0, 1),
    "dev": (2, 10),
    "paper": (5, 30),
}
TIMING_RE = re.compile(r"([^=,\s]+)=([^,]+)")
SUMMARY_FIELDS = (
    "matrix",
    "requested_operator",
    "operator_name",
    "operator_family",
    "status",
    "failure_class",
    "claim_status",
    "avg_ms",
    "min_ms",
    "speedup_vs_cusparse",
    "speedup_vs_original_lb",
    "speedup_vs_original_nolb",
    "value_payload_bytes",
    "position_payload_bytes",
    "value_codec",
    "value_codec_effective",
    "value_codec_auto_selected",
    "value_codec_auto_unique_count",
    "value_codec_auto_candidate_payload_bytes",
    "value_codec_exact",
    "value_codebook_size",
    "verify_gate",
    "verify_cpu",
)
RESULT_FIELDS = (
    "matrix",
    "matrix_bytes",
    "operator_name",
    "requested_operator",
    "operator_family",
    "operator_promotion_status",
    "operator_runner",
    "verify_gate",
    "layout",
    "effective_layout",
    "value_codec",
    "value_codec_effective",
    "value_codec_auto_selected",
    "value_codec_auto_unique_count",
    "value_codec_auto_unique_overflow",
    "value_codec_auto_raw_payload_bytes",
    "value_codec_auto_candidate_payload_bytes",
    "status",
    "failure_class",
    "rows",
    "cols",
    "nnz",
    "work_items",
    "gpu_num_blocks",
    "position_payload_bytes",
    "value_exact",
    "value_codebook_size",
    "value_payload_bytes",
    "value_bytes_per_nnz",
    "avg_ms",
    "min_ms",
    "gpu_convert_event_ms",
    "value_codec_auto_select_event_ms",
    "col_bitmap_pack_event_ms",
    "load_balance_pack_event_ms",
    "value_codec_auto_select_host_ms",
    "value_pack_host_ms",
    "cusparse_preprocess_event_ms",
    "verify_cpu",
    "spmv_output_rel_err",
    "elapsed_s",
    "error",
)


def select_operators(operator_filter):
    if operator_filter is None or operator_filter.strip() in {"", "all"}:
        return list(DEFAULT_OPERATORS)

    requested = {item.strip() for item in operator_filter.split(",") if item.strip()}
    unknown = sorted(requested.difference(OPERATORS_BY_NAME))
    if unknown:
        raise ValueError("unknown operators: " + ",".join(unknown))
    return [op for op in DEFAULT_OPERATORS if op.name in requested]


def resolve_operators(operator_preset, operator_filter):
    if operator_filter is not None and operator_filter.strip() not in {"", "preset"}:
        return select_operators(operator_filter)

    preset = operator_preset or "paper_perf"
    if preset not in OPERATOR_PRESETS:
        raise ValueError(
            "unknown operator preset: "
            + preset
            + " (valid: "
            + ",".join(sorted(OPERATOR_PRESETS))
            + ")"
        )
    return select_operators(",".join(OPERATOR_PRESETS[preset]))


def resolve_timing(timing_preset, warmup, iters):
    if timing_preset not in TIMING_PRESETS:
        raise ValueError(
            "unknown timing preset: "
            + timing_preset
            + " (valid: "
            + ",".join(sorted(TIMING_PRESETS))
            + ")"
        )
    preset_warmup, preset_iters = TIMING_PRESETS[timing_preset]
    return (
        preset_warmup if warmup is None else warmup,
        preset_iters if iters is None else iters,
    )


def apply_timing_defaults(args):
    args.warmup, args.iters = resolve_timing(args.timing_preset, args.warmup, args.iters)


def build_sweep_plan(args):
    data_dir = Path(args.data_dir)
    matrices = tuple(load_matrices(data_dir, args.matrix_list, args.matrix_filter, args.matrix_limit))
    operators = tuple(resolve_operators(args.operator_preset, args.operators))
    warmup, iters = resolve_timing(args.timing_preset, args.warmup, args.iters)
    out_prefix = Path(args.out_prefix)

    return SweepPlan(
        bin=args.bin,
        data_dir=data_dir,
        matrix_list=args.matrix_list,
        out_prefix=out_prefix,
        csv_path=out_prefix.with_suffix(".csv"),
        summary_path=out_prefix.with_suffix(".summary.csv"),
        log_path=out_prefix.with_suffix(".log"),
        precision=args.precision,
        operator_preset=args.operator_preset,
        operator_filter=args.operators,
        matrix_filter=args.matrix_filter,
        matrix_limit=args.matrix_limit,
        cusparse_alg=args.cusparse_alg,
        timing_preset=args.timing_preset,
        warmup=warmup,
        iters=iters,
        verify_cpu=args.verify_cpu,
        matrices=matrices,
        operators=operators,
    )


def format_plan_summary(plan):
    verify_cpu = "true" if plan.verify_cpu else "false"
    return (
        f"PLAN matrices={len(plan.matrices)} operators={len(plan.operators)} "
        f"rows={plan.expected_rows} operator_preset={plan.operator_preset} "
        f"operators={plan.operator_filter} timing_preset={plan.timing_preset} "
        f"warmup={plan.warmup} iters={plan.iters} verify_cpu={verify_cpu}"
    )


def parse_key_values(stdout):
    result = {}
    for line in stdout.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        result[key.strip()] = value.strip()
    for key, value in TIMING_RE.findall(result.get("timing_ms", "")):
        result[f"timing.{key}"] = value.strip()
    return result


def load_matrices(data_dir, matrix_list, matrix_filter=None, matrix_limit=None):
    if matrix_list is None:
        matrices = sorted(data_dir.glob("*.mtx"), key=lambda p: (p.stat().st_size, p.name))
    else:
        matrices = []
        with Path(matrix_list).open() as f:
            for raw_line in f:
                name = raw_line.strip()
                if not name or name.startswith("#"):
                    continue
                path = data_dir / name
                if not path.exists():
                    raise FileNotFoundError(f"matrix listed but not found: {path}")
                matrices.append(path)

    if matrix_filter:
        pattern = re.compile(matrix_filter)
        matrices = [path for path in matrices if pattern.search(path.name)]
    if matrix_limit is not None:
        matrices = matrices[:matrix_limit]
    return matrices


def hybrid_binary_path(args, operator=None):
    if operator is not None and getattr(operator, "binary", None):
        return operator.binary
    override = getattr(args, "hybrid_bin", None)
    if override:
        return override
    return str(Path(args.bin).parent / "struct_gpu_csr_bitbsr_spmv")


def make_hybrid_command(args, matrix, operator):
    cmd = [
        hybrid_binary_path(args, operator),
        str(matrix),
        "--mode",
        "fused_lb",
        "--long-row-split",
        "256",
        "--precision",
        args.precision,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--operator-label",
        operator.name,
    ]
    if args.verify_cpu:
        cmd.append("--verify-cpu")
    return cmd


def make_command(args, matrix, operator):
    if operator.layout == "csr_bitbsr_fused_lb":
        return make_hybrid_command(args, matrix, operator)
    cmd = [
        args.bin,
        str(matrix),
        "--precision",
        args.precision,
        "--layout",
        operator.layout,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    if operator.layout == "cusparse_csr":
        cmd += ["--cusparse-alg", args.cusparse_alg]
    else:
        cmd += ["--value-codec", operator.value_codec]
    if args.verify_cpu:
        cmd.append("--verify-cpu")
    return cmd


def run_one(args, matrix, operator):
    cmd = make_command(args, matrix, operator)
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - started
    parsed = parse_key_values(proc.stdout)
    row = {
        "matrix": matrix.name,
        "matrix_bytes": str(matrix.stat().st_size),
        "operator_name": parsed.get("operator_name", operator.name),
        "requested_operator": operator.name,
        "operator_family": parsed.get("operator_family", operator.family),
        "operator_promotion_status": parsed.get("operator_promotion_status", operator.promotion_status),
        "operator_runner": parsed.get("operator_runner", ""),
        "verify_gate": parsed.get("verify_gate", operator.verify_gate),
        "layout": operator.layout,
        "effective_layout": parsed.get("spmv_effective_layout", ""),
        "value_codec": parsed.get("value_codec", operator.value_codec or "original"),
        "value_codec_effective": parsed.get("value_codec_effective", parsed.get("value_codec", operator.value_codec or "original")),
        "value_codec_auto_selected": parsed.get("value_codec_auto_selected", "false"),
        "value_codec_auto_unique_count": parsed.get("value_codec_auto_unique_count", ""),
        "value_codec_auto_unique_overflow": parsed.get("value_codec_auto_unique_overflow", ""),
        "value_codec_auto_raw_payload_bytes": parsed.get("value_codec_auto_raw_payload_bytes", ""),
        "value_codec_auto_candidate_payload_bytes": parsed.get("value_codec_auto_candidate_payload_bytes", ""),
        "status": "ok" if proc.returncode == 0 else f"fail:{proc.returncode}",
        "rows": parsed.get("shape", "").split(" x ")[0] if " x " in parsed.get("shape", "") else "",
        "cols": parsed.get("shape", "").split(" x ")[1] if " x " in parsed.get("shape", "") else "",
        "nnz": parsed.get("nnz", ""),
        "work_items": parsed.get("gpu_work_items", ""),
        "gpu_num_blocks": parsed.get("gpu_num_blocks", ""),
        "position_payload_bytes": parsed.get("position_codec_payload_bytes", ""),
        "value_exact": parsed.get("value_codec_exact", ""),
        "value_codebook_size": parsed.get("value_codec_codebook_size", ""),
        "value_payload_bytes": parsed.get("value_codec_payload_bytes", ""),
        "value_bytes_per_nnz": parsed.get("value_codec_bytes_per_nnz", ""),
        "avg_ms": parsed.get("timing.gpu_spmv_avg_event", ""),
        "min_ms": parsed.get("timing.gpu_spmv_min_event", ""),
        "gpu_convert_event_ms": parsed.get("timing.gpu_convert_event", ""),
        "value_codec_auto_select_event_ms": parsed.get("timing.value_codec_auto_select_event", ""),
        "col_bitmap_pack_event_ms": parsed.get("timing.col_bitmap_pack_event", ""),
        "load_balance_pack_event_ms": parsed.get("timing.load_balance_pack_event", ""),
        "value_codec_auto_select_host_ms": parsed.get("timing.value_codec_auto_select_host", ""),
        "value_pack_host_ms": parsed.get("timing.value_codec_pack_host", ""),
        "cusparse_preprocess_event_ms": parsed.get("timing.cusparse_preprocess_event", ""),
        "verify_cpu": parsed.get("verify_cpu", ""),
        "spmv_output_rel_err": parsed.get("spmv_output_rel_err", ""),
        "elapsed_s": f"{elapsed:.3f}",
        "error": proc.stderr.strip(),
    }
    row["failure_class"] = classify_failure(row)
    return row


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def speedup_value(row, baseline):
    row_ms = safe_float(row.get("avg_ms"))
    baseline_ms = safe_float(baseline.get("avg_ms") if baseline else None)
    if row_ms is None or baseline_ms is None or row_ms <= 0.0 or baseline_ms <= 0.0:
        return None
    return baseline_ms / row_ms


def speedup(row, baseline):
    value = speedup_value(row, baseline)
    if value is None:
        return ""
    return f"{value:.6f}"


def classify_failure(row):
    if row.get("status") == "ok":
        return "none"

    family = row.get("operator_family", "")
    verify_gate = row.get("verify_gate", "")
    verify_cpu = row.get("verify_cpu", "")
    avg_ms = safe_float(row.get("avg_ms"))
    has_verify_failure = "fail" in verify_cpu.lower() or "max_rel" in verify_cpu.lower()

    if has_verify_failure or (verify_gate == "spmv_max_rel" and avg_ms is not None):
        return "accuracy_fail"
    if family == "lossless_value_specialized" and avg_ms is None:
        return "unsupported_value_domain"
    if avg_ms is None:
        return "run_fail_before_timing"
    return "run_fail_after_timing"


def classify_claim_status(row, original_lb):
    failure_class = row.get("failure_class") or classify_failure(row)
    if failure_class != "none":
        return failure_class

    requested_operator = row.get("requested_operator", "")
    family = row.get("operator_family", "")
    promotion_status = row.get("operator_promotion_status", "")

    if family in {"baseline", "external_baseline"} or promotion_status == "external_baseline" or requested_operator == "cusparse_default":
        return "external_baseline"
    if requested_operator == "original_lb" or promotion_status == "promoted":
        return "promoted_fallback"
    if family == "non_load_balanced_traversal":
        return "diagnostic_fallback"

    vs_original_lb = speedup_value(row, original_lb)
    if vs_original_lb is None:
        return "accuracy_pass_no_fallback"
    if vs_original_lb > 1.0:
        return "accuracy_pass_runtime_win"
    return "accuracy_pass_runtime_loss"


def write_summary(rows, out_path):
    by_matrix = {}
    for row in rows:
        by_matrix.setdefault(row["matrix"], {})[row["requested_operator"]] = row

    def write_summary_row(writer, matrix, row, cusparse, original_lb, original_nolb):
        writer.writerow(
            {
                "matrix": matrix,
                "requested_operator": row["requested_operator"],
                "operator_name": row["operator_name"],
                "operator_family": row["operator_family"],
                "status": row["status"],
                "failure_class": row.get("failure_class") or classify_failure(row),
                "claim_status": classify_claim_status(row, original_lb),
                "avg_ms": row["avg_ms"],
                "min_ms": row["min_ms"],
                "speedup_vs_cusparse": speedup(row, cusparse),
                "speedup_vs_original_lb": speedup(row, original_lb),
                "speedup_vs_original_nolb": speedup(row, original_nolb),
                "value_payload_bytes": row["value_payload_bytes"],
                "position_payload_bytes": row["position_payload_bytes"],
                "value_codec": row.get("value_codec", ""),
                "value_codec_effective": row.get("value_codec_effective", ""),
                "value_codec_auto_selected": row.get("value_codec_auto_selected", ""),
                "value_codec_auto_unique_count": row.get("value_codec_auto_unique_count", ""),
                "value_codec_auto_candidate_payload_bytes": row.get(
                    "value_codec_auto_candidate_payload_bytes", ""
                ),
                "value_codec_exact": row["value_exact"],
                "value_codebook_size": row.get("value_codebook_size", ""),
                "verify_gate": row["verify_gate"],
                "verify_cpu": row["verify_cpu"],
            }
        )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for matrix in sorted(by_matrix):
            item = by_matrix[matrix]
            cusparse = item.get("cusparse_default")
            original_lb = item.get("original_lb")
            original_nolb = item.get("original_nolb")
            for operator in DEFAULT_OPERATORS:
                row = item.get(operator.name)
                if row is None:
                    continue
                write_summary_row(writer, matrix, row, cusparse, original_lb, original_nolb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default="structural/build/struct_gpu_spmv")
    parser.add_argument("--hybrid-bin", default=None)
    parser.add_argument("--data-dir", default="data_all")
    parser.add_argument("--matrix-list", default="structural/matrix_lists/canonical25.txt")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--precision", default="double")
    parser.add_argument("--operator-preset", default="paper_perf",
                        choices=sorted(OPERATOR_PRESETS))
    parser.add_argument("--operators", default="preset")
    parser.add_argument("--matrix-filter", default=None)
    parser.add_argument("--matrix-limit", type=int, default=None)
    parser.add_argument("--cusparse-alg", default="default")
    parser.add_argument("--timing-preset", default="dev",
                        choices=sorted(TIMING_PRESETS))
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--verify-cpu", action="store_true")
    args = parser.parse_args()
    plan = build_sweep_plan(args)
    plan.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with plan.csv_path.open("w", newline="") as csv_file, plan.log_path.open("w") as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        print(format_plan_summary(plan), file=log_file, flush=True)
        done = 0
        for i, matrix in enumerate(plan.matrices, 1):
            print(f"MATRIX {i}/{len(plan.matrices)} {matrix.name} bytes={matrix.stat().st_size}", file=log_file, flush=True)
            for operator in plan.operators:
                done += 1
                row = run_one(plan, matrix, operator)
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"  [{done}/{plan.expected_rows}] {operator.name} {row['status']} avg={row['avg_ms']} "
                    f"min={row['min_ms']} gate={row['verify_gate']} elapsed={row['elapsed_s']}s",
                    file=log_file,
                    flush=True,
                )
        write_summary(rows, plan.summary_path)
        print(f"DONE matrices={len(plan.matrices)} operators={len(plan.operators)} rows={len(rows)}", file=log_file, flush=True)
        print(f"CSV {plan.csv_path}", file=log_file, flush=True)
        print(f"SUMMARY {plan.summary_path}", file=log_file, flush=True)


if __name__ == "__main__":
    main()
