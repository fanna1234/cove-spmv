#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


DASP_RE = re.compile(
    r"DASP\((?P<precision>Double|Half)\):\s+"
    r"(?P<pre_ms>[0-9.]+)\s+ms\s+"
    r"(?P<spmv_ms>[0-9.]+)\s+ms\s+"
    r"(?P<gflops>[0-9.]+)\s+GFlop/s"
)
SHAPE_RE = re.compile(
    r"input matrix A:\s+\(\s*(?P<rows>[0-9]+),\s*(?P<cols>[0-9]+)\s*\)\s+nnz = (?P<nnz>[0-9]+)"
)
SPADEN_SHAPE_RE = re.compile(r"Shape:\s+(?P<rows>[0-9]+)\s*[xX×]\s*(?P<cols>[0-9]+)")
SPADEN_NNZ_RE = re.compile(r"NNZ:\s+(?P<nnz>[0-9]+)")
SPADEN_CSR_RE = re.compile(r"spmv_warp16:\s+(?P<ms>[0-9.]+)\s+ms\s+(?P<gflops>[0-9.]+)\s+Gflops")
SPADEN_BITMAP_RE = re.compile(
    r"\[Bitmap Spaden\]\s+SpMV:\s+(?P<ms>[0-9.]+)\s+ms,\s+(?P<gflops>[0-9.]+)\s+Gflops"
)
SPADEN_CONVERT_RE = re.compile(r"Conversion time:\s+(?P<ms>[0-9.]+)\s+ms")
SPADEN_STORAGE_RE = re.compile(r"Bitmap:\s+[0-9.]+\s+MB\s+\((?P<ratio>[0-9.]+)% of CSR\)")
SPADEN_ERROR_RE = re.compile(r"(?:Relative|Absolute) error:\s+(?P<error>[0-9.eE+-]+)")
TIMING_RE = re.compile(r"([^=,\s]+)=([^,]+)")


CSV_FIELDS = [
    "matrix",
    "baseline",
    "baseline_group",
    "precision_group",
    "status",
    "gpu",
    "rows",
    "cols",
    "nnz",
    "pre_ms",
    "spmv_ms",
    "avg_ms",
    "min_ms",
    "gflops",
    "conversion_ms",
    "storage_ratio",
    "ref_baseline",
    "ref_spmv_ms",
    "ref_gflops",
    "error_metric",
    "elapsed_s",
    "command",
    "stderr",
    "stdout_tail",
]


BASELINE_GROUPS = {
    "cusparse_double": ("main", "double"),
    "cusparse_float": ("main_float", "float"),
    "dasp_double": ("main", "double"),
    "dasp_half": ("main_reduced_precision", "half"),
    "packsell_pack16": ("structural_related", "half"),
    "packsell_pack32": ("structural_related", "float"),
    "spaden_bitmap_float": ("structural_related", "float"),
}


DEFAULT_BASELINES = ",".join(BASELINE_GROUPS)


def short_tail(text, lines=8):
    return "\n".join(text.strip().splitlines()[-lines:])


def base_row(matrix, baseline, gpu, command, stderr, stdout, elapsed_s):
    group, precision = BASELINE_GROUPS[baseline]
    return {
        "matrix": matrix,
        "baseline": baseline,
        "baseline_group": group,
        "precision_group": precision,
        "status": "fail",
        "gpu": gpu,
        "rows": "",
        "cols": "",
        "nnz": "",
        "pre_ms": "",
        "spmv_ms": "",
        "avg_ms": "",
        "min_ms": "",
        "gflops": "",
        "conversion_ms": "",
        "storage_ratio": "",
        "ref_baseline": "",
        "ref_spmv_ms": "",
        "ref_gflops": "",
        "error_metric": "",
        "elapsed_s": f"{elapsed_s:.3f}",
        "command": command,
        "stderr": stderr.strip(),
        "stdout_tail": short_tail(stdout),
    }


def parse_key_values(stdout):
    parsed = {}
    for line in stdout.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        parsed[key.strip()] = value.strip()
    for key, value in TIMING_RE.findall(parsed.get("timing_ms", "")):
        parsed[f"timing.{key}"] = value.strip()
    return parsed


def parse_cusparse_stdout(matrix, baseline, gpu, stdout, stderr, returncode, elapsed_s, command):
    row = base_row(matrix, baseline, gpu, command, stderr, stdout, elapsed_s)
    parsed = parse_key_values(stdout)
    shape = parsed.get("shape", "")
    if " x " in shape:
        row["rows"], row["cols"] = shape.split(" x ", 1)
    row["nnz"] = parsed.get("nnz", "")
    row["avg_ms"] = parsed.get("timing.gpu_spmv_avg_event", "")
    row["min_ms"] = parsed.get("timing.gpu_spmv_min_event", "")
    row["spmv_ms"] = row["avg_ms"]
    row["pre_ms"] = parsed.get("timing.cusparse_preprocess_event", "")
    row["status"] = "ok" if returncode == 0 and row["spmv_ms"] else f"fail:{returncode}"
    if returncode == 0 and not row["spmv_ms"]:
        row["status"] = "fail:parse"
    return row


def parse_dasp_stdout(matrix, baseline, gpu, stdout, stderr, returncode, elapsed_s, command):
    row = base_row(matrix, baseline, gpu, command, stderr, stdout, elapsed_s)
    match = DASP_RE.search(stdout)
    shape = SHAPE_RE.search(stdout)
    if shape:
        row["rows"] = shape.group("rows")
        row["cols"] = shape.group("cols")
        row["nnz"] = shape.group("nnz")
    if match:
        row["pre_ms"] = match.group("pre_ms")
        row["spmv_ms"] = match.group("spmv_ms")
        row["avg_ms"] = match.group("spmv_ms")
        row["gflops"] = match.group("gflops")
        row["status"] = "ok" if returncode == 0 and "compute succeed!" in stdout else f"fail:{returncode}"
    elif returncode != 0:
        row["status"] = f"fail:{returncode}"
    else:
        row["status"] = "fail:parse"
    return row


def parse_packsell_stdout(matrix, baseline, gpu, stdout, stderr, returncode, elapsed_s, command, iters):
    row = base_row(matrix, baseline, gpu, command, stderr, stdout, elapsed_s)
    data_lines = [line for line in stdout.splitlines() if line.strip() and not line.startswith("matrix,")]
    parsed_line = None
    for line in reversed(data_lines):
        if line.count(",") >= 10:
            parsed_line = line
            break
    if parsed_line is None:
        row["status"] = "fail:parse"
        return row

    fields = next(csv.reader([parsed_line]))
    if len(fields) < 11:
        row["status"] = "fail:parse"
        return row
    _, rows, cols, nnz, _actnnz, _method, _sigma, _d, elapsed, gflops, error = fields[:11]
    row["rows"] = rows
    row["cols"] = cols
    row["nnz"] = nnz
    row["gflops"] = gflops
    row["error_metric"] = error
    try:
        per_iter_ms = float(elapsed) / float(iters) * 1000.0
        row["spmv_ms"] = f"{per_iter_ms:.6f}"
        row["avg_ms"] = row["spmv_ms"]
    except (TypeError, ValueError, ZeroDivisionError):
        row["spmv_ms"] = ""
    row["status"] = "ok" if returncode == 0 and row["spmv_ms"] else f"fail:{returncode}"
    return row


def parse_spaden_stdout(matrix, baseline, gpu, stdout, stderr, returncode, elapsed_s, command):
    row = base_row(matrix, baseline, gpu, command, stderr, stdout, elapsed_s)
    shape = SPADEN_SHAPE_RE.search(stdout)
    nnz = SPADEN_NNZ_RE.search(stdout)
    csr = SPADEN_CSR_RE.search(stdout)
    bitmap = SPADEN_BITMAP_RE.search(stdout)
    convert = SPADEN_CONVERT_RE.search(stdout)
    storage = SPADEN_STORAGE_RE.search(stdout)
    error = SPADEN_ERROR_RE.search(stdout)
    if shape:
        row["rows"] = shape.group("rows")
        row["cols"] = shape.group("cols")
    if nnz:
        row["nnz"] = nnz.group("nnz")
    if csr:
        row["ref_baseline"] = "spaden_csr_warp16_float"
        row["ref_spmv_ms"] = csr.group("ms")
        row["ref_gflops"] = csr.group("gflops")
    if bitmap:
        row["spmv_ms"] = bitmap.group("ms")
        row["avg_ms"] = bitmap.group("ms")
        row["gflops"] = bitmap.group("gflops")
    if convert:
        row["conversion_ms"] = convert.group("ms")
    if storage:
        row["storage_ratio"] = storage.group("ratio")
    if error:
        row["error_metric"] = error.group("error")
    if returncode == 0 and "PASSED" in stdout and row["spmv_ms"]:
        row["status"] = "ok"
    elif returncode == 0 and not row["spmv_ms"]:
        row["status"] = "fail:parse"
    else:
        row["status"] = f"fail:{returncode}"
    return row


def read_matrix_list(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def resolve_matrices(data_dir, matrix_list):
    data_root = Path(data_dir)
    return [data_root / name for name in read_matrix_list(matrix_list)]


def matrix_id(matrix_path, data_dir):
    try:
        return matrix_path.resolve().relative_to(Path(data_dir).resolve()).as_posix()
    except ValueError:
        return matrix_path.name


def balance_shards(paths, shard_count):
    shards = [[] for _ in range(shard_count)]
    totals = [0 for _ in range(shard_count)]
    for path in sorted(paths, key=lambda p: (p.stat().st_size if p.exists() else 0, p.name), reverse=True):
        idx = min(range(shard_count), key=lambda i: (totals[i], i))
        shards[idx].append(path)
        totals[idx] += path.stat().st_size if path.exists() else 0
    return shards


def shell_quote(path):
    return str(path)


def build_command(args, baseline, matrix_path):
    repo = Path(args.repo_root)
    matrix = shell_quote(matrix_path)
    if baseline in {"cusparse_double", "cusparse_float"}:
        precision = "double" if baseline == "cusparse_double" else "float"
        return [
            str(repo / "structural/build/struct_gpu_spmv"),
            matrix,
            "--precision",
            precision,
            "--layout",
            "cusparse_csr",
            "--cusparse-alg",
            "default",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
    if baseline == "dasp_double":
        return [str(Path(args.dasp_root) / "spmv_double"), matrix, "1"]
    if baseline == "dasp_half":
        return [str(Path(args.dasp_root) / "spmv_half"), matrix, "1"]
    if baseline == "packsell_pack16":
        return [
            str(Path(args.packsell_spmv_dir) / "bin/pack16.out"),
            matrix,
            str(args.warmup),
            str(args.iters),
            "0",
        ]
    if baseline == "packsell_pack32":
        return [
            str(Path(args.packsell_spmv_dir) / "bin/pack32.out"),
            matrix,
            str(args.warmup),
            str(args.iters),
            "0",
        ]
    if baseline == "spaden_bitmap_float":
        return [
            str(Path(args.spaden_root) / "build/examples/spmv_bitmap"),
            "-i",
            matrix,
            "-e",
            str(args.iters),
        ]
    raise ValueError(f"unknown baseline: {baseline}")


def parse_row(args, matrix_name, baseline, gpu, stdout, stderr, returncode, elapsed_s, command_text):
    if baseline in {"cusparse_double", "cusparse_float"}:
        return parse_cusparse_stdout(
            matrix_name, baseline, gpu, stdout, stderr, returncode, elapsed_s, command_text
        )
    if baseline in {"dasp_double", "dasp_half"}:
        return parse_dasp_stdout(
            matrix_name, baseline, gpu, stdout, stderr, returncode, elapsed_s, command_text
        )
    if baseline in {"packsell_pack16", "packsell_pack32"}:
        return parse_packsell_stdout(
            matrix_name,
            baseline,
            gpu,
            stdout,
            stderr,
            returncode,
            elapsed_s,
            command_text,
            args.iters,
        )
    if baseline == "spaden_bitmap_float":
        return parse_spaden_stdout(
            matrix_name, baseline, gpu, stdout, stderr, returncode, elapsed_s, command_text
        )
    raise ValueError(f"unknown baseline: {baseline}")


def run_command(cmd, gpu, timeout_s):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cuda_root = env.get("CUDA_HOME") or env.get("CUDA_ROOT") or "/usr/local/cuda"
    env["PATH"] = f"{cuda_root}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cuda_root}/lib64:" + env.get("LD_LIBRARY_PATH", "")
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
        elapsed_s = time.time() - started
        return proc.returncode, proc.stdout, proc.stderr, elapsed_s
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.time() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, stdout, stderr + f"\ntimeout after {timeout_s}s", elapsed_s


def run_worker(args, gpu, matrices, baselines, shard_csv, shard_log):
    rows = []
    with shard_csv.open("w", newline="") as csv_file, shard_log.open("w") as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        print(f"gpu={gpu} matrices={len(matrices)} baselines={','.join(baselines)}", file=log_file, flush=True)
        data_dir = Path(args.data_dir)
        for matrix_path in matrices:
            matrix_name = matrix_id(matrix_path, data_dir)
            if not matrix_path.exists():
                for baseline in baselines:
                    row = base_row(matrix_name, baseline, gpu, "", f"missing matrix: {matrix_path}", "", 0.0)
                    row["status"] = "missing_matrix"
                    writer.writerow(row)
                    rows.append(row)
                continue
            for baseline in baselines:
                cmd = build_command(args, baseline, matrix_path)
                command_text = " ".join(cmd)
                print(f"RUN matrix={matrix_name} baseline={baseline} cmd={command_text}", file=log_file, flush=True)
                rc, stdout, stderr, elapsed_s = run_command(cmd, gpu, args.timeout_s)
                row = parse_row(args, matrix_name, baseline, gpu, stdout, stderr, rc, elapsed_s, command_text)
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)
                print(
                    f"DONE matrix={matrix_name} baseline={baseline} status={row['status']} spmv_ms={row['spmv_ms']} elapsed_s={row['elapsed_s']}",
                    file=log_file,
                    flush=True,
                )
    return rows


def write_summary(rows, path):
    fields = [
        "baseline",
        "precision_group",
        "baseline_group",
        "ok",
        "total",
        "median_spmv_ms",
        "geomean_spmv_ms",
        "min_spmv_ms",
        "max_spmv_ms",
    ]
    by_baseline = {}
    for row in rows:
        by_baseline.setdefault(row["baseline"], []).append(row)

    def safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def median(values):
        values = sorted(values)
        n = len(values)
        if not n:
            return ""
        mid = n // 2
        if n % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0

    def geomean(values):
        import math

        positives = [v for v in values if v > 0]
        if not positives:
            return ""
        return math.exp(sum(math.log(v) for v in positives) / len(positives))

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for baseline in sorted(by_baseline):
            items = by_baseline[baseline]
            values = [safe_float(row["spmv_ms"]) for row in items if row["status"] == "ok"]
            values = [v for v in values if v is not None]
            group, precision = BASELINE_GROUPS[baseline]
            writer.writerow(
                {
                    "baseline": baseline,
                    "precision_group": precision,
                    "baseline_group": group,
                    "ok": sum(1 for row in items if row["status"] == "ok"),
                    "total": len(items),
                    "median_spmv_ms": f"{median(values):.6f}" if values else "",
                    "geomean_spmv_ms": f"{geomean(values):.6f}" if values else "",
                    "min_spmv_ms": f"{min(values):.6f}" if values else "",
                    "max_spmv_ms": f"{max(values):.6f}" if values else "",
                }
            )


def write_matrix_summary(rows, path):
    baselines = sorted({row["baseline"] for row in rows})
    fields = ["matrix"] + [f"{baseline}_ms" for baseline in baselines] + [
        f"{baseline}_status" for baseline in baselines
    ]
    by_matrix = {}
    for row in rows:
        by_matrix.setdefault(row["matrix"], {})[row["baseline"]] = row
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for matrix in sorted(by_matrix):
            item = by_matrix[matrix]
            out = {"matrix": matrix}
            for baseline in baselines:
                row = item.get(baseline, {})
                out[f"{baseline}_ms"] = row.get("spmv_ms", "")
                out[f"{baseline}_status"] = row.get("status", "missing")
            writer.writerow(out)


def parse_baselines(raw):
    baselines = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in baselines if item not in BASELINE_GROUPS]
    if unknown:
        raise ValueError("unknown baselines: " + ",".join(unknown))
    return baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-dir", default="data_canonical25")
    parser.add_argument("--matrix-list", default="structural/matrix_lists/canonical25.txt")
    parser.add_argument("--out-dir", default="repro/baselines/results/canonical25_2026-06-06")
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--dasp-root", default="repro/dasp/src/DASP")
    parser.add_argument("--packsell-spmv-dir", default="repro/packsell/src/PackSELL/spmv")
    parser.add_argument("--spaden-root", default="repro/spaden/src/Spaden-ICPP24-main")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    args.repo_root = str(repo_root)
    args.data_dir = str((repo_root / args.data_dir).resolve())
    args.matrix_list = str((repo_root / args.matrix_list).resolve())
    args.out_dir = str((repo_root / args.out_dir).resolve())
    args.dasp_root = str((repo_root / args.dasp_root).resolve())
    args.packsell_spmv_dir = str((repo_root / args.packsell_spmv_dir).resolve())
    args.spaden_root = str((repo_root / args.spaden_root).resolve())
    # DASP appends paper-era CSV records under the process cwd.
    # Keep the directory available even when matrix payloads live outside repo_root/data.
    (repo_root / "data").mkdir(exist_ok=True)

    baselines = parse_baselines(args.baselines)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    logs_dir = out_dir / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    matrices = resolve_matrices(args.data_dir, args.matrix_list)
    shards = balance_shards(matrices, len(gpus))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "repo_root": args.repo_root,
        "data_dir": args.data_dir,
        "matrix_list": args.matrix_list,
        "out_dir": args.out_dir,
        "baselines": baselines,
        "gpus": gpus,
        "warmup": args.warmup,
        "iters": args.iters,
        "timeout_s": args.timeout_s,
        "shards": [[matrix_id(path, args.data_dir) for path in shard] for shard in shards],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    all_rows = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []
        for gpu, shard in zip(gpus, shards):
            futures.append(
                executor.submit(
                    run_worker,
                    args,
                    gpu,
                    shard,
                    baselines,
                    shards_dir / f"gpu{gpu}.csv",
                    logs_dir / f"gpu{gpu}.log",
                )
            )
        for future in as_completed(futures):
            all_rows.extend(future.result())

    combined_path = out_dir / "combined.csv"
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["matrix"], r["baseline"])):
            writer.writerow(row)

    write_summary(all_rows, out_dir / "summary.csv")
    write_matrix_summary(all_rows, out_dir / "matrix_summary.csv")
    print(f"Wrote {combined_path}")
    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'matrix_summary.csv'}")
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
