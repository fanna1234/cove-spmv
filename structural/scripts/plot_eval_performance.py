#!/usr/bin/env python3
"""Evaluation performance figures from the locked result CSVs.

E1 (value1000, RTX PRO 6000): speedup-over-cuSPARSE-double survival curves
    ("fraction of matrices with speedup >= x") for COVE joint, COVE hybrid,
    CSR5, DASP double/half, PackSELL pack16/32, Spaden, cuSPARSE float.
    Per-matrix sources:
      - external baselines: repro/baselines/results/blackwell/.../combined.csv
      - COVE hybrid: cove_core_value1000 shards (speedup_vs_cusparse, hybrid_lb)
      - COVE joint: hybrid speedup x joint_vs_hybrid (bfp8, gate-pass only;
        gate-fail matrices stay on the hybrid = the dispatch semantics)
      - CSR5: csr5_value1000 shards (csr5_spmv_ms) / cusparse_double ms
    COVE curves use paired post-warmup minima; external curves retain each
    implementation's native statistic and are labeled as such in the plot.

E2 (le10gib, RTX PRO 6000): scale scatter.
    (a) hybrid-vs-cuSPARSE speedup vs nnz (coverage, 3624 matrices)
    (b) joint-vs-hybrid (bfp8) vs nnz (the stacking concentrates at scale)

Usage: plot_eval_performance.py <repo_root> <outdir>
"""
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from paper_style import setup_style, save_fig

BASELINES = [
    ("cusparse_float", "cuSPARSE FP32", "#9aa3ab", ":"),
    ("dasp_double", "DASP FP64", "#538000", "--"),
    ("dasp_half", "DASP FP16*", "#538000", ":"),
    ("packsell_pack16", "PackSELL p16", "#4b5563", "--"),
    ("packsell_pack32", "PackSELL p32", "#4b5563", ":"),
    ("spaden_bitmap_float", "Spaden FP32", "#9467bd", "--"),
]


def base(name):
    return os.path.basename(name)


def iter_sharded_rows(pattern, matrix_list):
    """Yield ``(full_matrix_id, row)`` for fixed-stride shard CSVs.

    Legacy COVE, joint, and CSR5 result CSVs stored only the basename. The
    frozen matrix list and shard order recover the original Group/Entry path,
    including same-basename matrices that must remain distinct.
    """
    paths = [p for p in glob.glob(pattern) if ".summary." not in base(p)]
    indexed = []
    for path in paths:
        match = re.search(r"shard(\d+)\.csv$", base(path))
        if match:
            indexed.append((int(match.group(1)), path))
    if not indexed:
        return
    indexed.sort()
    nshards = len(indexed)
    names = [line.strip() for line in open(matrix_list)
             if line.strip() and not line.startswith("#")]
    for _, path in indexed:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidates = []
        for offset in range(nshards):
            expected = names[offset::nshards]
            if not expected or len(rows) % len(expected) != 0:
                continue
            rows_per_matrix = len(rows) // len(expected)
            if all(
                base(rows[index * rows_per_matrix].get("matrix", ""))
                == base(matrix_id)
                for index, matrix_id in enumerate(expected)
            ):
                candidates.append((expected, rows_per_matrix))
        if len(candidates) != 1:
            raise ValueError(f"{path}: cannot infer one matrix-list shard offset")
        expected, rows_per_matrix = candidates[0]
        for index, matrix_id in enumerate(expected):
            chunk = rows[index * rows_per_matrix:(index + 1) * rows_per_matrix]
            expected_base = base(matrix_id)
            if any(base(row.get("matrix", "")) != expected_base for row in chunk):
                raise ValueError(f"{path}: shard order mismatch at {matrix_id}")
            for row in chunk:
                yield matrix_id, row


def read_combined(path):
    ms = defaultdict(dict)  # baseline -> full Group/Entry/file.mtx -> spmv_ms
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            try:
                v = float(row["spmv_ms"])
            except (ValueError, TypeError):
                continue
            if v > 0:
                ms[row["baseline"]][row["matrix"]] = v
    return ms


def read_core_speedups(pattern, matrix_list, operator):
    """Return same-row cuSPARSE/operator speedups keyed by full matrix ID."""
    by_matrix = defaultdict(dict)
    for matrix_id, row in iter_sharded_rows(pattern, matrix_list):
        if row.get("status") != "ok":
            continue
        op = row.get("operator_name")
        requested = row.get("requested_operator")
        if not ((op == operator and requested == operator)
                or (op == "cusparse_csr" and requested == "cusparse_default")):
            continue
        try:
            by_matrix[matrix_id][op] = float(row["min_ms"])
        except (ValueError, TypeError, KeyError):
            continue
    out = {}
    for matrix_id, values in by_matrix.items():
        if values.get(operator, 0) > 0 and "cusparse_csr" in values:
            out[matrix_id] = values["cusparse_csr"] / values[operator]
    return out


def read_joint(pattern, matrix_list, codec="bfp8"):
    """Full matrix ID -> (joint_vs_hybrid, pass, nnz)."""
    out = {}
    for matrix_id, row in iter_sharded_rows(pattern, matrix_list):
        if row.get("codec") != codec or row.get("status") != "ok":
            continue
        try:
            out[matrix_id] = (float(row["joint_speedup_vs_hybrid"]),
                              row.get("pass_1e-2") == "1",
                              int(row["nnz"]))
        except (ValueError, TypeError):
            continue
    return out


def read_csr5(pattern, matrix_list):
    out = {}
    for matrix_id, row in iter_sharded_rows(pattern, matrix_list):
        if row.get("status") != "ok" or row.get("check") != "PASS":
            continue
        try:
            out[matrix_id] = float(row["csr5_spmv_ms"])
        except (ValueError, TypeError):
            continue
    return out


def read_core_full(pattern, matrix_list):
    """Return raw operator measurements keyed by full matrix-list identity."""
    out = defaultdict(dict)
    for matrix_id, row in iter_sharded_rows(pattern, matrix_list):
        if row.get("status") != "ok":
            continue
        try:
            out[matrix_id][row["operator_name"]] = float(row["min_ms"])
            out[matrix_id]["nnz"] = int(row["nnz"])
        except (ValueError, TypeError, KeyError):
            continue
    return out


def survival(vals):
    v = np.sort(np.asarray(vals))
    y = 1.0 - np.arange(len(v)) / float(len(v))
    return v, y


def fig_e1(root, outdir):
    matrix_list = os.path.join(
        root, "structural/matrix_lists/"
        "suitesparse_auto_select_value1000_2026-06-07.txt")
    comb = read_combined(os.path.join(
        root, "repro/baselines/results/blackwell/"
        "suitesparse_auto_select_value1000_2026-06-07/combined.csv"))
    cd = comb["cusparse_double"]
    hyb = read_core_speedups(os.path.join(
        root, "structural/results/studies/cove_core_value1000_2026-06-09/"
        "shard*.csv"), matrix_list, "hybrid_lb")
    joint = read_joint(os.path.join(
        root, "structural/results/studies/cove_joint_value1000_2026-06-09/*.csv"),
        matrix_list)
    csr5 = read_csr5(os.path.join(
        root, "structural/results/studies/csr5_value1000_2026-06-09/shard*.csv"),
        matrix_list)

    fig, ax = plt.subplots(figsize=(3.25, 2.1))
    n_report = {}

    for key, label, color, ls in BASELINES:
        sp = [cd[m] / v for m, v in comb[key].items() if m in cd]
        n_report[label] = len(sp)
        x, y = survival(sp)
        ax.step(x, y, where="post", color=color, ls=ls, lw=1.1, label=label)

    sp = [cd[m] / v for m, v in csr5.items() if m in cd]
    n_report["CSR5"] = len(sp)
    x, y = survival(sp)
    ax.step(x, y, where="post", color="#9a3412", ls="--", lw=1.2, label="CSR5 FP64")

    sp = list(hyb.values())
    n_report["COVE hybrid"] = len(sp)
    x, y = survival(sp)
    ax.step(x, y, where="post", color="#2f6fbb", lw=1.6, label="COVE hybrid (lossless)")

    spj = []
    for m, s in hyb.items():
        j = joint.get(m)
        spj.append(s * j[0] if (j and j[1]) else s)  # gate-fail -> stays hybrid
    n_report["COVE joint"] = len(spj)
    x, y = survival(spj)
    ax.step(x, y, where="post", color="#8c1515", lw=1.8, label="COVE joint (dispatched)")

    ax.axvline(1.0, color="0.6", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 20)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("speedup over cuSPARSE-double (RTX PRO 6000)")
    ax.set_ylabel("fraction of matrices $\\geq x$")
    ax.text(0.98, 0.97, "COVE: paired minima\nexternals: native statistics",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.3,
            color="0.35")
    ax.legend(fontsize=5.6, loc="lower left", framealpha=0.9, ncol=1)
    save_fig(fig, os.path.join(outdir, "eval_value1000_profile"))
    print("E1 counts:", n_report)


def fig_e2(root, outdir):
    # (a)(b) scale panels + (c) cross-device stacking

    matrix_list = os.path.join(
        root, "structural/matrix_lists/"
        "suitesparse_real_paper_le10gib_2026-06-07.txt")
    core = read_core_full(os.path.join(
        root, "structural/results/studies/cove_core_le10gib_2026-06-09/shard*.csv"),
        matrix_list)
    joint = read_joint(os.path.join(
        root, "structural/results/studies/cove_joint_le10gib_2026-06-09/*.csv"),
        matrix_list)

    nnz_h, sp_h = [], []
    for m, d in core.items():
        if "hybrid_lb" in d and "cusparse_csr" in d and "nnz" in d and d["hybrid_lb"] > 0:
            nnz_h.append(d["nnz"])
            sp_h.append(d["cusparse_csr"] / d["hybrid_lb"])

    nnz_j, sp_j, pass_j = [], [], []
    for m, (s, ok, nz) in joint.items():
        nnz_j.append(nz); sp_j.append(s); pass_j.append(ok)

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.05),
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9],
                                              wspace=0.24))
    ax = axes[0]
    ax.scatter(nnz_h, sp_h, s=2.5, alpha=0.35, color="#2f6fbb", linewidths=0)
    ax.axhline(1.0, color="0.4", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4])
    ax.set_yticklabels(["0.5", "1", "2", "4"])
    ax.set_xlabel("nonzeros", fontsize=6.4)
    ax.set_ylabel("speedup vs cuSPARSE", fontsize=6.4)
    ax.tick_params(labelsize=5.6)
    ax.set_title(f"(a) lossless position hybrid (n={len(sp_h)})", fontsize=7)

    ax = axes[1]
    nnz_j = np.asarray(nnz_j); sp_j = np.asarray(sp_j)
    ax.scatter(nnz_j, sp_j, s=2.5, alpha=0.35, color="#8c1515", linewidths=0)
    # per-decade median: the stacking concentrates at scale
    edges = 10.0 ** np.arange(1, 10)
    mid, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (nnz_j >= lo) & (nnz_j < hi)
        if sel.sum() >= 5:
            mid.append(math.sqrt(lo * hi)); med.append(np.median(sp_j[sel]))
    ax.plot(mid, med, color="k", lw=1.3, marker="o", ms=2.5)
    ax.text(mid[-1] * 1.5, med[-1] + 0.16, "per-decade\nmedian", fontsize=5.6,
            color="k", ha="center", linespacing=1.2)
    ax.axhline(1.0, color="0.4", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("nonzeros", fontsize=6.4)
    ax.set_ylabel("speedup vs FP64 hybrid", fontsize=6.4)
    ax.tick_params(labelsize=5.6)
    ax.set_ylim(0.42, 2.66)
    ax.set_title(f"(b) value stacking on top, BFP8 (n={len(sp_j)})", fontsize=7)

    # ---- (c) the stacking across devices --------------------------------
    axc = axes[2]
    MATS = ["F1", "Ga41As41H72", "Si41Ge41H72"]
    DEV = [("RTX PRO 6000 (1/64)", "#8c1515",
            "structural/results/studies/cove_joint_canonical25_2026-06-09.csv"),
           ("H100 (1/2)", "#2c5f9e",
            "structural/results/studies/cove_joint_stars_h100_2026-06-09.csv"),
           ("A800 (1/2)", "#7f7f7f",
            "structural/results/studies/cove_joint_stars_a800_2026-06-09.csv")]
    import csv as _csv
    xpos = np.arange(len(MATS))
    w = 0.26
    for di, (label, color, rel) in enumerate(DEV):
        d = {}
        for row in _csv.DictReader(open(os.path.join(root, rel), newline="")):
            if "codec" in row and row.get("codec") != "bfp8":
                continue
            if "status" in row and row.get("status") != "ok":
                continue
            m = row["matrix"].replace(".mtx", "")
            if m in MATS and m not in d:
                d[m] = float(row["joint_speedup_vs_hybrid"])
        vals = [d[m] for m in MATS]
        bars = axc.bar(xpos + (di - 1) * w, vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            axc.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                         ha="center", va="bottom", fontsize=5.2)
    axc.axhline(1.0, color="0.4", lw=0.6)
    axc.set_xticks(xpos)
    axc.set_xticklabels(["F1", "Ga41As41", "Si41Ge41"], fontsize=6)
    axc.set_ylabel("speedup vs FP64 hybrid", fontsize=6.4)
    axc.set_ylim(0, 3.95)
    axc.tick_params(labelsize=5.6)
    axc.legend(fontsize=4.9, loc="upper left", framealpha=0.95,
               handlelength=1.1, labelspacing=0.3, borderpad=0.35)
    axc.set_title("(c) stacking across devices", fontsize=7)
    # annotate giants
    for name, label, off in [("Queen_4147.mtx", "Queen_4147", (-3, 5)),
                             ("nlpkkt240.mtx", "nlpkkt240", (-3, -10)),
                             ("indochina-2004.mtx", "indochina", (-3, 5))]:
        match = next((value for key, value in joint.items()
                      if base(key) == name), None)
        if match:
            s, ok, nz = match
            ax.annotate(label, (nz, s), fontsize=5.5, xytext=off,
                        textcoords="offset points", ha="right")
    save_fig(fig, os.path.join(outdir, "eval_le10gib_scale"))
    print(f"E2: hybrid n={len(sp_h)}, joint n={len(sp_j)}")


def main():
    setup_style()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    root, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    fig_e1(root, outdir)
    fig_e2(root, outdir)


if __name__ == "__main__":
    main()
