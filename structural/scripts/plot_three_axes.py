#!/usr/bin/env python3
"""The three claims in one row: (a) speed, (b) memory, (c) accuracy.

(a) survival curves of per-matrix speedup over cuSPARSE-double (value1000,
    RTX PRO 6000, all 8 baselines + COVE hybrid/joint);
(b) position metadata bytes/nnz, including COVE work descriptors;
(c) the value menu on the storage-accuracy plane (medians, common success set).

Usage: plot_three_axes.py <repo_root> <out (no ext)>
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from paper_style import setup_style, save_fig
import plot_eval_performance as pe

PBLUE, GREEN, DGREEN, CHAR = "#2c5f9e", "#76B900", "#538000", "#1c2128"


def panel_speed(ax, root):
    matrix_list = os.path.join(
        root, "structural/matrix_lists/"
        "suitesparse_auto_select_value1000_2026-06-07.txt")
    comb = pe.read_combined(os.path.join(
        root, "repro/baselines/results/blackwell/"
        "suitesparse_auto_select_value1000_2026-06-07/combined.csv"))
    cd = comb["cusparse_double"]
    hyb = pe.read_core_speedups(os.path.join(
        root, "structural/results/studies/cove_core_value1000_2026-06-09/"
        "shard*.csv"), matrix_list, "hybrid_lb")
    joint = pe.read_joint(os.path.join(
        root, "structural/results/studies/cove_joint_value1000_2026-06-09/*.csv"),
        matrix_list)
    csr5 = pe.read_csr5(os.path.join(
        root, "structural/results/studies/csr5_value1000_2026-06-09/shard*.csv"),
        matrix_list)
    for key, label, color, ls in pe.BASELINES:
        x, y = pe.survival([cd[m] / v for m, v in comb[key].items() if m in cd])
        ax.step(x, y, where="post", color=color, ls=ls, lw=0.9, label=label)
    x, y = pe.survival([cd[m] / v for m, v in csr5.items() if m in cd])
    ax.step(x, y, where="post", color="#9a3412", ls="--", lw=1.0,
            label="CSR5 FP64")
    x, y = pe.survival(list(hyb.values()))
    ax.step(x, y, where="post", color=PBLUE, lw=1.5, label="COVE hybrid")
    spj = [s * j[0] if (j := joint.get(m)) and j[1] else s
           for m, s in hyb.items()]
    x, y = pe.survival(spj)
    ax.step(x, y, where="post", color="#8c1515", lw=1.7, label="COVE joint")
    ax.axvline(1.0, color="0.6", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 20)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("speedup over cuSPARSE-double", fontsize=6.2)
    ax.set_ylabel("fraction of matrices $\\geq x$", fontsize=6.2)
    ax.text(0.98, 0.97, "COVE: minima\nexternals: native stats",
            transform=ax.transAxes, ha="right", va="top", fontsize=4.7,
            color="0.35")
    ax.tick_params(labelsize=5.4)
    ax.legend(fontsize=4.8, loc="lower left", framealpha=0.9,
              handlelength=1.6, labelspacing=0.32, borderpad=0.4)
    ax.set_title("(a) speed (value1000, RTX PRO 6000)", fontsize=7)


def panel_memory(ax, root):
    packed_bytes = work_bytes = csr_bytes = total_nnz = 0.0
    matrix_count = 0
    for p in glob.glob(os.path.join(
            root, "structural/results/studies/cove_core_value1000_2026-06-09/shard*.csv")):
        if ".summary." in os.path.basename(p):
            continue
        for row in csv.DictReader(open(p, newline="")):
            if (row.get("requested_operator") != "original_lb"
                    or row.get("operator_name") != "original_lb"
                    or row.get("status") != "ok"):
                continue
            try:
                nnz = int(row["nnz"]); rows = int(row["rows"])
                packed = float(row["position_payload_bytes"])
                work_items = int(row["work_items"])
            except (ValueError, KeyError):
                continue
            if nnz <= 0 or packed <= 0:
                continue
            packed_bytes += packed
            work_bytes += 16.0 * work_items
            csr_bytes += 4.0 * nnz + 4.0 * (rows + 1)
            total_nnz += nnz
            matrix_count += 1
    packed_bpn = packed_bytes / total_nnz
    work_bpn = work_bytes / total_nnz
    cove_bpn = packed_bpn + work_bpn
    csr_bpn = csr_bytes / total_nnz
    ax.barh([0], [packed_bpn], color=PBLUE, height=0.55,
            edgecolor=CHAR, linewidth=0.35)
    ax.barh([0], [work_bpn], left=[packed_bpn], color="#f3b37a",
            height=0.55, hatch="..", edgecolor=CHAR, linewidth=0.35)
    ax.barh([1], [csr_bpn], color="#d9dde2", height=0.55,
            hatch="//", edgecolor=CHAR, linewidth=0.45)
    ax.text(packed_bpn / 2, 0, f"words {packed_bpn:.2f}", ha="center",
            va="center", fontsize=5.4, color="white", fontweight="bold")
    ax.text(cove_bpn + 0.08, 0, f"+ work = {cove_bpn:.2f}", ha="left",
            va="center", fontsize=5.4, color=CHAR)
    ax.text(csr_bpn / 2, 1, f"row + column = {csr_bpn:.2f}", ha="center",
            va="center", fontsize=5.4, color=CHAR)
    ax.set_yticks([0, 1], ["COVE", "CSR"])
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("position bytes / nnz", fontsize=6.2)
    ax.tick_params(axis="y", length=0, labelsize=5.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"(b) position metadata (n={matrix_count})", fontsize=7)


CODECS = {
    "original_lb": ("FP64", "#59636e", "s", (-6, 7)),
    "bf16_lb": ("BF16", PBLUE, "o", (6, 2)),
    "fp16_lb": ("FP16", PBLUE, "^", (6, -9)),
    "bfp8_lb": ("BFP8", DGREEN, "o", (-4, 7)),
    "bfp8_outlier_lb": ("BFP8+outlier", "#8c1515", "D", (4, -10)),
    "bfp4_lb": ("BFP4", DGREEN, "v", (-16, 8)),
}


def panel_accuracy(ax, root):
    data = {k: {} for k in CODECS}
    all_keys = set()
    for p in glob.glob(os.path.join(
            root, "structural/results/studies/cove_core_value1000_2026-06-09/shard*.csv")):
        if ".summary." in os.path.basename(p):
            continue
        for row in csv.DictReader(open(p, newline="")):
            key = (row.get("matrix", ""), row.get("matrix_bytes", ""))
            if key[0] and key[1]:
                all_keys.add(key)
            op = row.get("operator_name")
            if (op not in CODECS or row.get("requested_operator") != op
                    or row.get("status") != "ok"):
                continue
            try:
                b = float(row["value_bytes_per_nnz"])
                e = float(row["spmv_output_rel_err"])
            except (ValueError, KeyError):
                continue
            if b <= 0 or e < 0:
                continue
            data[op][key] = (b, max(e, 1e-17))
    family = ["original_lb", "bf16_lb", "bfp8_lb", "bfp8_outlier_lb"]
    common = set.intersection(*(set(data[f]) for f in family))
    total = len(all_keys)
    for op, (label, color, marker, off) in CODECS.items():
        rows = data[op]
        keys = common if op in family else set(rows)
        vals = [rows[m] for m in keys if m in rows]
        bx = float(np.median([v[0] for v in vals]))
        ey = float(np.median([v[1] for v in vals]))
        cov = len(rows) / float(total)
        if op not in family and cov < 0.95:
            label += f" ({cov:.0%})"
        if ey < 1e-12:
            ey = 1.15e-5
            label += " (err 0)"
        ax.scatter([bx], [ey], s=22, color=color, marker=marker, zorder=3)
        ax.annotate(label, (bx, ey), xytext=off, textcoords="offset points",
                    fontsize=5.4, color=color)
    ax.axhline(1e-2, color="0.45", lw=0.7, ls="--")
    ax.text(8.6, 1.45e-2, "$\\varepsilon$ gate", fontsize=5.4, color="0.35",
            ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.6, 11)
    ax.set_ylim(8e-6, 4e-2)
    ax.set_xlabel("median value bytes / nnz", fontsize=6.2)
    ax.set_ylabel("median output rel. error", fontsize=6.2)
    ax.tick_params(labelsize=5.4)
    ax.set_title("(c) accuracy vs storage (value1000)", fontsize=7)


def main(root, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.15),
                             gridspec_kw=dict(width_ratios=[1.12, 1.0, 0.98],
                                              wspace=0.34))
    panel_speed(axes[0], root)
    panel_memory(axes[1], root)
    panel_accuracy(axes[2], root)
    save_fig(fig, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
