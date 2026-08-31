#!/usr/bin/env python3
"""DASP-style per-matrix showcase: largest fully-covered value1000 matrices,
speedup over cuSPARSE-FP64 for COVE and all per-matrix baselines.

Usage: plot_showcase.py <summary.csv> <combined.csv> <joint_shards_dir> <csr5_dir> <out>
"""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from paper_style import setup_style, save_fig

INK = "#1c2128"
SERIES = [
    ("cove_joint", "COVE joint (BFP8)", "#8c1515"),
    ("cove_hybrid", "COVE hybrid (FP64)", "#2f6fbb"),
    ("csr5", "CSR5 (FP64)", "#59636e"),
    ("dasp_double", "DASP (FP64)", "#9aa3ab"),
    ("packsell_pack32", "PackSELL p32", "#c4cad1"),
    ("spaden_bitmap_float", "Spaden (FP32)$^{*}$", "#e0c9a6"),
]


def main(summary, combined, joint_dir, csr5_dir, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False, "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    summ = {r["matrix"].split("/")[-1].replace(".mtx", ""): r
            for r in csv.DictReader(open(summary))}
    nnz = {}
    for r in csv.DictReader(open(combined)):
        m = os.path.basename(r["matrix"]).replace(".mtx", "")
        try:
            nnz[m] = int(float(r["nnz"]))
        except ValueError:
            pass
    cove = {}
    for p in glob.glob(os.path.join(joint_dir, "shard*.csv")):
        for r in csv.DictReader(open(p)):
            if r.get("codec") != "bfp8":
                continue
            m = os.path.basename(r["matrix"]).replace(".mtx", "")
            try:
                cove[m] = (float(r["joint_ms"]), float(r["hybrid_fp64_ms"]))
            except ValueError:
                pass
    c5 = {}
    for p in glob.glob(os.path.join(csr5_dir, "shard*.csv")):
        for r in csv.DictReader(open(p)):
            if r.get("status") == "ok":
                try:
                    c5[r["matrix"].replace(".mtx", "")] = float(r["csr5_spmv_ms"])
                except ValueError:
                    pass

    def ok(r, k):
        return r.get(k + "_status") == "ok" and r.get(k + "_ms") not in ("", "-1")

    rows = []
    for m, r in summ.items():
        if m not in cove or m not in c5 or m not in nnz:
            continue
        if not all(ok(r, k) for k in ("cusparse_double", "dasp_double",
                                      "packsell_pack32", "spaden_bitmap_float")):
            continue
        ref = float(r["cusparse_double_ms"])
        j, h = cove[m]
        rows.append((nnz[m], m, {
            "cove_joint": ref / j, "cove_hybrid": ref / h, "csr5": ref / c5[m],
            "dasp_double": ref / float(r["dasp_double_ms"]),
            "packsell_pack32": ref / float(r["packsell_pack32_ms"]),
            "spaden_bitmap_float": ref / float(r["spaden_bitmap_float_ms"])}))
    rows.sort(reverse=True, key=lambda t: t[0])
    rows = rows[:10]
    print("showcase:", [m for _, m, _ in rows])

    fig, ax = plt.subplots(figsize=(7.05, 1.46))
    n, k = len(rows), len(SERIES)
    width = 0.80 / k
    xs = np.arange(n)
    FLOOR = 0.05
    for si, (key, lab, color) in enumerate(SERIES):
        vals = [max(d[key], FLOOR) for _, _, d in rows]
        hatch = "///" if "Spaden" in lab else None
        ax.bar(xs + (si - (k - 1) / 2) * width, vals, width * 0.94, color=color,
               edgecolor="white", linewidth=0.3, label=lab, hatch=hatch, zorder=3)
    ax.axhline(1.0, color="#39414b", lw=0.8, zorder=4)
    ax.text(n - 0.52, 1.04, "cuSPARSE FP64", fontsize=5.2, ha="right",
            va="bottom", color="#39414b", zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 4.6)
    ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4])
    ax.set_yticklabels(["0.1", "0.25", "0.5", "1", "2", "4"], fontsize=5.6)
    ax.set_ylabel("speedup vs cuSPARSE-FP64", fontsize=6.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([m for _, m, _ in rows], fontsize=5.6, rotation=16,
                       ha="right")
    ax.set_xlim(-0.55, n - 0.45)
    ax.grid(axis="y", color="#d0d7de", lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(ncol=6, fontsize=5.4, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 1.16), handlelength=1.1,
              handletextpad=0.4, columnspacing=0.9)
    fig.tight_layout(pad=0.3)
    save_fig(fig, out)


if __name__ == "__main__":
    main(*sys.argv[1:6])
