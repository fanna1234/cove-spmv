#!/usr/bin/env python3
"""Fig 4: cross-device stacking bars (replaces tab:crossdev).

Grouped bars: x = flagship matrices, one bar per device, y = BFP8 joint
speedup over that device's tuned FP64 hybrid (the stacking factor), values
annotated. Data: cove_joint_canonical25 (RTX PRO 6000) + cove_joint_stars_{h100,
a800} CSVs, codec=bfp8.

Usage: plot_crossdev_stacking.py <repo_root> <out (no ext)>
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from paper_style import setup_style, save_fig

MATS = ["F1.mtx", "Ga41As41H72.mtx", "Si41Ge41H72.mtx"]
DEVICES = [
    ("RTX PRO 6000 (FP64 1/64)", "#8c1515",
     "structural/results/studies/cove_joint_canonical25_2026-06-09.csv"),
    ("H100 (1/2)", "#2f6fbb",
     "structural/results/studies/cove_joint_stars_h100_2026-06-09.csv"),
    ("A800 (1/2)", "#7f7f7f",
     "structural/results/studies/cove_joint_stars_a800_2026-06-09.csv"),
]


def read_stacking(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "codec" in row and row.get("codec") != "bfp8":
                continue
            if "status" in row and row.get("status") != "ok":
                continue
            m = row["matrix"]
            m = m if m.endswith(".mtx") else m + ".mtx"
            if m in MATS and m not in out:
                out[m] = float(row["joint_speedup_vs_hybrid"])
    return out


def main(root, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    data = []
    for label, color, rel in DEVICES:
        d = read_stacking(os.path.join(root, rel))
        data.append((label, color, [d[m] for m in MATS]))
        print(label, "->", {m: round(d[m], 2) for m in MATS})

    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    x = np.arange(len(MATS))
    w = 0.26
    for i, (label, color, vals) in enumerate(data):
        bars = ax.bar(x + (i - 1) * w, vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=6)
    ax.axhline(1.0, color="0.4", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m[:-4] for m in MATS], fontsize=7.5)
    ax.set_ylabel("joint speedup\nvs FP64 hybrid", fontsize=7.5)
    ax.set_ylim(0, 3.7)
    ax.legend(fontsize=5.8, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.04), framealpha=0.95,
              columnspacing=0.9, handlelength=1.2)
    save_fig(fig, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
