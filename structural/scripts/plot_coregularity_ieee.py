#!/usr/bin/env python3
"""IEEE-styled co-regularity scatter (Fig 5a spec): block fill vs
null-controlled value gain on le10gib, quadrant shading, corner stats,
the two Fig-1 matrices marked.

Usage: plot_coregularity_ieee.py <coregularity_le10gib.csv> <out (no ext)>
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from paper_style import setup_style, save_fig

MARK = {"HB/bcsstk24/bcsstk24.mtx": ("bcsstk24", (-40, 10))}


def main(csv_path, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    fill, gain, marks = [], [], {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or not row.get("gain_i8"):
                continue
            x, y = float(row["mat_fill"]), float(row["gain_i8"])
            fill.append(x); gain.append(y)
            if row["matrix"] in MARK:
                marks[row["matrix"]] = (x, y)
    fill = np.asarray(fill); gain = np.asarray(gain)
    gain_c = np.clip(gain, -1.0, 1.0)
    pos = float(np.mean(gain > 0)) * 100

    from scipy.stats import spearmanr
    rho = spearmanr(fill, gain).statistic

    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    ax.axhspan(0, 1.05, facecolor="#8c1515", alpha=0.045)
    ax.axhspan(-1.05, 0, facecolor="#2f6fbb", alpha=0.045)
    ax.scatter(fill[gain > 0], gain_c[gain > 0], s=3, alpha=0.45,
               color="#8c1515", linewidths=0)
    ax.scatter(fill[gain <= 0], gain_c[gain <= 0], s=3, alpha=0.45,
               color="#2f6fbb", linewidths=0)
    ax.axhline(0, color="0.3", lw=0.7)

    for key, (x, y) in marks.items():
        name, off = MARK[key]
        yc = float(np.clip(y, -1, 1))
        ax.scatter([x], [yc], s=26, facecolor="none", edgecolor="k", linewidths=0.9)
        ax.annotate(f"{name} ({y:+.2f})", (x, yc), xytext=off,
                    textcoords="offset points", fontsize=6.2)

    ax.text(0.97, 0.96,
            f"n={len(fill)}   $\\rho$(fill, gain)=+{rho:.2f}\n"
            f"{pos:.0f}% co-regular (gain>0)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4)
    ax.set_xlabel("block fill (position compressibility $\\rightarrow$)")
    ax.set_ylabel("value coherence gain\n$1 - \\mathrm{real/null}$ (clipped)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.05, 1.05)
    save_fig(fig, out)
    print(f"n={len(fill)} rho={rho:.3f} pos={pos:.1f}% marks={list(marks)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
