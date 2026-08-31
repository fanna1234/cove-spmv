#!/usr/bin/env python3
"""CG end-to-end figure: (a) residual floor per variant; (b) lossless speedup.

Usage: plot_cg_solver.py <cg_real_spd.csv> <out (no ext)>
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from paper_style import setup_style, save_fig

INK = "#1c2128"
VARIANTS = [
    ("cusparse_double", "FP64", "#59636e"),
    ("cove_lossless", "COVE\nlossless", "#2f6fbb"),
    ("cusparse_float", "FP32", "#9aa3ab"),
    ("cove_bf16", "BF16", "#538000"),
    ("cove_bfp8", "BFP8", "#76B900"),
]


def main(csv_path, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False, "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    rows = list(csv.DictReader(open(csv_path, newline="")))
    mats = sorted(set(r["matrix"] for r in rows))
    get = {(r["matrix"], r["variant"]): r for r in rows}

    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 1.55),
                               gridspec_kw={"width_ratios": [1.25, 1.0],
                                            "wspace": 0.42})
    # (a) residual floors, representative matrix (cant)
    REF = "cant"
    xs = []
    for i, (var, lab, color) in enumerate(VARIANTS):
        r = get.get((REF, var))
        if r is None:
            continue
        y = float(r["floor"])
        a.bar(i, y, width=0.62, color=color, edgecolor="white", linewidth=0.4)
        xs.append(i)
    a.set_yscale("log")
    a.set_ylim(2e-9, 6e-2)
    a.set_xticks(range(len(VARIANTS)))
    a.set_xticklabels([l for _, l, _ in VARIANTS], fontsize=5.2)
    a.tick_params(axis="y", labelsize=5.4)
    a.set_ylabel("CG residual floor (cant)", fontsize=6.0)
    a.set_title("(a) floor $\\approx$ codec error", fontsize=6.8, pad=3)
    a.annotate("bit-exact\nFP64", xy=(0.95, 1.6e-8), xytext=(-0.30, 2.5e-6),
               fontsize=5.4, color="#2f6fbb", ha="left", linespacing=1.2,
               arrowprops=dict(arrowstyle="-", lw=0.5, color="#2f6fbb"))
    a.annotate("$\\approx$ codec error", xy=(3.05, 2.4e-3), xytext=(1.45, 2.2e-2),
               fontsize=5.4, color="#538000", ha="left",
               arrowprops=dict(arrowstyle="-", lw=0.5, color="#538000"))
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.grid(axis="y", color="#d0d7de", lw=0.4, alpha=0.6)
    a.set_axisbelow(True)

    # (b) end-to-end speedup of the lossless operator
    sp = []
    for m in mats:
        r0, r1 = get[(m, "cusparse_double")], get[(m, "cove_lossless")]
        if r0["spd_ok"] != "1" or r1["spd_ok"] != "1":
            continue
        t0, t1 = float(r0["time_1e-6_ms"]), float(r1["time_1e-6_ms"])
        if t0 > 0 and t1 > 0:
            sp.append((m, t0 / t1))
    xs = np.arange(len(sp))
    b.bar(xs, [v for _, v in sp], width=0.62, color="#2f6fbb",
          edgecolor="white", linewidth=0.4)
    for x, (_, v) in zip(xs, sp):
        b.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
               fontsize=5.4, color=INK)
    b.axhline(1.0, color="#39414b", lw=0.7)
    b.set_xticks(xs)
    b.set_xticklabels([m for m, _ in sp], fontsize=5.2, rotation=20)
    b.tick_params(axis="y", labelsize=5.4)
    b.set_ylim(0, 1.45)
    b.set_ylabel("speedup vs cuSPARSE", fontsize=6.2)
    b.set_title("(b) end-to-end, tol $10^{-6}$", fontsize=6.8, pad=3)
    for spn in ("top", "right"):
        b.spines[spn].set_visible(False)
    b.grid(axis="y", color="#d0d7de", lw=0.4, alpha=0.6)
    b.set_axisbelow(True)

    fig.tight_layout(pad=0.3)
    save_fig(fig, out)
    print("matrices:", [m for m, _ in sp], "speedups:",
          [f"{v:.3f}" for _, v in sp])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
