#!/usr/bin/env python3
"""Value-domain census over le10gib (the insight pre-experiment, page-1 figure).

One segmented population bar: fraction of all 3624 real SuiteSparse matrices
whose value stream is constant / binary / <=16 / <=256 / <=65536 distinct
values / continuous. Cardinality from the auto-codec interning columns of the
cove_core_le10gib shards (value_codec_auto_unique_count, overflow > 65536).

Usage: plot_value_census.py <shards_dir> <out (no ext)>
"""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import patches
from paper_style import setup_style, save_fig

SEGS = [  # (label, upper bound, color)
    ("constant\n(implicit)", 1, "#76B900"),
    ("binary", 2, "#9fce4e"),
    ("$\\leq$16", 16, "#cde39b"),
    ("$\\leq$256", 256, "#b9cfe9"),
    ("$\\leq$65536", 65536, "#7da7d9"),
    ("continuous", float("inf"), "#8c1515"),
]


def main(shard_dir, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    best = {}
    for p in glob.glob(os.path.join(shard_dir, "shard*.csv")):
        if ".summary." in os.path.basename(p):
            continue
        for row in csv.DictReader(open(p, newline="")):
            try:
                u = int(row.get("value_codec_auto_unique_count", ""))
            except ValueError:
                continue
            ov = row.get("value_codec_auto_unique_overflow", "") == "true"
            if u == 0 and not ov:
                continue
            v = float("inf") if ov else u
            try:
                key = (row["matrix"], row["matrix_bytes"])
            except KeyError:
                continue
            if v > best.get(key, -1):
                best[key] = v
    n = len(best)
    counts = []
    lo = 0
    for label, ub, color in SEGS:
        c = sum(1 for v in best.values() if lo < v <= ub)
        counts.append(c)
        lo = ub
    print("n =", n, dict(zip([s[0].replace("\n", " ") for s in SEGS], counts)))

    fig, ax = plt.subplots(figsize=(3.4, 1.28))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    x = 0.0
    narrow_i = 0
    y0, hh = 0.34, 0.34
    for (label, _, color), c in zip(SEGS, counts):
        w = 100.0 * c / n
        ax.add_patch(patches.Rectangle((x, y0), w, hh, facecolor=color,
                                       edgecolor="white", lw=0.6))
        pct = 100.0 * c / n
        if pct >= 4.5:
            ax.text(x + w / 2, y0 + hh / 2, f"{pct:.0f}%", ha="center",
                    va="center", fontsize=6.4, color="#1c2128",
                    fontweight="bold")
        deep = False
        if w < 8.0:
            deep = narrow_i % 2 == 1
            narrow_i += 1
        ax.text(x + w / 2, y0 - (0.30 if deep else 0.10), label, ha="center",
                va="top", fontsize=5.6, color="#1c2128", linespacing=1.1)
        x += w

    # cumulative brackets above the bar
    def bracket(x0, x1, y, text, color):
        ax.plot([x0, x0, x1, x1], [y - 0.035, y, y, y - 0.035], color=color,
                lw=0.9, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, y + 0.045, text, ha="center", va="bottom",
                fontsize=6.0, color=color)

    c45 = 100.0 * (counts[0] + counts[1]) / n
    c91 = 100.0 * sum(counts[:5]) / n
    bracket(0, c45, 0.80, "no value array needed: 45%", "#538000")
    bracket(c45 + 1.5, c91, 0.80, "exact, 1–16-bit code: +45%", "#2c5f9e")
    bracket(c91 + 1.5, 100, 0.80, "9%", "#8c1515")
    save_fig(fig, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
