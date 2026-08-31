#!/usr/bin/env python3
"""COVE value-axis storage<->accuracy Pareto figure.

Reads structural/results/studies/accuracy_robust_2026-06-09.csv (produced by
structural/src/bench_accuracy_robust.cu) and saves a two-panel scatter:

  Panel A: x = bytes_per_nnz, y = rel_maxrow_worst_rand (log scale)
  Panel B: x = bytes_per_nnz, y = rel_l2_worst_rand     (log scale)

One color/marker per value codec; each point is one matrix. The eps = 1e-2
accuracy line is drawn in both panels. Accuracy is the WORST over the RANDOM
subset of the x-ensemble (uniform(-1,1) + N(0,1)), using the production decode +
fp32 accumulation. Structured / cancellation inputs (all-ones, legacy/zero-mean
fixed-x) collapse ||y0|| on cancellation-heavy matrices and inflate the error for
EVERY reduced-precision codec -- a matrix x input conditioning effect, not a codec
property -- so they are reported separately as a conditioning caveat (CSV
rel_*_worst_struct columns) and excluded from this headline figure. The
quantization-only bfp8 (fp64 accumulation) point is shown as a hollow marker so
the fp32-accumulation contribution (gap to the filled bfp8 point) is explicit.

Usage:
  /opt/conda/bin/python structural/scripts/plot_accuracy_pareto.py \
      [csv_path] [out_png]
"""

import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CSV = "structural/results/studies/accuracy_robust_2026-06-09.csv"
DEFAULT_PNG = "structural/results/studies/accuracy_pareto_2026-06-09.png"
EPS = 1e-2

# Per-codec plot style. The fp64-accum bfp8 is a separate, hollow series so the
# quantization-only vs full (fp32-accum) decomposition is visible.
STYLE = {
    ("bfp8", "fp32"): dict(label="bfp8 (fp32 accum)", color="#1f77b4", marker="o",
                           filled=True),
    ("bfp8", "fp64"): dict(label="bfp8 (fp64 accum = quant-only)", color="#1f77b4",
                           marker="o", filled=False),
    ("bfp8_outlier", "fp32"): dict(label="bfp8+outlier (P=0.5%)", color="#2ca02c",
                                   marker="^", filled=True),
    ("bfp4", "fp32"): dict(label="bfp4 (fp32 accum)", color="#d62728", marker="s",
                           filled=True),
    ("bf16", "fp32"): dict(label="bf16 (fp32 accum)", color="#9467bd", marker="D",
                           filled=True),
    ("fp16", "fp32"): dict(label="fp16 (fp32 accum)", color="#ff7f0e", marker="v",
                           filled=True),
}
# Plot order (filled codecs first, hollow decomposition point last).
ORDER = [
    ("bfp8", "fp32"),
    ("bfp8_outlier", "fp32"),
    ("bfp4", "fp32"),
    ("bf16", "fp32"),
    ("fp16", "fp32"),
    ("bfp8", "fp64"),
]


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def to_float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return math.nan
    return v


def scatter_panel(ax, rows, ycol, ylabel, title):
    n_overflow = 0
    for key in ORDER:
        codec, accum = key
        st = STYLE[key]
        xs, ys = [], []
        for r in rows:
            if r["codec"] != codec or r["accum"] != accum:
                continue
            x = to_float(r["bytes_per_nnz"])
            y = to_float(r[ycol])
            # Skip non-finite / non-positive points (e.g. fp16 OVERFLOWS on
            # matrices whose values exceed the fp16 max ~6.55e4 -> reported as
            # inf); count the inf overflows so the caption can flag them.
            if not (math.isfinite(x) and math.isfinite(y) and y > 0.0):
                if math.isinf(y):
                    n_overflow += 1
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        if st["filled"]:
            ax.scatter(xs, ys, s=70, c=st["color"], marker=st["marker"],
                       edgecolors="black", linewidths=0.5, label=st["label"],
                       alpha=0.85, zorder=3)
        else:
            ax.scatter(xs, ys, s=80, facecolors="none", edgecolors=st["color"],
                       marker=st["marker"], linewidths=1.6, label=st["label"],
                       zorder=4)

    ax.axhline(EPS, color="red", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(ax.get_xlim()[1], EPS * 1.12, r"$\epsilon = 10^{-2}$", color="red",
            ha="right", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("storage: bytes / nonzero")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    return n_overflow


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    out_png = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PNG
    if not os.path.exists(csv_path):
        sys.exit("CSV not found: %s" % csv_path)

    rows = load_rows(csv_path)
    matrices = sorted({r["matrix"] for r in rows})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    n_ov = scatter_panel(axA, rows, "rel_maxrow_worst_rand",
                         r"worst-over-random-x  $\max_r|\hat y_r-y_r|/\max_r|y_r|$",
                         "(A) max-row global-norm error")
    scatter_panel(axB, rows, "rel_l2_worst_rand",
                  r"worst-over-random-x  $\|\hat y-y\|_2/\|y\|_2$",
                  "(B) L2 relative error")

    # Single legend for the whole figure (panel A has every series).
    handles, labels = axA.get_legend_handles_labels()
    axA.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.92)

    note = ("Accuracy = worst over RANDOM x-ensemble (uniform + normal); "
            "structured/cancellation inputs (all-ones) reported separately as a "
            "conditioning caveat. Decode is the real BitBSR 8x4 operator with fp32 "
            "accumulation.\nHollow bfp8 = fp64-accumulation (quantization-only); "
            "gap to filled bfp8 = the fp32-accumulation contribution.  "
            "%d matrices: %s." % (len(matrices), ", ".join(matrices)))
    if n_ov:
        note += ("\nNote: fp16 overflows (|value| > 6.55e4) on some matrices and "
                 "is omitted there (reported as inf in the CSV).")
    fig.suptitle("COVE value-axis storage↔accuracy Pareto",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7.5, wrap=True)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("wrote %s  (%d matrices, %d rows)" % (out_png, len(matrices), len(rows)))


if __name__ == "__main__":
    main()
