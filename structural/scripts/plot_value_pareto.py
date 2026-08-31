#!/usr/bin/env python3
"""Fig 5b: the value menu on the storage-accuracy plane (value1000).

Per codec, median bytes-per-nonzero vs median SpMV output relative error
across the verified value1000 matrices (cove_core_value1000 shards,
RTX PRO 6000). eps=1e-2 gate dashed; the claim: BFP8+outlier dominates generic
BF16/FP16 on both axes; the lossless dictionary applies only to
low-cardinality matrices.

Usage: plot_value_pareto.py <shards_glob_dir> <out (no ext)>
"""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from paper_style import setup_style, save_fig

CODECS = {
    "original_lb": ("FP64 (exact)", "#59636e", "s"),
    "bf16_lb": ("BF16", "#2f6fbb", "o"),
    "fp16_lb": ("FP16", "#2f6fbb", "^"),
    "bfp8_lb": ("BFP8", "#538000", "o"),
    "bfp8_outlier_lb": ("BFP8+outlier", "#8c1515", "D"),
    "bfp4_lb": ("BFP4", "#538000", "v"),
}
ERR_FLOOR = 1e-17


def main(shard_dir, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    data = {k: {} for k in CODECS}  # codec -> matrix -> (bytes, err)
    all_keys = set()
    for path in glob.glob(os.path.join(shard_dir, "shard*.csv")):
        if ".summary." in os.path.basename(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
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
                except (ValueError, TypeError, KeyError):
                    continue
                if b <= 0 or e < 0:
                    continue
                data[op][key] = (b, max(e, ERR_FLOOR))

    # the comparable family is measured on the COMMON success set; partial-
    # coverage codecs (fp16 overflow, bfp4 gate, dict low-cardinality) are
    # plotted on their own survivors WITH the coverage stated -- otherwise
    # survivorship bias makes fp16 look better than bf16.
    family = ["original_lb", "bf16_lb", "bfp8_lb", "bfp8_outlier_lb"]
    common = set.intersection(*(set(data[f]) for f in family))
    total = len(all_keys)

    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    pts = {}
    for op, (label, color, marker) in CODECS.items():
        rows = data[op]
        if not rows:
            print(f"skip {op}: no rows")
            continue
        keys = common if op in family else set(rows)
        vals = [rows[m] for m in keys if m in rows]
        bx = float(np.median([v[0] for v in vals]))
        ey = float(np.median([v[1] for v in vals]))
        cov = len(rows) / float(total)
        if op not in family and cov < 0.95:
            label = f"{label} ({cov:.0%} cov.)"
        exact = ey < 1e-12
        if exact:
            ey = 1.15e-5
            label += r" $\downarrow$ err $= 0$"
        pts[op] = (bx, ey)
        ax.scatter([bx], [ey], s=30, color=color, marker=marker, zorder=3)
        CODECS[op] = (label, color, marker)
        print(f"{op}: n={len(vals)} cov={cov:.2f} B/nnz={bx:.2f} err={ey:.2e}")

    # annotate points
    offs = {"original_lb": (-8, 8), "bf16_lb": (7, 2), "fp16_lb": (7, -9),
            "bfp8_lb": (-6, 8), "bfp8_outlier_lb": (6, -11),
            "bfp4_lb": (-22, 9)}
    for op, (bx, ey) in pts.items():
        ax.annotate(CODECS[op][0], (bx, ey), xytext=offs.get(op, (6, 4)),
                    textcoords="offset points", fontsize=6.2,
                    color=CODECS[op][1])

    ax.axhline(1e-2, color="0.45", lw=0.8, ls="--")
    ax.text(7.6, 1.4e-2, r"$\varepsilon=10^{-2}$ gate", fontsize=6,
            color="0.35", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median value bytes per nonzero")
    ax.set_ylabel("median SpMV output rel. error")
    ax.set_xlim(0.6, 11)
    ax.set_ylim(8e-6, 4e-2)
    save_fig(fig, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
