#!/usr/bin/env python3
"""Memory anatomy: total bytes-per-nonzero decomposition (the third axis).

Horizontal stacked bars, position bytes (blue) + value bytes (green), for
CSR-FP64 / CSR-FP32 / COVE structural-only / COVE joint, nnz-weighted over
value1000. Position bytes from the bitmap layout (original_lb
position_payload_bytes); CSR index bytes = 4*nnz + 4*(rows+1); joint value
bytes from the joint sweep (bfp8 bytes_per_nnz, includes scales).

Usage: plot_memory_anatomy.py <repo_root> <out (no ext)>
"""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import patches
from paper_style import setup_style, save_fig

PBLUE, GREEN, DGREEN, CHAR = "#2c5f9e", "#76B900", "#538000", "#1c2128"


def main(root, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    tot_pos = tot_idx = tot_nnz = 0
    for p in glob.glob(os.path.join(
            root, "structural/results/studies/cove_core_value1000_2026-06-09/shard*.csv")):
        for row in csv.DictReader(open(p, newline="")):
            if row.get("operator_name") != "original_lb" or row.get("status") != "ok":
                continue
            try:
                nnz = int(row["nnz"]); rows = int(row["rows"])
                pb = float(row["position_payload_bytes"])
            except (ValueError, KeyError):
                continue
            if nnz <= 0 or pb <= 0:
                continue
            tot_pos += pb
            tot_idx += 4.0 * nnz + 4.0 * (rows + 1)
            tot_nnz += nnz
    vj = vn = 0
    for p in glob.glob(os.path.join(
            root, "structural/results/studies/cove_joint_value1000_2026-06-09/*.csv")):
        for row in csv.DictReader(open(p, newline="")):
            if row.get("codec") == "bfp8" and row.get("status") == "ok":
                try:
                    nnz = int(row["nnz"]); b = float(row["bytes_per_nnz"])
                except ValueError:
                    continue
                vj += b * nnz; vn += nnz
    pos_cove = tot_pos / tot_nnz
    pos_csr = tot_idx / tot_nnz
    val_joint = vj / vn
    print(f"pos COVE {pos_cove:.2f}  pos CSR {pos_csr:.2f}  val joint {val_joint:.2f}")

    bars = [
        ("CSR FP64", pos_csr, 8.0),
        ("CSR FP32", pos_csr, 4.0),
        ("COVE structural-only", pos_cove, 8.0),
        ("COVE joint (BFP8)", pos_cove, val_joint),
    ]
    base = pos_csr + 8.0

    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(-1.35, len(bars) - 0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.4)
    ax.invert_yaxis()
    ax.tick_params(left=False, bottom=True, labelsize=6)
    ax.set_xlabel("total bytes per nonzero (nnz-weighted, value1000)",
                  fontsize=6.6)
    for i, (name, pb, vb) in enumerate(bars):
        ax.barh(i, pb, color=PBLUE, height=0.62, edgecolor="white", lw=0.5)
        ax.barh(i, vb, left=pb, color=GREEN, height=0.62, edgecolor="white",
                lw=0.5)
        if pb > 1.6:
            ax.text(pb / 2, i, f"{pb:.1f}", ha="center", va="center",
                    fontsize=5.8, color="white", fontweight="bold")
        if vb > 1.6:
            ax.text(pb + vb / 2, i, f"{vb:.1f}", ha="center", va="center",
                    fontsize=5.8, color=CHAR, fontweight="bold")
        tot = pb + vb
        ratio = base / tot
        tag = "1.0$\\times$" if abs(ratio - 1) < 1e-9 else f"{ratio:.1f}$\\times$ smaller"
        ax.text(tot + 0.25, i, tag, ha="left", va="center", fontsize=6.0,
                color=DGREEN if ratio > 2 else "0.35",
                fontweight="bold" if ratio > 2 else "normal")
    # legend row above the bars
    ax.add_patch(patches.Rectangle((4.6, -1.15), 0.42, 0.34, color=PBLUE))
    ax.text(5.15, -0.98, "position bytes", fontsize=5.9, va="center")
    ax.add_patch(patches.Rectangle((8.2, -1.15), 0.42, 0.34, color=GREEN))
    ax.text(8.75, -0.98, "value bytes", fontsize=5.9, va="center")
    save_fig(fig, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
