#!/usr/bin/env python3
"""Position-metadata accounting on the value1000 successful set.

The plot compares COVE packed block words plus 16-byte work descriptors against
CSR row and column indices. It intentionally does not combine value bytes from a
different success set into a synthetic total-storage headline.

Usage: plot_memory_anatomy.py <repo_root> <out (no ext)>
"""

import csv
import glob
import os
import sys

import matplotlib.pyplot as plt

from paper_style import save_fig, setup_style


BLUE = "#2c5f9e"
ORANGE = "#f3b37a"
GRAY = "#d9dde2"
INK = "#1c2128"


def main(root, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})
    packed_bytes = work_bytes = csr_bytes = total_nnz = 0.0
    matrix_count = 0
    pattern = os.path.join(
        root, "structural/results/studies/"
        "cove_core_value1000_2026-06-09/shard*.csv")
    for path in glob.glob(pattern):
        if ".summary." in os.path.basename(path):
            continue
        for row in csv.DictReader(open(path, newline="")):
            if (row.get("requested_operator") != "original_lb"
                    or row.get("operator_name") != "original_lb"
                    or row.get("status") != "ok"):
                continue
            try:
                nnz = int(row["nnz"])
                rows = int(row["rows"])
                packed = float(row["position_payload_bytes"])
                work_items = int(row["work_items"])
            except (KeyError, ValueError):
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
    print(f"n={matrix_count} COVE={cove_bpn:.4f} B/nnz "
          f"(words={packed_bpn:.4f}, work={work_bpn:.4f}) "
          f"CSR={csr_bpn:.4f} B/nnz")

    fig, ax = plt.subplots(figsize=(3.3, 1.5))
    ax.barh([0], [packed_bpn], color=BLUE, height=0.55,
            edgecolor=INK, linewidth=0.35)
    ax.barh([0], [work_bpn], left=[packed_bpn], color=ORANGE,
            height=0.55, hatch="..", edgecolor=INK, linewidth=0.35)
    ax.barh([1], [csr_bpn], color=GRAY, height=0.55,
            hatch="//", edgecolor=INK, linewidth=0.45)
    ax.text(packed_bpn / 2, 0, f"words {packed_bpn:.2f}", ha="center",
            va="center", fontsize=6, color="white", fontweight="bold")
    ax.text(cove_bpn + 0.08, 0, f"+ work = {cove_bpn:.2f}", ha="left",
            va="center", fontsize=6, color=INK)
    ax.text(csr_bpn / 2, 1, f"row + column = {csr_bpn:.2f}", ha="center",
            va="center", fontsize=6, color=INK)
    ax.set_yticks([0, 1], ["COVE", "CSR"])
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("position bytes per nonzero")
    ax.set_title(f"Position metadata on value1000 (n={matrix_count})")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    save_fig(fig, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
