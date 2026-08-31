#!/usr/bin/env python3
"""Fig 1 (motivation): make position-value co-regularity VISIBLE at the 8x4 unit.

Two matrices: cant (co-regular) and bcsstk24 (anti-co-regular). For each, a
zoomed diagonal window with the 8x4 COVE grid:
  top row    = nonzero magnitude in decades below the matrix max,
               log10(|a| / max|a|); entries more than 4 decades down render
               as one light-gray "~0" tone (they carry no output-error mass);
  bottom row = per-block INT8+shared-scale relative L1 error, the
               codec-grounded quantity behind the measured coherence gain
               (NOT raw log-span, which the weighting confound inverts).
Real window vs the SAME window after a global value permutation (the null).
Whole-matrix gain_i8 is recomputed with the EXACT definition of
analyze_coregularity_batch.py (|v|, L1, per-block max scale, 4-shuffle null)
so the annotated numbers match the evaluation text.

Usage: plot_fig1_motivation.py <cant.mtx> <bcsstk24.mtx> <coreg_csv> <out>
"""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.io import mmread
from paper_style import setup_style, save_fig

BR, BC = 8, 4
WIN_R, WIN_C = 32, 64  # 4x16 grid of 8x4 blocks (wide, for figure*)
NSHUF = 4
DECADES = 4.0


def load(path):
    m = mmread(path).tocoo()
    a = np.abs(np.asarray(m.data, dtype=np.float64))
    keep = np.isfinite(a) & (a > 0)
    return m.row[keep].astype(np.int64), m.col[keep].astype(np.int64), \
        a[keep], m.shape


def _relerr(a_s, blk_start, blk_nnz, levels=127, perm=None):
    v = a_s if perm is None else a_s[perm]
    smax = np.maximum.reduceat(v, blk_start)
    scale = np.repeat(np.where(smax > 0, smax, 1.0), blk_nnz)
    q = np.clip(np.rint(v / scale * levels), -levels, levels)
    return float(np.abs(v - q * scale / levels).sum() / v.sum())


def matrix_gain_i8(r, c, a, ncols):
    nbc = int(ncols // BC) + 1
    bkey = (r // BR) * np.int64(nbc) + (c // BC)
    order = np.argsort(bkey, kind="stable")
    a_s, bkey_s = a[order], bkey[order]
    blk_start = np.flatnonzero(np.r_[True, bkey_s[1:] != bkey_s[:-1]])
    blk_nnz = np.diff(np.r_[blk_start, a_s.size]).astype(np.int64)
    rng = np.random.default_rng(0)
    rr = _relerr(a_s, blk_start, blk_nnz)
    nn = float(np.mean([_relerr(a_s, blk_start, blk_nnz,
                                perm=rng.permutation(a_s.size))
                        for _ in range(NSHUF)]))
    return 1.0 - rr / nn


def value_grid(r, c, a, r0, c0):
    sel = (r >= r0) & (r < r0 + WIN_R) & (c >= c0) & (c < c0 + WIN_C)
    g = np.full((WIN_R, WIN_C), np.nan)
    g[r[sel] - r0, c[sel] - c0] = a[sel]
    return g


def block_relerr_map(g):
    """Per-8x4-block INT8 shared-max-scale relative L1 error."""
    out = np.full((g.shape[0] // BR, g.shape[1] // BC), np.nan)
    for bi in range(out.shape[0]):
        for bj in range(out.shape[1]):
            v = g[bi * BR:(bi + 1) * BR, bj * BC:(bj + 1) * BC]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            scale = v.max()
            q = np.clip(np.rint(v / scale * 127.0), -127, 127)
            out[bi, bj] = np.abs(v - q * scale / 127.0).sum() / v.sum()
    return out


def pick_window(r, c, a, shape, by_coherence):
    """Block-aligned diagonal window; for the co-regular case prefer windows
    whose real per-block error is low (representative of the coherent mass)."""
    n = min(shape)
    cand = []
    step = max(BR, (n - WIN_R) // 600 // BR * BR or BR)
    for r0 in range(0, n - WIN_R, step):
        c0 = (r0 + WIN_R // 2 - WIN_C // 2) // BC * BC
        if c0 < 0 or c0 + WIN_C > shape[1]:
            continue
        cnt = np.sum((r >= r0) & (r < r0 + WIN_R) & (c >= c0) & (c < c0 + WIN_C))
        cand.append((int(cnt), int(r0), int(c0)))
    cand.sort(reverse=True)
    if not by_coherence:
        return cand[0][1], cand[0][2]
    dense = [x for x in cand if x[0] >= 0.6 * cand[0][0]][:80]
    best, best_med = dense[0][1:], np.inf
    for cnt, r0, c0 in dense:
        err = block_relerr_map(value_grid(r, c, a, r0, c0))
        med = np.nanmedian(err)
        if np.isfinite(med) and med < best_med:
            best, best_med = (r0, c0), med
    return best


def draw_grid(ax, nr, nc, lw=0.4, color="0.55"):
    for x in range(0, nc + 1, BC):
        ax.axvline(x - 0.5, color=color, lw=lw, zorder=3)
    for y in range(0, nr + 1, BR):
        ax.axhline(y - 0.5, color=color, lw=lw, zorder=3)


def main(cant_path, bcsstk_path, out):
    setup_style()
    plt.rcParams.update({"text.usetex": False,  # TinyTeX lacks type1cm
                         "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix"})

    rng = np.random.default_rng(1)
    cases = []
    for path, name, coh in [(cant_path, "cant (FEM)", True),
                            (bcsstk_path, "bcsstk24 (stiffness)", False)]:
        r, c, a, shape = load(path)
        gain = matrix_gain_i8(r, c, a, shape[1])
        anull = a[rng.permutation(a.size)]
        r0, c0 = pick_window(r, c, a, shape, by_coherence=coh)
        rel = np.log10(np.maximum(a, 1e-300) / a.max())
        greal = value_grid(r, c, rel, r0, c0)
        gnull = value_grid(r, c, rel[rng.permutation(rel.size)], r0, c0)
        ereal = block_relerr_map(value_grid(r, c, a, r0, c0))
        enull = block_relerr_map(value_grid(r, c, anull, r0, c0))
        print(f"{name}: window=({r0},{c0}) gain_i8={gain:+.2f}")
        cases.append((name, gain, greal, gnull, ereal, enull))

    from matplotlib import gridspec
    fig = plt.figure(figsize=(7.05, 2.35))
    gs = gridspec.GridSpec(2, 7, width_ratios=[1, 1, 1, 1, 0.035, 0.30, 1.55],
                           height_ratios=[2.0, 1.0], wspace=0.06, hspace=0.06)

    evmin, evmax = 3e-4, 3e-2
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("0.92")

    ims = imbs = None
    bot_axes = []
    for k, (name, gain, greal, gnull, ereal, enull) in enumerate(cases):
        for j, (la, err, tag) in enumerate([(greal, ereal, "real values"),
                                            (gnull, enull, "shuffled null")]):
            col = 2 * k + j
            ax = fig.add_subplot(gs[0, col])
            ims = ax.imshow(la, cmap=cmap, interpolation="nearest",
                            norm=Normalize(-DECADES, 0.0))
            draw_grid(ax, WIN_R, WIN_C)
            ax.set_title(tag, fontsize=7, pad=2, color="0.25")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.text(1.03, 1.42, f"({'ab'[k]}) {name}",
                        transform=ax.transAxes, ha="center",
                        fontsize=8.2, weight="bold")
                verd = "co-regular" if gain > 0 else "anti-co-regular"
                vcol = "#538000" if gain > 0 else "#8c1515"
                ax.text(1.03, 1.22, verd, transform=ax.transAxes, ha="center",
                        fontsize=6.8, color=vcol, weight="bold")
            if col == 0:
                ax.set_ylabel("magnitude\n(decades below max)", fontsize=6.5)

            axb = fig.add_subplot(gs[1, col])
            bot_axes.append(axb)
            imbs = axb.imshow(err, cmap="magma_r", interpolation="nearest",
                              norm=LogNorm(evmin, evmax), aspect="auto")
            for x in range(err.shape[1] + 1):
                axb.axvline(x - 0.5, color="0.78", lw=0.3)
            for y in range(err.shape[0] + 1):
                axb.axhline(y - 0.5, color="0.78", lw=0.3)
            axb.set_xticks([]); axb.set_yticks([])
            med = np.nanmedian(err)
            axb.set_xlabel(f"median {med:.1e}", fontsize=6.2, labelpad=1.5)
            if col == 0:
                axb.set_ylabel("per-block INT8 err.\n(darker = larger)",
                               fontsize=6.0)


    cax1 = fig.add_subplot(gs[0, 4])
    fig.colorbar(ims, cax=cax1, extend="min")
    cax1.tick_params(labelsize=5.5)
    cax2 = fig.add_subplot(gs[1, 4])
    fig.colorbar(imbs, cax=cax2)
    cax2.tick_params(labelsize=5.5)

    # ---- right panel: the whole collection (permutation-null gain) ----------
    axs = fig.add_subplot(gs[:, 6])
    fill, gv = [], []
    mark = None
    for row in csv.DictReader(open(sys.argv[3], newline="")):
        if row["status"] != "ok" or not row.get("gain_i8"):
            continue
        x, y = float(row["mat_fill"]), float(row["gain_i8"])
        fill.append(x); gv.append(y)
        if row["matrix"] == "HB/bcsstk24/bcsstk24.mtx":
            mark = (x, y)
    fill = np.asarray(fill); gv = np.asarray(gv)
    gc = np.clip(gv, -1, 1)
    axs.axhspan(0, 1.05, facecolor="#8c1515", alpha=0.05)
    axs.axhspan(-1.05, 0, facecolor="#2c5f9e", alpha=0.05)
    axs.scatter(fill[gv > 0], gc[gv > 0], s=2.5, alpha=0.45, color="#8c1515",
                linewidths=0)
    axs.scatter(fill[gv <= 0], gc[gv <= 0], s=2.5, alpha=0.45, color="#2c5f9e",
                linewidths=0)
    axs.axhline(0, color="0.3", lw=0.7)
    if mark:
        yc = float(np.clip(mark[1], -1, 1))
        axs.scatter([mark[0]], [yc], s=24, facecolor="none", edgecolor="k",
                    linewidths=0.9)
        axs.annotate("bcsstk24", (mark[0], yc), xytext=(7, 8),
                     textcoords="offset points", fontsize=5.8,
                     bbox=dict(facecolor="white", alpha=0.75,
                               edgecolor="none", pad=0.8))
    axs.text(0.96, 0.95, f"n={len(fill)}\n$\\rho$=+0.28\n61% gain>0",
             transform=axs.transAxes, ha="right", va="top", fontsize=5.8,
             linespacing=1.35)
    bb = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0)
    axs.text(0.03, 0.93, "co-regular", transform=axs.transAxes,
             ha="left", va="top", fontsize=6.0, color="#8c1515", bbox=bb)
    axs.text(0.03, 0.07, "anti-co-regular", transform=axs.transAxes,
             ha="left", va="bottom", fontsize=6.0, color="#2c5f9e", bbox=bb)
    axs.set_xlabel("block fill", fontsize=6.5)
    axs.set_ylabel("value gain (clipped)", fontsize=6.5)
    axs.tick_params(labelsize=5.5)
    axs.set_xlim(0, 1.0)
    axs.set_ylim(-1.05, 1.05)
    axs.set_title("(c) the whole collection (le10gib)", fontsize=7, pad=2,
                  color="0.25")
    axs.yaxis.tick_right()
    axs.yaxis.set_label_position("right")

    for k, gain in enumerate([c[1] for c in cases]):
        p0 = bot_axes[2 * k].get_position()
        p1 = bot_axes[2 * k + 1].get_position()
        fig.text((p0.x0 + p1.x1) / 2, -0.015,
                 "whole-matrix gain " + f"{gain:+.2f}".replace("-", "\u2212"),
                 ha="center", fontsize=7.5, weight="bold",
                 color="#538000" if gain > 0 else "#8c1515")

    save_fig(fig, out)


if __name__ == "__main__":
    main(*sys.argv[1:3], sys.argv[4])
