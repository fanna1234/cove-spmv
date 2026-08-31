#!/usr/bin/env python3
"""Co-regularity test: is value-compressibility SPATIALLY organized at the 8x4 unit?

Tests the COVE title claim "Co-Regular Value-Position Encoding": do the same 8x4
blocks that are position-favorable (dense) also hold value-favorable (tight-range)
nonzeros, BEYOND what each block's nonzero count alone explains?

Per block, value spread = log2(max|v|) - log2(min|v|) over its nonzeros = the
exponent span the shared BFP block-scale must cover (lower => fewer wasted scale
bits => more value-compressible at a given mantissa width).

CONFOUND: a block with few nonzeros mechanically shows a smaller observed span
(fewer samples => smaller spread), which would *fake* co-regularity for sparse
blocks. We remove it with a PERMUTATION NULL: shuffle the value array across all
nonzero positions (sparsity pattern AND per-block counts fixed) and recompute.
  real_span << null_span  =>  values are genuinely co-regular with position.
  coherence_gain = 1 - sum_w(real_span) / sum_w(null_span)   (nnz-weighted)
Because the shuffle preserves each block's count, the sample-size bias is present
identically in real and null, so any gap is real spatial value organization.

We also stratify blocks by fill to test the stronger "denser => more value-
coherent" form (real/null ratio should drop as fill rises) and aggregate across
matrices (cross-matrix scatter: mat_fill vs coherence_gain).

Run with /opt/conda/bin/python.  Usage: analyze_coregularity.py a.mtx b.mtx ...
Env: COVE_NSHUF (default 4), COVE_OUT (csv/png prefix dir).
"""
import os
import sys
import math
import numpy as np
from scipy.io import mmread

BR, BC = 8, 4
CELLS = BR * BC
NSHUF = int(os.environ.get("COVE_NSHUF", "4"))
OUTDIR = os.environ.get("COVE_OUT", ".")
TAU = 0.20  # dense/sparse split used elsewhere in this repo
FILL_BINS = np.array([0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0001])
RNG = np.random.default_rng(0)


def seg_minmax(values_sorted, seg_start):
    """Per-segment max and min over contiguous runs given start indices."""
    seg_max = np.maximum.reduceat(values_sorted, seg_start)
    seg_min = np.minimum.reduceat(values_sorted, seg_start)
    return seg_max, seg_min


def analyze(path):
    m = mmread(path).tocoo()
    r = m.row.astype(np.int64)
    c = m.col.astype(np.int64)
    v = np.asarray(m.data, dtype=np.float64)
    nnz_struct = r.size

    # value axis is about nonzero magnitudes: drop explicit/zero/non-finite values
    a = np.abs(v)
    keep = np.isfinite(a) & (a > 0.0)
    n_zero = int(nnz_struct - int(keep.sum()))
    r, c, a = r[keep], c[keep], a[keep]
    nnz = a.size
    if nnz == 0:
        return dict(name=None, novalue=True)

    g = np.log2(a)  # exponent of each |value|
    global_span = float(g.max() - g.min())

    nbc = int(c.max() // BC) + 1
    bkey = (r // BR) * np.int64(nbc) + (c // BC)
    order = np.argsort(bkey, kind="stable")
    bkey_s = bkey[order]
    g_s = g[order]
    blk_start = np.flatnonzero(np.concatenate(([True], bkey_s[1:] != bkey_s[:-1])))
    blk_nnz = np.diff(np.concatenate((blk_start, [nnz]))).astype(np.int64)
    num_blocks = blk_start.size
    fill = blk_nnz.astype(np.float64) / CELLS

    # real per-block exponent span
    smax, smin = seg_minmax(g_s, blk_start)
    real_span = smax - smin
    w = blk_nnz.astype(np.float64)
    wsum = w.sum()
    real_mean = float((w * real_span).sum() / wsum)

    # permutation null: shuffle values across all positions, keep block counts
    null_means = []
    bin_real = np.zeros(len(FILL_BINS) - 1)
    bin_null = np.zeros(len(FILL_BINS) - 1)
    bin_w = np.zeros(len(FILL_BINS) - 1)
    bin_idx = np.clip(np.digitize(fill, FILL_BINS) - 1, 0, len(FILL_BINS) - 2)
    for b in range(len(FILL_BINS) - 1):
        sel = bin_idx == b
        bin_w[b] = w[sel].sum()
        bin_real[b] = (w[sel] * real_span[sel]).sum()
    for _ in range(NSHUF):
        g_perm = RNG.permutation(g_s)
        nmax, nmin = seg_minmax(g_perm, blk_start)
        nspan = nmax - nmin
        null_means.append(float((w * nspan).sum() / wsum))
        for b in range(len(FILL_BINS) - 1):
            sel = bin_idx == b
            bin_null[b] += (w[sel] * nspan[sel]).sum()
    null_mean = float(np.mean(null_means))
    bin_null /= NSHUF

    coherence_gain = float(1.0 - real_mean / null_mean) if null_mean > 1e-12 else float("nan")
    frac_nnz_multi = float(w[blk_nnz >= 2].sum() / wsum)
    mat_fill = nnz / (num_blocks * CELLS)

    # confounded within-matrix rank corr over informative (>=2 nnz) blocks
    multi = blk_nnz >= 2
    spear = float("nan")
    if multi.sum() >= 100:
        from scipy.stats import spearmanr
        nm = int(multi.sum())
        if nm > 200000:
            pick = RNG.choice(np.flatnonzero(multi), 200000, replace=False)
        else:
            pick = np.flatnonzero(multi)
        rho, _ = spearmanr(fill[pick], real_span[pick])
        spear = float(rho)

    return dict(
        name=path.split("/")[-1].replace(".mtx", ""),
        novalue=global_span < 1e-9,
        nnz=nnz, n_zero=n_zero, num_blocks=num_blocks, mat_fill=mat_fill,
        frac_nnz_multi=frac_nnz_multi, global_span=global_span,
        real_mean=real_mean, null_mean=null_mean, coherence_gain=coherence_gain,
        spearman=spear,
        bin_real=bin_real, bin_null=bin_null, bin_w=bin_w,
    )


def main():
    paths = sys.argv[1:]
    os.makedirs(OUTDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, "coregularity_canonical25.csv")
    bin_path = os.path.join(OUTDIR, "coregularity_fillbins.csv")
    cf = open(csv_path, "w")
    bf = open(bin_path, "w")
    cf.write("matrix,nnz,n_zero,num_blocks,mat_fill,frac_nnz_multi,global_span,"
             "real_span,null_span,coherence_gain,spearman_fill_span,novalue\n")
    bf.write("matrix,fill_lo,fill_hi,real_span,null_span,ratio,nnz_weight\n")
    hdr = (f"{'matrix':16s} {'nnz':>10s} {'mfill':>6s} {'multi%':>6s} "
           f"{'gspan':>6s} {'real':>6s} {'null':>6s} {'gain':>6s} {'spear':>6s}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = []
    for p in paths:
        nm = p.split("/")[-1].replace(".mtx", "")
        try:
            d = analyze(p)
        except Exception as exc:  # noqa: BLE001
            print(f"{nm:16s} ERROR {exc}", flush=True)
            cf.write(f"{nm},ERROR,{exc}\n"); cf.flush()
            continue
        if d.get("name") is None:
            print(f"{nm:16s} EMPTY (no nonzero values)", flush=True)
            continue
        rows.append(d)
        tag = " [pattern/const]" if d["novalue"] else ""
        print(f"{d['name']:16s} {d['nnz']:10d} {d['mat_fill']:6.3f} "
              f"{100*d['frac_nnz_multi']:6.1f} {d['global_span']:6.1f} "
              f"{d['real_mean']:6.2f} {d['null_mean']:6.2f} "
              f"{d['coherence_gain']:6.3f} {d['spearman']:6.2f}{tag}", flush=True)
        cf.write(f"{d['name']},{d['nnz']},{d['n_zero']},{d['num_blocks']},"
                 f"{d['mat_fill']:.6f},{d['frac_nnz_multi']:.6f},{d['global_span']:.4f},"
                 f"{d['real_mean']:.4f},{d['null_mean']:.4f},{d['coherence_gain']:.4f},"
                 f"{d['spearman']:.4f},{int(d['novalue'])}\n")
        cf.flush()
        for b in range(len(FILL_BINS) - 1):
            if d["bin_w"][b] <= 0:
                continue
            rr = d["bin_real"][b] / d["bin_w"][b]
            nn = d["bin_null"][b] / d["bin_w"][b]
            ratio = rr / nn if nn > 1e-12 else float("nan")
            bf.write(f"{d['name']},{FILL_BINS[b]:.4f},{FILL_BINS[b+1]:.4f},"
                     f"{rr:.4f},{nn:.4f},{ratio:.4f},{d['bin_w'][b]:.0f}\n")
        bf.flush()
    cf.close(); bf.close()

    # aggregate summary
    valued = [d for d in rows if not d["novalue"] and math.isfinite(d["coherence_gain"])]
    if valued:
        print("\n=== AGGREGATE (value-bearing matrices) ===", flush=True)
        dense = [d for d in valued if d["mat_fill"] >= TAU]
        sparse = [d for d in valued if d["mat_fill"] < TAU]
        def agg(group, label):
            if not group:
                return
            # nnz-weighted coherence gain across the group
            tr = sum(d["real_mean"] * d["nnz"] for d in group)
            tn = sum(d["null_mean"] * d["nnz"] for d in group)
            gain = 1 - tr / tn
            gains = [d["coherence_gain"] for d in group]
            print(f"{label:22s} n={len(group):2d}  nnz-wt coherence_gain={gain:6.3f}  "
                  f"median={np.median(gains):6.3f}  range=[{min(gains):.3f},{max(gains):.3f}]",
                  flush=True)
        agg(valued, "ALL")
        agg(dense, "dense (fill>=0.2)")
        agg(sparse, "sparse (fill<0.2)")

    # cross-matrix scatter + fill-binned curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if valued:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            mf = np.array([d["mat_fill"] for d in valued])
            cg = np.array([d["coherence_gain"] for d in valued])
            ax[0].scatter(mf, cg, c=["#c0392b" if x >= TAU else "#2980b9" for x in mf], s=40)
            for d in valued:
                ax[0].annotate(d["name"], (d["mat_fill"], d["coherence_gain"]),
                               fontsize=6, alpha=0.7)
            ax[0].axvline(TAU, ls="--", c="gray", lw=0.8)
            ax[0].set_xlabel("matrix fill (position compressibility)")
            ax[0].set_ylabel("value coherence gain  (1 - real/null span)")
            ax[0].set_title("Co-regularity: dense matrices -> more value-coherent?")
            # fill-binned aggregate ratio across all valued matrices
            tot_real = np.zeros(len(FILL_BINS) - 1)
            tot_null = np.zeros(len(FILL_BINS) - 1)
            for d in valued:
                tot_real += d["bin_real"]
                tot_null += d["bin_null"]
            mids = 0.5 * (FILL_BINS[:-1] + FILL_BINS[1:])
            ok = tot_null > 0
            ax[1].plot(mids[ok], (tot_real[ok] / tot_null[ok]), "o-")
            ax[1].axhline(1.0, ls="--", c="gray", lw=0.8)
            ax[1].set_xlabel("block fill")
            ax[1].set_ylabel("real/null span ratio (lower = more coherent)")
            ax[1].set_title("Denser blocks -> tighter values than chance?")
            fig.tight_layout()
            png = os.path.join(OUTDIR, "coregularity.png")
            fig.savefig(png, dpi=130)
            print(f"\nsaved {png}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})", flush=True)
    print(f"\nwrote {csv_path}\nwrote {bin_path}", flush=True)


if __name__ == "__main__":
    main()
