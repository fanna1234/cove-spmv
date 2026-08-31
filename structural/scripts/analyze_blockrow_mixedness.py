#!/usr/bin/env python3
"""Per-block-row fill (internal mixedness) analysis for the 8x4 BitBSR layout.

Decides whether whole-matrix BitBSR-vs-CSR dispatch suffices, or whether
per-block-row hybrid (bitmap+CSR) storage is justified.

For each matrix it reports the distribution of per-block-row fill
(= nnz_in_block_row / (blocks_in_block_row * 32)) and, crucially, the fraction
of nonzeros that whole-matrix dispatch would "mis-serve" (put on the side of the
fill crossover tau that disagrees with the block-row's own best format).
Small mis-served% across matrices => whole-matrix dispatch is near-optimal and
hybrid storage is not worth its complexity.
"""
import sys
import numpy as np
from scipy.io import mmread

BR, BC = 8, 4
CELLS = BR * BC
TAU = 0.20  # empirical fill crossover: BitBSR wins above, cuSPARSE below


def analyze(path):
    m = mmread(path).tocoo()
    r = m.row.astype(np.int64)
    c = m.col.astype(np.int64)
    nnz = r.size
    brow = r // BR
    bcol = c // BC
    n_brow = int(brow.max()) + 1
    span = int(bcol.max()) + 1
    nnz_per_br = np.bincount(brow, minlength=n_brow)
    key = brow * np.int64(span) + bcol
    ukey = np.unique(key)
    ub_brow = (ukey // np.int64(span)).astype(np.int64)
    blocks_per_br = np.bincount(ub_brow, minlength=n_brow)
    nz = blocks_per_br > 0
    fill = np.zeros(n_brow)
    fill[nz] = nnz_per_br[nz] / (blocks_per_br[nz] * CELLS)
    total_blocks = int(blocks_per_br.sum())
    mat_fill = nnz / (total_blocks * CELLS)
    mat_dense = mat_fill >= TAU
    dense_br = nz & (fill >= TAU)
    frac_br_dense = dense_br.sum() / max(1, int(nz.sum()))
    frac_nnz_dense = nnz_per_br[dense_br].sum() / nnz
    # mis-served nnz under one decision for the whole matrix
    misserved = (1.0 - frac_nnz_dense) if mat_dense else frac_nnz_dense
    p = np.percentile(fill[nz], [10, 50, 90])
    return dict(nnz=nnz, mat_fill=mat_fill,
                cls=("dense" if mat_dense else "sparse"),
                p10=p[0], p50=p[1], p90=p[2],
                br_dense=100 * frac_br_dense,
                nnz_dense=100 * frac_nnz_dense,
                misserved=100 * misserved)


if __name__ == "__main__":
    print(f"{'matrix':18s} {'nnz':>9s} {'mfill':>6s} {'cls':>6s} "
          f"{'p10':>5s} {'p50':>5s} {'p90':>5s} {'brDense%':>8s} "
          f"{'nnzDense%':>9s} {'misserv%':>8s}")
    for path in sys.argv[1:]:
        name = path.split("/")[-1].replace(".mtx", "")
        try:
            d = analyze(path)
            print(f"{name:18s} {d['nnz']:9d} {d['mat_fill']:6.3f} {d['cls']:>6s} "
                  f"{d['p10']:5.2f} {d['p50']:5.2f} {d['p90']:5.2f} "
                  f"{d['br_dense']:8.1f} {d['nnz_dense']:9.1f} {d['misserved']:8.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:18s} ERROR {exc}")
