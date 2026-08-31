#!/usr/bin/env python3
"""Per-block VALUE structure for the 8x4 unit (the value half of co-design).

For each 8x4 block, how many distinct nonzero values does it hold? If most blocks
are const (1) or few-distinct, a per-block local value model (one base + tiny
dict / residual) is a big LOSSLESS win - decoded with no global gather (unlike
the global codebook that was latency-bound). Run with /opt/conda/bin/python.
"""
import sys
import numpy as np
from scipy.io import mmread

BR, BC = 8, 4

def analyze(path):
    m = mmread(path).tocoo()
    r = m.row.astype(np.int64); c = m.col.astype(np.int64); v = m.data.astype(np.float64)
    nnz = r.size
    nbc = int(c.max() // BC) + 1
    bkey = (r // BR) * nbc + (c // BC)
    order = np.lexsort((v, bkey))
    bkey_s = bkey[order]; v_s = v[order]
    ublk, blk_start, blk_nnz = np.unique(bkey_s, return_index=True, return_counts=True)
    nblk = ublk.size
    same_block = np.concatenate(([False], bkey_s[1:] == bkey_s[:-1]))
    same_val = np.concatenate(([False], v_s[1:] == v_s[:-1]))
    is_dup = same_block & same_val
    cum = np.concatenate(([0], np.cumsum(is_dup)))
    ends = np.concatenate((blk_start[1:], [nnz]))
    uniq = blk_nnz - (cum[ends] - cum[blk_start])      # distinct values per block
    fr = lambda cond: 100.0 * np.count_nonzero(cond) / nblk
    raw_bytes = nnz * 8
    # local per-block dict: uniq doubles/block + 1 byte id/nnz
    local_bytes = (uniq * 8 + blk_nnz).sum()
    # global dict: global_u doubles + ceil(log2 global_u) bits/nnz
    import math as _m
    global_u = int(np.unique(v).size)
    gbits = max(1, _m.ceil(_m.log2(max(global_u, 2))))
    global_bytes = global_u * 8 + nnz * gbits / 8.0
    sum_block_u = int(uniq.sum())
    locality = sum_block_u / max(global_u, 1)   # avg #blocks each distinct value spans
    return dict(nblk=nblk, apb=nnz / nblk, const=fr(uniq == 1), le4=fr(uniq <= 4),
                global_u=global_u, locality=locality,
                localratio=100.0 * local_bytes / raw_bytes,
                globalratio=100.0 * global_bytes / raw_bytes)

if __name__ == "__main__":
    print(f"{'matrix':16s} {'const%':>6s} {'<=4%':>6s} {'global_uniq':>11s} "
          f"{'locality(x)':>11s} {'globalDict%':>11s} {'localDict%':>10s}")
    for p in sys.argv[1:]:
        name = p.split("/")[-1].replace(".mtx", "")
        try:
            d = analyze(p)
            print(f"{name:16s} {d['const']:6.1f} {d['le4']:6.1f} {d['global_u']:11d} "
                  f"{d['locality']:11.1f} {d['globalratio']:11.1f} {d['localratio']:10.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:16s} ERROR {exc}")
