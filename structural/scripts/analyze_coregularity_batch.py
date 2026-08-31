#!/usr/bin/env python3
"""Co-regularity PREVALENCE sweep over a SuiteSparse denominator (parallel).

Scales the null-controlled co-regularity metric from canonical25 to a full
paper denominator, to show co-regularity is a prevalent, predictable property
(not cherry-picked) and to produce the headline taxonomy figure (fill ×
value-coherence).

Per matrix: gain_i8 = 1 - relerr_real/relerr_null for the BFP int8 per-block
max-scale codec (relerr = Σ|v-v̂|/Σ|v|); NULL permutes |v| across positions
(pattern + per-block counts fixed). Same for int4. Also mat_fill, nnz, gspan,
SuiteSparse group. Parallel (multiprocessing), resumable, size-capped+logged.

Usage: analyze_coregularity_batch.py LIST DATA_ROOT OUT_CSV [P=48] [MAXGB=8]
Run with /opt/conda/bin/python (needs numpy+scipy).
"""
import os
import sys
import csv
import numpy as np
from scipy.io import mmread
from multiprocessing import Pool

BR, BC, CELLS, NSHUF = 8, 4, 32, 3
ROOT = ""
MAXGB = 8.0
FIELDS = ["matrix", "status", "group", "nnz", "num_blocks", "mat_fill", "gspan",
          "real_i8", "null_i8", "gain_i8", "real_i4", "null_i4", "gain_i4"]


def _relerr(a_s, blk_start, blk_nnz, levels, perm=None):
    v = a_s if perm is None else a_s[perm]
    smax = np.maximum.reduceat(v, blk_start)
    scale = np.repeat(np.where(smax > 0, smax, 1.0), blk_nnz)
    q = np.clip(np.rint(v / scale * levels), -levels, levels)
    return float(np.abs(v - q * scale / levels).sum() / v.sum())


def one(relpath):
    path = os.path.join(ROOT, relpath)
    group = relpath.split("/")[0]
    base = {"matrix": relpath, "group": group}
    try:
        sz = os.path.getsize(path)
    except OSError:
        return {**base, "status": "missing"}
    if sz > MAXGB * (1 << 30):
        return {**base, "status": f"skip_size_{sz >> 20}MB"}
    try:
        m = mmread(path).tocoo()
        r = m.row.astype(np.int64)
        c = m.col.astype(np.int64)
        a = np.abs(np.asarray(m.data, dtype=np.float64))
        keep = np.isfinite(a) & (a > 0.0)
        r, c, a = r[keep], c[keep], a[keep]
        nnz = a.size
        if nnz == 0:
            return {**base, "status": "empty"}
        gspan = float(np.log2(a.max()) - np.log2(a.min())) if nnz > 1 else 0.0
        nbc = int(c.max() // BC) + 1
        bkey = (r // BR) * np.int64(nbc) + (c // BC)
        order = np.argsort(bkey, kind="stable")
        a_s = a[order]
        bkey_s = bkey[order]
        blk_start = np.flatnonzero(np.concatenate(([True], bkey_s[1:] != bkey_s[:-1])))
        blk_nnz = np.diff(np.concatenate((blk_start, [nnz]))).astype(np.int64)
        out = {**base, "status": "ok", "nnz": nnz, "num_blocks": int(blk_start.size),
               "mat_fill": round(nnz / (blk_start.size * CELLS), 5), "gspan": round(gspan, 3)}
        if gspan < 1e-9:
            out["status"] = "pattern"
            return out
        rng = np.random.default_rng(0)
        for tag, lv in (("i8", 127), ("i4", 7)):
            rr = _relerr(a_s, blk_start, blk_nnz, lv)
            nn = float(np.mean([_relerr(a_s, blk_start, blk_nnz, lv, rng.permutation(nnz))
                                for _ in range(NSHUF)]))
            out[f"real_{tag}"] = round(rr, 6)
            out[f"null_{tag}"] = round(nn, 6)
            out[f"gain_{tag}"] = round(1 - rr / nn, 4) if nn > 1e-12 else ""
        return out
    except Exception as e:  # noqa: BLE001
        return {**base, "status": f"error:{type(e).__name__}:{str(e)[:60]}"}


def main():
    global ROOT, MAXGB
    listf, ROOT, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    P = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    MAXGB = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
    rels = [x.strip() for x in open(listf) if x.strip()]

    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done.add(row["matrix"])
    todo = [r for r in rels if r not in done]
    new_file = not os.path.exists(out_csv)
    print(f"total={len(rels)} done={len(done)} todo={len(todo)} P={P} MAXGB={MAXGB}", flush=True)

    f = open(out_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", restval="")
    if new_file:
        w.writeheader()
    n_ok = n_pat = n_err = 0
    with Pool(P) as pool:
        for i, res in enumerate(pool.imap_unordered(one, todo, chunksize=4), 1):
            w.writerow(res)
            st = res.get("status", "")
            if st == "ok":
                n_ok += 1
            elif st == "pattern":
                n_pat += 1
            elif st not in ("ok", "pattern"):
                n_err += 1
            if i % 50 == 0 or i == len(todo):
                f.flush()
                print(f"  {i}/{len(todo)}  ok={n_ok} pattern={n_pat} other={n_err}", flush=True)
    f.close()
    print(f"DONE → {out_csv}", flush=True)


if __name__ == "__main__":
    main()
