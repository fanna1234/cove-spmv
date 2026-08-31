#!/usr/bin/env python3
"""Run the ported CSR5 baseline (bhSPARSE Benchmark_SpMV_using_CSR5, double) over a
matrix list and record per-matrix CSR5 conversion + SpMV time, for the head-to-head.

CSR5 is the load-balance standard. We do NOT claim to beat it (spec section 6); this run
quantifies where CSR5 wins (irregular matrices) vs where COVE wins (large/dense).
Schema is self-contained so it can be merged with the 7-config baseline table.
"""
import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

RE_NNZ = re.compile(r"nnz\s*=\s*(\d+)")
RE_CONV = re.compile(r"CSR->CSR5 time\s*=\s*([0-9.eE+-]+)\s*ms")
RE_SPMV = re.compile(r"CSR5-based SpMV time\s*=\s*([0-9.eE+-]+)\s*ms")
RE_GF = re.compile(r"CSR5-based SpMV.*GFlops\s*=\s*([0-9.eE+-]+)")
RE_BW = re.compile(r"CSR5-based SpMV.*Bandwidth\s*=\s*([0-9.eE+-]+)")
RE_CHECK = re.compile(r"Check\.\.\.\s*(PASS|FAIL)", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--matrix-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    data = Path(a.data_dir)
    names = [l.strip() for l in Path(a.matrix_list).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    if a.limit:
        names = names[:a.limit]
    names = names[a.shard::a.nshards]

    cols = ["matrix", "nnz", "csr5_convert_ms", "csr5_spmv_ms", "csr5_gflops",
            "csr5_bandwidth", "check", "status"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, name in enumerate(names):
            m = data / name
            base = Path(name).name
            row = {"matrix": base, "status": "missing"}
            if m.exists():
                try:
                    p = subprocess.run([a.bin, str(m)], text=True,
                                       capture_output=True, timeout=a.timeout)
                    o = p.stdout + p.stderr
                    def g(rx):
                        mm = rx.search(o)
                        return mm.group(1) if mm else ""
                    spmv = g(RE_SPMV)
                    row = {
                        "matrix": base,
                        "nnz": g(RE_NNZ),
                        "csr5_convert_ms": g(RE_CONV),
                        "csr5_spmv_ms": spmv,
                        "csr5_gflops": g(RE_GF),
                        "csr5_bandwidth": g(RE_BW),
                        "check": (RE_CHECK.search(o).group(1).upper()
                                  if RE_CHECK.search(o) else ""),
                        "status": "ok" if (p.returncode == 0 and spmv) else f"fail:{p.returncode}",
                    }
                except subprocess.TimeoutExpired:
                    row = {"matrix": base, "status": "timeout"}
            w.writerow(row)
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"[csr5 shard{a.shard}] {i + 1}/{len(names)} {time.time()-t0:.0f}s",
                      flush=True)
    print(f"[csr5 shard{a.shard}] DONE {len(names)} in {time.time()-t0:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
