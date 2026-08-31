#!/usr/bin/env python3
"""Run the JOINT operator (hybrid position dispatch + per-block value codec, together)
over a matrix list, at scale.

This is the operator the paper calls COVE innovation (1): one fused load-balanced kernel
that applies the hybrid fused-LB structure AND a value codec in the same pass. The binary
self-times both the joint kernel and the fp64 hybrid in one invocation, so each row carries
the stacking ratio (joint vs the tuned fp64 hybrid) directly.

Output schema matches cove_joint_canonical25_2026-06-09.csv (+ a `codec` column so several
codecs can share one file). One process handles a stride shard (names[shard::nshards]);
launch N processes for N-way parallelism on a single GPU.
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path


def parse_kv(stdout):
    d = {}
    for line in stdout.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--matrix-list", required=True)
    ap.add_argument("--codecs", default="bfp8,bfp8_outlier,auto")
    ap.add_argument("--long-row-split", default="256")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--warmup", default="5")
    ap.add_argument("--iters", default="30")
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
    codecs = [c for c in a.codecs.split(",") if c]

    cols = ["matrix_id", "matrix", "matrix_bytes", "codec", "nnz",
            "joint_ms", "hybrid_fp64_ms",
            "joint_speedup_vs_hybrid", "bytes_per_nnz", "rel_err", "pass_1e-2",
            "joint_value_codec", "status"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, name in enumerate(names):
            m = data / name
            base = Path(name).name
            if not m.exists():
                for c in codecs:
                    w.writerow({"matrix_id": name, "matrix": base,
                                "codec": c, "status": "missing"})
                f.flush()
                continue
            for c in codecs:
                cmd = [a.bin, str(m), "--mode", "fused_lb",
                       "--long-row-split", a.long_row_split,
                       "--value-codec", c, "--warmup", a.warmup, "--iters", a.iters]
                if a.verify:
                    cmd.append("--verify-cpu")
                try:
                    p = subprocess.run(cmd, text=True, capture_output=True,
                                       timeout=a.timeout)
                    d = parse_kv(p.stdout)
                    status = "ok" if p.returncode == 0 else f"fail:{p.returncode}"
                except subprocess.TimeoutExpired:
                    d, status = {}, "timeout"
                w.writerow({
                    "matrix_id": name,
                    "matrix": base,
                    "matrix_bytes": m.stat().st_size,
                    "codec": c,
                    "nnz": d.get("nnz", ""),
                    "joint_ms": d.get("joint_min_ms", ""),
                    "hybrid_fp64_ms": d.get("joint_hybrid_fp64_min_ms", ""),
                    "joint_speedup_vs_hybrid": d.get("joint_speedup_vs_hybrid_fp64", ""),
                    "bytes_per_nnz": d.get("joint_bytes_per_nnz", ""),
                    "rel_err": d.get("joint_rel_err", ""),
                    "pass_1e-2": d.get("joint_verify_pass_1e-2", ""),
                    "joint_value_codec": d.get("joint_value_codec", ""),
                    "status": status,
                })
                f.flush()
            if (i + 1) % 25 == 0:
                print(f"[shard{a.shard}] {i + 1}/{len(names)} {time.time() - t0:.0f}s",
                      flush=True)
    print(f"[shard{a.shard}] DONE {len(names)} mat x {len(codecs)} codecs "
          f"in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
