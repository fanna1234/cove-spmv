#!/usr/bin/env python3
"""Summarize value-frequency concentration: does codebook-the-frequent-head help?"""
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "structural/results/studies/value_freq_concentration_2026-06-09.csv"
rows = list(csv.DictReader(open(path)))
fl = lambda r, k: float(r[k])
it = lambda r, k: int(r[k])
n = len(rows)
print("=== %d matrices ===" % n)

low = [r for r in rows if it(r, "cardinality") <= 256]
high = [r for r in rows if it(r, "cardinality") > 256]
print("low-card (<=256, dict already exact): %d  |  high-card (>256, bfp8 territory): %d" % (len(low), len(high)))
print()


def stat(items, key):
    vs = sorted(fl(r, key) for r in items)
    if not vs:
        return
    print("  %-12s min=%5.1f  median=%5.1f  mean=%5.1f  max=%5.1f" %
          (key, vs[0], vs[len(vs) // 2], sum(vs) / len(vs), vs[-1]))


print("HIGH-card matrices -- top-K %-of-nnz coverage (concentrated => freq-codebook helps):")
for k in ["cov_top16", "cov_top64", "cov_top256", "cov_top1024"]:
    stat(high, k)
print()

c90 = [r for r in high if fl(r, "cov_top256") >= 90]
c50 = [r for r in high if fl(r, "cov_top256") >= 50]
print("HIGH-card with top256 >= 90%% of nnz (freq-layered CLEAR win): %d/%d" % (len(c90), len(high)))
print("HIGH-card with top256 >= 50%% of nnz (partial win):           %d/%d" % (len(c50), len(high)))
print()
print("HIGH-card + CONCENTRATED (top256>=90), best first:")
for r in sorted(c90, key=lambda r: -fl(r, "cov_top256"))[:12]:
    print("  %-24s card=%9s top1=%5s top16=%5s top256=%6s" %
          (r["name"], r["cardinality"], r["cov_top1"], r["cov_top16"], r["cov_top256"]))
print()
print("HIGH-card + SPREAD (top256<50, freq-codebook NO help):")
for r in sorted([r for r in high if fl(r, "cov_top256") < 50], key=lambda r: fl(r, "cov_top256"))[:10]:
    print("  %-24s card=%9s top256=%6s top1024=%6s" %
          (r["name"], r["cardinality"], r["cov_top256"], r["cov_top1024"]))
