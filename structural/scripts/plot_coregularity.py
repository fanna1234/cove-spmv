#!/usr/bin/env python3
"""Regenerate the co-regularity figure with labels consistent with the paper:
le10gib denominator, rho=+0.27 read as WEAK-BUT-POSITIVE coupling (the basis of
"Co-Regular"), not "orthogonal". Usage: plot_coregularity.py <csv> <out.png>"""
import sys, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fill, gain = [], []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        g = r.get("gain_i8", "")
        if g not in ("", "NA", "nan", "None"):
            try:
                fill.append(float(r["mat_fill"])); gain.append(float(g))
            except ValueError:
                pass
n = len(gain)
pos = sum(1 for b in gain if b > 0)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); rk = [0] * len(v)
        for i, idx in enumerate(order):
            rk[idx] = i
        return rk
    rx, ry = rank(x), rank(y); m = len(x)
    mx, my = sum(rx) / m, sum(ry) / m
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(m))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(m)) *
                    sum((ry[i] - my) ** 2 for i in range(m)))
    return num / den if den else 0.0


rho = spearman(fill, gain)
med = sorted(gain)[n // 2]

fig, ax1 = plt.subplots(1, 1, figsize=(5.4, 4.0))  # single panel -> fits one column
cols = ['#c0392b' if b > 0 else '#2980b9' for b in gain]
ax1.scatter(fill, gain, s=7, c=cols, alpha=0.5, linewidths=0)
ax1.axhline(0, color='k', lw=0.8)
ax1.set_ylim(-1.05, 1.05)  # clip rare extreme anti-co-regular outliers for readability
ax1.set_xlabel("matrix fill  (position compressibility →)")
ax1.set_ylabel("value coherence  gain = 1 − real/null")
ax1.set_title("Co-regularity (le10gib, n=%d): Spearman +%.2f, %d%% co-regular" %
              (n, rho, round(100 * pos / n)))
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=150, bbox_inches='tight')
print("n=%d pos=%d (%.0f%%) rho=+%.3f median=%.3f -> %s" %
      (n, pos, 100 * pos / n, rho, med, sys.argv[2]))
