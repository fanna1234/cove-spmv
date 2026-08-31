#!/usr/bin/env python3
"""Merge sharded family-sweep CSVs and summarize the COVE complete-core results.
Usage: cove_scale_summary.py shard0.csv shard1.csv ...   (or a glob)"""
import csv, math, sys
rows=[]
for p in sys.argv[1:]:
    try:
        rows+=list(csv.DictReader(open(p)))
    except FileNotFoundError: pass
CU="cusparse_csr"
ops=["original_lb","hybrid_lb","bfp8_lb","bfp8_outlier_lb","bfp4_lb","bf16_lb","fp16_lb","auto_lossless_lb"]
def gm(xs):
    xs=[v for v in xs if v and v>0]
    return math.exp(sum(map(math.log,xs))/len(xs)) if xs else float("nan")
def F(r,k):
    try: return float(r[k])
    except: return None
idx={(r["matrix"],r["operator_name"]):r for r in rows}
mats=sorted({r["matrix"] for r in rows})
print(f"total rows={len(rows)} matrices={len(mats)}")
print(f"{'operator':18s} {'gmean_x/cusp':>12s} {'wins/ok':>10s} {'verifyPASS':>11s} {'medB/nnz':>9s}")
for nm in ops+[CU]:
    sp=[];wins=0;ok=0;pas=0;ver=0;bpn=[]
    for m in mats:
        r=idx.get((m,nm)); cu=idx.get((m,CU))
        if not r or r.get("status")!="ok": continue
        ok+=1
        mm=F(r,"min_ms"); cm=F(cu,"min_ms") if cu else None
        if mm and cm: sp.append(cm/mm); wins+=(mm<cm)
        v=r.get("verify_cpu")
        if v in ("PASS","FAIL"): ver+=1; pas+=(v=="PASS")
        b=F(r,"value_bytes_per_nnz")
        if b: bpn.append(b)
    bpn.sort(); med=bpn[len(bpn)//2] if bpn else float("nan")
    print(f"{nm:18s} {gm(sp):>11.3f}x {f'{wins}/{ok}':>10s} {(f'{pas}/{ver}' if ver else '-'):>11s} {med:>9.3f}")
# hybrid regime: wins among dense vs sparse not available without fill; just report win fraction
hy=[(F(idx[(m,'hybrid_lb')],'min_ms'), F(idx.get((m,CU),{}),'min_ms'), m) for m in mats if (m,'hybrid_lb') in idx and idx[(m,'hybrid_lb')].get('status')=='ok']
hsp=[(cm/mm,m) for mm,cm,m in hy if mm and cm]
print(f"\nhybrid_lb: geomean {gm([s for s,_ in hsp]):.3f}x over {len(hsp)} matrices; wins {sum(1 for s,_ in hsp if s>1)}/{len(hsp)} ({100*sum(1 for s,_ in hsp if s>1)/max(1,len(hsp)):.0f}%)")
