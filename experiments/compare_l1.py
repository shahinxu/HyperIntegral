"""Side by side: the proposed method with and without the L1 term.

Reads the l1=0 tree in results/paper and the rerun in results/paper_l1, and
reports every (scene, variant) they share. Read-only.
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

try:
    from scipy import stats as sps
except ImportError:
    sps = None

MAIN_NOISE = 0.02


def load(root, noise=MAIN_NOISE):
    """(suite, scene, variant) -> {seed: macro}"""
    out = defaultdict(dict)
    for f in glob.glob(f"{root}/*/results.jsonl"):
        suite = os.path.basename(os.path.dirname(f)).replace("_neuronal", "")
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("macro_auc") is None:
                continue
            m = r.get("meta") or {}
            if m.get("method") != "integral_pinn_nn":
                continue
            if noise is not None and m.get("noise") != noise:
                continue
            out[(suite, m.get("scene"), m.get("variant"))][m.get("seed")] = r["macro_auc"]
    return out


def fmt(d):
    v = list(d.values())
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.3f} (n=1)"
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f} (n={len(v)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/paper")
    ap.add_argument("--l1", default="results/paper_l1")
    ap.add_argument("--only", default="", help="substring filter on the variant")
    a = ap.parse_args()

    b, l = load(a.base), load(a.l1)
    keys = sorted(set(b) & set(l), key=str)
    if not keys:
        print("no overlapping runs yet")
        return
    print(f"{'suite':12s} {'scene':11s} {'variant':16s} "
          f"{'l1=0':24s} {'l1>0':24s} {'Δ':>7s} {'paired p':>9s}")
    for k in keys:
        if a.only and a.only not in (k[2] or ""):
            continue
        bv, lv = b[k], l[k]
        seeds = sorted(set(bv) & set(lv))
        delta = p = "-"
        if len(seeds) >= 2:
            x = [lv[s] for s in seeds]
            y = [bv[s] for s in seeds]
            delta = f"{st.mean(x) - st.mean(y):+.3f}"
            if sps is not None:
                p = f"{sps.ttest_rel(x, y).pvalue:.4f}"
        print(f"{k[0]:12s} {k[1]:11s} {str(k[2]):16s} "
              f"{fmt(bv):24s} {fmt(lv):24s} {delta:>7s} {p:>9s}")


if __name__ == "__main__":
    main()
