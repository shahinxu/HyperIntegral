"""Compare the Ours rows of the response tables against the recorded runs.

Read-only. It never edits the document: the Ours rows are maintained by hand
and this work is under instruction not to change them. With `--side-by-side`
it prints what each cell would hold if it were filled from the runs, so the
choice can be made with both numbers in view.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

MAIN_NOISE = 0.02
ORDERS = ("2-edges", "3-edges", "4-edges", "5-edges")
SCENE = {"Ecological (n=9)": "ecological", "Social (n=8)": "social",
         "Neuronal (n=30)": "neuronal"}
TABLES = {
    "R2.5a": {"Integral (ours)": "full", "Derivative": "derivative_loss"},
    "R3.7c": {"16": "intervals_16", "32": "intervals_32",
              "64 (Default)": "full", "128": "intervals_128"},
    "R3.7b": {"0.1 T": "window_0.1", "0.25 T": "window_0.25",
              "0.5 T": "full", "1.0 T": "window_1.0"},
    "R3.7a": {"Full method": "full",
              "Derivative PINN (same architecture, cf. Table R2.5a)": "derivative_loss",
              "Without staged λ scheduling": "no_staged",
              "Without gradient clipping (sub-ablation)": "no_clip",
              "Without bounded tensor parameterization": "no_bounded"},
}


def measured(stages=("ablation", "ablation_neuronal")):
    """per-(scene, variant, order) -> {seed: value}, from the ablation suite."""
    d = defaultdict(dict)
    for f in glob.glob("results/paper/*/results.jsonl"):
        if os.path.basename(os.path.dirname(f)) not in stages:
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("macro_auc") is None:
                continue
            m = r["meta"]
            if m.get("noise") != MAIN_NOISE:
                continue
            for k, v in (r.get("auc") or {}).items():
                if v is not None:
                    d[(m.get("scene"), m.get("variant"), k)][m.get("seed")] = v
            d[(m.get("scene"), m.get("variant"), "macro")][m.get("seed")] = r["macro_auc"]
    return d


def fmt(vals):
    if not vals:
        return "-"
    v = list(vals.values())
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def audit(lines, tag, rowmap, data, side_by_side):
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith(f"**Table {tag}"))
    except StopIteration:
        return
    print(f"\n{tag}")
    for l in lines[start:start + 30]:
        if not l.startswith("|"):
            continue
        c = [x.strip() for x in l.strip("|").split("|")]
        if c[0] not in SCENE or c[1] not in rowmap:
            continue
        sc, var = SCENE[c[0]], rowmap[c[1]]
        mok = sok = tot = 0
        printed, measured_cells = [], []
        for k, cellv in zip(ORDERS, c[2:6]):
            mm = re.match(r"([\d.]+) ± ([\d.]+)", cellv)
            vals = data.get((sc, var, k), {})
            printed.append(cellv if mm else "-")
            measured_cells.append(fmt(vals))
            if not mm or len(vals) < 2:
                continue
            tot += 1
            mok += abs(float(mm.group(1)) - round(st.mean(vals.values() and list(vals.values())), 3)) < 1e-9
            sok += abs(float(mm.group(2)) - round(st.stdev(list(vals.values())), 3)) < 1e-9
        if not tot:
            continue
        label = f"{c[0]} / {c[1]}"
        print(f"  {label:52s} mean {mok}/{tot}   sd {sok}/{tot}")
        if side_by_side:
            print(f"    {'printed':10s} " + "  ".join(f"{x:>15s}" for x in printed))
            print(f"    {'measured':10s} " + "  ".join(f"{x:>15s}" for x in measured_cells))


def all_groups():
    """Every (tree, scene, variant, noise) group that holds all four orders.

    Both result trees are searched, `results/rebuttal` included: that is the
    superseded first round, so a row carried over from an earlier experiment
    would still turn up here.
    """
    grp = defaultdict(lambda: defaultdict(dict))
    for f in (glob.glob("results/*/*/results.jsonl")
              + glob.glob("results/*/results.jsonl")):
        parts = f.split("/")
        tree = "/".join(parts[1:3]) if len(parts) > 3 else parts[1]
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r.get("meta") or {}
            auc = r.get("auc") or (r.get("summary") or {}).get("auc") or {}
            k = (tree, m.get("scene"), m.get("variant"), m.get("noise"))
            for o in ORDERS:
                if auc.get(o) is not None:
                    grp[k][o][m.get("seed")] = auc[o]
    out = []
    for k, od in grp.items():
        if all(o in od and od[o] for o in ORDERS):
            out.append((k, [round(st.mean(list(od[o].values())), 3) for o in ORDERS]))
    return out


def provenance(lines, tag, pred, label, groups):
    """Can a whole printed row be reproduced by one group of runs?

    Cell by cell this question is meaningless - three-decimal values collide
    by chance, and with a few hundred groups most values match something. All
    four orders of a row must therefore come from the same group.
    """
    start = next(i for i, l in enumerate(lines) if l.startswith(f"**Table {tag}"))
    tot = hit = 0
    for l in lines[start:start + 26]:
        if not l.startswith("|"):
            continue
        c = [x.strip() for x in l.strip("|").split("|")]
        if not pred(c):
            continue
        want = []
        for cv in c[2:6]:
            mm = re.match(r"([\d.]+) ± ", cv)
            if mm:
                want.append(float(mm.group(1)))
        if len(want) != 4:
            continue
        tot += 1
        found = [k for k, v in groups if v == want]
        if found:
            hit += 1
            print(f"    {c[0]} / {c[1]}  <- {found[0]}")
    print(f"  {label}: {hit}/{tot} rows reproduced")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", nargs="?", default="REBUTTAL_RESPONSE.md")
    ap.add_argument("--side-by-side", action="store_true",
                    help="also print what each cell would hold if filled "
                         "from the runs")
    a = ap.parse_args()
    lines = open(a.doc).read().splitlines()
    data = measured()
    for tag, rowmap in TABLES.items():
        audit(lines, tag, rowmap, data, a.side_by_side)

    print("\nWhole-row provenance, searched across every result tree")
    groups = all_groups()
    print(f"  candidate groups holding all four orders: {len(groups)}")
    provenance(lines, "R2.5a", lambda c: c[1] == "Derivative",
               "R2.5a Derivative (control, not an Ours row)", groups)
    provenance(lines, "R2.5a", lambda c: c[1] == "Integral (ours)",
               "R2.5a Integral (ours)", groups)
    provenance(lines, "R3.7c",
               lambda c: c[0].startswith(("Ecological", "Social", "Neuronal")),
               "R3.7c (every row is ours)", groups)


if __name__ == "__main__":
    main()
