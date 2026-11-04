"""Compare the Ours rows of R2.5a and R3.7c against the recorded runs.

Reports, per cell, whether the printed mean and standard deviation can be
reproduced from results/paper. Read-only: it never edits the document.
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

MAIN_NOISE = 0.02
ORDERS = ("2-edges", "3-edges", "4-edges", "5-edges")
SCENE = {"Ecological (n=9)": "ecological", "Social (n=8)": "social",
         "Neuronal (n=30)": "neuronal"}
R375C_ROW = {"16": "intervals_16", "32": "intervals_32",
             "64 (Default)": "full", "128": "intervals_128"}
R25A_ROW = {"Integral (ours)": "full", "Derivative": "derivative_loss"}


def measured():
    d = defaultdict(dict)
    for f in glob.glob("results/paper/*/results.jsonl"):
        if os.path.basename(os.path.dirname(f)) == "smoke":
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
    return d


def audit(lines, tag, rowmap, data):
    start = next(i for i, l in enumerate(lines) if l.startswith(f"**Table {tag}"))
    print(f"\n{tag}")
    for l in lines[start:start + 24]:
        if not l.startswith("|"):
            continue
        c = [x.strip() for x in l.strip("|").split("|")]
        if c[0] not in SCENE or c[1] not in rowmap:
            continue
        mok = sok = tot = 0
        for k, cellv in zip(ORDERS, c[2:6]):
            mm = re.match(r"([\d.]+) ± ([\d.]+)", cellv)
            vals = list(data.get((SCENE[c[0]], rowmap[c[1]], k), {}).values())
            if not mm or len(vals) < 2:
                continue
            tot += 1
            mok += abs(float(mm.group(1)) - round(st.mean(vals), 3)) < 1e-9
            sok += abs(float(mm.group(2)) - round(st.stdev(vals), 3)) < 1e-9
        if tot:
            print(f"  {c[0] + ' / ' + c[1]:36s} mean {mok}/{tot}   sd {sok}/{tot}")


def main():
    doc = sys.argv[1] if len(sys.argv) > 1 else "REBUTTAL_RESPONSE.md"
    lines = open(doc).read().splitlines()
    data = measured()
    audit(lines, "R2.5a", R25A_ROW, data)
    audit(lines, "R3.7c", R375C_ROW, data)


if __name__ == "__main__":
    main()
