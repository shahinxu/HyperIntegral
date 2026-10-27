"""Report how much of each response table is actually filled.

Only data rows count: the header row is skipped explicitly, because several
headers contain numbers (h=0.1 T, sigma=0.00) that a naive scan mistakes for
results.
"""
import re
import sys
from pathlib import Path

TAGS = ["R1.4", "R1.5a", "R1.5b", "R2.5a", "R2.5b", "R2.5c", "R2.6a",
        "R2.6b", "R2.6c", "R3.6a", "R3.6b", "R3.7a", "R3.7b", "R3.7c"]
NAMES = {
    "R1.4": "vs regression baselines", "R1.5a": "vs explicit sparsity",
    "R1.5b": "L1 weight sweep", "R2.5a": "integral vs derivative",
    "R2.5b": "HyperPINN config audit", "R2.5c": "sigma x h grid",
    "R2.6a": "run-to-run variability", "R2.6b": "tuning protocol",
    "R2.6c": "best config per method", "R3.6a": "CI + significance",
    "R3.6b": "noise sensitivity", "R3.7a": "component ablations",
    "R3.7b": "integral window length", "R3.7c": "number of intervals",
}


def bounds(lines, tag):
    s = next(i for i, l in enumerate(lines) if l.lstrip().startswith(f"**Table {tag}"))
    e = s + 1
    while e < len(lines) and not lines[e].lstrip().startswith("**Table") \
            and not lines[e].startswith(("## ", "### ")):
        e += 1
    return s, e


def main():
    doc = Path(sys.argv[1] if len(sys.argv) > 1 else "REBUTTAL_RESPONSE.md")
    lines = doc.read_text().splitlines()
    tf = tt = 0
    for tag in TAGS:
        try:
            s, e = bounds(lines, tag)
        except StopIteration:
            continue
        rows = [l for l in lines[s:e]
                if l.startswith("|") and not re.match(r"^\|[\s\-:|]+\|$", l)]
        rows = rows[1:]                      # drop the header row
        filled = sum(1 for l in rows
                     if any(re.search(r"\d\.\d", c)
                            for c in l.strip().strip("|").split("|")[2:]))
        total = len(rows)
        tf += filled; tt += total
        pct = filled / total if total else 0
        bar = "█" * round(pct * 20) + "░" * (20 - round(pct * 20))
        mark = "  <- done" if total and filled == total else ""
        print(f"{tag:<6} {NAMES[tag]:<28} {bar} {filled:>3}/{total:<3}{mark}")
    pct = tf / tt if tt else 0
    print("-" * 72)
    print(f"{'TOTAL':<35} {'█' * round(pct*20) + '░' * (20-round(pct*20))} {tf:>3}/{tt:<3} ({pct:.0%})")


if __name__ == "__main__":
    main()
