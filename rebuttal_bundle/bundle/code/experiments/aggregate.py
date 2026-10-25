"""Aggregate sweep results into reviewer-facing tables.

Reads one or more ``results.jsonl`` files produced by ``experiments.sweep``
and reports, for every (variant, scene, noise) cell:

  * mean +- standard deviation over seeds
  * a 95% confidence interval for the mean (Student-t)
  * a paired comparison against a reference variant (paired t-test and the
    distribution-free Wilcoxon signed-rank test), computed on the per-seed
    pairs so that the shared data seed is controlled for

Output is a Markdown file ready to paste into the response letter, plus a
tidy CSV of the per-run numbers.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats as sps
except ImportError:  # keep the aggregator usable without scipy
    sps = None


def load_rows(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("macro_auc") is None:
                continue
            meta = r.get("meta", {})
            rows.append({
                "variant": meta.get("variant", meta.get("method", "?")),
                "method": meta.get("method", "?"),
                "scene": meta.get("scene", "?"),
                "n_nodes": meta.get("n_nodes"),
                "noise": meta.get("noise"),
                "n_samples": meta.get("n_samples"),
                "seed": meta.get("seed"),
                "macro_auc": float(r["macro_auc"]),
                "auc": r.get("auc", {}),
                "seconds": r.get("seconds"),
                "job": r.get("job"),
            })
    return rows


def ci95(vals):
    """95% CI half-width for the mean. Falls back to a normal quantile if
    scipy is unavailable; returns nan for a single observation."""
    n = len(vals)
    if n < 2:
        return float("nan")
    sem = np.std(vals, ddof=1) / math.sqrt(n)
    tcrit = sps.t.ppf(0.975, n - 1) if sps is not None else 1.96
    return float(tcrit * sem)


def cell_key(r, group_by):
    return tuple(r[k] for k in group_by)


def summarize(rows, group_by):
    cells = defaultdict(list)
    for r in rows:
        cells[cell_key(r, group_by)].append(r)
    out = {}
    for key, rs in cells.items():
        vals = np.array([r["macro_auc"] for r in rs], dtype=float)
        out[key] = {
            "n": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "ci95": ci95(vals),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "by_seed": {r["seed"]: r["macro_auc"] for r in rs},
        }
    return out


def paired_test(a_by_seed, b_by_seed):
    """Paired comparison on seeds present in both arms."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if len(seeds) < 2 or sps is None:
        return None
    a = np.array([a_by_seed[s] for s in seeds])
    b = np.array([b_by_seed[s] for s in seeds])
    if np.allclose(a, b):
        return {"n_pairs": len(seeds), "delta": 0.0, "t_p": 1.0, "wilcoxon_p": 1.0}
    res = {"n_pairs": len(seeds), "delta": float((a - b).mean())}
    res["t_p"] = float(sps.ttest_rel(a, b).pvalue)
    try:
        res["wilcoxon_p"] = float(sps.wilcoxon(a, b).pvalue)
    except ValueError:
        res["wilcoxon_p"] = float("nan")
    return res


def fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def stars(p):
    if p is None or math.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def write_markdown(summary, group_by, reference, out_path, title):
    lines = [f"# {title}", ""]
    # Group rows by everything except the variant so each table is one setting.
    var_idx = group_by.index("variant")
    settings = defaultdict(dict)
    for key, stat in summary.items():
        setting = tuple(v for i, v in enumerate(key) if i != var_idx)
        settings[setting][key[var_idx]] = stat

    setting_names = [g for g in group_by if g != "variant"]
    for setting, variants in sorted(settings.items(), key=lambda kv: str(kv[0])):
        label = ", ".join(f"{n}={v}" for n, v in zip(setting_names, setting))
        lines += [f"## {label}", "",
                  "| Variant | n seeds | Macro-AUC (mean ± std) | 95% CI | Δ vs ref | paired t p | Wilcoxon p | sig |",
                  "|---|---|---|---|---|---|---|---|"]
        ref_stat = variants.get(reference)
        for vname, stat in sorted(variants.items(), key=lambda kv: -kv[1]["mean"]):
            cmp = (paired_test(stat["by_seed"], ref_stat["by_seed"])
                   if ref_stat is not None and vname != reference else None)
            lo = stat["mean"] - stat["ci95"] if not math.isnan(stat["ci95"]) else float("nan")
            hi = stat["mean"] + stat["ci95"] if not math.isnan(stat["ci95"]) else float("nan")
            ci_txt = "n/a" if math.isnan(lo) else f"[{lo:.4f}, {hi:.4f}]"
            mark = " **(ref)**" if vname == reference else ""
            lines.append(
                f"| {vname}{mark} | {stat['n']} | {stat['mean']:.4f} ± {stat['std']:.4f} | {ci_txt} | "
                f"{fmt(cmp['delta']) if cmp else '—'} | {fmt(cmp['t_p'], 4) if cmp else '—'} | "
                f"{fmt(cmp['wilcoxon_p'], 4) if cmp else '—'} | {stars(cmp['t_p']) if cmp else '—'} |"
            )
        lines.append("")
    if sps is None:
        lines.append("> scipy not available: CIs use a normal quantile and no tests were run.")
    Path(out_path).write_text("\n".join(lines))
    return out_path


def write_csv(rows, out_path):
    order_keys = sorted({k for r in rows for k in (r["auc"] or {})})
    fields = ["job", "variant", "method", "scene", "n_nodes", "noise", "n_samples",
              "seed", "macro_auc", "seconds"] + order_keys
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fields if k not in order_keys}
            for k in order_keys:
                row[k] = (r["auc"] or {}).get(k)
            w.writerow(row)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="results.jsonl file(s) from experiments.sweep")
    ap.add_argument("--group_by", nargs="+",
                    default=["variant", "scene", "n_nodes", "noise"])
    ap.add_argument("--reference", default="full",
                    help="Variant name used as the paired-comparison reference.")
    ap.add_argument("--title", default="Experimental results")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    rows = load_rows(args.results)
    if not rows:
        raise SystemExit("No completed runs found in the given results files.")
    print(f"Loaded {len(rows)} completed runs")
    summary = summarize(rows, args.group_by)
    write_markdown(summary, args.group_by, args.reference, args.out_md, args.title)
    print(f"Wrote {args.out_md}")
    if args.out_csv:
        write_csv(rows, args.out_csv)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
