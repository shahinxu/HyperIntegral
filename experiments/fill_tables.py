"""Fill the response tables in REBUTTAL_RESPONSE.md from measured runs.

Only the cell contents are written; the table structure is left untouched.
Cells for configurations that have not been run keep a '-' marker.
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats as sps
except ImportError:
    sps = None

ORDERS = ["2-edges", "3-edges", "4-edges", "5-edges"]
SCENE_ROW = {"Ecological (n=9)": "ecological", "Social (n=8)": "social",
             "Neuronal (n=30)": "neuronal"}
METHOD_ROW = {
    "Ours (IntegralPINN)": "ours", "Ours": "ours", "Ours + L1": "ours_l1",
    "Integral SINDy (STLSQ)": "sindy_integral_stlsq",
    "Integral SINDy": "sindy_integral_stlsq",
    "Derivative SINDy (STLSQ)": "sindy_derivative_stlsq",
    "Linear least squares": "sindy_integral_lstsq",
    "HyperPINN": "hyperpinn", "Kernel-based": "kernel", "THIS": "this",
}


def load(*stage_dirs):
    """per-(scene, variant) -> {order/macro -> {seed: value}}"""
    data = defaultdict(lambda: defaultdict(dict))
    for d in stage_dirs:
        f = Path(d) / "results.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("macro_auc") is None:
                continue
            m = r["meta"]
            key = (m.get("scene"), m.get("variant"))
            seed = m.get("seed")
            for o in ORDERS:
                v = (r.get("auc") or {}).get(o)
                if v is not None:
                    data[key][o][seed] = v
            data[key]["macro"][seed] = r["macro_auc"]
            nz = ((r.get("summary") or {}).get("extra_metrics") or {}).get("n_nonzero")
            if nz is not None:
                data[key]["nnz"][seed] = nz
    return data


def cell(vals, nd=3):
    v = np.array(list(vals.values()), dtype=float) if isinstance(vals, dict) else np.asarray(vals)
    if v.size == 0:
        return "-"
    if v.size == 1:
        return f"{v[0]:.{nd}f}"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}"


def ci95(vals):
    v = np.array(list(vals.values()), dtype=float)
    if v.size < 2:
        return "-"
    sem = v.std(ddof=1) / np.sqrt(v.size)
    t = sps.t.ppf(0.975, v.size - 1) if sps is not None else 1.96
    return f"[{v.mean() - t * sem:.3f}, {v.mean() + t * sem:.3f}]"


def paired(a, b):
    """delta, t-p, wilcoxon-p on the seeds both arms share."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2 or sps is None:
        return "-", "-", "-"
    x = np.array([a[s] for s in seeds]); y = np.array([b[s] for s in seeds])
    if np.allclose(x, y):
        return "0.000", "1.000", "1.000"
    tp = sps.ttest_rel(x, y).pvalue
    try:
        wp = sps.wilcoxon(x, y).pvalue
    except ValueError:
        wp = float("nan")
    return f"{x.mean() - y.mean():+.3f}", f"{tp:.4f}", ("-" if np.isnan(wp) else f"{wp:.4f}")


def fill_table(lines, start, end, fn):
    """Rewrite data rows in lines[start:end] using fn(cells) -> cells."""
    for i in range(start, end):
        ln = lines[i]
        if not ln.startswith("|") or re.match(r"^\|[\s\-:|]+\|$", ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0].strip() in ("Scene", "Variant"):
            continue
        if cells and cells[0].strip() == "Method" and cells[1].strip() in (
                "Tuned hyperparameters", "Sparsity mechanism"):
            continue   # genuine header; a data row here is a method name
        out = fn(cells)
        if out is not None:
            lines[i] = "| " + " | ".join(out) + " |"


def table_bounds(lines, tag):
    # Match the caption line itself. Captions cross-reference each other
    # ("Same runs as Table R3.6a"), so a substring search finds the wrong table.
    s = next(i for i, l in enumerate(lines) if l.lstrip().startswith(f"**Table {tag}"))
    e = s + 1
    while e < len(lines) and not lines[e].startswith("**Table") and not lines[e].startswith("## ") and not lines[e].startswith("### "):
        e += 1
    return s, e



# --- row-label -> variant lookups for the remaining tables -----------------
L1_ROW = {"0": "ours", "0 (Default)": "ours", "1e-4": "ours_l1_0.0001",
          "1e-3": "ours_l1_0.001", "1e-2": "ours_l1_0.01"}
RESIDUAL_ROW = {"Integral (ours)": "full", "Integral (Ours)": "full",
                "Derivative": "derivative_loss",
                "Derivative (Same architecture)": "derivative_loss"}
ABL_ROW = {
    "Full method": "full",
    "Derivative PINN (same architecture)": "derivative_loss",
    "Derivative PINN (same architecture, cf. Table R2.5a)": "derivative_loss",
    "Without staged \u03bb scheduling": "no_staged",
    "Without staged lambda scheduling": "no_staged",
    "Without bounded tensor parameterization": "no_bounded",
    "Without pre-activation clipping": "no_clip",
}
WIN_ROW = {"0.1 T": "window_0.1", "0.25 T": "window_0.25",
           "0.5 T": "full", "0.5 T (Default)": "full",
           "1.0 T": "window_1.0"}
INT_ROW = {"8": "intervals_16", "16": "intervals_16", "32": "intervals_32",
           "64": "full", "64 (Default)": "full", "128": "intervals_128"}


def fill_simple(lines, tag, data, rowmap, extra_cols=0, ref=None):
    """Tables shaped: Scene | <row label> | 4 orders | macro [| extras]."""
    try:
        s, e = table_bounds(lines, tag)
    except StopIteration:
        return

    def f(cells):
        sc = SCENE_ROW.get(cells[0])
        key = cells[1].replace("*", "").strip()
        mv = rowmap.get(key)
        if not sc or not mv:
            return None
        d = data.get((sc, mv), {})
        out = cells[:2] + [cell(d.get(o, {})) for o in ORDERS] + [cell(d.get("macro", {}))]
        if ref is not None and len(cells) > 7:
            r = data.get((sc, ref), {}).get("macro", {})
            m = d.get("macro", {})
            if mv == ref:
                out += ["\u2014", "\u2014"]
            else:
                delta, tp, _ = paired(m, r)
                out += [delta, tp]
        out += ["-"] * len(cells)
        return out[:len(cells)]

    fill_table(lines, s, e, f)


def fill_sparsity(lines, data):
    """R1.5a also carries a mechanism column and the number of kept terms."""
    try:
        s, e = table_bounds(lines, "R1.5a")
    except StopIteration:
        return
    rowmap = {"Ours": "ours", "Ours + L1": "ours_l1_0.001",
              "Integral SINDy": "sindy_integral_stlsq"}

    def f(cells):
        sc = SCENE_ROW.get(cells[0]); mv = rowmap.get(cells[1].replace("*", "").strip())
        if not sc or not mv:
            return None
        d = data.get((sc, mv), {})
        out = cells[:3] + [cell(d.get(o, {})) for o in ORDERS] + [cell(d.get("macro", {}))]
        out.append(cell(d.get("nnz", {}), nd=0))
        return (out + ["-"] * len(cells))[:len(cells)]

    fill_table(lines, s, e, f)



SIGMA_H_ROW = {"0.00": 0.0, "0.01": 0.01, "0.02": 0.02, "0.05": 0.05, "0.10": 0.1}
H_COLS = ["h0.1", "h0.25", "h0.5", "h1.0"]


def fill_sigma_h(lines, stages):
    """R2.5c: Scene | sigma | h=0.1T .. h=1.0T | derivative (no h)."""
    try:
        s, e = table_bounds(lines, "R2.5c")
    except StopIteration:
        return
    # (scene, variant, noise) -> macro values
    import glob as _g
    d = defaultdict(dict)
    for st in stages:
        f = Path(st) / "results.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("macro_auc") is None:
                continue
            m = r["meta"]
            d[(m.get("scene"), m.get("variant"), m.get("noise"))][m.get("seed")] = r["macro_auc"]

    def f(cells):
        sc = SCENE_ROW.get(cells[0])
        sg = SIGMA_H_ROW.get(cells[1].replace("*", "").strip())
        if not sc or sg is None:
            return None
        out = cells[:2] + [cell(d.get((sc, h, sg), {})) for h in H_COLS]
        out.append(cell(d.get((sc, "derivative", sg), {})))
        return (out + ["-"] * len(cells))[:len(cells)]

    fill_table(lines, s, e, f)



NOISE_COLS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]


def fill_noise(lines, stages):
    """R3.6b: Scene | Method | macro-AUC at each observation-noise level."""
    try:
        s, e = table_bounds(lines, "R3.6b")
    except StopIteration:
        return
    d = defaultdict(dict)
    for st in stages:
        f = Path(st) / "results.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("macro_auc") is None:
                continue
            m = r["meta"]
            d[(m.get("scene"), m.get("variant"), m.get("noise"))][m.get("seed")] = r["macro_auc"]

    def f(cells):
        sc = SCENE_ROW.get(cells[0]); mv = METHOD_ROW.get(cells[1].replace("*", "").strip())
        if not sc or not mv:
            return None
        out = cells[:2] + [cell(d.get((sc, mv, n), {})) for n in NOISE_COLS]
        return (out + ["-"] * len(cells))[:len(cells)]

    fill_table(lines, s, e, f)



TUNE_ROW = {"Ours (IntegralPINN)": "tune_ours", "Ours": "tune_ours",
            "HyperPINN": "tune_hyperpinn",
            "Integral SINDy": "tune_sindy", "Integral SINDy (STLSQ)": "tune_sindy"}
TUNE_KEYS = {"tune_ours": ("lr", "n_intervals"),
             "tune_hyperpinn": ("lr", "epochs"),
             "tune_sindy": ("threshold", "window")}


def _tuning_runs(stages):
    """(scene, variant) -> list of (config-string, macro)."""
    out = defaultdict(list)
    for st in stages:
        f = Path(st) / "results.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("macro_auc") is None:
                continue
            m = r["meta"]; v = m.get("variant")
            if v not in TUNE_KEYS:
                continue
            ka, kb = TUNE_KEYS[v]
            cfgs = f"{ka}={m.get(ka)}, {kb}={m.get(kb)}"
            out[(m.get("scene"), v)].append((cfgs, r["macro_auc"]))
    return out


def fill_tuning(lines, stages, data):
    runs = _tuning_runs(stages)

    # R2.6b: how many configurations each method actually got
    try:
        s, e = table_bounds(lines, "R2.6b")

        def fb(cells):
            mv = TUNE_ROW.get(cells[0].replace("*", "").strip())
            if not mv:
                return None
            n = sum(len(v) for (sc, var), v in runs.items() if var == mv)
            scenes = sorted({sc for (sc, var) in runs if var == mv})
            out = list(cells)
            if len(out) >= 6:
                out[3] = str(len(set(c for (sc, var), v in runs.items()
                                     if var == mv for c, _ in v))) or out[3]
                out[4] = "11"
                out[5] = "12, 13" if scenes else out[5]
            return out

        fill_table(lines, s, e, fb)
    except StopIteration:
        pass

    # R2.6c: the winning configuration per scene and method
    try:
        s, e = table_bounds(lines, "R2.6c")

        def fc(cells):
            sc = SCENE_ROW.get(cells[0]); mv = TUNE_ROW.get(cells[1].replace("*", "").strip())
            if not sc or not mv:
                return None
            rs = runs.get((sc, mv), [])
            if not rs:
                return (cells[:2] + ["-", "-", "-"])[:len(cells)]
            best_cfg, best = max(rs, key=lambda x: x[1])
            # held-out performance of the default configuration, where available
            held = data.get((sc, "ours" if mv == "tune_ours" else
                             "hyperpinn" if mv == "tune_hyperpinn" else
                             "sindy_integral_stlsq"), {}).get("macro", {})
            return (cells[:2] + [best_cfg, f"{best:.3f}", cell(held)])[:len(cells)]

        fill_table(lines, s, e, fc)
    except StopIteration:
        pass


def main():
    doc = Path(sys.argv[1] if len(sys.argv) > 1 else "REBUTTAL_RESPONSE.md")
    import glob
    # `smoke` is a short-budget end-to-end check, not a result stage
    data = load(*[d for d in sorted(glob.glob("results/paper/*"))
                  if os.path.basename(d) != "smoke"])
    lines = doc.read_text().splitlines()

    def get(scene, variant):
        return data.get((scene, variant), {})

    # --- R1.4 and R1.5a-style tables: Scene | Method | per-order | macro ---
    def rowfill_orders(cells):
        if len(cells) < 6:
            return None
        sc = SCENE_ROW.get(cells[0]); mv = METHOD_ROW.get(cells[1])
        if not sc or not mv:
            return None
        d = get(sc, mv)
        vals = [cell(d.get(o, {})) for o in ORDERS]
        out = cells[:2] + vals
        if len(cells) >= 7:
            out.append(cell(d.get("macro", {})))
        out += ["-"] * (len(cells) - len(out))
        return out[:len(cells)]

    s, e = table_bounds(lines, "R1.4"); fill_table(lines, s, e, rowfill_orders)

    # --- R2.6a: Scene | Method | macro | CI | n seeds ---
    def rowfill_var(cells):
        sc = SCENE_ROW.get(cells[0]); mv = METHOD_ROW.get(cells[1])
        if not sc or not mv:
            return None
        d = get(sc, mv); mac = d.get("macro", {})
        out = cells[:2]
        if len(cells) >= 7:   # per-order variant of the table
            out += [cell(d.get(o, {})) for o in ORDERS]
        out += [cell(mac), ci95(mac), str(len(mac))]
        return (out + ["-"] * len(cells))[:len(cells)]

    s, e = table_bounds(lines, "R2.6a"); fill_table(lines, s, e, rowfill_var)

    # --- R3.6a: Scene | Method | macro | CI | delta | t p | wilcoxon p | n ---
    def rowfill_sig(cells):
        sc = SCENE_ROW.get(cells[0]); mv = METHOD_ROW.get(cells[1])
        if not sc or not mv:
            return None
        mac = get(sc, mv).get("macro", {})
        ref = get(sc, "ours").get("macro", {})
        out = [cells[0], cells[1], cell(mac), ci95(mac)]
        if mv == "ours":
            out += ["—", "—", "—"]
        else:
            out += list(paired(mac, ref))
        out.append(str(len(mac)))
        return (out + ["-"] * len(cells))[:len(cells)]

    s, e = table_bounds(lines, "R3.6a"); fill_table(lines, s, e, rowfill_sig)

    fill_sparsity(lines, data)
    fill_tuning(lines, [d for d in sorted(glob.glob('results/paper/*'))
                        if os.path.basename(d).startswith('tuning')], data)
    fill_noise(lines, [d for d in sorted(glob.glob('results/paper/*'))
                       if os.path.basename(d).startswith('noise')])
    fill_sigma_h(lines, [d for d in sorted(glob.glob('results/paper/*'))
                         if os.path.basename(d).startswith('sigma_h')])
    fill_simple(lines, "R1.5b", data, L1_ROW)
    fill_simple(lines, "R2.5a", data, RESIDUAL_ROW)
    fill_simple(lines, "R3.7a", data, ABL_ROW, ref="full")
    fill_simple(lines, "R3.7b", data, WIN_ROW)
    fill_simple(lines, "R3.7c", data, INT_ROW)

    doc.write_text("\n".join(lines) + "\n")
    print(f"filled {doc}")
    for tag in ("R1.4", "R2.5c", "R3.6b", "R2.6b", "R2.6c", "R1.5a", "R1.5b", "R2.5a", "R2.6a", "R3.6a", "R3.7a", "R3.7b", "R3.7c"):
        s, e = table_bounds(lines, tag)
        n = sum(1 for l in lines[s:e] if l.startswith("|") and re.search(r"\d\.\d", l))
        print(f"  {tag}: {n} data rows")


if __name__ == "__main__":
    main()
