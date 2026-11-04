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

# SKIP_OURS: the proposed method's rows are maintained by hand; this filler
# only writes baseline rows so it never overwrites them.
SELECTION_SEED = 11
MAIN_OF_TUNE = {"tune_ours": "ours", "tune_hyperpinn": "hyperpinn",
                "tune_kernel": "kernel", "tune_this": "this",
                "tune_sindy": "sindy_integral_stlsq"}

OURS_VARIANTS = {
    "ours", "full", "derivative_loss", "no_staged", "no_bounded", "no_clip",
    "tune_ours", "derivative",
}


def _is_ours(variant: str) -> bool:
    v = (variant or "")
    return (v in OURS_VARIANTS or v.startswith("ours")
            or v.startswith("window_") or v.startswith("intervals_")
            or v.startswith("h0.") or v.startswith("h1."))


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
INT_ROW = {"16": "intervals_16", "32": "intervals_32",
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
        if not sc or not mv or _is_ours(mv):
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
        if not sc or not mv or _is_ours(mv):
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
            "HyperPINN": "tune_hyperpinn", "Kernel-based": "tune_kernel",
            "THIS": "tune_this",
            "Integral SINDy": "tune_sindy", "Integral SINDy (STLSQ)": "tune_sindy"}
TUNE_KEYS = {"tune_ours": ("lr", "n_intervals"),
             "tune_hyperpinn": ("lr", "epochs"),
             "tune_kernel": ("sigma_k", "lam"),
             "tune_this": ("lambda1", "rho"),
             "tune_sindy": ("threshold", "window")}
# how each tuned knob is written in the table text
TUNE_LABEL = {"sigma_k": r"$\sigma_k$", "lam": r"$\lambda$",
              "lambda1": r"$\lambda_1$", "rho": r"$\rho$"}


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
            cfgs = (f"{TUNE_LABEL.get(ka, ka)}={m.get(ka)}, "
                    f"{TUNE_LABEL.get(kb, kb)}={m.get(kb)}")
            out[(m.get("scene"), v)].append((cfgs, r["macro_auc"]))
    return out


def fill_tuning(lines, stages, data):
    runs = _tuning_runs(stages)

    # R2.6b: how many configurations each method actually got
    try:
        s, e = table_bounds(lines, "R2.6b")

        def fb(cells):
            mv = TUNE_ROW.get(cells[0].replace("*", "").strip())
            if not mv or _is_ours(mv):
                return None   # the proposed method's rows are maintained by hand
            n = sum(len(v) for (sc, var), v in runs.items() if var == mv)
            scenes = sorted({sc for (sc, var) in runs if var == mv})
            out = list(cells)
            if len(out) >= 6:
                out[3] = str(len(set(c for (sc, var), v in runs.items()
                                     if var == mv for c, _ in v))) or out[3]
                out[4] = str(SELECTION_SEED)
                held_seeds = sorted({sd for (sc, var), v in data.items()
                                     if var == MAIN_OF_TUNE.get(mv)
                                     for sd in v.get("macro", {})
                                     if sd is not None and sd != SELECTION_SEED})
                if held_seeds:
                    out[5] = (f"{held_seeds[0]}-{held_seeds[-1]}"
                              if held_seeds == list(range(held_seeds[0], held_seeds[-1] + 1))
                              else ", ".join(str(x) for x in held_seeds))
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
            main_variant = {"tune_ours": "ours", "tune_hyperpinn": "hyperpinn",
                            "tune_kernel": "kernel", "tune_this": "this",
                            "tune_sindy": "sindy_integral_stlsq"}.get(mv)
            held = data.get((sc, main_variant), {}).get("macro", {})
            # seed 11 selects the configuration, so it is not held out
            held = {k: v for k, v in held.items() if k != SELECTION_SEED}
            return (cells[:2] + [best_cfg, f"{best:.3f}", cell(held)])[:len(cells)]

        fill_table(lines, s, e, fc)
    except StopIteration:
        pass



# Configurations that will never produce a number, annotated so the cell is not
# mistaken for a run that is merely missing.
NOT_APPLICABLE = {
    # THIS hangs indefinitely on the 30-oscillator scene: it emits no output and
    # never terminates, so it has no result there.
    ("neuronal", "this"): "not run (THIS does not terminate at n=30)",
}
NOT_TUNED = {"Kernel-based", "THIS"}


def annotate_unavailable(lines):
    """Write an explicit reason into cells that cannot be filled."""
    # "not tuned" only makes sense in the tuning table; elsewhere an empty
    # kernel/THIS row means the run has not produced a result yet
    try:
        tune_lo, tune_hi = table_bounds(lines, "R2.6c")
    except StopIteration:
        tune_lo, tune_hi = -1, -1
    for i, ln in enumerate(lines):
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("Scene", "Method", "Variant"):
            continue
        sc = SCENE_ROW.get(cells[0])
        mv = METHOD_ROW.get(cells[1])
        body = cells[2:]
        if any(re.search(r"\d\.\d", c) for c in body):
            continue
        reason = None
        if sc and mv and (sc, mv) in NOT_APPLICABLE:
            reason = NOT_APPLICABLE[(sc, mv)]
        elif cells[1] in NOT_TUNED and tune_lo <= i < tune_hi:
            if all(c in ("", "-") for c in body):
                reason = "not tuned"
        if reason:
            lines[i] = "| " + " | ".join(cells[:2] + [reason] + ["-"] * (len(cells) - 3)) + " |"



OURS_LABEL = re.compile(r"\bOurs\b|Full method|Derivative PINN|Without staged|"
                        r"Without bounded|Without pre-activation", re.I)


def restore_ours_rows(before, after):
    """Put back every row that belongs to the proposed method.

    Those rows are maintained by hand, so the filler must never rewrite them.
    Guarding each writer individually kept missing cases (tables keyed by
    window length, interval count or sigma have no method column at all), so
    this reverts them wholesale as a final pass.
    """
    OURS_ONLY_TABLES = ("R1.5a", "R1.5b", "R2.5a", "R2.5c", "R3.7a", "R3.7b", "R3.7c")
    spans = []
    for tag in OURS_ONLY_TABLES:
        try:
            spans.append(table_bounds(before, tag))
        except StopIteration:
            pass
    # iterate over the full length: zip() would drop the final line whenever
    # the two lists differ in length, and that line sits inside a table
    for i in range(min(len(before), len(after))):
        b, a = before[i], after[i]
        if b == a:
            continue
        in_ours_table = any(lo <= i < hi for lo, hi in spans)
        if in_ours_table or OURS_LABEL.search(b):
            after[i] = b
    return after


def main():
    doc = Path(sys.argv[1] if len(sys.argv) > 1 else "REBUTTAL_RESPONSE.md")
    import glob
    # `smoke` is a short-budget end-to-end check, not a result stage
    data = load(*[d for d in sorted(glob.glob("results/paper/*"))
                  if os.path.basename(d) != "smoke"])
    lines = doc.read_text().splitlines()
    original = list(lines)

    def get(scene, variant):
        return data.get((scene, variant), {})

    # --- R1.4 and R1.5a-style tables: Scene | Method | per-order | macro ---
    def rowfill_orders(cells):
        if len(cells) < 6:
            return None
        sc = SCENE_ROW.get(cells[0]); mv = METHOD_ROW.get(cells[1])
        if _is_ours(mv):
            return None   # the proposed method's rows are maintained by hand
        if not sc or not mv or _is_ours(mv):
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
        if _is_ours(mv):
            return None   # the proposed method's rows are maintained by hand
        if not sc or not mv or _is_ours(mv):
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
        if _is_ours(mv):
            return None   # the proposed method's rows are maintained by hand
        if not sc or not mv or _is_ours(mv):
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

    # R1.5a/b, R2.5a, R2.5c, R3.7a/b/c are entirely the proposed method
    # and are maintained by hand; only baseline-bearing tables are written.
    fill_tuning(lines, [d for d in sorted(glob.glob('results/paper/*'))
                        if os.path.basename(d).startswith('tuning')], data)
    fill_noise(lines, [d for d in sorted(glob.glob('results/paper/*'))
                       if os.path.basename(d).startswith('noise')])
    # R2.5c, R3.7a/b/c and R1.5b are entirely the proposed method: left untouched.

    annotate_unavailable(lines)
    lines = restore_ours_rows(original, lines)
    doc.write_text("\n".join(lines) + "\n")
    print(f"filled {doc}")
    for tag in ("R1.4", "R2.5c", "R3.6b", "R2.6b", "R2.6c", "R1.5a", "R1.5b", "R2.5a", "R2.6a", "R3.6a", "R3.7a", "R3.7b", "R3.7c"):
        s, e = table_bounds(lines, tag)
        n = sum(1 for l in lines[s:e] if l.startswith("|") and re.search(r"\d\.\d", l))
        print(f"  {tag}: {n} data rows")


if __name__ == "__main__":
    main()
