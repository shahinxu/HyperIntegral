"""Experiment sweep driver for the rebuttal experiments.

Runs a named suite of jobs (method x scene x seed x setting) across the
available GPUs, then collects every run's ``summary.json`` into a single
``results.jsonl`` for aggregation.

Suites
------
main      : proposed method vs all baselines, multiple seeds, multiple noise levels
ablation  : integral-vs-derivative, staged lambda, bounded parametrization,
            integral window length, number of sampled intervals
sigma_h   : joint sweep over observation noise (sigma) and integration step (h)
tuning    : equal-budget hyperparameter search for the baselines

Usage
-----
    python -m experiments.sweep --suite ablation --seeds 1 2 3 --gpus 2 3 \\
        --out results/rebuttal/ablation
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Rossler presets are selected through an environment variable.
ROSSLER_PRESETS = {8: "n8", 16: "n16", 32: "n32", 64: "n64", 100: "n100"}


def rossler_env(n_nodes):
    env = dict(os.environ)
    preset = ROSSLER_PRESETS.get(n_nodes)
    if preset:
        env["ROSSLER_HYPERGRAPH_PRESET"] = preset
    return env


class Job:
    def __init__(self, name, module, cli, meta, env=None):
        self.name = name
        self.module = module
        self.cli = cli
        self.meta = meta
        self.env = env or dict(os.environ)

    def command(self, gpu_id, out_root):
        cli = list(self.cli) + [
            "--gpu_id", str(gpu_id),
            "--results_root", str(out_root),
            "--run_tag", self.name,
        ]
        return [PYTHON, "-m", self.module] + cli


# --------------------------------------------------------------------------
# Suite definitions
# --------------------------------------------------------------------------

def _integral_job(name, scene, n_nodes, max_order, n_samples, noise, seed,
                  n_epochs, lr, extra=(), meta_extra=None, cfg_max_cand=0):
    cli = ["--scene", scene, "--n_samples", str(n_samples), "--noise", str(noise),
           "--seed", str(seed), "--n_epochs", str(n_epochs), "--lr", str(lr)]
    if scene == "rossler":
        cli += ["--n_nodes", str(n_nodes), "--max_order", str(max_order)]
    if cfg_max_cand:
        cli += ["--max_candidates_per_order", str(cfg_max_cand)]
    cli += list(extra)
    meta = {"method": "integral_pinn", "scene": scene, "n_nodes": n_nodes,
            "noise": noise, "seed": seed, "n_samples": n_samples}
    meta.update(meta_extra or {})
    return Job(name, "models.integral.train", cli, meta, rossler_env(n_nodes))


def _sindy_job(name, scene, n_nodes, max_order, n_samples, noise, seed,
               form, estimator, threshold, window, meta_extra=None, cfg_max_cand=0):
    cli = ["--scene", scene, "--n_samples", str(n_samples), "--noise", str(noise),
           "--seed", str(seed), "--form", form, "--estimator", estimator,
           "--threshold", str(threshold), "--window", str(window)]
    if scene == "rossler":
        cli += ["--n_nodes", str(n_nodes), "--max_order", str(max_order)]
    if cfg_max_cand:
        cli += ["--max_candidates_per_order", str(cfg_max_cand)]
    meta = {"method": f"sindy_{form}_{estimator}", "scene": scene,
            "n_nodes": n_nodes, "noise": noise, "seed": seed,
            "n_samples": n_samples, "threshold": threshold, "window": window}
    meta.update(meta_extra or {})
    return Job(name, "models.sindy.train", cli, meta, rossler_env(n_nodes))


def suite_main(cfg):
    jobs = []
    for scene, n_nodes, max_order in cfg["scenes"]:
        for noise in cfg["noises"]:
            for seed in cfg["seeds"]:
                base = f"{scene}{n_nodes}_noise{noise}_seed{seed}"
                jobs.append(_integral_job(
                    f"ours_{base}", scene, n_nodes, max_order, cfg["n_samples"],
                    noise, seed, cfg["n_epochs"], cfg["lr"],
                    meta_extra={"variant": "ours"}, cfg_max_cand=cfg["max_cand"]))
                for form, est in (("integral", "stlsq"), ("derivative", "stlsq"),
                                  ("integral", "lstsq")):
                    jobs.append(_sindy_job(
                        f"sindy_{form}_{est}_{base}", scene, n_nodes, max_order,
                        cfg["n_samples"], noise, seed, form, est,
                        cfg["sindy_threshold"], cfg["sindy_window"],
                        meta_extra={"variant": f"sindy_{form}_{est}"},
                        cfg_max_cand=cfg["max_cand"]))
    return jobs


def suite_ablation(cfg):
    """One switch flipped at a time, against the full method."""
    variants = {
        "full":              [],
        "derivative_loss":   ["--loss_mode", "derivative"],
        "no_bounded":        ["--bounded", "0"],
        "staged_lambda":     ["--lambda_schedule", "staged"],
        "window_0.1":        ["--window_frac", "0.1"],
        "window_0.25":       ["--window_frac", "0.25"],
        "window_0.5":        ["--window_frac", "0.5"],
        "intervals_8":       ["--n_intervals", "8"],
        "intervals_64":      ["--n_intervals", "64"],
        "intervals_128":     ["--n_intervals", "128"],
    }
    jobs = []
    for scene, n_nodes, max_order in cfg["scenes"]:
        for noise in cfg["noises"]:
            for seed in cfg["seeds"]:
                for vname, extra in variants.items():
                    jobs.append(_integral_job(
                        f"{vname}_{scene}{n_nodes}_noise{noise}_seed{seed}",
                        scene, n_nodes, max_order, cfg["n_samples"], noise, seed,
                        cfg["n_epochs"], cfg["lr"], extra,
                        meta_extra={"variant": vname}))
    return jobs


def suite_sigma_h(cfg):
    """Theorems 1-2 support: error vs observation noise sigma and step size h.

    h is controlled through n_samples: a coarser simulation grid means a
    larger effective integration step for the quadrature.
    """
    jobs = []
    for scene, n_nodes, max_order in cfg["scenes"]:
        for n_samples in cfg["sigma_h_samples"]:
            for noise in cfg["sigma_h_noises"]:
                for seed in cfg["seeds"]:
                    name = f"ours_{scene}{n_nodes}_T{n_samples}_sigma{noise}_seed{seed}"
                    jobs.append(_integral_job(
                        name, scene, n_nodes, max_order, n_samples, noise, seed,
                        cfg["n_epochs"], cfg["lr"],
                        meta_extra={"variant": "ours", "sigma": noise, "h_n_samples": n_samples}))
    return jobs


def suite_tuning(cfg):
    """Equal-budget hyperparameter search, reported so that the baselines
    cannot be dismissed as under-tuned."""
    jobs = []
    seed = cfg["seeds"][0]
    for scene, n_nodes, max_order in cfg["scenes"]:
        for noise in cfg["noises"]:
            # Baseline grid: sparsity threshold x integral window.
            for thr in [1e-4, 1e-3, 1e-2, 1e-1]:
                for win in [5, 20, 50, 100]:
                    jobs.append(_sindy_job(
                        f"tune_sindy_thr{thr}_win{win}_{scene}{n_nodes}_noise{noise}",
                        scene, n_nodes, max_order, cfg["n_samples"], noise, seed,
                        "integral", "stlsq", thr, win,
                        meta_extra={"variant": "tuning_sindy"}))
            # Our grid of the same size: learning rate x number of intervals.
            for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
                for ni in [8, 32, 64, 128]:
                    jobs.append(_integral_job(
                        f"tune_ours_lr{lr}_ni{ni}_{scene}{n_nodes}_noise{noise}",
                        scene, n_nodes, max_order, cfg["n_samples"], noise, seed,
                        cfg["n_epochs"], lr, ["--n_intervals", str(ni)],
                        meta_extra={"variant": "tuning_ours", "lr": lr, "n_intervals": ni}))
    return jobs


def suite_diagnostic(cfg):
    """Why does the integral residual not beat the derivative residual?

    Hypothesis: with Phi and f evaluated at the noisy state, errors-in-variables
    in the regressor dominate the noise in the matched increment, so changing
    the residual operator cannot help.  Two controls test this:

      oracle_regressor : Phi and f built from the noise-free trajectory, so the
                         only remaining noise is on the residual's left-hand
                         side -- the regime the integral form is meant to fix.
      no_resample      : stay on the native (coarse) observation grid instead of
                         interpolating onto 500 points, so the finite-difference
                         step is large and derivative estimation is genuinely
                         ill-posed.

    Each cell is run with both loss modes so the integral-vs-derivative gap can
    be read off directly.
    """
    controls = {
        "plain":          [],
        "oracle":         ["--oracle_regressor", "1"],
        "coarse":         ["--no_resample", "1"],
        "oracle_coarse":  ["--oracle_regressor", "1", "--no_resample", "1"],
    }
    jobs = []
    for scene, n_nodes, max_order in cfg["scenes"]:
        for n_samples in cfg["diag_samples"]:
            for noise in cfg["noises"]:
                for seed in cfg["seeds"]:
                    for cname, cextra in controls.items():
                        for loss in ("integral", "derivative"):
                            name = (f"{cname}_{loss}_{scene}{n_nodes}_T{n_samples}"
                                    f"_noise{noise}_seed{seed}")
                            jobs.append(_integral_job(
                                name, scene, n_nodes, max_order, n_samples, noise,
                                seed, cfg["n_epochs"], cfg["lr"],
                                cextra + ["--loss_mode", loss],
                                meta_extra={"variant": f"{cname}_{loss}",
                                            "control": cname, "loss_mode": loss,
                                            "n_samples_grid": n_samples}))
    return jobs


def suite_sparsity(cfg):
    """R?-5, second half: how does the proposed parametrization compare with
    explicitly sparsity-promoting estimation, and does adding an explicit L1
    penalty to our objective help?"""
    jobs = []
    for scene, n_nodes, max_order in cfg["scenes"]:
        for noise in cfg["noises"]:
            for seed in cfg["seeds"]:
                base = f"{scene}{n_nodes}_noise{noise}_seed{seed}"
                for l1 in cfg["l1_weights"]:
                    vname = "ours" if l1 == 0 else f"ours_l1_{l1}"
                    extra = [] if l1 == 0 else ["--l1", str(l1)]
                    jobs.append(_integral_job(
                        f"{vname}_{base}", scene, n_nodes, max_order, cfg["n_samples"],
                        noise, seed, cfg["n_epochs"], cfg["lr"], extra,
                        meta_extra={"variant": vname, "l1": l1},
                        cfg_max_cand=cfg["max_cand"]))
                jobs.append(_sindy_job(
                    f"sindy_integral_stlsq_{base}", scene, n_nodes, max_order,
                    cfg["n_samples"], noise, seed, "integral", "stlsq",
                    cfg["sindy_threshold"], cfg["sindy_window"],
                    meta_extra={"variant": "sindy_integral_stlsq"},
                    cfg_max_cand=cfg["max_cand"]))
    return jobs



# --- paper baselines (HyperPINN family) -----------------------------------
# cp / ITC_0 / ITC_sup only support max_order <= 3, so any comparison that
# includes them has to be run at max_order 3.
HYPERPINN_METHODS = {
    "hyperpinn":              {"scene_arg": "scene", "python": True,  "max_order": 5},
    "hyperpinn_base":         {"scene_arg": "preset", "python": True,  "max_order": 5},
    "hyperpinn_cp":           {"scene_arg": "preset", "python": True,  "max_order": 3},
    "hyperpinn_tucker":       {"scene_arg": "preset", "python": False, "max_order": 5},
    "hyperpinn_tensor_train": {"scene_arg": "preset", "python": False, "max_order": 5},
    "hyperpinn_ITC_0":        {"scene_arg": "preset", "python": True,  "max_order": 3},
    "hyperpinn_ITC_sup":      {"scene_arg": "preset", "python": True,  "max_order": 3},
}


def _hyperpinn_job(name, method, n_nodes, max_order, n_samples, noise, seed,
                   n_epochs, lr, extra=(), meta_extra=None):
    spec = HYPERPINN_METHODS[method]
    cli = ["--n_samples", str(n_samples), "--noise", str(noise),
           "--max_order", str(max_order), "--n_epochs", str(n_epochs),
           "--lr", str(lr), "--seed", str(seed)]
    if spec["scene_arg"] == "scene":
        cli += ["--scene", "rossler", "--n_nodes", str(n_nodes)]
    else:
        cli += ["--preset", f"n{n_nodes}"]
    if spec["python"]:
        cli += ["--python", PYTHON]
    cli += list(extra)
    meta = {"method": method, "scene": "rossler", "n_nodes": n_nodes,
            "noise": noise, "seed": seed, "n_samples": n_samples,
            "max_order": max_order}
    meta.update(meta_extra or {})
    return Job(name, f"models.{method}.run", cli, meta, rossler_env(n_nodes))


def suite_baselines(cfg):
    """The paper's own baselines against our method, matched budget.

    Run at max_order 3 by default because the low-rank variants (cp, ITC) cap
    there; running everything at the same order is what makes the comparison
    legitimate.
    """
    jobs = []
    mo = cfg["baseline_max_order"]
    for scene, n_nodes, _ in cfg["scenes"]:
        if scene != "rossler":
            continue
        for noise in cfg["noises"]:
            for seed in cfg["seeds"]:
                base = f"rossler{n_nodes}_noise{noise}_seed{seed}"
                jobs.append(_integral_job(
                    f"ours_{base}", scene, n_nodes, mo, cfg["n_samples"], noise,
                    seed, cfg["n_epochs"], cfg["lr"],
                    meta_extra={"variant": "ours", "max_order": mo},
                    cfg_max_cand=cfg["max_cand"]))
                for method in cfg["baseline_methods"]:
                    if HYPERPINN_METHODS[method]["max_order"] < mo:
                        continue
                    jobs.append(_hyperpinn_job(
                        f"{method}_{base}", method, n_nodes, mo, cfg["n_samples"],
                        noise, seed, cfg["baseline_epochs"], cfg["baseline_lr"],
                        meta_extra={"variant": method}))
    return jobs


def suite_baseline_tuning(cfg):
    """Equal-budget grid for every paper baseline and for our method:
    4 learning rates x 2 epoch budgets = 8 configurations each."""
    jobs = []
    mo = cfg["baseline_max_order"]
    seed = cfg["seeds"][0]
    lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    budgets = [7000, 14000]
    for scene, n_nodes, _ in cfg["scenes"]:
        if scene != "rossler":
            continue
        for noise in cfg["noises"]:
            for method in cfg["baseline_methods"]:
                if HYPERPINN_METHODS[method]["max_order"] < mo:
                    continue
                for lr in lrs:
                    for ep in budgets:
                        jobs.append(_hyperpinn_job(
                            f"tune_{method}_lr{lr}_ep{ep}_n{n_nodes}_noise{noise}",
                            method, n_nodes, mo, cfg["n_samples"], noise, seed, ep, lr,
                            meta_extra={"variant": f"tune_{method}", "lr": lr, "epochs": ep}))
            for lr in lrs:
                for ep in budgets:
                    jobs.append(_integral_job(
                        f"tune_ours_lr{lr}_ep{ep}_n{n_nodes}_noise{noise}",
                        scene, n_nodes, mo, cfg["n_samples"], noise, seed, ep, lr,
                        meta_extra={"variant": "tune_ours", "lr": lr, "epochs": ep},
                        cfg_max_cand=cfg["max_cand"]))
    return jobs


SUITES = {
    "main": suite_main,
    "baselines": suite_baselines,
    "baseline_tuning": suite_baseline_tuning,
    "sparsity": suite_sparsity,
    "diagnostic": suite_diagnostic,
    "ablation": suite_ablation,
    "sigma_h": suite_sigma_h,
    "tuning": suite_tuning,
}


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def run_job(job, gpu_id, out_root, log_dir, timeout):
    cmd = job.command(gpu_id, out_root)
    log_path = Path(log_dir) / f"{job.name}.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        try:
            rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=job.env,
                                stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
            log.write("\n*** TIMEOUT ***\n")
    return {"job": job.name, "returncode": rc, "seconds": round(time.time() - t0, 1),
            "meta": job.meta, "log": str(log_path)}


def collect(out_root, records, out_file):
    """Attach each run's summary.json to its job record."""
    rows = []
    for rec in records:
        summary = None
        for path in Path(out_root).rglob(f"{rec['job']}/summary.json"):
            summary = json.loads(path.read_text())
            summary["_dir"] = str(path.parent)
            break
        row = dict(rec)
        row["summary"] = summary
        if summary:
            row["macro_auc"] = summary.get("extra_metrics", {}).get("macro_auc")
            row["auc"] = summary.get("auc")
        rows.append(row)
    with open(out_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--jobs_per_gpu", type=int, default=1)
    ap.add_argument("--scenes", type=str, nargs="+", default=["rossler:8:5"],
                    help="scene:n_nodes:max_order triples, e.g. rossler:8:5 social:8:5")
    ap.add_argument("--noises", type=float, nargs="+", default=[0.0])
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--n_epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sindy_threshold", type=float, default=0.01)
    ap.add_argument("--sindy_window", type=int, default=20)
    ap.add_argument("--sigma_h_samples", type=int, nargs="+", default=[100, 200, 300, 600])
    ap.add_argument("--sigma_h_noises", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    ap.add_argument("--baseline_max_order", type=int, default=5,
                    help="Order at which the baseline comparison runs. Only the "
                         "cp/ITC variants cap at 3; HyperPINN itself runs at 5.")
    ap.add_argument("--baseline_methods", type=str, nargs="+",
                    default=["hyperpinn_base"],
                    help="Which baselines to include. Defaults to hyperpinn_base, which IS "
                         "HyperPINN -- the only baseline the reviewers name. On Rossler it is "
                         "identical to `hyperpinn` except for a vmap/jacrev time derivative "
                         "that is simply faster; `hyperpinn` is the slower multi-scene build. "
                         "The low-rank tensor-decomposition variants are out of scope.")
    ap.add_argument("--baseline_epochs", type=int, default=14000)
    ap.add_argument("--baseline_lr", type=float, default=5e-4)
    ap.add_argument("--max_cand", type=int, default=0,
                    help="Subsample negative candidates to this many per order (0 = full).")
    ap.add_argument("--l1_weights", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2],
                    help="L1 weights for the sparsity suite (0 = our method unchanged).")
    ap.add_argument("--diag_samples", type=int, nargs="+", default=[50, 100, 300],
                    help="Native grid sizes for the diagnostic suite (smaller = coarser dt).")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    scenes = []
    for spec in args.scenes:
        parts = spec.split(":")
        scenes.append((parts[0], int(parts[1]), int(parts[2])))

    cfg = {"seeds": args.seeds, "scenes": scenes, "noises": args.noises,
           "n_samples": args.n_samples, "n_epochs": args.n_epochs, "lr": args.lr,
           "sindy_threshold": args.sindy_threshold, "sindy_window": args.sindy_window,
           "sigma_h_samples": args.sigma_h_samples, "sigma_h_noises": args.sigma_h_noises,
           "diag_samples": args.diag_samples, "max_cand": args.max_cand,
           "l1_weights": args.l1_weights,
           "baseline_max_order": args.baseline_max_order,
           "baseline_epochs": args.baseline_epochs,
           "baseline_lr": args.baseline_lr,
           "baseline_methods": args.baseline_methods}

    jobs = SUITES[args.suite](cfg)
    out_root = Path(args.out)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Suite '{args.suite}': {len(jobs)} jobs over GPUs {args.gpus}")

    if args.dry_run:
        for j in jobs:
            print(" ", " ".join(j.command(args.gpus[0], out_root)))
        return

    slots = [g for g in args.gpus for _ in range(args.jobs_per_gpu)]
    records = []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = {}
        for i, job in enumerate(jobs):
            futures[pool.submit(run_job, job, slots[i % len(slots)], out_root,
                                log_dir, args.timeout)] = job
        from concurrent.futures import as_completed
        done = 0
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            done += 1
            status = "ok" if rec["returncode"] == 0 else f"FAIL({rec['returncode']})"
            print(f"[{done}/{len(jobs)}] {rec['job']}: {status} {rec['seconds']}s")

    results_file = out_root / "results.jsonl"
    rows = collect(out_root, records, results_file)
    n_ok = sum(1 for r in rows if r.get("macro_auc") is not None)
    print(f"\nWrote {results_file}  ({n_ok}/{len(rows)} runs produced a summary)")


if __name__ == "__main__":
    main()
