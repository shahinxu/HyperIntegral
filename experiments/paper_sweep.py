"""Runs the rebuttal experiment matrix on the paper's three benchmarks.

Scenes and sizes follow the paper: ecological n=9 (372 candidates),
social contagion n=8 (210), neuronal n=30 (174406).  The proposed method is
the neural-surrogate IntegralPINN (models.integral.train_nn); baselines are
HyperPINN, the kernel-based method, THIS, and the SINDy variants.
"""
import argparse, json, os, subprocess, sys, time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
N30 = str(ROOT / "lib_neuronal_synchronization/presets/neuronal_synchronization_n30_hypergraph.json")

SCENES = {
    "ecological": {"env": {"ECOLOGY_HYPERGRAPH_PRESET": "n9"}, "n": 9, "cap": 0},
    "social":     {"env": {}, "n": 8, "cap": 0},
    # The full neuronal candidate set is 174406; a dense ridge solve over it is
    # ~121 GB, so the kernel baseline cannot run on it.  Every method on this
    # scene therefore uses the same subsampled pool (3000 negatives per order,
    # all true hyperedges kept), which keeps the comparison like-for-like.
    "neuronal":   {"env": {"NEURONAL_HYPERGRAPH_FILE": N30}, "n": 30, "cap": 3000},
}


def cap_for(scene):
    return SCENES[scene].get("cap", 0)


# Each worker inherits a machine with 256 cores, so OpenMP and the BLAS
# libraries each start a thread per core: one job was measured at 127 threads.
# The sandbox cgroup allows 8192 pids in total, so a dozen concurrent jobs
# exhaust it and the next one dies with
#   libgomp: Thread creation failed: Resource temporarily unavailable
# before it reaches its first epoch. Capping the pools also removes the
# oversubscription that made every surviving job several times slower.
THREADS_PER_JOB = int(os.environ.get("SWEEP_THREADS_PER_JOB", "4"))


def env_for(scene):
    e = dict(os.environ)
    e.update(SCENES[scene]["env"])
    e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "JULIA_NUM_THREADS"):
        e[var] = str(THREADS_PER_JOB)
    e["PATH"] = str(Path.home() / ".local/julia-1.10.5/bin") + os.pathsep + e.get("PATH", "")
    return e


def ours(tag, scene, noise, seed, cfg, extra=(), meta=None):
    cli = ["--scene", scene, "--n_samples", str(cfg["n_samples"]), "--noise", str(noise),
           "--seed", str(seed), "--n_epochs", str(cfg["n_epochs"]),
           "--lr", str(cfg["lr"]), "--stage1_epochs", str(cfg["stage1"]),
           "--physics_ramp_epochs", str(cfg["ramp"]), "--eval_every", "0"]
    _cap = cfg["max_cand"] or cap_for(scene)
    if _cap:
        cli += ["--max_candidates_per_order", str(_cap)]
    cli += list(extra)
    m = {"method": "integral_pinn_nn", "variant": "ours", "scene": scene,
         "noise": noise, "seed": seed, "n_nodes": SCENES[scene]["n"]}
    m.update(meta or {})
    return dict(name=tag, module="models.integral.train_nn", cli=cli, meta=m, scene=scene)


def sindy(tag, scene, noise, seed, cfg, form, est, meta=None):
    cli = ["--scene", scene, "--n_samples", str(cfg["n_samples"]), "--noise", str(noise),
           "--seed", str(seed), "--form", form, "--estimator", est,
           "--threshold", str(cfg["thr"]), "--window", str(cfg["win"])]
    _cap = cfg["max_cand"] or cap_for(scene)
    if _cap:
        cli += ["--max_candidates_per_order", str(_cap)]
    m = {"method": f"sindy_{form}_{est}", "variant": f"sindy_{form}_{est}", "scene": scene,
         "noise": noise, "seed": seed, "n_nodes": SCENES[scene]["n"]}
    m.update(meta or {})
    return dict(name=tag, module="models.sindy.train", cli=cli, meta=m, scene=scene)


def hyperpinn(tag, scene, noise, seed, cfg):
    # n=30 is far slower per epoch; the full budget cannot finish inside the
    # per-job timeout, so every neuronal HyperPINN run used to die at -9.
    epochs = cfg["bl_epochs"]
    if scene == "neuronal":
        epochs = min(epochs, cfg.get("bl_epochs_neuronal", 3000))
    cli = ["--scene", scene, "--n_samples", str(cfg["n_samples"]), "--noise", str(noise),
           "--seed", str(seed), "--max_order", "5", "--n_epochs", str(epochs),
           "--lr", str(cfg["bl_lr"]), "--python", PY]
    _cap = cfg["max_cand"] or cap_for(scene)
    if _cap:
        cli += ["--max_candidates_per_order", str(_cap)]
    return dict(name=tag, module="models.hyperpinn.run", cli=cli, scene=scene,
                meta={"method": "hyperpinn", "variant": "hyperpinn", "scene": scene,
                      "noise": noise, "seed": seed, "n_nodes": SCENES[scene]["n"]})


def kernel(tag, scene, noise, seed, cfg):
    return dict(name=tag, module="models.kernel.train", scene=scene,
                cli=["--scene", scene, "--n_samples", str(cfg["n_samples"]),
                     "--noise", str(noise), "--seed", str(seed),
                     "--n_epochs", str(cfg["bl_epochs"]),
                     "--max_candidates_per_order", str(cfg["max_cand"] or cap_for(scene))],
                meta={"method": "kernel", "variant": "kernel", "scene": scene,
                      "noise": noise, "seed": seed, "n_nodes": SCENES[scene]["n"]})


def this_method(tag, scene, noise, seed, cfg):
    """THIS is a Julia implementation driven through models/this/run.py.

    The neuronal scene used to be excluded here: THIS appeared never to
    terminate at n=30. It was three quadratic accumulation patterns in the
    Julia code, not a hang - see bugs 27-29 in BUGFIX_LOG.md. With those
    fixed a neuronal run finishes in minutes, so the scene is no longer
    skipped.
    """
    return dict(name=tag, module="models.this.run", scene=scene,
                cli=["--scene", scene, "--n_samples", str(cfg["n_samples"]),
                     "--noise", str(noise), "--seed", str(seed), "--python", PY,
                     "--max_candidates_per_order",
                     str(cfg["max_cand"] or cap_for(scene))],
                meta={"method": "this", "variant": "this", "scene": scene,
                      "noise": noise, "seed": seed, "n_nodes": SCENES[scene]["n"]})


ABLATIONS = {
    "full":            [],
    "derivative_loss": ["--loss_mode", "derivative"],
    "no_staged":       ["--physics_ramp_epochs", "0"],
    "no_bounded":      ["--bounded", "0"],
    "no_clip":         ["--no_clip", "1"],
    "window_0.1":      ["--window_frac", "0.1"],
    "window_0.25":     ["--window_frac", "0.25"],
    "window_1.0":      ["--window_frac", "1.0"],
    "intervals_16":    ["--n_intervals", "16"],
    "intervals_32":    ["--n_intervals", "32"],
    "intervals_128":   ["--n_intervals", "128"],
}


def suite_main(cfg):
    jobs = []
    for sc in cfg["scenes"]:
        for noise in cfg["noises"]:
            for seed in cfg["seeds"]:
                b = f"{sc}_noise{noise}_seed{seed}"
                jobs.append(ours(f"ours_{b}", sc, noise, seed, cfg))
                jobs.append(hyperpinn(f"hyperpinn_{b}", sc, noise, seed, cfg))
                jobs.append(kernel(f"kernel_{b}", sc, noise, seed, cfg))
                jobs.append(this_method(f"this_{b}", sc, noise, seed, cfg))
                for form, est in (("integral", "stlsq"), ("derivative", "stlsq"),
                                  ("integral", "lstsq")):
                    jobs.append(sindy(f"sindy_{form}_{est}_{b}", sc, noise, seed, cfg, form, est))
    return jobs


def suite_ablation(cfg):
    jobs = []
    for sc in cfg["scenes"]:
        for noise in cfg["noises"]:
            for seed in cfg["seeds"]:
                for v, extra in ABLATIONS.items():
                    jobs.append(ours(f"{v}_{sc}_noise{noise}_seed{seed}", sc, noise, seed,
                                     cfg, extra, meta={"variant": v}))
    return jobs



NOISES_SWEEP = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
WINDOWS = [0.1, 0.25, 0.5, 1.0]
L1_WEIGHTS = [0.0, 1e-4, 1e-3, 1e-2]


def suite_noise(cfg):
    """Table R3.6b: macro-AUC against observation noise, all methods."""
    jobs = []
    for sc in cfg["scenes"]:
        for noise in NOISES_SWEEP:
            for seed in cfg["seeds"]:
                b = f"{sc}_noise{noise}_seed{seed}"
                jobs.append(ours(f"ours_{b}", sc, noise, seed, cfg))
                jobs.append(hyperpinn(f"hyperpinn_{b}", sc, noise, seed, cfg))
                jobs.append(kernel(f"kernel_{b}", sc, noise, seed, cfg))
                jobs.append(this_method(f"this_{b}", sc, noise, seed, cfg))
                jobs.append(sindy(f"sindy_integral_stlsq_{b}", sc, noise, seed, cfg,
                                  "integral", "stlsq"))
    return jobs


def suite_sigma_h(cfg):
    """Table R2.5c: sigma x integral window length, plus a derivative column."""
    jobs = []
    for sc in cfg["scenes"]:
        for noise in NOISES_SWEEP[:5]:
            for seed in cfg["seeds"]:
                for h in WINDOWS:
                    jobs.append(ours(f"h{h}_{sc}_noise{noise}_seed{seed}", sc, noise, seed,
                                     cfg, ["--window_frac", str(h)],
                                     meta={"variant": f"h{h}", "window_frac": h}))
                jobs.append(ours(f"deriv_{sc}_noise{noise}_seed{seed}", sc, noise, seed, cfg,
                                 ["--loss_mode", "derivative"],
                                 meta={"variant": "derivative"}))
    return jobs


def suite_sparsity(cfg):
    """Tables R1.5a / R1.5b: explicit L1 on top of the proposed parameterization."""
    jobs = []
    for sc in cfg["scenes"]:
        for noise in cfg["noises"]:
            for seed in cfg["seeds"]:
                for l1 in L1_WEIGHTS:
                    name = "ours" if l1 == 0 else f"ours_l1_{l1}"
                    extra = [] if l1 == 0 else ["--l1", str(l1)]
                    jobs.append(ours(f"{name}_{sc}_noise{noise}_seed{seed}", sc, noise, seed,
                                     cfg, extra, meta={"variant": name, "l1": l1}))
                jobs.append(sindy(f"sindy_integral_stlsq_{sc}_noise{noise}_seed{seed}",
                                  sc, noise, seed, cfg, "integral", "stlsq"))
    return jobs


def suite_tuning(cfg):
    """Tables R2.6b / R2.6c: 16 configurations per method, one tuning seed."""
    jobs = []
    seed = cfg["seeds"][0]
    for sc in cfg["scenes"]:
        for noise in cfg["noises"]:
            for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
                for ni in [16, 32, 64, 128]:
                    j = ours(f"tune_ours_lr{lr}_ni{ni}_{sc}", sc, noise, seed, cfg,
                             ["--lr", str(lr), "--n_intervals", str(ni)],
                             meta={"variant": "tune_ours", "lr": lr, "n_intervals": ni})
                    # the ours() helper already injected cfg lr; keep the override last
                    jobs.append(j)
            _EP_GRID = [7000, 14000, 21000, 28000]
            for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
                for ep in _EP_GRID:
                    # hyperpinn() clamps the neuronal budget to
                    # bl_epochs_neuronal because the full budget cannot finish
                    # inside the per-job timeout at n=30. Clamping collapsed
                    # the whole epoch axis of this grid to one value, so all 16
                    # configurations ran identically and only the 4 learning
                    # rates actually differed. Scale the grid into the feasible
                    # budget instead, keeping four distinct points.
                    ep_eff = ep
                    if sc == "neuronal":
                        ep_eff = max(1, ep * cfg["bl_epochs_neuronal"] // _EP_GRID[-1])
                    c2 = dict(cfg, bl_epochs=ep_eff, bl_epochs_neuronal=ep_eff, bl_lr=lr)
                    jobs.append(dict(hyperpinn(f"tune_hyperpinn_lr{lr}_ep{ep}_{sc}",
                                               sc, noise, seed, c2),
                                     meta={"method": "hyperpinn", "variant": "tune_hyperpinn",
                                           "scene": sc, "noise": noise, "seed": seed,
                                           "n_nodes": SCENES[sc]["n"], "lr": lr,
                                           "epochs": ep_eff}))
            # Kernel and THIS were previously left untuned while every other
            # method got a 16-point grid, which is exactly the imbalance the
            # reviewer asks about. Give them grids of the same size.
            for sg in [0.1, 1.0, 10.0, 100.0]:
                for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
                    j = kernel(f"tune_kernel_sg{sg}_lam{lam}_{sc}", sc, noise, seed, cfg)
                    # the grid point has to reach the model, not just the
                    # record: without these flags all 16 runs were identical
                    j["cli"] = j["cli"] + ["--sigma_k", str(sg), "--lam", str(lam)]
                    jobs.append(dict(
                        j,
                        meta={"method": "kernel", "variant": "tune_kernel", "scene": sc,
                              "noise": noise, "seed": seed, "n_nodes": SCENES[sc]["n"],
                              "sigma_k": sg, "lam": lam}))
            if True:
                for l1 in [1e-4, 1e-3, 1e-2, 1e-1]:
                    for rho in [1e-4, 1e-3, 1e-2, 1e-1]:
                        j = this_method(f"tune_this_l1{l1}_rho{rho}_{sc}",
                                        sc, noise, seed, cfg)
                        j["cli"] = j["cli"] + ["--lambda1", str(l1), "--rho", str(rho)]
                        jobs.append(dict(
                            j,
                            meta={"method": "this", "variant": "tune_this", "scene": sc,
                                  "noise": noise, "seed": seed, "n_nodes": SCENES[sc]["n"],
                                  "lambda1": l1, "rho": rho}))
            for thr in [1e-4, 1e-3, 1e-2, 1e-1]:
                for win in [5, 20, 50, 100]:
                    c2 = dict(cfg, thr=thr, win=win)
                    jobs.append(dict(sindy(f"tune_sindy_thr{thr}_win{win}_{sc}",
                                           sc, noise, seed, c2, "integral", "stlsq"),
                                     meta={"method": "sindy_integral_stlsq",
                                           "variant": "tune_sindy", "scene": sc,
                                           "noise": noise, "seed": seed,
                                           "n_nodes": SCENES[sc]["n"],
                                           "threshold": thr, "window": win}))
    return jobs


TUNING_DIRS = ["results/paper/tuning", "results/paper/tuning_neuronal"]


def _best_configs(dirs=None):
    """(scene, tuned-variant) -> the meta of its highest-scoring run."""
    best = {}
    for d in (dirs or TUNING_DIRS):
        f = Path(d) / "results.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("macro_auc") is None:
                continue
            m = r.get("meta") or {}
            v = m.get("variant", "")
            if not v.startswith("tune_"):
                continue
            k = (m.get("scene"), v)
            if k not in best or r["macro_auc"] > best[k][0]:
                best[k] = (r["macro_auc"], m)
    return {k: m for k, (_, m) in best.items()}


def suite_holdout(cfg):
    """Re-run each baseline's selected configuration on the held-out seeds.

    Table R2.6c reports a best configuration per method and, beside it, a
    macro-AUC over the held-out seeds. That second column used to be read off
    the main suite, which runs the *default* hyperparameters - for Integral
    SINDy the table named threshold=1e-4 while the number came from runs at
    threshold=1e-2. This suite evaluates the configuration the table actually
    names. The proposed method is excluded: its rows are maintained by hand.
    """
    best = _best_configs(cfg.get("tuning_dirs"))
    jobs = []
    for sc in cfg["scenes"]:
        for noise in cfg["noises"]:
            for seed in cfg["seeds"]:
                b = f"{sc}_noise{noise}_seed{seed}"
                m = best.get((sc, "tune_hyperpinn"))
                if m:
                    ep = int(m["epochs"])
                    c2 = dict(cfg, bl_lr=m["lr"], bl_epochs=ep, bl_epochs_neuronal=ep)
                    jobs.append(dict(
                        hyperpinn(f"best_hyperpinn_{b}", sc, noise, seed, c2),
                        meta={"method": "hyperpinn", "variant": "best_hyperpinn",
                              "scene": sc, "noise": noise, "seed": seed,
                              "n_nodes": SCENES[sc]["n"], "lr": m["lr"], "epochs": ep}))
                m = best.get((sc, "tune_kernel"))
                if m:
                    j = kernel(f"best_kernel_{b}", sc, noise, seed, cfg)
                    j["cli"] = j["cli"] + ["--sigma_k", str(m["sigma_k"]),
                                           "--lam", str(m["lam"])]
                    jobs.append(dict(j, meta={
                        "method": "kernel", "variant": "best_kernel", "scene": sc,
                        "noise": noise, "seed": seed, "n_nodes": SCENES[sc]["n"],
                        "sigma_k": m["sigma_k"], "lam": m["lam"]}))
                m = best.get((sc, "tune_this"))
                if m:
                    j = this_method(f"best_this_{b}", sc, noise, seed, cfg)
                    j["cli"] = j["cli"] + ["--lambda1", str(m["lambda1"]),
                                           "--rho", str(m["rho"])]
                    jobs.append(dict(j, meta={
                        "method": "this", "variant": "best_this", "scene": sc,
                        "noise": noise, "seed": seed, "n_nodes": SCENES[sc]["n"],
                        "lambda1": m["lambda1"], "rho": m["rho"]}))
                m = best.get((sc, "tune_sindy"))
                if m:
                    c2 = dict(cfg, thr=m["threshold"], win=m["window"])
                    jobs.append(dict(
                        sindy(f"best_sindy_{b}", sc, noise, seed, c2,
                              "integral", "stlsq"),
                        meta={"method": "sindy_integral_stlsq",
                              "variant": "best_sindy", "scene": sc, "noise": noise,
                              "seed": seed, "n_nodes": SCENES[sc]["n"],
                              "threshold": m["threshold"], "window": m["window"]}))
    return jobs


SUITES = {"main": suite_main, "ablation": suite_ablation, "holdout": suite_holdout,
          "noise": suite_noise, "sigma_h": suite_sigma_h,
          "sparsity": suite_sparsity, "tuning": suite_tuning}


_WRITE_LOCK = threading.Lock()


def find_summary(out_root, job):
    """Return the summary.json of a completed run, if it exists."""
    for p in Path(out_root).rglob(f"{job}/summary.json"):
        return p
    return None


def record(out_root, rec):
    """Append one finished run to results.jsonl immediately.

    The sweep used to write the whole file only at the end, so an interrupted
    run lost every completed job. Appending as we go makes the sweep resumable.
    """
    rec = dict(rec)
    p = find_summary(out_root, rec["job"])
    if p:
        s = json.loads(p.read_text()); s["_dir"] = str(p.parent)
        rec["summary"] = s
        rec["macro_auc"] = s.get("extra_metrics", {}).get("macro_auc")
        rec["auc"] = s.get("auc")
        if rec["macro_auc"] is None:
            # baselines that do not report a macro themselves: average the
            # per-order AUCs here, the same way repair_results does, so the
            # run counts as done and does not get rerun on every resume
            a = [v for v in (rec["auc"] or {}).values() if v is not None]
            rec["macro_auc"] = sum(a) / len(a) if a else None
    else:
        rec["summary"] = None
    with _WRITE_LOCK:
        with open(Path(out_root) / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def load_done(out_root):
    """Jobs that already produced a summary, so a resumed sweep can skip them."""
    done = set()
    rf = Path(out_root) / "results.jsonl"
    if rf.exists():
        for line in rf.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("macro_auc") is not None:
                done.add(r["job"])
    for p in Path(out_root).rglob("summary.json"):
        done.add(p.parent.name)
    return done


def run_job(j, gpu, out_root, log_dir, timeout):
    cmd = [PY, "-m", j["module"]] + j["cli"] + [
        "--gpu_id", str(gpu), "--results_root", str(out_root), "--run_tag", j["name"]]
    log = Path(log_dir) / f"{j['name']}.log"
    t0 = time.time()
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n\n"); f.flush()
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env_for(j["scene"]),
                                stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9; f.write("\n*** TIMEOUT ***\n")
    return {"job": j["name"], "returncode": rc, "seconds": round(time.time()-t0, 1),
            "meta": j["meta"], "log": str(log)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--scenes", nargs="+", default=["ecological", "social", "neuronal"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--noises", type=float, nargs="+", default=[0.02])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--jobs_per_gpu", type=int, default=2)
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--n_epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stage1", type=int, default=2500)
    ap.add_argument("--ramp", type=int, default=2000)
    ap.add_argument("--max_cand", type=int, default=0)
    ap.add_argument("--bl_epochs", type=int, default=14000)
    ap.add_argument("--bl_epochs_neuronal", type=int, default=3000,
                    help="HyperPINN epoch budget on the neuronal scene.")
    ap.add_argument("--bl_lr", type=float, default=5e-4)
    ap.add_argument("--thr", type=float, default=0.01)
    ap.add_argument("--win", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore existing results and rerun every job.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--tuning_dirs", nargs="*", default=None,
                    help="Where the holdout suite reads the selected "
                         "configurations from (default: both tuning sweeps).")
    ap.add_argument("--skip_variants", nargs="*", default=[],
                    help="Variants to leave out of this run, e.g. ours "
                         "tune_ours. The proposed method's rows are maintained "
                         "by hand, so rerunning them only costs GPU time.")
    a = ap.parse_args()

    cfg = dict(scenes=a.scenes, seeds=a.seeds, noises=a.noises, n_samples=a.n_samples,
               n_epochs=a.n_epochs, lr=a.lr, stage1=a.stage1, ramp=a.ramp,
               max_cand=a.max_cand, bl_epochs=a.bl_epochs, bl_lr=a.bl_lr,
               thr=a.thr, win=a.win, bl_epochs_neuronal=a.bl_epochs_neuronal,
               tuning_dirs=a.tuning_dirs)
    jobs = SUITES[a.suite](cfg)
    if a.skip_variants:
        skip = set(a.skip_variants)
        before = len(jobs)
        jobs = [j for j in jobs if j["meta"].get("variant") not in skip]
        print(f"skipping {before - len(jobs)} job(s) of variant(s) "
              f"{', '.join(sorted(skip))}")
    out = Path(a.out); logs = out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    print(f"suite={a.suite}: {len(jobs)} jobs on GPUs {a.gpus}")
    if a.dry_run:
        for j in jobs[:5]:
            print("  ", " ".join([PY, "-m", j["module"]] + j["cli"]))
        return

    # SHORT_FIRST: run cheap jobs before expensive ones. The sandbox is torn
    # down every 15-20 minutes, so a long job (HyperPINN needs ~40 min) never
    # finishes; ordering by cost means each window still produces results.
    _COST = {"sindy_integral_lstsq": 1, "sindy_derivative_stlsq": 2,
             "sindy_integral_stlsq": 3, "best_sindy": 3, "this": 4,
             "best_this": 4, "kernel": 5, "best_kernel": 5,
             "ours": 6, "hyperpinn": 9, "best_hyperpinn": 9}
    jobs.sort(key=lambda j: _COST.get(j["meta"].get("variant", ""), 7))

    done = load_done(out) if not a.fresh else set()
    if done:
        before = len(jobs)
        jobs = [j for j in jobs if j["name"] not in done]
        print(f"resuming: {before - len(jobs)} job(s) already complete, {len(jobs)} to run")

    slots = [g for g in a.gpus for _ in range(a.jobs_per_gpu)]
    # THIS now writes into a per-run export directory, so its jobs can run
    # concurrently with everything else instead of being serialized.
    this_jobs = []
    recs = []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futs = {pool.submit(run_job, j, slots[i % len(slots)], out, logs, a.timeout): j
                for i, j in enumerate(jobs)}
        n_done = 0
        for f in as_completed(futs):
            r = record(out, f.result()); recs.append(r); n_done += 1
            print(f"[{n_done}/{len(jobs)}] {r['job']}: "
                  f"{'ok' if r['returncode']==0 else 'FAIL'} {r['seconds']}s", flush=True)

    this_jobs = [j for j in this_jobs if j["name"] not in done]
    for i, j in enumerate(this_jobs):
        r = record(out, run_job(j, slots[i % len(slots)], out, logs, a.timeout))
        recs.append(r)
        print(f"[THIS {i+1}/{len(this_jobs)}] {r['job']}: "
              f"{'ok' if r['returncode']==0 else 'FAIL'} {r['seconds']}s", flush=True)

    rf = out / "results.jsonl"
    total = [json.loads(l) for l in rf.read_text().splitlines() if l.strip()] if rf.exists() else []
    ok = sum(1 for r in total if r.get("macro_auc") is not None)
    print(f"\n{rf}: {ok}/{len(total)} runs have a summary")


if __name__ == "__main__":
    main()
