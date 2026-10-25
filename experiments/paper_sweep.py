"""Runs the rebuttal experiment matrix on the paper's three benchmarks.

Scenes and sizes follow the paper: ecological n=9 (372 candidates),
social contagion n=8 (210), neuronal n=30 (174406).  The proposed method is
the neural-surrogate IntegralPINN (models.integral.train_nn); baselines are
HyperPINN, the kernel-based method, THIS, and the SINDy variants.
"""
import argparse, json, os, subprocess, sys, time
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


def env_for(scene):
    e = dict(os.environ)
    e.update(SCENES[scene]["env"])
    e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
    cli = ["--scene", scene, "--n_samples", str(cfg["n_samples"]), "--noise", str(noise),
           "--seed", str(seed), "--max_order", "5", "--n_epochs", str(cfg["bl_epochs"]),
           "--lr", str(cfg["bl_lr"]), "--python", PY]
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
    It serializes on a per-scene export directory, so it is run with
    jobs_per_gpu = 1 for its scene."""
    return dict(name=tag, module="models.this.run", scene=scene,
                cli=["--scene", scene, "--n_samples", str(cfg["n_samples"]),
                     "--noise", str(noise), "--seed", str(seed), "--python", PY],
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


SUITES = {"main": suite_main, "ablation": suite_ablation}


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
    ap.add_argument("--bl_lr", type=float, default=5e-4)
    ap.add_argument("--thr", type=float, default=0.01)
    ap.add_argument("--win", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    cfg = dict(scenes=a.scenes, seeds=a.seeds, noises=a.noises, n_samples=a.n_samples,
               n_epochs=a.n_epochs, lr=a.lr, stage1=a.stage1, ramp=a.ramp,
               max_cand=a.max_cand, bl_epochs=a.bl_epochs, bl_lr=a.bl_lr,
               thr=a.thr, win=a.win)
    jobs = SUITES[a.suite](cfg)
    out = Path(a.out); logs = out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    print(f"suite={a.suite}: {len(jobs)} jobs on GPUs {a.gpus}")
    if a.dry_run:
        for j in jobs[:5]:
            print("  ", " ".join([PY, "-m", j["module"]] + j["cli"]))
        return

    slots = [g for g in a.gpus for _ in range(a.jobs_per_gpu)]
    # THIS shells out to Julia through a per-scene export directory shared by
    # all of its runs, so those jobs cannot overlap; run them last, serially.
    this_jobs = [j for j in jobs if j["module"] == "models.this.run"]
    jobs = [j for j in jobs if j["module"] != "models.this.run"]
    recs = []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futs = {pool.submit(run_job, j, slots[i % len(slots)], out, logs, a.timeout): j
                for i, j in enumerate(jobs)}
        done = 0
        for f in as_completed(futs):
            r = f.result(); recs.append(r); done += 1
            print(f"[{done}/{len(jobs)}] {r['job']}: "
                  f"{'ok' if r['returncode']==0 else 'FAIL'} {r['seconds']}s", flush=True)

    for i, j in enumerate(this_jobs):
        r = run_job(j, slots[i % len(slots)], out, logs, a.timeout)
        recs.append(r)
        print(f"[THIS {i+1}/{len(this_jobs)}] {r['job']}: "
              f"{'ok' if r['returncode']==0 else 'FAIL'} {r['seconds']}s", flush=True)

    rows = []
    for r in recs:
        s = None
        for p in out.rglob(f"{r['job']}/summary.json"):
            s = json.loads(p.read_text()); s["_dir"] = str(p.parent); break
        row = dict(r, summary=s)
        if s:
            row["macro_auc"] = s.get("extra_metrics", {}).get("macro_auc")
            row["auc"] = s.get("auc")
        rows.append(row)
    rf = out / "results.jsonl"
    rf.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ok = sum(1 for r in rows if r.get("macro_auc") is not None)
    print(f"\nWrote {rf} ({ok}/{len(rows)} produced a summary)")


if __name__ == "__main__":
    main()
