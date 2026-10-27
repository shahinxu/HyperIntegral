import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

THIS_ROOT = Path(__file__).resolve().parent

from hypergraph.outputs import write_standard_summary
from hypergraph.scene_registry import SCENE_REGISTRY, get_scene_model


SCENE_TO_SCRIPT = {
    "ecological": "ecology-run.jl",
    "neuronal": "neuronal-run.jl",
    "social": "social-run.jl",
}

SCENE_TO_METRICS = {
    "ecological": Path("ecology-data/this_ecology_metrics.txt"),
    "neuronal": Path("neuronal-data/this_neuronal_metrics.txt"),
    "social": Path("social-data/this_social_metrics.txt"),
}

FILE_DRIVEN_SCENES = {"ecological", "neuronal", "social"}


def parse_metrics_file(path: Path):
    auc_scores = {}
    extra = {}
    p_auc = re.compile(r"(?:^|\s)(order=\d+|(?:train|test): order=\d+).*?auc=([0-9.]+|NaN)")
    p_rel = re.compile(r"^(relerr(?:_train|_test)?)=(.+)$")
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m_rel = p_rel.match(ln.strip())
        if m_rel:
            key = m_rel.group(1)
            try:
                extra[key] = float(m_rel.group(2))
            except ValueError:
                extra[key] = m_rel.group(2)
        m_auc = p_auc.search(ln)
        if m_auc:
            key = m_auc.group(1).replace(" ", "")
            val = m_auc.group(2)
            auc_scores[key] = None if val == "NaN" else float(val)
    return auc_scores, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_TO_SCRIPT.keys()))
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--results_root", type=str, default="results_this")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_candidates_per_order", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--bin_thresh", type=float, default=1e-4)
    args = parser.parse_args()

    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".local/julia-1.10.5/bin") + os.pathsep + env.get("PATH", "")
    # The Julia driver shells out to the scene export script; it needs the
    # project on sys.path and an interpreter that has the dependencies.
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    interpreter = args.python or sys.executable
    env["PATH"] = str(Path(interpreter).parent) + os.pathsep + env["PATH"]

    scene_prefix = {
        "ecological": "ECO",
        "neuronal": "NS",
        "social": "SC",
    }[args.scene]

    # Always pin the order. social-run.jl otherwise defaults SC_ORDER to 3,
    # which silently evaluated THIS at a lower order than every other method.
    _mo = args.max_order
    if _mo is None:
        _hm, _ = get_scene_model(args.scene)
        _mo = _hm.get_default_params()["max_order"]
    env[f"{scene_prefix}_ORDER"] = str(_mo)
    # neuronal-run.jl defaults NS_NODES to 9 (the ecological size); pin the
    # scene's real node count so THIS runs the same system as everyone else.
    _hm2, _ = get_scene_model(args.scene)
    env[f"{scene_prefix}_NODES"] = str(
        args.n_nodes if args.n_nodes is not None else _hm2.get_default_params()["n_nodes"])
    env[f"{scene_prefix}_SAMPLES"] = str(args.n_samples)
    env[f"{scene_prefix}_NOISE"] = str(args.noise)
    env[f"{scene_prefix}_PY"] = interpreter
    env[f"{scene_prefix}_BIN_THRESH"] = str(args.bin_thresh)
    env[f"{scene_prefix}_SEED"] = str(args.seed)
    # social-run.jl reads SC_GRAPH_SEED / SC_TRAIN_SEED / SC_TEST_SEED, not
    # SC_SEED, so every seed produced an identical run (std = 0 across seeds).
    env[f"{scene_prefix}_GRAPH_SEED"] = str(args.seed)
    env[f"{scene_prefix}_TRAIN_SEED"] = str(args.seed + 100)
    env[f"{scene_prefix}_TEST_SEED"] = str(args.seed + 200)
    env["HG_DATA_SEED"] = str(args.seed)
    env[f"{scene_prefix}_MAX_CAND"] = str(args.max_candidates_per_order)

    cmd = ["julia", "--project=.", SCENE_TO_SCRIPT[args.scene]]
    proc = subprocess.run(cmd, cwd=str(THIS_ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

    metrics_path = THIS_ROOT / SCENE_TO_METRICS[args.scene]
    out_dir = Path(args.results_root) / SCENE_REGISTRY[args.scene].label / \
        f"sample_{args.n_samples}_noise_{args.noise}" / (args.run_tag or "default")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        raise RuntimeError(f"Metrics file not found: {metrics_path}")

    HypergraphModel, _ = get_scene_model(args.scene)
    defaults = HypergraphModel.get_default_params()

    auc_scores, extra_metrics = parse_metrics_file(metrics_path)
    # Normalise to the "<k>-edges" convention used by every other method.
    # Some scenes (social) report a train/test split; the other methods are all
    # evaluated in-sample on the data they fit, so the comparable column is the
    # train split. The held-out split is kept under extra_metrics.
    def _order_key(k):
        return f"{k.rsplit('=', 1)[1]}-edges"
    split = {k: v for k, v in auc_scores.items() if ":" in k}
    if split:
        extra_metrics.update({f"auc_{k.replace(':', '_').replace('=', '')}": v
                              for k, v in split.items()})
        auc_scores = {_order_key(k): v for k, v in split.items() if k.startswith("train:")}
    else:
        auc_scores = {(_order_key(k) if k.startswith("order=") else k): v
                      for k, v in auc_scores.items()}
    write_standard_summary(
        save_dir=str(out_dir),
        method="this",
        scene=SCENE_REGISTRY[args.scene].label,
        config={
            "scene": args.scene,
            "n_nodes": defaults["n_nodes"],
            "max_order": defaults["max_order"],
            "n_samples": args.n_samples,
            "noise": args.noise,
            "bin_thresh": args.bin_thresh,
        },
        auc_scores=auc_scores,
        extra_metrics={**extra_metrics,
                       "macro_auc": (sum(v for v in auc_scores.values() if v is not None)
                                     / max(1, len([v for v in auc_scores.values() if v is not None])))
                       if any(v is not None for v in auc_scores.values()) else None},
    )
    print(f"[unified this] summary written: {metrics_path.parent / 'summary.json'}")


if __name__ == "__main__":
    main()
