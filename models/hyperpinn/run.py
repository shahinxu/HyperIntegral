import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(__file__).resolve().parent
HYPERPINN_ROOT = BASELINE_ROOT / "HyperPINN"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypergraph.outputs import write_standard_summary
from hypergraph.scene_registry import SCENE_REGISTRY, get_scene_model


SCENE_TO_SCRIPT = {
    "ecological": "HyperPINN_Ecosystem.py",
    "neuronal": "HyperPINN_neuronal_synchronization.py",
    "rossler": "HyperPINN_Rossler.py",
    "social": "HyperPINN_SocialContagion.py",
}

FILE_DRIVEN_SCENES = {"ecological", "neuronal", "social"}


def _safe_max_order(n_nodes: int | None, requested_max_order: int | None) -> int | None:
    if n_nodes is None or requested_max_order is None:
        return requested_max_order

    # HyperPINN enumerates all k-combinations; large N with high k is intractable.
    if n_nodes >= 40:
        return min(requested_max_order, 3)
    if n_nodes >= 25:
        return min(requested_max_order, 4)
    return requested_max_order


def parse_auc_file(path: Path):
    auc_scores = {}
    if not path.exists():
        return auc_scores
    p = re.compile(r"(\d+-edges)\s*[:=]\s*([0-9.]+|N/A)")
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = p.search(ln)
        if not m:
            continue
        auc_scores[m.group(1)] = None if m.group(2) == "N/A" else float(m.group(2))
    return auc_scores


def extract_results_dir(stdout: str):
    p = re.compile(r"Results will be saved to:\s*(.+)")
    for ln in stdout.splitlines():
        m = p.search(ln)
        if m:
            return m.group(1).strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_TO_SCRIPT.keys()))
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--results_root", type=str, default="results/hyperpinn")
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--bin_thresh", type=float, default=1e-4)
    args = parser.parse_args()

    effective_max_order = _safe_max_order(args.n_nodes, args.max_order)
    if args.scene not in FILE_DRIVEN_SCENES and args.max_order is not None and effective_max_order is not None and effective_max_order < args.max_order:
        print(
            f"[hyperpinn] max_order={args.max_order} is unsafe for n_nodes={args.n_nodes}; "
            f"using max_order={effective_max_order} to avoid OOM."
        )

    script = SCENE_TO_SCRIPT[args.scene]
    script_path = HYPERPINN_ROOT / script
    cmd = [sys.executable, str(script_path), "--M", str(args.n_samples), "--gpu_id", str(args.gpu_id), "--noise", str(args.noise)]
    if args.scene not in FILE_DRIVEN_SCENES and args.n_nodes is not None:
        cmd.extend(["--N", str(args.n_nodes)])
    if args.scene not in FILE_DRIVEN_SCENES and effective_max_order is not None:
        cmd.extend(["--max_order", str(effective_max_order)])

    # Run from project root so HyperPINN writes results into top-level results_* directories.
    env = os.environ.copy()
    env["HYPERPINN_RESULTS_ROOT"] = args.results_root

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

    results_dir = extract_results_dir(proc.stdout)
    if results_dir is None:
        raise RuntimeError("Could not infer results directory from baseline output")

    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
        if not results_path.exists():
            results_path = BASELINE_ROOT / results_dir

    HypergraphModel, _ = get_scene_model(args.scene)
    defaults = HypergraphModel.get_default_params()

    auc_scores = parse_auc_file(results_path / "auc_scores.txt")
    write_standard_summary(
        save_dir=str(results_path),
        method="baseline_hyperpinn",
        scene=SCENE_REGISTRY[args.scene].label,
        config={
            "scene": args.scene,
            "n_samples": args.n_samples,
            "n_nodes": defaults["n_nodes"] if args.scene in FILE_DRIVEN_SCENES else args.n_nodes,
            "max_order": defaults["max_order"] if args.scene in FILE_DRIVEN_SCENES else effective_max_order,
            "gpu_id": args.gpu_id,
            "noise": args.noise,
        },
        auc_scores=auc_scores,
    )
    print(f"[unified baseline] summary written: {results_path / 'summary.json'}")


if __name__ == "__main__":
    main()
