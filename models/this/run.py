import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

THIS_ROOT = PROJECT_ROOT / "THIS"

from hyperpinn_unified.outputs import write_standard_summary
from hyperpinn_unified.scene_registry import SCENE_REGISTRY


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
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--bin_thresh", type=float, default=1e-4)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("PATH", str(Path.home() / ".local/julia-1.10.5/bin") + os.pathsep + env.get("PATH", ""))

    scene_prefix = {
        "ecological": "ECO",
        "neuronal": "NS",
        "social": "SC",
    }[args.scene]

    if args.max_order is not None:
        env[f"{scene_prefix}_ORDER"] = str(args.max_order)
    env[f"{scene_prefix}_SAMPLES"] = str(args.n_samples)
    env[f"{scene_prefix}_NOISE"] = str(args.noise)
    if args.python:
        env[f"{scene_prefix}_PY"] = args.python
    env[f"{scene_prefix}_BIN_THRESH"] = str(args.bin_thresh)

    cmd = ["julia", "--project=.", SCENE_TO_SCRIPT[args.scene]]
    proc = subprocess.run(cmd, cwd=str(THIS_ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

    metrics_path = THIS_ROOT / SCENE_TO_METRICS[args.scene]
    if not metrics_path.exists():
        raise RuntimeError(f"Metrics file not found: {metrics_path}")

    auc_scores, extra_metrics = parse_metrics_file(metrics_path)
    write_standard_summary(
        save_dir=str(metrics_path.parent),
        method="this",
        scene=SCENE_REGISTRY[args.scene].label,
        config={
            "scene": args.scene,
            "max_order": args.max_order,
            "n_samples": args.n_samples,
            "noise": args.noise,
            "bin_thresh": args.bin_thresh,
        },
        auc_scores=auc_scores,
        extra_metrics=extra_metrics,
    )
    print(f"[unified this] summary written: {metrics_path.parent / 'summary.json'}")


if __name__ == "__main__":
    main()
