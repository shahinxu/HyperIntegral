import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_DRIVEN_SCENES = {"ecological", "neuronal", "social"}


def main():
    parser = argparse.ArgumentParser(description="Unified implicit model runner")
    parser.add_argument("--scene", required=True, choices=["ecological", "neuronal", "rossler", "social"])
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--results_root", type=str, default="results/implicit")
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--head_hidden", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--max_candidates_per_order", type=int, default=4096)
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--bin_thresh", type=float, default=1e-4)
    args = parser.parse_args()

    # Keep a common interface across all methods; unsupported knobs are ignored.
    cmd = [
        sys.executable,
        "-m",
        "models.Implicit.train",
        "--scene",
        args.scene,
        "--n_samples",
        str(args.n_samples),
        "--n_epochs",
        str(args.n_epochs),
        "--lr",
        str(args.lr),
        "--gpu_id",
        str(args.gpu_id),
        "--noise",
        str(args.noise),
        "--n_trajectories",
        str(args.n_trajectories),
        "--embed_dim",
        str(args.embed_dim),
        "--head_hidden",
        str(args.head_hidden),
        "--eval_every",
        str(args.eval_every),
        "--max_candidates_per_order",
        str(args.max_candidates_per_order),
        "--results_root",
        args.results_root,
    ]
    if args.scene not in FILE_DRIVEN_SCENES and args.max_order is not None:
        cmd.extend(["--max_order", str(args.max_order)])
    if args.scene not in FILE_DRIVEN_SCENES and args.n_nodes is not None:
        cmd.extend(["--n_nodes", str(args.n_nodes)])

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
