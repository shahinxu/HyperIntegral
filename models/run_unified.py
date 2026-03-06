import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Unified entrypoint for HyperPINN methods")
    parser.add_argument("--method", required=True, choices=["integral", "kernel", "baseline", "this"])
    parser.add_argument("--scene", required=True, choices=["ecological", "neuronal", "rossler", "social"])
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--python", type=str, default=None, help="Python executable for THIS bridge exporters")
    parser.add_argument("--bin_thresh", type=float, default=1e-4)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        "-m",
        f"models.{args.method}.run",
        "--scene",
        args.scene,
        "--n_samples",
        str(args.n_samples),
        "--noise",
        str(args.noise),
        "--gpu_id",
        str(args.gpu_id),
        "--n_epochs",
        str(args.n_epochs),
        "--lr",
        str(args.lr),
        "--n_trajectories",
        str(args.n_trajectories),
        "--bin_thresh",
        str(args.bin_thresh),
    ]
    if args.n_nodes is not None:
        cmd.extend(["--n_nodes", str(args.n_nodes)])
    if args.max_order is not None:
        cmd.extend(["--max_order", str(args.max_order)])
    if args.results_root is not None:
        cmd.extend(["--results_root", args.results_root])
    if args.python:
        cmd.extend(["--python", args.python])

    proc = subprocess.run(cmd, cwd=str(project_root))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
