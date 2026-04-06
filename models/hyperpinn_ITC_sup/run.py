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
    "rossler": "HyperPINN_Rossler.py",
}

FILE_DRIVEN_SCENES = set()

# Supported Rossler hypergraph presets
ROSSLER_PRESETS = {
    "n8": {"preset": "n8", "n_nodes": 8},
    "n16": {"preset": "n16", "n_nodes": 16},
    "n32": {"preset": "n32", "n_nodes": 32},
    "n64": {"preset": "n64", "n_nodes": 64},
    "n100": {"preset": "n100", "n_nodes": 100},
    "n300": {"preset": "n300", "n_nodes": 300},
    "n500": {"preset": "n500", "n_nodes": 500},
    "n1000": {"preset": "n1000", "n_nodes": 1000},
}


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


def run_and_stream(cmd: list[str], *, cwd: str, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)

    return proc.wait(), "".join(output_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default=None, choices=sorted(ROSSLER_PRESETS.keys()),
                        help="Rossler hypergraph preset (n8, n16, n32, n64, n100)")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--max_order", type=int, default=3)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--results_root", type=str, default="results/hyperpinn")
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--bin_thresh", type=float, default=1e-4)
    parser.add_argument("--itc_rank", type=int, default=16)
    parser.add_argument("--observed_fraction", type=float, default=0.5)
    parser.add_argument("--completion_neg_ratio", type=float, default=1.0)
    parser.add_argument("--completion_seed", type=int, default=42)
    args = parser.parse_args()

    scene = "rossler"

    HypergraphModel, _ = get_scene_model(scene)
    defaults = HypergraphModel.get_default_params()

    # Handle preset selection
    if args.preset is not None:
        preset_config = ROSSLER_PRESETS[args.preset]
        effective_n_nodes = preset_config["n_nodes"]
        os.environ["ROSSLER_HYPERGRAPH_PRESET"] = preset_config["preset"]
    else:
        effective_n_nodes = args.n_nodes if args.n_nodes is not None else defaults["n_nodes"]

    effective_max_order = int(args.max_order)
    if effective_max_order not in (2, 3):
        raise ValueError(f"hyperpinn_ITC only supports max_order in {{2, 3}}, got {effective_max_order}.")

    script = SCENE_TO_SCRIPT[scene]
    script_path = HYPERPINN_ROOT / script
    python_executable = args.python or sys.executable
    cmd = [
        python_executable,
        "-u",
        str(script_path),
        "--M",
        str(args.n_samples),
        "--gpu_id",
        str(args.gpu_id),
        "--noise",
        str(args.noise),
        "--epochs",
        str(args.n_epochs),
        "--lr",
        str(args.lr),
        "--itc_rank",
        str(args.itc_rank),
        "--observed_fraction",
        str(args.observed_fraction),
        "--completion_neg_ratio",
        str(args.completion_neg_ratio),
        "--completion_seed",
        str(args.completion_seed),
    ]
    if effective_n_nodes is not None:
        cmd.extend(["--N", str(effective_n_nodes)])
    if effective_max_order is not None:
        cmd.extend(["--max_order", str(effective_max_order)])

    env = os.environ.copy()
    env["HYPERPINN_RESULTS_ROOT"] = args.results_root
    env["PYTHONUNBUFFERED"] = "1"

    returncode, combined_output = run_and_stream(cmd, cwd=str(PROJECT_ROOT), env=env)
    if returncode != 0:
        raise SystemExit(returncode)

    results_dir = extract_results_dir(combined_output)
    if results_dir is None:
        raise RuntimeError("Could not infer results directory from baseline output")

    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
        if not results_path.exists():
            results_path = BASELINE_ROOT / results_dir

    auc_scores = parse_auc_file(results_path / "auc_scores.txt")
    write_standard_summary(
        save_dir=str(results_path),
        method="hyperpinn_itc",
        scene=SCENE_REGISTRY[scene].label,
        config={
            "scene": scene,
            "preset": args.preset,
            "n_samples": args.n_samples,
            "n_nodes": effective_n_nodes,
            "max_order": effective_max_order,
            "gpu_id": args.gpu_id,
            "noise": args.noise,
            "itc_rank": args.itc_rank,
            "observed_fraction": args.observed_fraction,
            "completion_neg_ratio": args.completion_neg_ratio,
        },
        auc_scores=auc_scores,
    )
    print(f"[unified baseline] summary written: {results_path / 'summary.json'}")


if __name__ == "__main__":
    main()
