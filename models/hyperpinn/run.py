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



def parse_auc_from_stdout(stdout: str):
    """The inner scripts only print their AUCs, they never write auc_scores.txt,
    so the summary used to come out empty.  Take the last reported line."""
    pat = re.compile(r"AUC \((\d+-edges)\):\s*([0-9.]+|nan|N/A)")
    best = {}
    for ln in stdout.splitlines():
        found = pat.findall(ln)
        if found:
            best = {k: (None if v in ("nan", "N/A") else float(v)) for k, v in found}
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_TO_SCRIPT.keys()))
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_candidates_per_order", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--results_root", type=str, default="results/hyperpinn")
    parser.add_argument("--python", type=str, default=None)
    parser.add_argument("--bin_thresh", type=float, default=1e-4)
    args = parser.parse_args()

    HypergraphModel, _ = get_scene_model(args.scene)
    defaults = HypergraphModel.get_default_params()

    effective_max_order = defaults["max_order"] if args.max_order is None else args.max_order

    script = SCENE_TO_SCRIPT[args.scene]
    script_path = HYPERPINN_ROOT / script
    cmd = [sys.executable, "-u", str(script_path), "--M", str(args.n_samples), "--gpu_id", str(args.gpu_id), "--noise", str(args.noise)]
    if args.scene not in FILE_DRIVEN_SCENES and args.n_nodes is not None:
        cmd.extend(["--N", str(args.n_nodes)])
    if args.scene not in FILE_DRIVEN_SCENES and effective_max_order is not None:
        cmd.extend(["--max_order", str(effective_max_order)])

    cmd.extend(["--seed", str(args.seed)])
    if args.max_candidates_per_order:
        cmd.extend(["--max_candidates_per_order", str(args.max_candidates_per_order)])

    cmd.extend(["--epochs", str(args.n_epochs), "--lr", str(args.lr)])

    if args.scene == "rossler" and args.n_nodes is not None:

        # Without this the n100 preset is loaded whatever --n_nodes says.

        os.environ["ROSSLER_HYPERGRAPH_PRESET"] = f"n{args.n_nodes}"

    env = os.environ.copy()
    env["HYPERPINN_RESULTS_ROOT"] = args.results_root
    if args.run_tag:
        env["HYPERPINN_RUN_TAG"] = args.run_tag
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

    auc_scores = parse_auc_from_stdout(combined_output) or parse_auc_file(results_path / "auc_scores.txt")
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
            "seed": args.seed,
        },
        auc_scores=auc_scores,
    )
    print(f"[unified baseline] summary written: {results_path / 'summary.json'}")


if __name__ == "__main__":
    main()
