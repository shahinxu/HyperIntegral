import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(__file__).resolve().parent
HYPERPINN_ROOT = BASELINE_ROOT / 'HyperPINN'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypergraph.outputs import write_standard_summary
from hypergraph.scene_registry import SCENE_REGISTRY

SCENE_TO_SCRIPT = {
    'rossler': 'HyperPINN_Rossler.py',
}

# Supported Rossler hypergraph presets
ROSSLER_PRESETS = {
    "n8": {"preset": "n8", "n_nodes": 8},
    "n16": {"preset": "n16", "n_nodes": 16},
    "n32": {"preset": "n32", "n_nodes": 32},
    "n64": {"preset": "n64", "n_nodes": 64},
    "n100": {"preset": "n100", "n_nodes": 100},
}


def parse_auc_file(path: Path):
    auc_scores = {}
    if not path.exists():
        return auc_scores
    pattern = re.compile(r'(\d+-edges)\s*[:=]\s*([0-9.]+|N/A)')
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        match = pattern.search(line)
        if not match:
            continue
        auc_scores[match.group(1)] = None if match.group(2) == 'N/A' else float(match.group(2))
    return auc_scores


def extract_results_dir(stdout: str):
    pattern = re.compile(r'Results will be saved to:\s*(.+)')
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1).strip()
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
        print(line, end='', flush=True)

    return proc.wait(), ''.join(output_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default=None, choices=sorted(ROSSLER_PRESETS.keys()),
                        help='Rossler hypergraph preset (n8, n16, n32, n64, n100)')
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--n_nodes', type=int, default=100)
    parser.add_argument('--max_order', type=int, default=3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--tucker_rank', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=14000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--tmax', type=float, default=20.0)
    parser.add_argument('--results_root', type=str, default='results/hyperpinn_tucker')
    args = parser.parse_args()

    scene = 'rossler'

    # Handle preset selection
    if args.preset is not None:
        preset_config = ROSSLER_PRESETS[args.preset]
        effective_n_nodes = preset_config["n_nodes"]
        os.environ["ROSSLER_HYPERGRAPH_PRESET"] = preset_config["preset"]
    else:
        effective_n_nodes = args.n_nodes

    effective_max_order = int(args.max_order)
    if effective_max_order not in (2, 3):
        raise ValueError(f'hyperpinn_tucker only supports max_order in {{2, 3}}, got {effective_max_order}.')

    script_path = HYPERPINN_ROOT / SCENE_TO_SCRIPT[scene]
    cmd = [
        sys.executable,
        '-u',
        str(script_path),
        '--M',
        str(args.n_samples),
        '--N',
        str(effective_n_nodes),
        '--max_order',
        str(effective_max_order),
        '--gpu_id',
        str(args.gpu_id),
        '--noise',
        str(args.noise),
        '--tucker_rank',
        str(args.tucker_rank),
        '--epochs',
        str(args.n_epochs),
        '--lr',
        str(args.lr),
        '--tmax',
        str(args.tmax),
    ]

    env = os.environ.copy()
    env['HYPERPINN_RESULTS_ROOT'] = args.results_root
    env['PYTHONUNBUFFERED'] = '1'

    returncode, combined_output = run_and_stream(cmd, cwd=str(PROJECT_ROOT), env=env)
    if returncode != 0:
        raise SystemExit(returncode)

    results_dir = extract_results_dir(combined_output)
    if results_dir is None:
        raise RuntimeError('Could not infer results directory from baseline output')

    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
        if not results_path.exists():
            results_path = BASELINE_ROOT / results_dir

    auc_scores = parse_auc_file(results_path / 'auc_scores.txt')
    write_standard_summary(
        save_dir=str(results_path),
        method='baseline_hyperpinn_tucker',
        scene=SCENE_REGISTRY[scene].label,
        config={
            'scene': scene,
            'preset': args.preset,
            'n_samples': args.n_samples,
            'n_nodes': effective_n_nodes,
            'max_order': effective_max_order,
            'gpu_id': args.gpu_id,
            'noise': args.noise,
            'tucker_rank': args.tucker_rank,
            'tmax': args.tmax,
            'n_epochs': args.n_epochs,
            'lr': args.lr,
        },
        auc_scores=auc_scores,
    )
    print(f"[unified baseline] summary written: {results_path / 'summary.json'}")


if __name__ == '__main__':
    main()
