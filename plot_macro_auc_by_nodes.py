#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


METHOD_ALIASES = {
    "CP": {"baseline_hyperpinn_cp", "hyperpinn_cp"},
    "Tucker": {"baseline_hyperpinn_tucker", "hyperpinn_tucker"},
    "TT": {"baseline_hyperpinn_tensor_train", "hyperpinn_tensor_train", "baseline_hyperpinn_tt"},
    "ITC": {"hyperpinn_itc", "baseline_hyperpinn_itc"},
    "base": {"baseline_hyperpinn", "hyperpinn_base", "base_hyperpinn"},
}


@dataclass
class RunRecord:
    method_label: str
    node: int
    run_dir: Path
    timestamp: datetime
    auc2: float
    auc3: float
    macro_auc: float


def _parse_timestamp(summary: dict, run_dir: Path) -> datetime:
    ts = summary.get("timestamp")
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            pass
    # Fallback to folder name style: YYYYMMDD_HHMMSS
    try:
        return datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.min


def _find_auc_from_summary(summary: dict) -> Tuple[Optional[float], Optional[float]]:
    auc = summary.get("auc", {}) or {}
    if not isinstance(auc, dict):
        return None, None

    def pick(keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in auc:
                try:
                    return float(auc[key])
                except (TypeError, ValueError):
                    return None
        return None

    auc2 = pick(["2-edges", "order2", "2", "pairwise"])
    auc3 = pick(["3-edges", "order3", "3", "third-order"])
    return auc2, auc3


def _find_auc_from_text(run_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    txt_path = run_dir / "auc_scores.txt"
    if not txt_path.exists():
        return None, None

    auc2, auc3 = None, None
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line_lower = line.lower().strip()
        if ":" not in line_lower:
            continue
        key, value = [part.strip() for part in line_lower.split(":", 1)]
        try:
            val = float(value)
        except ValueError:
            continue
        if key.startswith("2-"):
            auc2 = val
        elif key.startswith("3-"):
            auc3 = val
    return auc2, auc3


def _map_method(method_value: str) -> Optional[str]:
    method_value = (method_value or "").strip()
    for label, aliases in METHOD_ALIASES.items():
        if method_value in aliases:
            return label
    return None


def collect_records(results_root: Path, scene: str) -> List[RunRecord]:
    records: List[RunRecord] = []
    for summary_path in results_root.rglob("summary.json"):
        run_dir = summary_path.parent
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        method = _map_method(str(summary.get("method", "")))
        if method is None:
            continue

        config = summary.get("config", {}) or {}
        if str(config.get("scene", "")) != scene:
            continue

        node = config.get("n_nodes")
        try:
            node = int(node)
        except (TypeError, ValueError):
            continue

        auc2, auc3 = _find_auc_from_summary(summary)
        if auc2 is None or auc3 is None:
            alt2, alt3 = _find_auc_from_text(run_dir)
            auc2 = auc2 if auc2 is not None else alt2
            auc3 = auc3 if auc3 is not None else alt3

        if auc2 is None or auc3 is None:
            continue

        macro_auc = 0.5 * (auc2 + auc3)
        records.append(
            RunRecord(
                method_label=method,
                node=node,
                run_dir=run_dir,
                timestamp=_parse_timestamp(summary, run_dir),
                auc2=auc2,
                auc3=auc3,
                macro_auc=macro_auc,
            )
        )

    return records


def select_latest(records: List[RunRecord]) -> Dict[Tuple[str, int], RunRecord]:
    latest: Dict[Tuple[str, int], RunRecord] = {}
    for rec in records:
        key = (rec.method_label, rec.node)
        old = latest.get(key)
        if old is None or rec.timestamp >= old.timestamp:
            latest[key] = rec
    return latest


def write_csv(out_csv: Path, nodes: List[int], latest: Dict[Tuple[str, int], RunRecord]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "node", "auc2", "auc3", "macro_auc", "run_dir", "timestamp"])
        for method in ["CP", "Tucker", "TT", "ITC", "base"]:
            for node in nodes:
                rec = latest.get((method, node))
                if rec is None:
                    writer.writerow([method, node, "", "", "", "", ""])
                else:
                    writer.writerow([
                        method,
                        node,
                        f"{rec.auc2:.6f}",
                        f"{rec.auc3:.6f}",
                        f"{rec.macro_auc:.6f}",
                        str(rec.run_dir),
                        rec.timestamp.isoformat() if rec.timestamp != datetime.min else "",
                    ])


def plot_macro_auc(nodes: List[int], latest: Dict[Tuple[str, int], RunRecord], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    METHOD_MARKERS = {
        "CP":     ("o", 14),   # circle
        "Tucker": ("s", 13),   # square
        "TT":     ("^", 14),   # triangle up
        "ITC":    ("D", 12),   # diamond
        "base":   ("P", 14),   # thick plus
    }

    plt.figure(figsize=(8, 7))
    ax = plt.gca()

    methods = ["CP", "Tucker", "TT", "ITC", "base"]
    for method in methods:
        y_vals: List[float] = []
        for node in nodes:
            rec = latest.get((method, node))
            y_vals.append(rec.macro_auc if rec is not None else math.nan)

        marker, ms = METHOD_MARKERS.get(method, ("o", 12))
        ax.plot(
            nodes, y_vals,
            marker=marker, markersize=ms,
            linewidth=2.5, label=method,
        )

    ax.set_xlabel("Node Count", fontsize=20, fontweight="bold", labelpad=10)
    ax.set_ylabel("Macro-AUC", fontsize=20, fontweight="bold", labelpad=10)
    ax.set_xscale("log")
    ax.set_xticks([10, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.tick_params(axis="both", labelsize=16, width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=15, title_fontsize=15, prop={"weight": "bold", "size": 15})
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def load_from_csv(csv_path: Path) -> Tuple[Dict[Tuple[str, int], RunRecord], List[int]]:
    """Load precomputed Macro-AUC values from a CSV with columns: method,node,macro_auc."""
    latest: Dict[Tuple[str, int], RunRecord] = {}
    nodes_set: set = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"].strip()
            node = int(row["node"].strip())
            macro_auc = float(row["macro_auc"].strip())
            nodes_set.add(node)
            latest[(method, node)] = RunRecord(
                method_label=method,
                node=node,
                run_dir=Path("."),
                timestamp=datetime.min,
                auc2=macro_auc,
                auc3=macro_auc,
                macro_auc=macro_auc,
            )
    return latest, sorted(nodes_set)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Macro-AUC vs node count for multiple methods.")
    parser.add_argument("--results-root", default="results", help="Root directory that contains experiment outputs.")
    parser.add_argument("--scene", default="rossler", help="Scene filter from summary config, e.g. rossler.")
    parser.add_argument("--nodes", default="8,16,32,64,100", help="Comma-separated node counts.")
    parser.add_argument("--out-png", default="results/figures/macro_auc_vs_nodes.png", help="Output figure path.")
    parser.add_argument("--out-csv", default="results/figures/macro_auc_vs_nodes.csv", help="Output table path.")
    parser.add_argument("--title", default="Macro-AUC vs Node Count", help="Plot title.")
    parser.add_argument("--from-csv", default=None, help="Load precomputed Macro-AUC directly from CSV (method,node,macro_auc). Skips results scan.")
    args = parser.parse_args()

    if args.from_csv is not None:
        latest, nodes = load_from_csv(Path(args.from_csv))
        records = []
    else:
        nodes = [int(x.strip()) for x in args.nodes.split(",") if x.strip()]
        results_root = Path(args.results_root)
        records = collect_records(results_root=results_root, scene=args.scene)
        latest = select_latest(records)

    write_csv(Path(args.out_csv), nodes, latest)
    plot_macro_auc(nodes, latest, Path(args.out_png), args.title)

    if args.from_csv is None:
        print(f"Collected runs: {len(records)}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved plot: {args.out_png}")

    for method in ["CP", "Tucker", "TT", "ITC", "base"]:
        for node in nodes:
            rec = latest.get((method, node))
            if rec is None:
                print(f"[MISSING] {method:>6} n={node}")
            else:
                print(
                    f"[OK] {method:>6} n={node} "
                    f"Macro-AUC={rec.macro_auc:.4f} (AUC2={rec.auc2:.4f}, AUC3={rec.auc3:.4f})"
                )


if __name__ == "__main__":
    main()
