#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class RunRecord:
    method_label: str
    node: int
    run_dir: Path
    timestamp: datetime
    auc2: float
    auc3: float
    macro_auc: float
    source: str = "measured"


def write_csv(out_csv: Path, nodes: List[int], latest: Dict[Tuple[str, int], RunRecord]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "node", "auc2", "auc3", "macro_auc", "source", "run_dir", "timestamp"])
        for method in ["CP", "Tucker", "TT", "ITC", "base"]:
            for node in nodes:
                rec = latest.get((method, node))
                if rec is None:
                    continue
                writer.writerow([
                    method,
                    node,
                    f"{rec.auc2:.6f}",
                    f"{rec.auc3:.6f}",
                    f"{rec.macro_auc:.6f}",
                    rec.source,
                    str(rec.run_dir),
                    rec.timestamp.isoformat() if rec.timestamp != datetime.min else "",
                ])


def plot_macro_auc(nodes: List[int], latest: Dict[Tuple[str, int], RunRecord], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    METHOD_MARKERS = {
        "ITC":    ("o", 14),   # circle
        "CP":     ("D", 12),   # diamond
        "Tucker": ("s", 13),   # square
        "TT":     ("^", 14),   # triangle up
        "base":   ("P", 14),   # thick plus
    }
    METHOD_COLORS = {
        "ITC": "#1f77b4",      # blue
        "CP": "#e3a008",       # amber
        "Tucker": "#2ca02c",   # green
        "TT": "#d62728",       # red
        "base": "#111111",     # near-black
    }

    plt.figure(figsize=(8, 7))
    ax = plt.gca()

    methods = ["CP", "Tucker", "TT", "ITC", "base"]
    for method in methods:
        x_vals: List[int] = []
        y_vals: List[float] = []
        estimated_x: List[int] = []
        estimated_y: List[float] = []
        measured_x: List[int] = []
        measured_y: List[float] = []
        for node in nodes:
            rec = latest.get((method, node))
            if rec is None:
                continue
            x_vals.append(node)
            y_vals.append(rec.macro_auc)
            if rec.source == "estimated":
                estimated_x.append(node)
                estimated_y.append(rec.macro_auc)
            else:
                measured_x.append(node)
                measured_y.append(rec.macro_auc)

        if not x_vals:
            continue

        marker, ms = METHOD_MARKERS.get(method, ("o", 12))
        line, = ax.plot(
            x_vals, y_vals,
            linewidth=2.5,
            linestyle="--" if estimated_x else "-",
            label=method,
            color=METHOD_COLORS.get(method),
        )
        color = line.get_color()
        if measured_x:
            ax.plot(
                measured_x, measured_y,
                linestyle="None",
                marker=marker,
                markersize=ms,
                markerfacecolor=color,
                markeredgecolor=color,
            )
        if estimated_x:
            ax.plot(
                estimated_x, estimated_y,
                linestyle="None",
                marker=marker,
                markersize=ms,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2.0,
            )

    ax.set_xlabel("Node Count", fontsize=20, fontweight="bold", labelpad=10)
    ax.set_ylabel("Macro-AUC", fontsize=20, fontweight="bold", labelpad=10)
    ax.set_xscale("log")
    ax.set_xticks(nodes)
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
    """Load precomputed Macro-AUC values from a CSV with columns: method,node,macro_auc[,source]."""
    latest: Dict[Tuple[str, int], RunRecord] = {}
    nodes_set: set = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"].strip()
            node = int(row["node"].strip())
            macro_text = (row.get("macro_auc") or "").strip()
            source = (row.get("source") or "measured").strip() or "measured"
            macro_auc = float(macro_text) if macro_text else None
            if macro_auc is None:
                continue
            nodes_set.add(node)
            latest[(method, node)] = RunRecord(
                method_label=method,
                node=node,
                run_dir=Path("."),
                timestamp=datetime.min,
                auc2=macro_auc,
                auc3=macro_auc,
                macro_auc=macro_auc,
                source=source,
            )
    return latest, sorted(nodes_set)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Macro-AUC vs node count using external CSV only.")
    parser.add_argument("--csv", default="results/figures/macro_auc_external.csv", help="External CSV data source (method,node,macro_auc[,source]).")
    parser.add_argument("--out-png", default="results/figures/macro_auc_vs_nodes.png", help="Output figure path.")
    parser.add_argument("--out-csv", default="results/figures/macro_auc_vs_nodes.csv", help="Output table path.")
    parser.add_argument("--title", default="Macro-AUC vs Node Count", help="Plot title.")
    args = parser.parse_args()

    latest, nodes = load_from_csv(Path(args.csv))

    write_csv(Path(args.out_csv), nodes, latest)
    plot_macro_auc(nodes, latest, Path(args.out_png), args.title)

    print(f"Loaded CSV: {args.csv}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved plot: {args.out_png}")

    for method in ["CP", "Tucker", "TT", "ITC", "base"]:
        for node in nodes:
            rec = latest.get((method, node))
            if rec is None:
                continue
            print(
                f"[OK] {method:>6} n={node} "
                f"Macro-AUC={rec.macro_auc:.4f} (AUC2={rec.auc2:.4f}, AUC3={rec.auc3:.4f})"
            )


if __name__ == "__main__":
    main()
