import argparse
import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib_social_contagion.hypergraph import HypergraphModel


def finite_difference_y(t: np.ndarray, x_tn: np.ndarray) -> np.ndarray:
    dt = np.gradient(t)
    return np.gradient(x_tn, axis=0) / dt[:, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--train_seed", type=int, default=123)
    parser.add_argument("--test_seed", type=int, default=456)
    parser.add_argument("--out_dir", type=str, default="social-data")
    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(THIS_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    defaults = HypergraphModel.get_default_params()
    n_nodes = defaults["n_nodes"]
    max_order = defaults["max_order"]
    edge_config = HypergraphModel.get_hyperedge_config(n_nodes, max_order=max_order)

    edges = [[i - 1, j - 1] for i, j in edge_config.get("edges", [])]
    triangles = [[i - 1, j - 1, k - 1] for i, j, k in edge_config.get("triangles", [])]
    quads = [[a - 1, b - 1, c - 1, d - 1] for a, b, c, d in edge_config.get("quads", [])]
    quints = [[a - 1, b - 1, c - 1, d - 1, e - 1] for a, b, c, d, e in edge_config.get("quints", [])]

    complex_dict = {
        "nodes": np.arange(n_nodes),
        "edges": edges,
        "triangles": triangles,
        "quads": quads,
        "quints": quints,
    }

    def export_split(split_name: str, split_seed: int):
        params = HypergraphModel.SCMParams(n_nodes=n_nodes, t_max=50.0, seed=split_seed)
        sim = HypergraphModel._simulate(params, complex_dict, n_steps=args.n_samples)
        t = np.asarray(sim["t"], dtype=np.float64)
        x = np.asarray(sim["X_observed"], dtype=np.float64).T  # [T, N], observed only

        if args.noise > 0:
            x = np.clip(x + np.random.randn(*x.shape) * args.noise, 0.0, 1.0)

        y = finite_difference_y(t, x)

        # THIS expects rows=nodes, cols=time
        X = x.T
        Y = y.T
        np.savetxt(os.path.join(out_dir, f"X_{split_name}.csv"), X, delimiter=",")
        np.savetxt(os.path.join(out_dir, f"Y_{split_name}.csv"), Y, delimiter=",")

    export_split("train", args.train_seed)
    export_split("test", args.test_seed)

    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_nodes": n_nodes,
                "max_order": max_order,
                "n_samples": args.n_samples,
                "train_seed": args.train_seed,
                "test_seed": args.test_seed,
                "edges": edge_config.get("edges", []),
                "triangles": edge_config.get("triangles", []),
                "quads": edge_config.get("quads", []),
                "quints": edge_config.get("quints", []),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Exported strict social contagion train/test data to: {out_dir}")


if __name__ == "__main__":
    main()
