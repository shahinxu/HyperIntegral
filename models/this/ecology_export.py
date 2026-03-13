import argparse
import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib_ecological_dynamics.hypergraph import HypergraphModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="ecology-data")
    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(THIS_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    defaults = HypergraphModel.get_default_params()
    n_nodes = defaults["n_nodes"]
    max_order = defaults["max_order"]
    edge_config = HypergraphModel.get_hyperedge_config(n_nodes, max_order=max_order, seed=args.seed)

    cached = HypergraphModel._cached_hypergraph
    if cached is None:
        raise RuntimeError("Ecological cache empty after get_hyperedge_config")

    t_full = np.asarray(cached["t"], dtype=np.float64)
    x_full = np.asarray(cached["X"], dtype=np.float64).T  # [T, N]

    if args.n_samples < len(t_full):
        idx = np.linspace(0, len(t_full) - 1, args.n_samples).astype(int)
        t = t_full[idx]
        x = x_full[idx]
    else:
        t = t_full
        x = x_full

    if args.noise > 0:
        x = x + np.random.randn(*x.shape) * args.noise

    # THIS expects rows=nodes, cols=time
    X = x.T

    # Derivative estimates for THIS input Y
    dt = np.gradient(t)
    y = np.gradient(x, axis=0) / dt[:, None]
    Y = y.T

    np.savetxt(os.path.join(out_dir, "X.csv"), X, delimiter=",")
    np.savetxt(os.path.join(out_dir, "Y.csv"), Y, delimiter=",")

    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_nodes": n_nodes,
                "max_order": max_order,
                "n_samples": int(X.shape[1]),
                "edges": edge_config.get("edges", []),
                "triangles": edge_config.get("triangles", []),
                "quads": edge_config.get("quads", []),
                "quints": edge_config.get("quints", []),
                "sexts": edge_config.get("sexts", []),
                "septs": edge_config.get("septs", []),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Exported ecology data to: {out_dir}")


if __name__ == "__main__":
    main()
