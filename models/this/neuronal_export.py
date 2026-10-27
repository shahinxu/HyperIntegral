import argparse
import json
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib_neuronal_synchronization.hypergraph import HypergraphModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--max_candidates_per_order", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="neuronal-data")
    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(THIS_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _seed = getattr(args, 'seed', None) or int(os.environ.get('HG_DATA_SEED', 42))
    np.random.seed(_seed)
    os.environ.setdefault("HG_DATA_SEED", str(_seed))
    defaults = HypergraphModel.get_default_params()
    n_nodes = args.n_nodes or defaults["n_nodes"]
    max_order = args.max_order or defaults["max_order"]
    edge_config = HypergraphModel.get_hyperedge_config(n_nodes, max_order=max_order)
    t, x_data = HypergraphModel.generate_training_data(
        n_nodes,
        edge_config,
        n_samples=args.n_samples,
        noise=args.noise,
    )

    x_tensor = torch.as_tensor(x_data, dtype=torch.float32)
    y_tensor = HypergraphModel.dynamic_f_batch(x_tensor, n_nodes)

    # Use theta as scalar node state for THIS (rows=nodes, cols=time)
    X = np.asarray(x_data[:, :, 0].T, dtype=np.float64)
    Y = np.asarray(y_tensor[:, :, 0].detach().cpu().numpy().T, dtype=np.float64)

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

    # Every other method on this scene is scored against a subsampled candidate
    # pool (all true hyperedges plus a capped number of negatives per order),
    # because the kernel baseline cannot solve the full 174406-column system.
    # Export the same pool so THIS is scored against identical negatives.
    if args.max_candidates_per_order:
        from models.integral.train import build_orderwise_support_sampled
        sup = build_orderwise_support_sampled(
            n_nodes, max_order, args.max_candidates_per_order, edge_config, seed=_seed)
        order_of = {"edges": 2, "triangles": 3, "quads": 4, "quints": 5}
        pool = {str(o): [[int(v) for v in e] for e in sup.get(k, [])]
                for k, o in order_of.items()}
        with open(os.path.join(out_dir, "candidates.json"), "w") as f:
            json.dump(pool, f)
        print("Exported candidate pool: "
              + ", ".join(f"order{o}={len(v)}" for o, v in sorted(pool.items())))

    print(f"Exported neuronal data to: {out_dir}")


if __name__ == "__main__":
    main()
