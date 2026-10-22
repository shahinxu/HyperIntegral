"""Sparse-regression baselines for hyperedge identification.

Two variants, both operating on exactly the same library of candidate
hyperedges and the same known basic dynamics f as the proposed method:

  * ``integral``   -- integral (weak-form) SINDy.  For a set of time windows
                      [t_i, t_j] the identity

                          x(t_j) - x(t_i) - \\int f dt = (\\int Phi dt) A

                      is a linear system in the interaction tensor A, solved by
                      sequentially-thresholded least squares (STLSQ).
  * ``derivative`` -- classical SINDy: the same STLSQ applied to the
                      collocation form  dx/dt - f = Phi A  with dx/dt obtained
                      by central finite differences.

This is the "why not just do linear regression / integral SINDy" baseline
requested by the reviewers.  Keeping the library, the f-term, the data
pipeline and the evaluation identical to the proposed method makes the
comparison attributable to the estimator alone.
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypergraph.outputs import write_standard_summary
from hypergraph.scene_registry import get_scene_model
from models.integral.train import (
    build_orderwise_support,
    build_orderwise_support_sampled,
    compute_auc_scores,
    infer_max_order_from_edge_config,
    infer_n_nodes_from_edge_config,
    macro_auc,
    plot_roc_curves,
    save_roc_curve_data,
    set_global_seed,
)

ORDER_KEYS = ["edges", "triangles", "quads", "quints"]
ORDER_SIZES = {"edges": 2, "triangles": 3, "quads": 4, "quints": 5}


def stlsq(G, b, threshold, ridge=1e-8, n_iters=20):
    """Sequentially-thresholded least squares (the SINDy estimator).

    G: [M, E] design matrix, b: [M] target.  Returns coefficients [E].
    Small coefficients are zeroed and permanently removed from the active set,
    which is what makes SINDy an explicitly sparsity-promoting method.
    """
    E = G.shape[1]
    GtG = G.T @ G + ridge * np.eye(E)
    Gtb = G.T @ b
    xi = np.linalg.solve(GtG, Gtb)
    active = np.ones(E, dtype=bool)
    for _ in range(n_iters):
        small = np.abs(xi) < threshold
        new_active = ~small
        if new_active.sum() == 0:
            break
        if np.array_equal(new_active, active) and _ > 0:
            break
        active = new_active
        xi = np.zeros(E)
        Ga = G[:, active]
        xi[active] = np.linalg.solve(
            Ga.T @ Ga + ridge * np.eye(active.sum()), Ga.T @ b
        )
    return xi


def least_squares(G, b, ridge=1e-8):
    """Plain (dense) ridge least squares: the 'it is just linear regression'
    estimator the reviewers point at, with no sparsity at all."""
    E = G.shape[1]
    return np.linalg.solve(G.T @ G + ridge * np.eye(E), G.T @ b)


def prepare_trajectory(HypergraphModel, N, edge_config, n_samples, noise, seed,
                       target_points=500):
    """Reproduce the data pipeline used by models/integral/train.py:
    simulate, then resample onto a uniform dense grid by linear interpolation
    (skipped for the discrete social-contagion scene)."""
    gen_kwargs = {"noise": noise}
    if "seed" in inspect.signature(HypergraphModel.generate_training_data).parameters:
        gen_kwargs["seed"] = seed
    t_data, x_data = HypergraphModel.generate_training_data(
        N, edge_config, n_samples, **gen_kwargs
    )
    is_discrete_social = "lib_social_contagion" in getattr(HypergraphModel, "__module__", "")

    if is_discrete_social:
        # Discrete Bernoulli observations: smooth, keep the native grid.
        window, pad = 5, 2
        x_padded = np.pad(x_data, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        x_smooth = np.zeros_like(x_data)
        for i in range(len(t_data)):
            x_smooth[i] = x_padded[i:i + window].mean(axis=0)
        return t_data, x_smooth, True

    from scipy.interpolate import interp1d

    state_dim = x_data.shape[2]
    t_new = np.linspace(t_data[0], t_data[-1], target_points)
    x_new = np.zeros((target_points, N, state_dim))
    for n in range(N):
        for d in range(state_dim):
            f = interp1d(t_data, x_data[:, n, d], kind="linear", fill_value="extrapolate")
            x_new[:, n, d] = f(t_new)
    return t_new, x_new, False


def precompute_phi_f(HypergraphModel, x_data, t_data, orderwise_support, N, device):
    """Phi[t, n, d, e] and f[t, n, d] for every observation time."""
    n_times, _, state_dim = x_data.shape
    n_hyperedges = sum(len(orderwise_support.get(k, [])) for k in ORDER_KEYS)

    edges_gpu = {}
    for key in ORDER_KEYS:
        edges = orderwise_support.get(key, [])
        if len(edges) > 0:
            edges_gpu[key] = torch.tensor(edges, dtype=torch.long, device=device) - 1
        else:
            edges_gpu[key] = torch.empty((0, ORDER_SIZES[key]), dtype=torch.long, device=device)

    x_gpu = torch.tensor(x_data, dtype=torch.float32, device=device)
    t_gpu = torch.tensor(t_data, dtype=torch.float32, device=device)
    Phi = torch.zeros((n_times, N, state_dim, n_hyperedges), device=device)
    F = torch.zeros((n_times, N, state_dim), device=device)
    accepts_time = len(inspect.signature(HypergraphModel.dynamic_f).parameters) >= 3
    for i in tqdm(range(n_times), desc="Precomputing Phi", leave=False):
        Phi[i] = HypergraphModel.dynamic_phi(x_gpu[i], edges_gpu, N, device)
        F[i] = (HypergraphModel.dynamic_f(x_gpu[i], N, t_gpu[i]) if accepts_time
                else HypergraphModel.dynamic_f(x_gpu[i], N))
    return x_gpu, t_gpu, Phi, F, n_hyperedges


def build_integral_system(x_gpu, t_gpu, Phi, F, window):
    """Rows: every window [i, i+window]. Returns (G [M,E], b [M])."""
    n_times = x_gpu.shape[0]
    dt = (t_gpu[1:] - t_gpu[:-1]).view(-1, 1, 1)
    cum_f = torch.cat([torch.zeros_like(F[:1]), torch.cumsum(F[:-1] * dt, dim=0)], dim=0)
    dt4 = dt.unsqueeze(-1)
    cum_phi = torch.cat(
        [torch.zeros_like(Phi[:1]), torch.cumsum(Phi[:-1] * dt4, dim=0)], dim=0
    )
    i = torch.arange(0, n_times - window, device=x_gpu.device)
    j = i + window
    b = (x_gpu[j] - x_gpu[i]) - (cum_f[j] - cum_f[i])        # [W, N, D]
    G = cum_phi[j] - cum_phi[i]                              # [W, N, D, E]
    E = G.shape[-1]
    return G.reshape(-1, E).cpu().numpy().astype(np.float64), b.reshape(-1).cpu().numpy().astype(np.float64)


def build_derivative_system(x_gpu, t_gpu, Phi, F):
    """Rows: every interior time point, central-difference derivative."""
    dxdt = (x_gpu[2:] - x_gpu[:-2]) / (t_gpu[2:] - t_gpu[:-2]).view(-1, 1, 1)
    b = dxdt - F[1:-1]
    G = Phi[1:-1]
    E = G.shape[-1]
    return G.reshape(-1, E).cpu().numpy().astype(np.float64), b.reshape(-1).cpu().numpy().astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="SINDy / least-squares baselines")
    parser.add_argument("--scene", type=str, default="rossler",
                        choices=["ecological", "neuronal", "rossler", "social"])
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--form", type=str, default="integral",
                        choices=["integral", "derivative"],
                        help="integral: weak-form (integral SINDy); derivative: classical SINDy.")
    parser.add_argument("--estimator", type=str, default="stlsq",
                        choices=["stlsq", "lstsq"],
                        help="stlsq: sparsity-promoting SINDy; lstsq: plain linear regression.")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="STLSQ sparsity threshold (the one hyperparameter to tune).")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--window", type=int, default=20,
                        help="Integral window length in grid steps.")
    parser.add_argument("--max_candidates_per_order", type=int, default=0,
                        help="Subsample negative candidates per order (0 = full support).")
    parser.add_argument("--results_root", type=str, default="results/sindy")
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    HypergraphModel, scene_spec = get_scene_model(args.scene)
    defaults = HypergraphModel.get_default_params()
    if args.scene in {"ecological", "neuronal", "social"}:
        cfg = HypergraphModel.get_hyperedge_config(defaults["n_nodes"], defaults["max_order"])
        N = infer_n_nodes_from_edge_config(cfg) or defaults["n_nodes"]
        max_order = infer_max_order_from_edge_config(cfg) or defaults["max_order"]
    else:
        N = args.n_nodes if args.n_nodes is not None else defaults["n_nodes"]
        max_order = args.max_order if args.max_order is not None else defaults["max_order"]

    edge_config = HypergraphModel.get_hyperedge_config(N, max_order)
    if args.max_candidates_per_order and args.max_candidates_per_order > 0:
        orderwise_support = build_orderwise_support_sampled(
            N, max_order, args.max_candidates_per_order, edge_config, seed=args.seed)
    else:
        orderwise_support = build_orderwise_support(N, max_order)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  N={N} max_order={max_order}")

    t_data, x_data, is_discrete = prepare_trajectory(
        HypergraphModel, N, edge_config, args.n_samples, args.noise, args.seed
    )
    x_gpu, t_gpu, Phi, F, n_hyperedges = precompute_phi_f(
        HypergraphModel, x_data, t_data, orderwise_support, N, device
    )
    print(f"Candidate library size E={n_hyperedges}, T={len(t_data)}")

    if args.form == "integral":
        G, b = build_integral_system(x_gpu, t_gpu, Phi, F, max(1, args.window))
    else:
        G, b = build_derivative_system(x_gpu, t_gpu, Phi, F)
    print(f"Design matrix: {G.shape}, target: {b.shape}")

    if args.estimator == "stlsq":
        A = stlsq(G, b, threshold=args.threshold, ridge=args.ridge)
    else:
        A = least_squares(G, b, ridge=args.ridge)
    nnz = int((np.abs(A) > 0).sum())
    print(f"Recovered {nnz}/{n_hyperedges} non-zero coefficients")

    tag = args.run_tag or f"{args.form}_{args.estimator}_seed{args.seed}"
    save_dir = os.path.join(
        args.results_root, scene_spec.label,
        f"sample_{args.n_samples}_noise_{args.noise}", tag,
    )
    os.makedirs(save_dir, exist_ok=True)

    # SINDy coefficients are signed; a hyperedge is detected by coefficient
    # magnitude.  Score by |A| (the baseline-favourable choice) and also
    # report the signed-score AUC that the proposed method is evaluated with.
    A_mag = np.abs(A).reshape(-1, 1)
    auc_scores, roc_data = compute_auc_scores(A_mag, edge_config, N, max_order, orderwise_support)
    auc_signed, _ = compute_auc_scores(A.reshape(-1, 1), edge_config, N, max_order, orderwise_support)

    print("\nAUC (|coefficient| score):")
    for k, v in auc_scores.items():
        print(f"  {k}: {'N/A' if v is None else f'{v:.4f}'}")

    plot_roc_curves(roc_data, auc_scores, save_dir, max_order)
    save_roc_curve_data(A_mag, edge_config, N, max_order, save_dir, orderwise_support)
    np.save(os.path.join(save_dir, "coefficients.npy"), A)

    write_standard_summary(
        save_dir=save_dir,
        method=f"sindy[{args.form}/{args.estimator}]",
        scene=scene_spec.label,
        config={
            "n_nodes": N, "max_order": max_order, "n_samples": args.n_samples,
            "noise": args.noise, "seed": args.seed, "form": args.form,
            "estimator": args.estimator, "threshold": args.threshold,
            "window": args.window, "ridge": args.ridge,
            "max_candidates_per_order": args.max_candidates_per_order,
        },
        auc_scores=auc_scores,
        extra_metrics={
            "macro_auc": macro_auc(auc_scores),
            "macro_auc_signed_score": macro_auc(auc_signed),
            "n_nonzero": nnz,
            "library_size": n_hyperedges,
        },
    )
    print(f"\nResults saved to {save_dir}/")


if __name__ == "__main__":
    main()
