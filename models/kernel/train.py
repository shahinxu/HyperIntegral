import argparse
import os
from datetime import datetime
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from hypergraph.scene_registry import get_scene_model
from hypergraph.outputs import write_standard_summary

HypergraphModel = None


def rbf_kernel_smooth_and_derivative(t: np.ndarray, y: np.ndarray, length_scale: float, ridge: float):
    t_col = t[:, None]
    dt = t_col - t_col.T
    k = np.exp(-(dt ** 2) / (2.0 * length_scale ** 2))
    alpha = np.linalg.solve(k + ridge * np.eye(len(t)), y)
    y_hat = k @ alpha
    dk_dt = -(dt / (length_scale ** 2)) * k
    dy_dt = dk_dt @ alpha
    return y_hat, dy_dt


def smooth_dataset_with_kernel(t: np.ndarray, x_data: np.ndarray, length_scale: float, ridge: float):
    _, n_nodes, state_dim = x_data.shape
    x_smooth = np.zeros_like(x_data)
    dx_dt = np.zeros_like(x_data)
    for node_idx in range(n_nodes):
        for coord_idx in range(state_dim):
            y = x_data[:, node_idx, coord_idx]
            y_hat, y_dot = rbf_kernel_smooth_and_derivative(t, y, length_scale, ridge)
            x_smooth[:, node_idx, coord_idx] = y_hat
            dx_dt[:, node_idx, coord_idx] = y_dot
    return x_smooth, dx_dt


def rbf_kernel_matrix(x1: np.ndarray, x2: np.ndarray, sigma: float):
    x1_norm = np.sum(x1 * x1, axis=1, keepdims=True)
    x2_norm = np.sum(x2 * x2, axis=1, keepdims=True).T
    sqdist = np.maximum(x1_norm + x2_norm - 2.0 * (x1 @ x2.T), 0.0)
    return np.exp(-sqdist / (2.0 * sigma * sigma))


def split_by_time(time_ids: np.ndarray, n_times: int):
    t1 = int(0.6 * n_times)
    t2 = int(0.8 * n_times)
    tr = np.where(time_ids < t1)[0]
    va = np.where((time_ids >= t1) & (time_ids < t2))[0]
    te = np.where(time_ids >= t2)[0]
    return tr, va, te


def build_candidate_pool(n_nodes: int, max_order: int, edge_config: dict, max_candidates_per_order: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    all_possible = HypergraphModel.generate_all_possible_hyperedges(n_nodes, max_order)

    pool = {"edges": [], "triangles": [], "quads": [], "quints": []}
    for key in ["edges", "triangles", "quads", "quints"]:
        candidates = all_possible.get(key, [])
        true_edges = edge_config.get(key, [])
        true_set = {tuple(sorted(e)) for e in true_edges}

        if len(candidates) <= max_candidates_per_order:
            selected = candidates
        else:
            false_candidates = [e for e in candidates if tuple(sorted(e)) not in true_set]
            n_keep_false = max(0, max_candidates_per_order - len(true_set))
            if n_keep_false > 0 and len(false_candidates) > n_keep_false:
                pick_idx = rng.choice(len(false_candidates), size=n_keep_false, replace=False)
                sampled_false = [false_candidates[i] for i in sorted(pick_idx.tolist())]
            else:
                sampled_false = false_candidates
            selected = [list(e) for e in sorted(true_set)] + sampled_false

        pool[key] = selected

    return pool


def build_affine_dataset(t, x_smooth, dx_dt, n_nodes, all_possible_torch):
    x_tensor = torch.tensor(x_smooth, dtype=torch.float32)
    dxdt_tensor = torch.tensor(dx_dt, dtype=torch.float32)

    y_rows = []
    f0_rows = []
    f_rows = []
    z_rows = []
    time_ids = []

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    t_scale = max(t_max - t_min, 1e-12)

    for ti in range(len(t)):
        x_t = x_tensor[ti]
        f0_t = HypergraphModel.dynamic_f(x_t, n_nodes).numpy()[:, 0]
        phi_t = HypergraphModel.dynamic_phi(x_t, all_possible_torch, n_nodes, torch.device("cpu")).numpy()[:, 0, :]

        for node in range(n_nodes):
            y_rows.append(float(dxdt_tensor[ti, node, 0].item()))
            f0_rows.append(float(f0_t[node]))
            f_rows.append(phi_t[node])
            z = np.concatenate(
                [
                    x_smooth[ti, :, 0],
                    np.array([(t[ti] - t_min) / t_scale, node / max(n_nodes - 1, 1)], dtype=np.float64),
                ]
            )
            z_rows.append(z)
            time_ids.append(ti)

    return (
        np.asarray(y_rows, dtype=np.float64),
        np.asarray(f0_rows, dtype=np.float64),
        np.asarray(f_rows, dtype=np.float64),
        np.asarray(z_rows, dtype=np.float64),
        np.asarray(time_ids, dtype=np.int64),
    )


def fit_affine_kernel_embedded(y, f0, fmat, z, train_idx, sigma, gamma, theta_ridge):
    y0 = y - f0
    ztr = z[train_idx]
    ftr = fmat[train_idx]
    ytr = y0[train_idx]

    ktr = rbf_kernel_matrix(ztr, ztr, sigma)
    psi = np.linalg.inv(ktr + gamma * np.eye(ktr.shape[0]))

    lhs = ftr.T @ psi @ ftr + theta_ridge * np.eye(ftr.shape[1])
    rhs = ftr.T @ psi @ ytr
    theta = np.linalg.solve(lhs, rhs)
    omega_k = psi @ (ytr - ftr @ theta)

    return theta, omega_k, ztr


def predict_affine_kernel_embedded(f0, fmat, z, theta, omega_k, ztr, sigma):
    kcross = rbf_kernel_matrix(z, ztr, sigma)
    return f0 + fmat @ theta + kcross @ omega_k


def compute_fit_percent(y_true, y_pred):
    num = np.linalg.norm(y_true - y_pred)
    den = np.linalg.norm(y_true - np.mean(y_true)) + 1e-12
    return 100.0 * (1.0 - num / den)


def compute_auc(theta: np.ndarray, candidate_pool: dict, edge_config: dict, max_order: int):
    key_to_label = {"edges": "2-edges", "triangles": "3-edges", "quads": "4-edges", "quints": "5-edges"}
    keys = ["edges", "triangles", "quads", "quints"]
    order_end = {2: 1, 3: 2, 4: 3, 5: 4}.get(max_order, 4)

    auc_scores = {}
    roc_data = {}

    start = 0
    for key in keys[:order_end]:
        label = key_to_label[key]
        possible = candidate_pool.get(key, [])
        true_set = {tuple(sorted(e)) for e in edge_config.get(key, [])}
        n = len(possible)
        theta_k = theta[start:start + n]
        start += n

        if n == 0:
            auc_scores[label] = None
            roc_data[label] = None
            continue

        score = np.abs(theta_k)
        score = score / (np.max(score) + 1e-12)
        y_true = np.array([1 if tuple(sorted(e)) in true_set else 0 for e in possible])

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, score)
            auc_scores[label] = auc(fpr, tpr)
            roc_data[label] = (fpr, tpr)
        else:
            auc_scores[label] = None
            roc_data[label] = None

    return auc_scores, roc_data


def plot_roc(roc_data, auc_scores, save_dir):
    plt.figure(figsize=(8, 6))
    for label, color in [("2-edges", "blue"), ("3-edges", "green"), ("4-edges", "red"), ("5-edges", "purple")]:
        if roc_data.get(label) is not None:
            fpr, tpr = roc_data[label]
            plt.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC={auc_scores[label]:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Social Contagion (Kernel-Embedded)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "roc_curves_order.png"), bbox_inches="tight", dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="social", choices=["ecological", "neuronal", "rossler", "social"])
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--results_root", type=str, default="results/kernel")
    args = parser.parse_args()

    global HypergraphModel
    HypergraphModel, scene_spec = get_scene_model(args.scene)

    defaults = HypergraphModel.get_default_params()
    n_nodes = args.n_nodes if args.n_nodes is not None else defaults["n_nodes"]
    max_order_default = defaults["max_order"]
    max_order = args.max_order if args.max_order is not None else max_order_default
    max_order = min(max_order, 5)

    edge_config = HypergraphModel.get_hyperedge_config(n_nodes, max_order=max_order)

    kernel_ridge = 1e-4
    theta_ridge = 1e-8
    sigma_list = [0.2, 0.5, 1.0, 2.0]
    gamma_list = [1e-6, 1e-5, 1e-4, 1e-3]
    max_candidates_per_order = 3000

    candidate_pool = build_candidate_pool(
        n_nodes=n_nodes,
        max_order=max_order,
        edge_config=edge_config,
        max_candidates_per_order=max_candidates_per_order,
        seed=42,
    )

    all_possible_torch = {
        "edges": torch.tensor(candidate_pool["edges"], dtype=torch.long) - 1 if len(candidate_pool["edges"]) > 0 else torch.empty((0, 2), dtype=torch.long),
        "triangles": torch.tensor(candidate_pool["triangles"], dtype=torch.long) - 1 if len(candidate_pool["triangles"]) > 0 else torch.empty((0, 3), dtype=torch.long),
        "quads": torch.tensor(candidate_pool["quads"], dtype=torch.long) - 1 if len(candidate_pool["quads"]) > 0 else torch.empty((0, 4), dtype=torch.long),
        "quints": torch.tensor(candidate_pool["quints"], dtype=torch.long) - 1 if len(candidate_pool["quints"]) > 0 else torch.empty((0, 5), dtype=torch.long),
        "sexts": torch.empty((0, 6), dtype=torch.long),
        "septs": torch.empty((0, 7), dtype=torch.long),
    }

    y_all, f0_all, f_all, z_all, tid_all = [], [], [], [], []

    for k in range(max(1, args.n_trajectories)):
        seed_k = 123 + k
        sig = inspect.signature(HypergraphModel.generate_training_data)
        kwargs = {
            "n_samples": args.n_samples,
            "noise": args.noise,
        }
        if "seed" in sig.parameters:
            kwargs["seed"] = seed_k
        t, x_data = HypergraphModel.generate_training_data(n_nodes, edge_config, **kwargs)
        time_range = float(t[-1] - t[0])
        kernel_length = 0.03 * time_range
        x_smooth, dx_dt = smooth_dataset_with_kernel(t, x_data, kernel_length, kernel_ridge)

        y, f0, fmat, z, tid = build_affine_dataset(t, x_smooth, dx_dt, n_nodes, all_possible_torch)
        y_all.append(y)
        f0_all.append(f0)
        f_all.append(fmat)
        z_all.append(z)
        tid_all.append(tid)

    y_vec = np.concatenate(y_all, axis=0)
    f0_vec = np.concatenate(f0_all, axis=0)
    f_mat = np.concatenate(f_all, axis=0)
    z_mat = np.concatenate(z_all, axis=0)
    time_ids = np.concatenate(tid_all, axis=0)

    tr, va, te = split_by_time(time_ids, len(t))

    best = None
    for sigma in sigma_list:
        for gamma in gamma_list:
            theta, omega_k, ztr = fit_affine_kernel_embedded(y_vec, f0_vec, f_mat, z_mat, tr, sigma, gamma, theta_ridge)
            yv = predict_affine_kernel_embedded(f0_vec[va], f_mat[va], z_mat[va], theta, omega_k, ztr, sigma)
            rmse = float(np.sqrt(np.mean((y_vec[va] - yv) ** 2)))
            if best is None or rmse < best["val_rmse"]:
                best = {"sigma": sigma, "gamma": gamma, "val_rmse": rmse}

    trva = np.concatenate([tr, va])
    theta, omega_k, ztr = fit_affine_kernel_embedded(
        y_vec,
        f0_vec,
        f_mat,
        z_mat,
        trva,
        best["sigma"],
        best["gamma"],
        theta_ridge,
    )

    yt = predict_affine_kernel_embedded(f0_vec[te], f_mat[te], z_mat[te], theta, omega_k, ztr, best["sigma"])
    test_rmse = float(np.sqrt(np.mean((y_vec[te] - yt) ** 2)))
    test_fit = compute_fit_percent(y_vec[te], yt)

    auc_scores, roc_data = compute_auc(theta, candidate_pool, edge_config, max_order)

    save_dir = os.path.join(
        args.results_root,
        scene_spec.label,
        f"sample_{args.n_samples}_noise_{args.noise}",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "auc_scores.txt"), "w") as f:
        f.write(f"SCENE={scene_spec.label}, LIB={scene_spec.module}\n")
        f.write(f"N={n_nodes}, max_order={max_order}, n_samples={args.n_samples}, noise={args.noise}, n_trajectories={args.n_trajectories}\n")
        f.write(
            f"candidate_pool(edges={len(candidate_pool['edges'])}, triangles={len(candidate_pool['triangles'])}, "
            f"quads={len(candidate_pool['quads'])}, quints={len(candidate_pool['quints'])}, cap_per_order={max_candidates_per_order})\n"
        )
        f.write(
            f"best sigma={best['sigma']}, gamma={best['gamma']}, val_rmse={best['val_rmse']:.6e}, test_rmse={test_rmse:.6e}, fit={test_fit:.4f}\n\n"
        )
        for key in ["2-edges", "3-edges", "4-edges", "5-edges"]:
            val = auc_scores.get(key)
            if val is None:
                f.write(f"{key}: N/A\n")
            else:
                f.write(f"{key}: {val:.6f}\n")

    np.save(os.path.join(save_dir, "theta_physics.npy"), theta.reshape(-1, 1))
    np.save(os.path.join(save_dir, "omega_kernel.npy"), omega_k.reshape(-1, 1))
    plot_roc(roc_data, auc_scores, save_dir)

    write_standard_summary(
        save_dir=save_dir,
        method="kernel_integral",
        scene=scene_spec.label,
        config={
            "n_nodes": n_nodes,
            "max_order": max_order,
            "n_samples": args.n_samples,
            "noise": args.noise,
            "n_trajectories": args.n_trajectories,
            "best_sigma": best["sigma"],
            "best_gamma": best["gamma"],
        },
        auc_scores=auc_scores,
        extra_metrics={
            "val_rmse": best["val_rmse"],
            "test_rmse": test_rmse,
            "test_fit_percent": test_fit,
        },
    )

    print("[Kernel Integral - unified]")
    print(f"  Scene: {scene_spec.label} ({scene_spec.module})")
    print(f"  Results saved to: {save_dir}")
    print(f"  best sigma={best['sigma']}, gamma={best['gamma']}, test_rmse={test_rmse:.6e}, fit={test_fit:.2f}%")
    for key in ["2-edges", "3-edges", "4-edges", "5-edges"]:
        val = auc_scores.get(key)
        if val is None:
            print(f"  {key}: AUC=N/A")
        else:
            print(f"  {key}: AUC={val:.4f}")


if __name__ == "__main__":
    main()
