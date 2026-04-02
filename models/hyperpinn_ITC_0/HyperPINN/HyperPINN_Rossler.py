import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch import optim as optim

from HyperPINNTopology import HyperPINNTopology

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib_rossler_oscillator.hypergraph import HypergraphModel as RosslerHypergraphModel


def get_labels_and_scores(all_edges, true_edges, probs):
    y_true = []
    y_score = []
    for idx, edge in enumerate(all_edges):
        edge = tuple(sorted(edge))
        y_true.append(1 if edge in true_edges else 0)
        y_score.append(probs[idx])
    return np.array(y_true), np.array(y_score)


def compute_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def plot_roc(y_true, y_score, label):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.2f})", linewidth=2)
    return fpr, tpr, auc_score


def evaluate_edges_triangles(model, observed_trajectory, all_2edges, true_2edges, all_3edges, true_3edges, max_order):
    with torch.no_grad():
        edge_logits = model.score_order_edges("edges", all_2edges, observed_trajectory, one_based=True)
        edge_probs = torch.sigmoid(edge_logits).detach().cpu().numpy()
        triangle_probs = np.zeros(0)
        if max_order >= 3:
            tri_logits = model.score_order_edges("triangles", all_3edges, observed_trajectory, one_based=True)
            triangle_probs = torch.sigmoid(tri_logits).detach().cpu().numpy()

    y_true_2, y_score_2 = get_labels_and_scores(all_2edges, true_2edges, edge_probs)
    y_true_3, y_score_3 = get_labels_and_scores(all_3edges, true_3edges, triangle_probs)
    return y_true_2, y_score_2, y_true_3, y_score_3


def summarize_topology_stats(model):
    topo_prefixes = ("node_encoder.", "factor_head2.", "factor_head3.")
    topo_exact = {"raw_lambda2", "raw_lambda3"}

    grad_parts = []
    param_parts = []
    total_grad_sq = 0.0
    total_param_sq = 0.0

    for name, param in model.named_parameters():
        if not (name.startswith(topo_prefixes) or name in topo_exact):
            continue
        if name == "raw_lambda3" and model.max_order < 3:
            continue

        param_norm = float(param.detach().norm().cpu())
        grad_norm = 0.0 if param.grad is None else float(param.grad.detach().norm().cpu())
        total_grad_sq += grad_norm * grad_norm
        total_param_sq += param_norm * param_norm
        grad_parts.append(f"{name}={grad_norm:.3e}")
        param_parts.append(f"{name}={param_norm:.3e}")

    total_grad = total_grad_sq ** 0.5
    total_param = total_param_sq ** 0.5
    return total_grad, total_param, grad_parts, param_parts


def save_true_hyperedge_figures(results_dir, N, true_2edges, true_3edges, max_order, name_prefix="true"):
    orders = [2, 3] if max_order >= 3 else [2]
    true_lists = [sorted(true_2edges), sorted(true_3edges)][:len(orders)]

    G = nx.Graph()
    for n in range(1, N + 1):
        G.add_node(n)
    pos = nx.circular_layout(G)
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, len(orders), figsize=(6 * len(orders), 6))
    axes = np.atleast_1d(axes).flatten()
    for ax, order, true_list in zip(axes, orders, true_lists):
        xs = [pos[n][0] for n in G.nodes()]
        ys = [pos[n][1] for n in G.nodes()]
        ax.scatter(xs, ys, s=140, color="tab:blue")
        for n in G.nodes():
            ax.text(pos[n][0], pos[n][1], str(n), fontsize=11, ha="center", va="center", color="white")
        for idx, edge in enumerate(true_list):
            nodes = [int(v) for v in edge]
            color = cmap(idx % cmap.N)
            if len(nodes) == 2:
                i, j = nodes
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=color, linewidth=2.0, alpha=0.9)
            else:
                poly_x = [pos[n][0] for n in nodes] + [pos[nodes[0]][0]]
                poly_y = [pos[n][1] for n in nodes] + [pos[nodes[0]][1]]
                ax.plot(poly_x, poly_y, color=color, linewidth=2.0, alpha=0.9)

        ax.set_title(f"Order={order} (#{len(true_list)})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{name_prefix}_hyperedges_all_orders.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    defaults = RosslerHypergraphModel.get_default_params()

    parser = argparse.ArgumentParser(description="Run Rossler Oscillators with inductive tensor completion HyperPINN")
    parser.add_argument("--M", type=int, default=150)
    parser.add_argument("--tmax", type=float, default=20)
    parser.add_argument("--N", type=int, default=defaults["n_nodes"])
    parser.add_argument("--max_order", type=int, default=3, choices=[2, 3])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--itc_rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=14000)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    N = args.N
    max_order = args.max_order
    noise = args.noise
    M = args.M

    edge_config = RosslerHypergraphModel.get_hyperedge_config(N, max_order)
    edge_list = np.array(edge_config.get("edges", [])) if edge_config.get("edges") else np.empty((0, 2), dtype=int)
    triangle_list = np.array(edge_config.get("triangles", [])) if edge_config.get("triangles") else np.empty((0, 3), dtype=int)
    if max_order < 3:
        triangle_list = np.empty((0, 3), dtype=int)

    all_possible = RosslerHypergraphModel.generate_all_possible_hyperedges(N, max_order)
    all_2edges = all_possible["edges"]
    all_3edges = all_possible["triangles"]

    true_2edges = set(tuple(sorted(edge)) for edge in edge_list)
    true_3edges = set(tuple(sorted(edge)) for edge in triangle_list)

    t_eval, X = RosslerHypergraphModel.generate_training_data(
        N,
        edge_config,
        n_samples=M + 1,
        noise=0.0,
        tmax=args.tmax,
        flatten=True,
    )

    if noise > 0:
        np.random.seed()
        X_noisy = X + np.random.randn(*X.shape) * noise
        print(f"Added Gaussian noise with std={noise:.6f} (absolute)")
    else:
        X_noisy = X

    t_data = torch.tensor(t_eval, dtype=torch.float32, requires_grad=True).unsqueeze(1)
    x_data = torch.tensor(X_noisy, dtype=torch.float32)

    n_cols = 4
    n_rows = int(np.ceil(N / n_cols))
    plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    X_plot = x_data.cpu().numpy() if noise > 0 else X
    for i in range(N):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(t_eval, X_plot[:, i], "b-", label=f"x_{i + 1}", alpha=0.7)
        plt.plot(t_eval, X_plot[:, i + N], "r-", label=f"y_{i + 1}", alpha=0.7)
        plt.plot(t_eval, X_plot[:, i + 2 * N], "g-", label=f"z_{i + 1}", alpha=0.7)
        if noise > 0:
            plt.plot(t_eval, X[:, i], "b--", alpha=0.3, linewidth=0.5)
            plt.plot(t_eval, X[:, i + N], "r--", alpha=0.3, linewidth=0.5)
            plt.plot(t_eval, X[:, i + 2 * N], "g--", alpha=0.3, linewidth=0.5)
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.title(f"Node {i + 1}")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = os.environ.get("HYPERPINN_RESULTS_ROOT", os.path.join("results", "hyperpinn"))
    results_dir = os.path.join(
        results_base,
        "rossler",
        f"itc_rank_{args.itc_rank}",
        f"order_{max_order}",
        f"sample_{M}_noise_{noise}",
        timestamp,
    )
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "rossler_oscillators.png"))
    plt.close()
    print(f"Results will be saved to: {results_dir}")
    print(f"Inductive Tensor Completion mode enabled: rank={args.itc_rank}, max_order={max_order}")

    save_true_hyperedge_figures(results_dir, N, true_2edges, true_3edges, max_order)

    if args.gpu_id is not None and args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using default GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = HyperPINNTopology(
        N=N,
        output_dim=3 * N,
        use_resnet=False,
        use_attention=False,
        use_pirate=True,
        max_order=max_order,
        itc_rank=args.itc_rank,
    ).to(device)
    model.lambda_l1_edges = 0.03
    model.lambda_l1_triangles = 0.05

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    t_data = t_data.float().to(device)
    x_data = x_data.float().to(device)

    stage1_epochs = max(1, int(0.25 * args.epochs))
    stage2_epochs = max(stage1_epochs + 1, int(0.65 * args.epochs))
    losses = []
    sparsity_stats = []

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        x_pred = model.forward(t_data)
        data_loss = torch.mean((x_pred - x_data) ** 2)
        physics_loss = model.physics_loss(t_data, x_data)
        sparsity_loss, sparsity_info = model.sparsity_regularization(x_data)

        if epoch < stage1_epochs:
            data_weight = 1.0
            physics_weight = 0.1
            sparsity_weight = 0.0
            print_prefix = "Stage 1 (Data Fitting)"
        elif epoch < stage2_epochs:
            progress = (epoch - stage1_epochs) / (stage2_epochs - stage1_epochs)
            data_weight = 1.0
            physics_weight = 0.1 * progress
            sparsity_weight = 0.0
            print_prefix = "Stage 2 (Physics Warmup)"
        else:
            progress = min(1.0, (epoch - stage2_epochs) / max(1, args.epochs - stage2_epochs))
            data_weight = 1.0 - 0.8 * progress
            physics_weight = 0.1 + 0.9 * progress
            sparsity_weight = 0.01 * progress
            print_prefix = "Stage 3 (Physics + Sparsity)"

        total_loss = (
            data_weight * data_loss
            + physics_weight * physics_loss
            + sparsity_weight * sparsity_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(total_loss.item())
        sparsity_stats.append(sparsity_info)

        if epoch % 500 == 0:
            topo_grad_norm, topo_param_norm, grad_parts, param_parts = summarize_topology_stats(model)
            print(f"\n{'=' * 80}")
            print(f"{print_prefix} | Epoch {epoch}, Total Loss: {total_loss.item():.6f}")
            print(
                f"  Data: {data_loss.item():.6f}, "
                f"Physics: {physics_loss.item():.6f}, Sparsity: {sparsity_loss.item():.6f}"
            )
            print(
                f"  L1 edges: {sparsity_info['l1_edges']:.2f}, "
                f"L1 triangles: {sparsity_info['l1_triangles']:.2f}, "
                f"Factor penalty: {sparsity_info['l2_factor_penalty']:.6f}"
            )
            print(f"  Topology grad norm: {topo_grad_norm:.3e} | param norm: {topo_param_norm:.3e}")
            print(f"  Topology grad detail: {', '.join(grad_parts)}")
            print(f"  Topology param detail: {', '.join(param_parts)}")

            with torch.no_grad():
                X_pred = model.forward(t_data).cpu().numpy()
            X_train = x_data.detach().cpu().numpy()

            fig, axes = plt.subplots(N, 3, figsize=(15, 2.5 * N))
            coord_names = ["x", "y", "z"]
            for node_idx in range(N):
                for coord_idx in range(3):
                    ax = axes[node_idx, coord_idx]
                    ax.plot(
                        t_eval,
                        X_train[:, node_idx + coord_idx * N],
                        "o",
                        label="Data",
                        markersize=4,
                        alpha=0.7,
                        color="blue",
                    )
                    ax.plot(
                        t_eval,
                        X_pred[:, node_idx + coord_idx * N],
                        "-",
                        label="NN prediction",
                        linewidth=1,
                        alpha=0.8,
                        color="red",
                    )
                    if node_idx == 0:
                        ax.set_title(f"{coord_names[coord_idx]}-coordinate", fontsize=12)
                    if coord_idx == 0:
                        ax.set_ylabel(f"Node {node_idx + 1}", fontsize=11)
                    if node_idx == N - 1:
                        ax.set_xlabel("Time", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    if node_idx == 0 and coord_idx == 0:
                        ax.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"nn_prediction_epoch{epoch}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            y_true_2, y_score_2, y_true_3, y_score_3 = evaluate_edges_triangles(
                model,
                x_data,
                all_2edges,
                true_2edges,
                all_3edges,
                true_3edges,
                max_order,
            )
            auc_str = f"  AUC (2-edges): {compute_auc(y_true_2, y_score_2):.4f}"
            if max_order >= 3:
                auc_str += f", AUC (3-edges): {compute_auc(y_true_3, y_score_3):.4f}"
            print(auc_str)

    y_true_2, y_score_2, y_true_3, y_score_3 = evaluate_edges_triangles(
        model,
        x_data,
        all_2edges,
        true_2edges,
        all_3edges,
        true_3edges,
        max_order,
    )

    y_true_list = []
    y_score_list = []
    plt.figure(figsize=(8, 6))
    plot_roc(y_true_2, y_score_2, "Pairwise")
    y_true_list.append(y_true_2)
    y_score_list.append(y_score_2)
    if max_order >= 3:
        plot_roc(y_true_3, y_score_3, "Third-order")
        y_true_list.append(y_true_3)
        y_score_list.append(y_score_3)

    if y_true_list:
        y_true_total = np.concatenate(y_true_list)
        y_score_total = np.concatenate(y_score_list)
        plot_roc(y_true_total, y_score_total, label="All")
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title("ROC Curves for Identified Hypergraphs", fontsize=17)
    plt.legend(fontsize=14, loc="lower right")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(results_dir, f"roc_curves_{max_order}_order.png"), bbox_inches="tight")
    plt.close()

    auc_lines = [f"2-edges: {compute_auc(y_true_2, y_score_2):.6f}"]
    if max_order >= 3:
        auc_lines.append(f"3-edges: {compute_auc(y_true_3, y_score_3):.6f}")
    with open(os.path.join(results_dir, "auc_scores.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(auc_lines) + "\n")


if __name__ == "__main__":
    main()
