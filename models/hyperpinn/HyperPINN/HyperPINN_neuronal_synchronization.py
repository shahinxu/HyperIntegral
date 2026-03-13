import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from HyperPINNTopology import HyperPINNTopology
import torch
from torch import optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime
import networkx as nx
import argparse

from lib_neuronal_synchronization.hypergraph import HypergraphModel as NSHypergraphModel

parser = argparse.ArgumentParser(description="Run HyperPINN on neuronal synchronization hypergraph dynamics")
parser.add_argument("--M", type=int, default=300, help="Number of time samples")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--noise", type=float, default=0.0)
args = parser.parse_args()

M = args.M
gpu_id = args.gpu_id
noise = args.noise
defaults = NSHypergraphModel.get_default_params()
N = defaults["n_nodes"]
max_order = defaults["max_order"]

# ---------------------------------------------------------------------------
# Ground-truth hypergraph and data generation (neuronal synchronization)
# ---------------------------------------------------------------------------
edge_config = NSHypergraphModel.get_hyperedge_config(N, max_order)

EdgeList = np.array(edge_config.get("edges", [])) if edge_config.get("edges") else np.empty((0, 2), dtype=int)
TriangleList = np.array(edge_config.get("triangles", [])) if edge_config.get("triangles") else np.empty((0, 3), dtype=int)
QuadList = np.array(edge_config.get("quads", [])) if edge_config.get("quads") else np.empty((0, 4), dtype=int)

all_2edges = edge_config.get("edges", [])
all_3edges = edge_config.get("triangles", [])
all_4edges = edge_config.get("quads", [])
all_5edges = edge_config.get("quints", [])
all_6edges = []
all_7edges = []

true_2edges = set(tuple(sorted(edge)) for edge in EdgeList)
true_3edges = set(tuple(sorted(triangle)) for triangle in TriangleList)
true_4edges = set(tuple(sorted(quad)) for quad in QuadList)
true_5edges = set(tuple(sorted(edge)) for edge in all_5edges)
true_6edges = set()
true_7edges = set()

# generate_training_data returns continuous phases theta/phi per node
t_eval, X = NSHypergraphModel.generate_training_data(N, edge_config, n_samples=M, noise=noise)
state_dim = X.shape[2]  # typically 2 (theta, phi)
X_noisy = X
t_data = torch.tensor(t_eval, dtype=torch.float32, requires_grad=True).unsqueeze(1)
x_data = torch.tensor(X_noisy.reshape(X_noisy.shape[0], -1), dtype=torch.float32)

architectures = [("Pirate", False, False, True)]
arch_name, use_resnet, use_attention, use_pirate = architectures[0]

n_cols = 4
n_rows = int(np.ceil(N / n_cols))
plt.figure(figsize=(4 * n_cols, 3 * n_rows))
X_plot = x_data.cpu().numpy()
coord_names = ["x"]
for i in range(N):
    plt.subplot(n_rows, n_cols, i + 1)
    for coord_idx in range(state_dim):
        idx = i + coord_idx * N
        color = "b"
        label = f"{coord_names[coord_idx]}_{i+1}" if coord_idx < len(coord_names) else f"state{coord_idx+1}_{i+1}"
        plt.plot(t_eval, X_plot[:, idx], color + "-", label=label, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title(f"Node {i+1}")
    plt.legend()
    plt.grid(True)
plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_base = os.environ.get("HYPERPINN_RESULTS_ROOT", os.path.join("results", "hyperpinn"))
results_dir = os.path.join(
    results_base,
    "neuronal",
    f"sample_{M}_noise_{noise}",
    timestamp,
)
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "neuronal_timeseries.png"))
print(f"Results will be saved to: {results_dir}")

# ---------------------------------------------------------------------------
# Utility: visualize true/pred hyperedges (reuse ecosystem helper pattern)
# ---------------------------------------------------------------------------

def _save_true_hyperedge_figures(
    results_dir,
    N,
    true_2edges,
    true_3edges,
    true_4edges,
    true_5edges,
    true_6edges,
    true_7edges,
    name_prefix: str = "true",
):
    orders = [2, 3, 4, 5, 6, 7]
    true_lists = [
        sorted(true_2edges),
        sorted(true_3edges),
        sorted(true_4edges),
        sorted(true_5edges),
        sorted(true_6edges),
        sorted(true_7edges),
    ]

    G = nx.Graph()
    for n in range(1, N + 1):
        G.add_node(n)
    pos = nx.circular_layout(G)
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax, order, true_list in zip(axes, orders, true_lists):
        xs = [pos[n][0] for n in G.nodes()]
        ys = [pos[n][1] for n in G.nodes()]
        ax.scatter(xs, ys, s=140, color="tab:blue")
        for n in G.nodes():
            ax.text(pos[n][0], pos[n][1], str(n), fontsize=11, ha="center", va="center", color="white")
        if len(true_list) > 0:
            for idx, e in enumerate(true_list):
                nodes = [int(v) for v in e]
                color = cmap(idx % cmap.N)
                if len(nodes) == 2:
                    i, j = nodes
                    x = [pos[i][0], pos[j][0]]
                    y = [pos[i][1], pos[j][1]]
                    ax.plot(x, y, color=color, linewidth=2.0, alpha=0.9)
                else:
                    poly_x = [pos[n][0] for n in nodes] + [pos[nodes[0]][0]]
                    poly_y = [pos[n][1] for n in nodes] + [pos[nodes[0]][1]]
                    ax.plot(poly_x, poly_y, color=color, linewidth=2.0, alpha=0.9)

        ax.set_title(f"Order={order} (#{len(true_list)})")
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.tight_layout()
    fname = f"{name_prefix}_hyperedges_all_orders.png"
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, fname), bbox_inches="tight", dpi=200)
    plt.close(fig)

_save_true_hyperedge_figures(
    results_dir,
    N,
    true_2edges,
    true_3edges,
    true_4edges,
    true_5edges,
    true_6edges,
    true_7edges,
)

# ---------------------------------------------------------------------------
# Device and model setup
# ---------------------------------------------------------------------------
if gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using default GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = HyperPINNTopology(
    N=N,
    output_dim=state_dim * N,
    use_resnet=use_resnet,
    use_attention=use_attention,
    use_pirate=use_pirate,
    max_order=max_order,
)
model = model.to(device)

# Precompute all possible hyperedges (0-based indices) aligned with model's internal ordering
all_edges_tensors = {}
if max_order >= 2 and hasattr(model, "edge_indices_t"):
    all_edges_tensors["edges"] = model.edge_indices_t.to(device)
if max_order >= 3 and hasattr(model, "triangle_indices_t"):
    all_edges_tensors["triangles"] = model.triangle_indices_t.to(device)
if max_order >= 4 and hasattr(model, "quad_indices_t"):
    all_edges_tensors["quads"] = model.quad_indices_t.to(device)

# ---------------------------------------------------------------------------
# Sparsity and optimizer
# ---------------------------------------------------------------------------
model.lambda_l1_edges = 0.03
model.lambda_l1_triangles = 0.05
model.lambda_l0_edges = 0.01
model.lambda_l0_triangles = 0.02
# higher-order lambdas unused here but present on the model
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

losses = []
sparsity_stats = []
t_data = t_data.float().to(device)
x_data = x_data.float().to(device)

epochs = 14000
stage1_epochs = 2500
stage2_epochs = 10000
adaptive_weights = True
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def get_labels_and_scores(all_edges, true_edges, probs):
    y_true = []
    y_score = []
    for idx, edge in enumerate(all_edges):
        edge = tuple(sorted(edge))
        y_true.append(1 if edge in true_edges else 0)
        y_score.append(probs[idx])
    return np.array(y_true), np.array(y_score)


def evaluate_edges_triangles(
    model,
    t_data,
    all_2edges,
    true_2edges,
    all_3edges,
    true_3edges,
    all_4edges,
    true_4edges,
    all_5edges,
    true_5edges,
    all_6edges,
    true_6edges,
    all_7edges,
    true_7edges,
    results_dir=None,
    epoch=None,
):
    with torch.no_grad():
        edge_probs, triangle_probs, quad_probs, quint_probs, sext_probs, sept_probs = model.get_sparse_weights(
            use_concrete=False, hard=False
        )
        edge_probs = edge_probs.cpu().numpy() if edge_probs is not None else np.zeros(len(all_2edges))
        triangle_probs = triangle_probs.cpu().numpy() if triangle_probs is not None else np.zeros(len(all_3edges))
        quad_probs = quad_probs.cpu().numpy() if quad_probs is not None else np.zeros(len(all_4edges))
        quint_probs = quint_probs.cpu().numpy() if quint_probs is not None else np.zeros(len(all_5edges))
        sext_probs = sext_probs.cpu().numpy() if sext_probs is not None else np.zeros(len(all_6edges))
        sept_probs = sept_probs.cpu().numpy() if sept_probs is not None else np.zeros(len(all_7edges))

    edge_scores = [abs(edge_probs[idx]) for idx, _ in enumerate(all_2edges)]
    triangle_scores = [abs(triangle_probs[idx]) for idx, _ in enumerate(all_3edges)]
    quad_scores = [abs(quad_probs[idx]) for idx, _ in enumerate(all_4edges)]
    quint_scores = [abs(quint_probs[idx]) for idx, _ in enumerate(all_5edges)]
    sext_scores = [abs(sext_probs[idx]) for idx, _ in enumerate(all_6edges)]
    sept_scores = [abs(sept_probs[idx]) for idx, _ in enumerate(all_7edges)]

    y_true_2, y_score_2 = get_labels_and_scores(all_2edges, true_2edges, edge_scores)
    y_true_3, y_score_3 = get_labels_and_scores(all_3edges, true_3edges, triangle_scores)
    y_true_4, y_score_4 = get_labels_and_scores(all_4edges, true_4edges, quad_scores)
    y_true_5, y_score_5 = get_labels_and_scores(all_5edges, true_5edges, quint_scores)
    y_true_6, y_score_6 = get_labels_and_scores(all_6edges, true_6edges, sext_scores)
    y_true_7, y_score_7 = get_labels_and_scores(all_7edges, true_7edges, sept_scores)

    if results_dir is not None:
        pred_2 = set(tuple(sorted(edge)) for idx, edge in enumerate(all_2edges) if edge_probs[idx] >= 0.5)
        pred_3 = set(tuple(sorted(edge)) for idx, edge in enumerate(all_3edges) if triangle_probs[idx] >= 0.5)
        pred_4 = set(tuple(sorted(edge)) for idx, edge in enumerate(all_4edges) if quad_probs[idx] >= 0.5)
        pred_5 = set(tuple(sorted(edge)) for idx, edge in enumerate(all_5edges) if quint_probs[idx] >= 0.5)
        pred_6 = set(tuple(sorted(edge)) for idx, edge in enumerate(all_6edges) if sext_probs[idx] >= 0.5)
        pred_7 = set(tuple(sorted(edge)) for idx, edge in enumerate(all_7edges) if sept_probs[idx] >= 0.5)
        prefix = "predicted"
        if epoch is not None:
            prefix = f"predicted_epoch{epoch}"
        _save_true_hyperedge_figures(
            results_dir, N, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, name_prefix=prefix
        )

    return (
        y_true_2,
        y_score_2,
        y_true_3,
        y_score_3,
        y_true_4,
        y_score_4,
        y_true_5,
        y_score_5,
        y_true_6,
        y_score_6,
        y_true_7,
        y_score_7,
    )


def compute_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def plot_roc(y_true, y_score, label):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.2f})", linewidth=2)
    return fpr, tpr, auc_score


# ---------------------------------------------------------------------------
# Physics loss for neuronal synchronization
# ---------------------------------------------------------------------------

def physics_loss_neuronal(model, t_data, N, max_order, device, all_edges_tensors):
    """Physics loss for neuronal synchronization using NSHypergraphModel.

    NN produces x_pred(t) with 2 components per node (theta, phi),
    flattened as [T, 2N]. We compute d/dt via autograd and match to

        dX/dt ≈ f(X) + Phi(X) @ A,

    where f and Phi come from lib_neuronal_synchronization.hypergraph.
    """

    if not t_data.requires_grad:
        t_data = t_data.clone().detach().requires_grad_(True).to(device)

    x_pred_flat = model.forward(t_data)  # [T, 2N]
    T, Dflat = x_pred_flat.shape
    assert Dflat == 2 * N, f"Neuronal mode expects output_dim = 2N, got {Dflat} for N={N}"

    # Reshape to [T, N, 2] for (theta, phi)
    x_pred = x_pred_flat.view(T, N, 2)

    # Time derivative via autograd (per component over output dimension)
    dx_dt_flat = torch.zeros_like(x_pred_flat)
    for i in range(Dflat):
        grad_i = torch.autograd.grad(
            x_pred_flat[:, i].sum(), t_data, create_graph=True, retain_graph=True
        )[0]
        dx_dt_flat[:, i] = grad_i.squeeze(-1)
    dx_dt = dx_dt_flat.view(T, N, 2)

    # Topology probabilities (global, per order)
    edge_probs, triangle_probs, quad_probs, quint_probs, sext_probs, sept_probs = model.get_sparse_weights(
        use_concrete=False, hard=False
    )

    weights_list = []
    if max_order >= 2 and edge_probs is not None and "edges" in all_edges_tensors:
        weights_list.append(edge_probs.to(device))
    if max_order >= 3 and triangle_probs is not None and "triangles" in all_edges_tensors:
        weights_list.append(triangle_probs.to(device))

    if len(weights_list) == 0:
        # Only drift part f(x), fully vectorized over time
        dx_expected = NSHypergraphModel.dynamic_f_batch(x_pred, N)  # [T, N, 2]
        residual = dx_dt - dx_expected
        return torch.mean(residual ** 2)

    A_all = torch.cat(weights_list)  # [E_total]

    # all_possible_edges using 0-based indices from the model
    all_possible_edges = {}
    if max_order >= 2 and "edges" in all_edges_tensors:
        all_possible_edges["edges"] = all_edges_tensors["edges"]
    if max_order >= 3 and "triangles" in all_edges_tensors:
        all_possible_edges["triangles"] = all_edges_tensors["triangles"]

    # Vectorized over all time steps: f(x_t) and Phi(x_t) @ A_all
    f_all = NSHypergraphModel.dynamic_f_batch(x_pred, N)  # [T, N, 2]
    Phi_all = NSHypergraphModel.dynamic_phi_batch(x_pred, all_possible_edges, N, device)  # [T, N, 2, E_total]
    interaction_all = torch.einsum("tnde,e->tnd", Phi_all, A_all)  # [T, N, 2]
    dx_expected = f_all + interaction_all

    residual = dx_dt - dx_expected
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for epoch in range(epochs):
    optimizer.zero_grad(set_to_none=True)
    x_pred = model.forward(t_data)

    physics_loss = physics_loss_neuronal(model, t_data, N, max_order, device, all_edges_tensors)
    data_loss = torch.mean((x_pred - x_data) ** 2)
    sparsity_loss, sparsity_info = model.sparsity_regularization()

    if adaptive_weights and epoch > 500:
        sparsity_weight = max(0.1, 1.0 * (0.99 ** (epoch - 500)))
    else:
        sparsity_weight = 1.0

    if epoch < stage1_epochs:
        physics_weight = 0.1
        data_weight = 1.0
        sparsity_weight = 0.0
        print_prefix = "Stage 1 (Data Fitting)"
    elif epoch < stage2_epochs:
        progress = (epoch - stage1_epochs) / (stage2_epochs - stage1_epochs)
        physics_weight = 0.01 + 0.99 * progress
        data_weight = 1.0 - 0.8 * progress
        sparsity_weight = 0.0
        print_prefix = "Stage 2 (Physics Learning)"
    else:
        progress = min(1.0, (epoch - stage2_epochs) / (epochs - stage2_epochs))
        physics_weight = 1.0
        data_weight = 0.2
        sparsity_weight = 0.01 * progress
        if hasattr(model, "temperature"):
            model.temperature = max(0.5, 1.0 * (0.995 ** ((epoch - stage2_epochs) // 100)))

    total_loss = physics_weight * physics_loss + data_weight * data_loss + sparsity_weight * sparsity_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    losses.append(total_loss.item())
    sparsity_stats.append(sparsity_info)

    if epoch % 500 == 0:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}")
        print(f"  Physics: {physics_loss.item():.6f}, Data: {data_loss.item():.6f}")
        print(f"  Sparsity: {sparsity_loss.item():.6f}")
        print(
            f"  L1 edges: {sparsity_info['l1_edges']:.2f},"
            f"  L1 triangles: {sparsity_info['l1_triangles']:.2f},"
        )

        (
            y_true_2,
            y_score_2,
            y_true_3,
            y_score_3,
            y_true_4,
            y_score_4,
            y_true_5,
            y_score_5,
            y_true_6,
            y_score_6,
            y_true_7,
            y_score_7,
        ) = evaluate_edges_triangles(
            model,
            t_data,
            all_2edges,
            true_2edges,
            all_3edges,
            true_3edges,
            all_4edges,
            true_4edges,
            all_5edges,
            true_5edges,
            all_6edges,
            true_6edges,
            all_7edges,
            true_7edges,
            results_dir=results_dir,
            epoch=epoch,
        )

        auc_str = ""
        if max_order >= 2:
            auc_2 = compute_auc(y_true_2, y_score_2)
            auc_str = f"  AUC (2-edges): {auc_2:.4f}"
        if max_order >= 3:
            auc_3 = compute_auc(y_true_3, y_score_3)
            auc_str += f", AUC (3-edges): {auc_3:.4f}"
        if max_order >= 4:
            auc_4 = compute_auc(y_true_4, y_score_4)
            auc_str += f", AUC (4-edges): {auc_4:.4f}"
        print(auc_str)

# ---------------------------------------------------------------------------
# Final evaluation and ROC plot
# ---------------------------------------------------------------------------
(
    y_true_2,
    y_score_2,
    y_true_3,
    y_score_3,
    y_true_4,
    y_score_4,
    y_true_5,
    y_score_5,
    y_true_6,
    y_score_6,
    y_true_7,
    y_score_7,
) = evaluate_edges_triangles(
    model,
    t_data,
    all_2edges,
    true_2edges,
    all_3edges,
    true_3edges,
    all_4edges,
    true_4edges,
    all_5edges,
    true_5edges,
    all_6edges,
    true_6edges,
    all_7edges,
    true_7edges,
    results_dir=results_dir,
    epoch="final",
)

plt.figure(figsize=(8, 6))
y_true_list = []
y_score_list = []
if max_order >= 2:
    plot_roc(y_true_2, y_score_2, "Pairwise")
    y_true_list.append(y_true_2)
    y_score_list.append(y_score_2)
if max_order >= 3:
    plot_roc(y_true_3, y_score_3, "Third-order")
    y_true_list.append(y_true_3)
    y_score_list.append(y_score_3)
if max_order >= 4:
    plot_roc(y_true_4, y_score_4, "Fourth-order")
    y_true_list.append(y_true_4)
    y_score_list.append(y_score_4)

if len(y_true_list) > 0:
    y_true_total = np.concatenate(y_true_list)
    y_score_total = np.concatenate(y_score_list)
    plot_roc(y_true_total, y_score_total, label="All")

plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.title("ROC Curves for Identified Hypergraphs (Social Contagion)", fontsize=17)
plt.legend(fontsize=14, loc="lower right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(results_dir, f"roc_curves_{max_order}_order.png"), bbox_inches="tight")
