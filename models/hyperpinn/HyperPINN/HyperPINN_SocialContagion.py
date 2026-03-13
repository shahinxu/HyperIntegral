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

from lib_social_contagion.hypergraph import HypergraphModel as SCMHypergraphModel

parser = argparse.ArgumentParser(description="Run HyperPINN on social contagion hypergraph dynamics")
parser.add_argument("--M", type=int, default=300, help="Number of time samples")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--noise", type=float, default=0.0)
args = parser.parse_args()

M = args.M
gpu_id = args.gpu_id
noise = args.noise
defaults = SCMHypergraphModel.get_default_params()
N = defaults["n_nodes"]
max_order = defaults["max_order"]

# ---------------------------------------------------------------------------
# Ground-truth hypergraph and data generation
# ---------------------------------------------------------------------------
edge_config = SCMHypergraphModel.get_hyperedge_config(N, max_order)

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

t_eval, X = SCMHypergraphModel.generate_training_data(N, edge_config, n_samples=M, noise=noise)
state_dim = X.shape[2]
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
    "social",
    f"sample_{M}_noise_{noise}",
    timestamp,
)
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "social_timeseries.png"))
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
# Physics loss for social contagion
# ---------------------------------------------------------------------------

def physics_loss_social(model, t_data, N, max_order, device, all_edges_tensors):
    """Physics loss for social contagion dynamics using SCMHypergraphModel.

    NN produces x_pred(t) (infection probability per node). We compute
    dx/dt via autograd and match it to

        dx/dt ≈ f(x) + Phi(x) @ A,

    where f(x) = -mu * x and Phi(x) encodes the contributions of all
    candidate edges/triangles/quads, and A is represented by the
    model's sparse hyperedge weights (edge/triangle/quad probabilities).
    """

    if not t_data.requires_grad:
        t_data = t_data.clone().detach().requires_grad_(True).to(device)

    x_pred = model.forward(t_data)  # [T, N]
    T, D = x_pred.shape
    assert D == N, f"Social contagion mode expects output_dim = N, got {D}"

    # Time derivative via autograd (per-node, small N so loop is fine)
    dx_dt_pred = torch.zeros_like(x_pred)
    for i in range(D):
        grad_i = torch.autograd.grad(
            x_pred[:, i].sum(), t_data, create_graph=True, retain_graph=True
        )[0]
        dx_dt_pred[:, i] = grad_i.squeeze(-1)

    # Topology probabilities (global, per order)
    edge_probs, triangle_probs, quad_probs, quint_probs, sext_probs, sept_probs = model.get_sparse_weights(
        use_concrete=False, hard=False
    )

    # Build concatenated weight vector A_all matching the ordering used
    # below: first all edges, then all triangles, then all quads.
    weights_list = []
    n_edges = 0
    n_tris = 0
    n_quads = 0
    if max_order >= 2 and edge_probs is not None and "edges" in all_edges_tensors:
        edge_probs = edge_probs.to(device)
        n_edges = edge_probs.shape[0]
        weights_list.append(edge_probs)
    if max_order >= 3 and triangle_probs is not None and "triangles" in all_edges_tensors:
        triangle_probs = triangle_probs.to(device)
        n_tris = triangle_probs.shape[0]
        weights_list.append(triangle_probs)
    if max_order >= 4 and quad_probs is not None and "quads" in all_edges_tensors:
        quad_probs = quad_probs.to(device)
        n_quads = quad_probs.shape[0]
        weights_list.append(quad_probs)
    # Parameters of the SCM dynamics (mu, beta, beta_delta)
    params = SCMHypergraphModel.SCMParams()

    if len(weights_list) == 0:
        # No candidate hyperedges; fall back to matching only f(x) = -mu x
        dx_expected = -params.mu * x_pred
        residual = dx_dt_pred - dx_expected
        return torch.mean(residual ** 2)

    A_all = torch.cat(weights_list)  # [E_total]

    # Prepare all_possible_edges dict (0-based indices)
    all_possible_edges = {}
    if max_order >= 2 and "edges" in all_edges_tensors:
        all_possible_edges["edges"] = all_edges_tensors["edges"]
    if max_order >= 3 and "triangles" in all_edges_tensors:
        all_possible_edges["triangles"] = all_edges_tensors["triangles"]
    if max_order >= 4 and "quads" in all_edges_tensors:
        all_possible_edges["quads"] = all_edges_tensors["quads"]

    # Vectorized computation of f(x) and Phi(x)A over all time steps.
    # x_all: [T, N]
    x_all = x_pred
    susceptible = 1.0 - x_all  # [T, N]

    # Drift part: f(x) = -mu * x
    f_all = -params.mu * x_all  # [T, N]

    # Interaction part: Phi(x) * A, with Phi shape [T, N, E_total]
    E_total = A_all.shape[0]
    Phi = torch.zeros((T, N, E_total), device=device, dtype=x_all.dtype)
    feature_offset = 0

    # 1. Pairwise edges
    edges = all_possible_edges.get("edges", [])
    if len(edges) > 0:
        edges_t = torch.as_tensor(edges, dtype=torch.long, device=device)  # [E2, 2]
        num_edges = edges_t.shape[0]
        i_idx = edges_t[:, 0]
        j_idx = edges_t[:, 1]

        feature_range = feature_offset + torch.arange(num_edges, device=device)  # [E2]

        # term_i, term_j: [T, E2]
        term_i = params.beta * susceptible[:, i_idx] * x_all[:, j_idx]
        term_j = params.beta * susceptible[:, j_idx] * x_all[:, i_idx]

        # Advanced indexing to assign terms for all times at once
        time_idx = torch.arange(T, device=device)[:, None]            # [T, 1]
        edge_ids = feature_range[None, :]                             # [1, E2]
        node_i = i_idx[None, :]                                       # [1, E2]
        node_j = j_idx[None, :]

        Phi[time_idx, node_i, edge_ids] = term_i
        Phi[time_idx, node_j, edge_ids] = term_j

        feature_offset += num_edges

    # 2. Triangles
    triangles = all_possible_edges.get("triangles", [])
    if len(triangles) > 0:
        tri_t = torch.as_tensor(triangles, dtype=torch.long, device=device)  # [E3, 3]
        num_tris = tri_t.shape[0]
        i_idx = tri_t[:, 0]
        j_idx = tri_t[:, 1]
        k_idx = tri_t[:, 2]

        feature_range = feature_offset + torch.arange(num_tris, device=device)  # [E3]

        term_i = params.beta_delta * susceptible[:, i_idx] * (x_all[:, j_idx] * x_all[:, k_idx])
        term_j = params.beta_delta * susceptible[:, j_idx] * (x_all[:, i_idx] * x_all[:, k_idx])
        term_k = params.beta_delta * susceptible[:, k_idx] * (x_all[:, i_idx] * x_all[:, j_idx])

        time_idx = torch.arange(T, device=device)[:, None]            # [T, 1]
        edge_ids = feature_range[None, :]                             # [1, E3]
        node_i = i_idx[None, :]
        node_j = j_idx[None, :]
        node_k = k_idx[None, :]

        Phi[time_idx, node_i, edge_ids] = term_i
        Phi[time_idx, node_j, edge_ids] = term_j
        Phi[time_idx, node_k, edge_ids] = term_k

        feature_offset += num_tris

    # 3. Quads (4-body interactions)
    quads = all_possible_edges.get("quads", [])
    if len(quads) > 0:
        quads_t = torch.as_tensor(quads, dtype=torch.long, device=device)  # [E4, 4]
        num_quads = quads_t.shape[0]
        i_idx = quads_t[:, 0]
        j_idx = quads_t[:, 1]
        k_idx = quads_t[:, 2]
        l_idx = quads_t[:, 3]

        feature_range = feature_offset + torch.arange(num_quads, device=device)  # [E4]

        term_i = params.beta_delta * susceptible[:, i_idx] * (x_all[:, j_idx] * x_all[:, k_idx] * x_all[:, l_idx])
        term_j = params.beta_delta * susceptible[:, j_idx] * (x_all[:, i_idx] * x_all[:, k_idx] * x_all[:, l_idx])
        term_k = params.beta_delta * susceptible[:, k_idx] * (x_all[:, i_idx] * x_all[:, j_idx] * x_all[:, l_idx])
        term_l = params.beta_delta * susceptible[:, l_idx] * (x_all[:, i_idx] * x_all[:, j_idx] * x_all[:, k_idx])

        time_idx = torch.arange(T, device=device)[:, None]            # [T, 1]
        edge_ids = feature_range[None, :]                             # [1, E4]
        node_i = i_idx[None, :]
        node_j = j_idx[None, :]
        node_k = k_idx[None, :]
        node_l = l_idx[None, :]

        Phi[time_idx, node_i, edge_ids] = term_i
        Phi[time_idx, node_j, edge_ids] = term_j
        Phi[time_idx, node_k, edge_ids] = term_k
        Phi[time_idx, node_l, edge_ids] = term_l

        feature_offset += num_quads

    # Now contract Phi with A_all: [T, N, E_total] @ [E_total] -> [T, N]
    Phi_flat = Phi.view(T * N, E_total)
    interaction_flat = torch.matmul(Phi_flat, A_all)  # [T * N]
    interaction = interaction_flat.view(T, N)

    dx_expected = f_all + interaction

    residual = dx_dt_pred - dx_expected
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for epoch in range(epochs):
    optimizer.zero_grad(set_to_none=True)
    x_pred = model.forward(t_data)

    physics_loss = physics_loss_social(model, t_data, N, max_order, device, all_edges_tensors)
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
