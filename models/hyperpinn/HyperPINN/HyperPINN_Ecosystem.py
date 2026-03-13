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
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from datetime import datetime
import networkx as nx
import argparse

from lib_ecological_dynamics.hypergraph import HypergraphModel

# Parse command-line arguments (ecological dynamics only)
parser = argparse.ArgumentParser(description='Run HyperPINN on ecological hypergraph dynamics')
parser.add_argument('--M', type=int, default=300, help='Number of time samples')
parser.add_argument('--N', type=int, default=8, help='Number of species')
parser.add_argument('--max_order', type=int, default=3)
parser.add_argument('--gpu_id', type=int, default=4)
parser.add_argument('--noise', type=float, default=0.0)
args = parser.parse_args()

N = args.N
max_order = args.max_order
gpu_id = args.gpu_id
noise = args.noise
M = args.M


# Use ecological dynamics from lib_ecological_dynamics to generate data and ground-truth hyperedges
edge_config = HypergraphModel.get_hyperedge_config(N, max_order)

EdgeList = np.array(edge_config.get('edges', [])) if edge_config.get('edges') else np.empty((0, 2), dtype=int)
TriangleList_full = np.array(edge_config.get('triangles', [])) if edge_config.get('triangles') else np.empty((0, 3), dtype=int)
QuadList_full = np.array(edge_config.get('quads', [])) if edge_config.get('quads') else np.empty((0, 4), dtype=int)
QuintList_full = np.array(edge_config.get('quints', [])) if edge_config.get('quints') else np.empty((0, 5), dtype=int)
SextList_full = np.array(edge_config.get('sexts', [])) if edge_config.get('sexts') else np.empty((0, 6), dtype=int)
SeptList_full = np.array(edge_config.get('septs', [])) if edge_config.get('septs') else np.empty((0, 7), dtype=int)

# Automatically adjust ground truth based on max_order
TriangleList = TriangleList_full if max_order >= 3 else np.empty((0, 3), dtype=int)
QuadList = QuadList_full if max_order >= 4 else np.empty((0, 4), dtype=int)
QuintList = QuintList_full if max_order >= 5 else np.empty((0, 5), dtype=int)
SextList = SextList_full if max_order >= 6 else np.empty((0, 6), dtype=int)
SeptList = SeptList_full if max_order >= 7 else np.empty((0, 7), dtype=int)

all_2edges = list(combinations(range(1, N+1), 2))
all_3edges = list(combinations(range(1, N+1), 3)) if max_order >= 3 else []
all_4edges = list(combinations(range(1, N+1), 4)) if max_order >= 4 else []
all_5edges = list(combinations(range(1, N+1), 5)) if max_order >= 5 else []
all_6edges = list(combinations(range(1, N+1), 6)) if max_order >= 6 else []
all_7edges = list(combinations(range(1, N+1), 7)) if max_order >= 7 else []

true_2edges = set(tuple(sorted(edge)) for edge in EdgeList)
true_3edges = set(tuple(sorted(triangle)) for triangle in TriangleList)
true_4edges = set(tuple(sorted(quad)) for quad in QuadList)
true_5edges = set(tuple(sorted(quint)) for quint in QuintList)
true_6edges = set(tuple(sorted(sext)) for sext in SextList)
true_7edges = set(tuple(sorted(sept)) for sept in SeptList)

# Time series from ecological simulator (single state dimension per node)
t_eval, X = HypergraphModel.generate_training_data(N, edge_config, n_samples=M, noise=noise)
# X has shape (T, N, 1)
state_dim = X.shape[2]
X_noisy = X  # noise handled inside generate_training_data
t_data = torch.tensor(t_eval, dtype=torch.float32, requires_grad=True).unsqueeze(1)
x_data = torch.tensor(X_noisy.reshape(X_noisy.shape[0], -1), dtype=torch.float32)
architectures = [
    ("ResNet", True, False, False),
    ("Attention", False, True, False),
    ("Pirate", False, False, True),
]

n_cols = 4
n_rows = int(np.ceil(N / n_cols))
plt.figure(figsize=(4*n_cols, 3*n_rows))
X_plot = x_data.cpu().numpy()
coord_names = ['x']

for i in range(N):
    plt.subplot(n_rows, n_cols, i+1)
    for coord_idx in range(state_dim):
        idx = i + coord_idx * N
        color = 'b'
        label = f'{coord_names[coord_idx]}_{i+1}' if coord_idx < len(coord_names) else f'state{coord_idx+1}_{i+1}'
        plt.plot(t_eval, X_plot[:, idx], color + '-', label=label, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title(f'Node {i+1}')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_base = os.environ.get("HYPERPINN_RESULTS_ROOT", os.path.join("results", "hyperpinn"))
results_dir = os.path.join(
    results_base,
    "ecological",
    f"sample_{M}_noise_{noise}",
    timestamp,
)
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "ecology_timeseries.png"))
print(f"Results will be saved to: {results_dir}")

def _save_true_hyperedge_figures(
    results_dir, 
    N, 
    true_2edges, 
    true_3edges, 
    true_4edges, 
    true_5edges, 
    true_6edges, 
    true_7edges,
    name_prefix: str = "true"
):
    orders = [2, 3, 4, 5, 6, 7]
    true_lists = [
        sorted(true_2edges),
        sorted(true_3edges),
        sorted(true_4edges),
        sorted(true_5edges),
        sorted(true_6edges),
        sorted(true_7edges)
    ]

    G = nx.Graph()
    for n in range(1, N+1):
        G.add_node(n)
    pos = nx.circular_layout(G)
    cmap = plt.get_cmap('tab20')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax, order, true_list in zip(axes, orders, true_lists):
        xs = [pos[n][0] for n in G.nodes()]
        ys = [pos[n][1] for n in G.nodes()]
        ax.scatter(xs, ys, s=140, color='tab:blue')
        for n in G.nodes():
            ax.text(pos[n][0], pos[n][1], str(n), fontsize=11, ha='center', va='center', color='white')
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

        ax.set_title(f'Order={order} (#{len(true_list)})')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')

    fig.tight_layout()
    fname = f"{name_prefix}_hyperedges_all_orders.png"
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, fname), bbox_inches='tight', dpi=200)
    plt.close(fig)

_save_true_hyperedge_figures(results_dir, N, true_2edges, true_3edges, true_4edges, true_5edges, true_6edges, true_7edges)

if gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using default GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

arch_name, use_resnet, use_attention, use_pirate = architectures[2]
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
if max_order >= 2 and hasattr(model, 'edge_indices_t'):
    all_edges_tensors['edges'] = model.edge_indices_t.to(device)
if max_order >= 3 and hasattr(model, 'triangle_indices_t'):
    all_edges_tensors['triangles'] = model.triangle_indices_t.to(device)
if max_order >= 4 and hasattr(model, 'quad_indices_t'):
    all_edges_tensors['quads'] = model.quad_indices_t.to(device)
if max_order >= 5 and hasattr(model, 'quint_indices_t'):
    all_edges_tensors['quints'] = model.quint_indices_t.to(device)
if max_order >= 6 and hasattr(model, 'sext_indices_t'):
    all_edges_tensors['sexts'] = model.sext_indices_t.to(device)
if max_order >= 7 and hasattr(model, 'sept_indices_t'):
    all_edges_tensors['septs'] = model.sept_indices_t.to(device)

model.lambda_l1_edges = 0.03      
model.lambda_l1_triangles = 0.05   
model.lambda_l0_edges = 0.01
model.lambda_l0_triangles = 0.02
model.lambda_l1_quads = 0.04
model.lambda_l0_quads = 0.015
model.lambda_l1_quints = 0.03
model.lambda_l0_quints = 0.01
model.lambda_l1_sexts = 0.025
model.lambda_l0_sexts = 0.008
model.lambda_l1_septs = 0.02
model.lambda_l0_septs = 0.005
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
losses = []
sparsity_stats = []
t_data = t_data.float().to(device)
x_data = x_data.float().to(device)

epochs = 14000
stage1_epochs = 2500   
stage2_epochs = 10000 
adaptive_weights = True
best_loss = float('inf')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
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
        edge_probs, triangle_probs, quad_probs, quint_probs, sext_probs, sept_probs = model.get_sparse_weights(use_concrete=False, hard=False)
        edge_probs = edge_probs.cpu().numpy() if edge_probs is not None else np.zeros(len(all_2edges))
        triangle_probs = triangle_probs.cpu().numpy() if triangle_probs is not None else np.zeros(len(all_3edges))
        quad_probs = quad_probs.cpu().numpy() if quad_probs is not None else np.zeros(len(all_4edges))
        quint_probs = quint_probs.cpu().numpy() if quint_probs is not None else np.zeros(len(all_5edges))
        sext_probs = sext_probs.cpu().numpy() if sext_probs is not None else np.zeros(len(all_6edges))
        sept_probs = sept_probs.cpu().numpy() if sept_probs is not None else np.zeros(len(all_7edges))

    # scores used for ROC/AUC (if values are hard 0/1 this still works)
    edge_scores = [abs(edge_probs[idx]) for idx, _ in enumerate(all_2edges)]
    triangle_scores = [abs(triangle_probs[idx]) for idx, _ in enumerate(all_3edges)]
    quad_scores = [abs(quad_probs[idx]) for idx, _ in enumerate(all_4edges)]
    quint_scores = [abs(quint_probs[idx]) for idx, _ in enumerate(all_5edges)]
    sext_scores = [abs(sext_probs[idx]) for idx, _ in enumerate(all_6edges)]
    sept_scores = [abs(sept_probs[idx]) for idx, _ in enumerate(all_7edges)]

    # build ground-truth label / score arrays
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
        prefix = f'predicted'
        if epoch is not None:
            prefix = f'predicted_epoch{epoch}'
        _save_true_hyperedge_figures(results_dir, N, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, name_prefix=prefix)

    return y_true_2, y_score_2, y_true_3, y_score_3, y_true_4, y_score_4, y_true_5, y_score_5, y_true_6, y_score_6, y_true_7, y_score_7

def compute_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def plot_roc(y_true, y_score, label):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.2f})',linewidth=2)
    return fpr, tpr, auc_score


def _accumulate_ecology_order(dx_comp, x_pred, node_idx, probs, order_weight, chunk_size=4096):
    """Accumulate one hyperedge order contribution in chunks to reduce peak memory."""
    if node_idx is None or probs is None or node_idx.numel() == 0:
        return

    t_len = x_pred.shape[0]
    e_total = node_idx.shape[0]
    node_arity = node_idx.shape[1]
    probs = probs.to(x_pred.device)

    for start in range(0, e_total, chunk_size):
        end = min(start + chunk_size, e_total)
        idx_chunk = node_idx[start:end]  # [E_chunk, k]

        # [T, E_chunk, k] -> [T, E_chunk]
        x_chunk = x_pred[:, idx_chunk]
        prod = torch.prod(x_chunk, dim=2)

        w = (order_weight * probs[start:end]).view(1, -1)
        contrib = -w * prod

        for pos in range(node_arity):
            nodes = idx_chunk[:, pos]
            dx_comp.scatter_add_(1, nodes.unsqueeze(0).expand(t_len, -1), contrib)

def physics_loss_ecology(model, t_data, N, max_order, device, all_edges_tensors):
    """Physics loss for ecological dynamics using HypergraphModel.

    We use the NN to produce x_pred(t), compute dx/dt via autograd,
    and match it to ecological dynamics: f(x) + Phi(x) @ A_hat, where
    A_hat are the sigmoid-transformed topology parameters.
    """
    # Ensure t_data has gradients for time-derivative computation
    if not t_data.requires_grad:
        t_data = t_data.clone().detach().requires_grad_(True).to(device)

    x_pred = model.forward(t_data)  # [T, N] for ecology (state_dim=1)
    T, D = x_pred.shape
    assert D == N, f"Ecology mode expects output_dim = N, got {D}"

    # Time derivative via autograd (vectorized over time, loop over dims only)
    dx_dt_pred = torch.zeros_like(x_pred)
    for i in range(D):
        grad_i = torch.autograd.grad(
            x_pred[:, i].sum(),
            t_data,
            create_graph=True,
            retain_graph=True,
        )[0]
        dx_dt_pred[:, i] = grad_i.squeeze(-1)

    # Topology probabilities per order (global over time)
    edge_probs, triangle_probs, quad_probs, quint_probs, sext_probs, sept_probs = \
        model.get_sparse_weights(use_concrete=False, hard=False)

    # Baseline ecological dynamics f(x,t) for all time points (no competition)
    f_all = HypergraphModel.dynamic_f_batch(x_pred, N, t_data).view(T, N)

    # Competition contributions from hyperedges, vectorized over time.
    # Chunking keeps memory bounded for large candidate sets.
    from lib_ecological_dynamics.hypergraph import HypergraphModel as _HM
    params = _HM._cached_params

    dx_comp = torch.zeros_like(x_pred)

    chunk_size = int(os.environ.get("HYPERPINN_PHYSICS_CHUNK", "4096"))

    if max_order >= 2 and 'edges' in all_edges_tensors:
        _accumulate_ecology_order(
            dx_comp,
            x_pred,
            all_edges_tensors['edges'],
            edge_probs,
            params.w2,
            chunk_size=chunk_size,
        )
    if max_order >= 3 and 'triangles' in all_edges_tensors:
        _accumulate_ecology_order(
            dx_comp,
            x_pred,
            all_edges_tensors['triangles'],
            triangle_probs,
            params.w3,
            chunk_size=chunk_size,
        )
    if max_order >= 4 and 'quads' in all_edges_tensors:
        _accumulate_ecology_order(
            dx_comp,
            x_pred,
            all_edges_tensors['quads'],
            quad_probs,
            params.w4,
            chunk_size=chunk_size,
        )
    if max_order >= 5 and 'quints' in all_edges_tensors:
        _accumulate_ecology_order(
            dx_comp,
            x_pred,
            all_edges_tensors['quints'],
            quint_probs,
            params.w5,
            chunk_size=chunk_size,
        )

    dx_dt_expected = f_all + dx_comp  # [T, N]
    residual = dx_dt_pred - dx_dt_expected
    return torch.mean(residual**2)

for epoch in range(epochs):
    optimizer.zero_grad(set_to_none=True)
    x_pred = model.forward(t_data)
    # Physics loss is defined purely from ecological dynamics using HypergraphModel.
    physics_loss = physics_loss_ecology(model, t_data, N, max_order, device, all_edges_tensors)
    data_loss = torch.mean((x_pred - x_data)**2)
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
        if hasattr(model, 'temperature'):
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
            f"  L1 quads: {sparsity_info['l1_quads']:.2f},"
            f"  L1 quints: {sparsity_info['l1_quints']:.2f},"
            f"  L1 sexts: {sparsity_info['l1_sexts']:.2f},"
            f"  L1 septs: {sparsity_info['l1_septs']:.2f}"
        )
        # (Optional) If needed, we could add 1D ecological prediction plots here.
        
        y_true_2, y_score_2, y_true_3, y_score_3, y_true_4, y_score_4, y_true_5, y_score_5, y_true_6, y_score_6, y_true_7, y_score_7 = \
            evaluate_edges_triangles(
                model, t_data,
                all_2edges, true_2edges,
                all_3edges, true_3edges,
                all_4edges, true_4edges,
                all_5edges, true_5edges,
                all_6edges, true_6edges,
                all_7edges, true_7edges,
                results_dir=results_dir,
                epoch=epoch,
            )
        # Only compute and display AUC for orders <= max_order
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
        if max_order >= 5:
            auc_5 = compute_auc(y_true_5, y_score_5)
            auc_str += f", AUC (5-edges): {auc_5:.4f}"
        if max_order >= 6:
            auc_6 = compute_auc(y_true_6, y_score_6)
            auc_str += f", AUC (6-edges): {auc_6:.4f}"
        if max_order >= 7:
            auc_7 = compute_auc(y_true_7, y_score_7)
            auc_str += f", AUC (7-edges): {auc_7:.4f}"
        print(auc_str)

y_true_2, y_score_2, y_true_3, y_score_3, y_true_4, y_score_4, y_true_5, y_score_5, y_true_6, y_score_6, y_true_7, y_score_7 = \
    evaluate_edges_triangles(
        model, t_data,
        all_2edges, true_2edges,
        all_3edges, true_3edges,
        all_4edges, true_4edges,
        all_5edges, true_5edges,
        all_6edges, true_6edges,
        all_7edges, true_7edges,
        results_dir=results_dir,
        epoch='final',
    )

y_true_list = []
y_score_list = []
plt.figure(figsize=(8, 6))
if max_order >= 2:
    plot_roc(y_true_2, y_score_2, 'Pairwise')
    y_true_list.append(y_true_2)
    y_score_list.append(y_score_2)
if max_order >= 3:
    plot_roc(y_true_3, y_score_3, 'Third-order')
    y_true_list.append(y_true_3)
    y_score_list.append(y_score_3)
if max_order >= 4:
    plot_roc(y_true_4, y_score_4, 'Fourth-order')
    y_true_list.append(y_true_4)
    y_score_list.append(y_score_4)
if max_order >= 5:
    plot_roc(y_true_5, y_score_5, 'Fifth-order')
    y_true_list.append(y_true_5)
    y_score_list.append(y_score_5)
if max_order >= 6:
    plot_roc(y_true_6, y_score_6, 'Sixth-order')
    y_true_list.append(y_true_6)
    y_score_list.append(y_score_6)
if max_order >= 7:
    plot_roc(y_true_7, y_score_7, 'Seventh-order')
    y_true_list.append(y_true_7)
    y_score_list.append(y_score_7)

if len(y_true_list) > 0:
    y_true_total = np.concatenate(y_true_list)
    y_score_total = np.concatenate(y_score_list)
    plot_roc(y_true_total, y_score_total, label='All') 
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
plt.title('ROC Curves for Identified Hypergraphs',fontsize=17)
plt.legend(fontsize=14, loc="lower right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(results_dir, f'roc_curves_{max_order}_order.png'), bbox_inches='tight')