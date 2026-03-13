import torch
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from math import comb
import os
from datetime import datetime
from tqdm import tqdm
import argparse
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypergraph.scene_registry import get_scene_model
from hypergraph.outputs import write_standard_summary

HypergraphModel = None


class ImplicitHypergraphNet(nn.Module):
    def __init__(self, n_nodes: int, max_order: int, embed_dim: int = 32, head_hidden: int = 64):
        super().__init__()
        self.n_nodes = n_nodes
        self.max_order = max_order
        self.node_emb = nn.Embedding(n_nodes, embed_dim)
        nn.init.normal_(self.node_emb.weight, std=0.1)

        self.heads = nn.ModuleDict()
        for order in range(2, max_order + 1):
            head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
            nn.init.constant_(head[-1].bias, -(float(order) + 1.0))
            self.heads[f"order_{order}"] = head

    def _all_embeddings(self) -> torch.Tensor:
        device = self.node_emb.weight.device
        return self.node_emb(torch.arange(self.n_nodes, device=device))

    def score(self, order: int, edge_indices: torch.Tensor) -> torch.Tensor:
        z = self._all_embeddings()
        pooled = z[edge_indices].sum(dim=1)  # permutation-invariant pooling
        return self.heads[f"order_{order}"](pooled).squeeze(1)

    def predict_all_scores(self, all_possible_edges_gpu: dict) -> torch.Tensor:
        parts = []
        order_map = {
            "edges": 2,
            "triangles": 3,
            "quads": 4,
            "quints": 5,
            "sexts": 6,
            "septs": 7,
        }
        for key in ["edges", "triangles", "quads", "quints", "sexts", "septs"]:
            order = order_map[key]
            if order > self.max_order:
                break
            edge_idx = all_possible_edges_gpu.get(key)
            if edge_idx is None or edge_idx.numel() == 0:
                continue
            parts.append(self.score(order, edge_idx))

        if not parts:
            return torch.empty(0, device=self.node_emb.weight.device)
        return torch.cat(parts, dim=0)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        return x + self.net(x)


class TimeResNet(nn.Module):
    def __init__(self, output_dim, hidden_dim=64, num_layers=8, dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(1, hidden_dim)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) 
            for _ in range(num_layers - 2)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.register_buffer('t_mean', torch.tensor(0.0))
        self.register_buffer('t_std', torch.tensor(1.0))
        self.register_buffer('x_mean', torch.zeros(output_dim))
        self.register_buffer('x_std', torch.ones(output_dim))

    def forward(self, t, normalize=False):
        if normalize:
            t = (t - self.t_mean) / (self.t_std + 1e-8)
        h = torch.tanh(self.input_layer(t.unsqueeze(-1)))
        for block in self.res_blocks:
            h = block(h)
        out = self.output_layer(h)
        if normalize:
            out = out * self.x_std + self.x_mean
        return out



def compute_auc_scores(A_learned, edge_config, N, max_order, all_possible=None):
    if all_possible is None:
        all_possible = HypergraphModel.generate_all_possible_hyperedges(N, max_order)
    
    A_flat = A_learned.flatten()
    
    auc_scores = {}
    roc_data = {}
    A_idx = 0  # Index in A
    
    order_names = ['edges', 'triangles', 'quads', 'quints', 'sexts', 'septs']
    order_labels = ['2-edges', '3-edges', '4-edges', '5-edges', '6-edges', '7-edges']
    
    for order_idx, (order_name, order_label) in enumerate(zip(order_names, order_labels)):
        order_num = order_idx + 2
        
        if order_num > max_order:
            break
        
        # Get all possible hyperedges and true hyperedges
        possible_edges = all_possible.get(order_name, [])
        true_edges = edge_config.get(order_name, [])
        
        if len(possible_edges) == 0:
            continue
        
        # Create ground truth labels
        y_true = []
        y_pred = []
        
        for edge in possible_edges:
            # Check if this hyperedge is in the true configuration
            is_true_edge = any(
                sorted(edge) == sorted(true_edge) 
                for true_edge in true_edges
            )
            y_true.append(1 if is_true_edge else 0)
            
            # Predicted value (using sigmoid)
            if A_idx < len(A_flat):
                pred_score = 1 / (1 + np.exp(-A_flat[A_idx]))
                y_pred.append(pred_score)
                A_idx += 1
            else:
                y_pred.append(0.0)
        
        # Compute AUC
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc_score = auc(fpr, tpr)
            auc_scores[order_label] = auc_score
            roc_data[order_label] = (fpr, tpr)
        else:
            auc_scores[order_label] = None
            roc_data[order_label] = None
    
    return auc_scores, roc_data


def evaluate_full_auc_scores(
    implicit_model: nn.Module,
    edge_config: dict,
    N: int,
    max_order: int,
    device: torch.device,
    score_batch_size: int = 8192,
):
    order_info = [
        ("edges", "2-edges", 2),
        ("triangles", "3-edges", 3),
        ("quads", "4-edges", 4),
        ("quints", "5-edges", 5),
        ("sexts", "6-edges", 6),
        ("septs", "7-edges", 7),
    ]

    auc_scores = {}
    roc_data = {}
    eval_stats = {}

    implicit_model.eval()
    with torch.no_grad():
        for order_name, order_label, order in order_info:
            if order > max_order:
                break

            total = comb(N, order)
            true_set = {tuple(sorted(map(int, e))) for e in edge_config.get(order_name, [])}
            n_true = len(true_set)

            if total <= 0:
                continue

            y_true = np.empty(total, dtype=np.uint8)
            y_pred = np.empty(total, dtype=np.float32)

            write_idx = 0
            batch_edges = []

            for edge in combinations(range(1, N + 1), order):
                batch_edges.append(edge)
                if len(batch_edges) >= score_batch_size:
                    edge_tensor = torch.tensor(batch_edges, dtype=torch.long, device=device) - 1
                    probs = torch.sigmoid(implicit_model.score(order, edge_tensor)).cpu().numpy().astype(np.float32)
                    bsz = len(batch_edges)
                    y_pred[write_idx:write_idx + bsz] = probs
                    y_true[write_idx:write_idx + bsz] = np.fromiter(
                        (1 if tuple(e) in true_set else 0 for e in batch_edges),
                        dtype=np.uint8,
                        count=bsz,
                    )
                    write_idx += bsz
                    batch_edges = []

            if batch_edges:
                edge_tensor = torch.tensor(batch_edges, dtype=torch.long, device=device) - 1
                probs = torch.sigmoid(implicit_model.score(order, edge_tensor)).cpu().numpy().astype(np.float32)
                bsz = len(batch_edges)
                y_pred[write_idx:write_idx + bsz] = probs
                y_true[write_idx:write_idx + bsz] = np.fromiter(
                    (1 if tuple(e) in true_set else 0 for e in batch_edges),
                    dtype=np.uint8,
                    count=bsz,
                )

            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auc_score = auc(fpr, tpr)
                auc_scores[order_label] = auc_score
                roc_data[order_label] = (fpr, tpr)
            else:
                auc_scores[order_label] = None
                roc_data[order_label] = None

            eval_stats[order_label] = {
                "possible": total,
                "true": n_true,
                "p_min": float(y_pred.min()) if y_pred.size > 0 else None,
                "p_max": float(y_pred.max()) if y_pred.size > 0 else None,
            }

    implicit_model.train()
    return auc_scores, roc_data, eval_stats


def build_training_candidate_pool(
    n_nodes: int,
    max_order: int,
    max_candidates_per_order: int = 4096,
    rng: np.random.Generator | None = None,
) -> dict:
    rng = rng or np.random.default_rng()
    order_to_key = {
        2: "edges",
        3: "triangles",
        4: "quads",
        5: "quints",
        6: "sexts",
        7: "septs",
    }
    pool = {k: [] for k in order_to_key.values()}

    max_order = min(max_order, 7)
    for order in range(2, max_order + 1):
        key = order_to_key[order]
        total = comb(n_nodes, order)

        if total <= max_candidates_per_order:
            sampled_edges = [list(edge) for edge in combinations(range(1, n_nodes + 1), order)]
            rng.shuffle(sampled_edges)
            pool[key] = sampled_edges
            continue

        target = max_candidates_per_order
        selected = set()
        while len(selected) < target:
            cand = tuple(sorted((rng.choice(n_nodes, size=order, replace=False) + 1).tolist()))
            selected.add(cand)

        sampled_edges = [list(edge) for edge in selected]
        rng.shuffle(sampled_edges)
        pool[key] = sampled_edges

    return pool


def candidate_pool_to_gpu(
    all_possible_edges: dict,
    order_keys: list[str],
    order_sizes: dict,
    device: torch.device,
):
    all_possible_edges_gpu = {}
    n_hyperedges = 0
    for key in order_keys:
        edges = all_possible_edges.get(key, [])
        n_hyperedges += len(edges)
        if len(edges) > 0:
            all_possible_edges_gpu[key] = torch.tensor(edges, dtype=torch.long, device=device) - 1
        else:
            all_possible_edges_gpu[key] = torch.empty((0, order_sizes[key]), dtype=torch.long, device=device)
    return all_possible_edges_gpu, n_hyperedges


def precompute_basic_dynamics(
    x_data_gpu: torch.Tensor,
    t_data_gpu: torch.Tensor,
    n_times: int,
    n_nodes: int,
    state_dim: int,
    device: torch.device,
    is_multi_trajectory: bool,
    n_trajectories: int,
    show_progress: bool = False,
) -> torch.Tensor:
    if is_multi_trajectory:
        f_all = torch.zeros((n_trajectories, n_times, n_nodes, state_dim), device=device)
    else:
        f_all = torch.zeros((n_times, n_nodes, state_dim), device=device)

    dynamic_params = inspect.signature(HypergraphModel.dynamic_f).parameters
    dynamic_accepts_time = len(dynamic_params) >= 3

    if is_multi_trajectory:
        outer_iter = range(n_trajectories)
        for k in outer_iter:
            time_iter = range(n_times)
            if show_progress:
                time_iter = tqdm(time_iter, desc=f"Precomputing f (traj {k+1}/{n_trajectories})", leave=False)
            for t_idx in time_iter:
                x_t = x_data_gpu[k, t_idx]
                if dynamic_accepts_time:
                    f_all[k, t_idx] = HypergraphModel.dynamic_f(x_t, n_nodes, t_data_gpu[t_idx])
                else:
                    f_all[k, t_idx] = HypergraphModel.dynamic_f(x_t, n_nodes)
    else:
        time_iter = range(n_times)
        if show_progress:
            time_iter = tqdm(time_iter, desc="Precomputing f", leave=False)
        for t_idx in time_iter:
            x_t = x_data_gpu[t_idx]
            if dynamic_accepts_time:
                f_all[t_idx] = HypergraphModel.dynamic_f(x_t, n_nodes, t_data_gpu[t_idx])
            else:
                f_all[t_idx] = HypergraphModel.dynamic_f(x_t, n_nodes)

    return f_all


def precompute_phi_for_pool(
    x_data_gpu: torch.Tensor,
    all_possible_edges_gpu: dict,
    n_times: int,
    n_nodes: int,
    state_dim: int,
    n_hyperedges: int,
    device: torch.device,
    is_multi_trajectory: bool,
    n_trajectories: int,
    show_progress: bool = False,
) -> torch.Tensor:
    if is_multi_trajectory:
        Phi_all = torch.zeros((n_trajectories, n_times, n_nodes, state_dim, n_hyperedges), device=device)
        for k in range(n_trajectories):
            time_iter = range(n_times)
            if show_progress:
                time_iter = tqdm(time_iter, desc=f"Precomputing Phi (traj {k+1}/{n_trajectories})", leave=False)
            for t_idx in time_iter:
                Phi_all[k, t_idx] = HypergraphModel.dynamic_phi(
                    x_data_gpu[k, t_idx], all_possible_edges_gpu, n_nodes, device
                )
        return Phi_all

    Phi_all = torch.zeros((n_times, n_nodes, state_dim, n_hyperedges), device=device)
    time_iter = range(n_times)
    if show_progress:
        time_iter = tqdm(time_iter, desc="Precomputing Phi", leave=False)
    for t_idx in time_iter:
        Phi_all[t_idx] = HypergraphModel.dynamic_phi(
            x_data_gpu[t_idx], all_possible_edges_gpu, n_nodes, device
        )
    return Phi_all


def plot_roc_curves(roc_data, auc_scores, save_dir, max_order):
    plt.figure(figsize=(8, 6))
    colors = {
        '2-edges': 'blue',
        '3-edges': 'green',
        '4-edges': 'red',
        '5-edges': 'purple',
        '6-edges': 'orange',
        '7-edges': 'brown'
    }
    
    for order_label in ['2-edges', '3-edges', '4-edges', '5-edges', '6-edges', '7-edges']:
        if order_label in roc_data and roc_data[order_label] is not None:
            fpr, tpr = roc_data[order_label]
            auc_val = auc_scores[order_label]
            color = colors.get(order_label, 'black')
            plt.plot(fpr, tpr, color=color, linewidth=2, 
                    label=f'{order_label} (AUC = {auc_val:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curves for Identified Hypergraphs', fontsize=17)
    plt.legend(fontsize=14, loc="lower right")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, f'roc_curves_{max_order}_order.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"ROC curves saved to {os.path.join(save_dir, f'roc_curves_{max_order}_order.png')}")


def train_implicit_model(
    N=8, 
    max_order=7, 
    n_epochs=40000, 
    lr=0.001, 
    batch_size=32, 
    gpu_id=6,
    use_nn=True,
    n_samples=11,
    noise=0.0,
    n_trajectories: int = 1,
    embed_dim: int = 32,
    head_hidden: int = 64,
    eval_every: int = 5000,
    max_candidates_per_order: int = 4096,
    results_root: str = "results/implicit",
    scene_label: str = "unknown",
):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    edge_config = HypergraphModel.get_hyperedge_config(N, max_order)
    rng = np.random.default_rng(42)
    order_keys = ['edges', 'triangles', 'quads', 'quints', 'sexts', 'septs']
    order_sizes = {
        'edges': 2,
        'triangles': 3,
        'quads': 4,
        'quints': 5,
        'sexts': 6,
        'septs': 7,
    }
    all_possible_edges = build_training_candidate_pool(
        n_nodes=N,
        max_order=max_order,
        max_candidates_per_order=max_candidates_per_order,
        rng=rng,
    )
    n_hyperedges = sum(len(all_possible_edges.get(key, [])) for key in order_keys)
    print(f"N={N}, max_order={max_order}")
    print(f"Training candidates per step={n_hyperedges}")
    print(f"  2-edges: {len(all_possible_edges.get('edges', []))}")
    print(f"  3-edges: {len(all_possible_edges.get('triangles', []))}")
    print(f"  4-edges: {len(all_possible_edges.get('quads', []))}")
    print(f"  5-edges: {len(all_possible_edges.get('quints', []))}")
    print(f"  6-edges: {len(all_possible_edges.get('sexts', []))}")
    print(f"  7-edges: {len(all_possible_edges.get('septs', []))}")
    print(f"\nTrue hyperedges:")
    print(f"  2-edges: {len(edge_config.get('edges', []))}")
    print(f"  3-edges: {len(edge_config.get('triangles', []))}")
    print(f"  4-edges: {len(edge_config.get('quads', []))}")
    print(f"  5-edges: {len(edge_config.get('quints', []))}")
    print(f"  6-edges: {len(edge_config.get('sexts', []))}")
    print(f"  7-edges: {len(edge_config.get('septs', []))}")
    
    print("\nGenerating training data...")
    try:
        model_module = HypergraphModel.__module__
    except Exception:
        model_module = ""
    is_discrete_social = "lib_social_contagion" in model_module

    if is_discrete_social and n_trajectories > 1:
        print(f"Using social contagion model with {n_trajectories} independent trajectories on the same hypergraph...")
        traj_t_list = []
        traj_x_list = []
        base_seed = 123
        for k in range(n_trajectories):
            seed_k = base_seed + k
            t_k, x_k = HypergraphModel.generate_training_data(N, edge_config, n_samples, noise=noise, seed=seed_k)
            traj_t_list.append(t_k)
            traj_x_list.append(x_k)
        t_data = traj_t_list[0]
        x_data_multi = np.stack(traj_x_list, axis=0)  # [K, T, N, D]
        n_times = len(t_data)
        print(f"Multi-trajectory data shape: {x_data_multi.shape}, Time points: {n_times}")
        state_dim = x_data_multi.shape[3]
        if noise > 0:
            print(f"Added Gaussian noise with std={noise}")
        save_dir = os.path.join(
            results_root,
            scene_label,
            f"sample_{n_samples}_noise_{noise}",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(save_dir, exist_ok=True)
        x_target = x_data_multi.copy()
        window = 5
        pad = window // 2
        x_smooth = np.zeros_like(x_data_multi)
        for k in range(n_trajectories):
            x_padded = np.pad(x_data_multi[k], ((pad, pad), (0, 0), (0, 0)), mode="edge")
            for t_idx in range(n_times):
                x_smooth[k, t_idx] = x_padded[t_idx:t_idx + window].mean(axis=0)
        x_data = x_smooth  # [K, T, N, D]
    else:
        t_data, x_data = HypergraphModel.generate_training_data(N, edge_config, n_samples, noise=noise)
        n_times = len(t_data)
        print(f"Data shape: {x_data.shape}, Time points: {n_times}")
        state_dim = x_data.shape[2]
        if noise > 0:
            print(f"Added Gaussian noise with std={noise}")

        if use_nn:
            print("\n" + "="*80)
            print("LINEAR INTERPOLATION MODE: Using piecewise linear curves")
            print("="*80)        
            print(f"\nUsing LINEAR INTERPOLATION (no neural network)")
            print(f"  Original {n_times} points will be connected by straight lines")
            print(f"  This GUARANTEES passing through all observation points!")
            print(f"\nDense resampling using linear interpolation to unified sampling rate 500/20...")
            total_time = float(t_data[-1] - t_data[0])  # should be ~20.0
            target_points = 500
            n_resampled = target_points
            t_data_resampled = np.linspace(t_data[0], t_data[-1], n_resampled)
            print(f"Original data points: {n_times} (sampling rate ~ {n_times/total_time:.2f} per time unit)")
            print(f"Resampled data points: {n_resampled} (sampling rate ~ {n_resampled/total_time:.2f} per time unit)")
            from scipy.interpolate import interp1d
            x_data_resampled = np.zeros((n_resampled, N, state_dim))
            
            for node_idx in range(N):
                for coord_idx in range(state_dim):
                    interp_func = interp1d(t_data, x_data[:, node_idx, coord_idx], 
                                          kind='linear', fill_value='extrapolate')
                    x_data_resampled[:, node_idx, coord_idx] = interp_func(t_data_resampled)
            
            print(f"Linear interpolation completed!")
            
            x_data_resampled_cpu = x_data_resampled
            
            fig, axes = plt.subplots(N, state_dim, figsize=(5 * state_dim, 2.5 * N))
            coord_names = [f'd{idx+1}' for idx in range(state_dim)]
            
            for node_idx in range(N):
                for coord_idx in range(state_dim):
                    ax = axes[node_idx, coord_idx] if state_dim > 1 else axes[node_idx]
                    
                    ax.plot(t_data, x_data[:, node_idx, coord_idx], 'o', 
                           label='Original (ODE)', markersize=4, alpha=0.7, color='blue')
                    
                    ax.plot(t_data_resampled, x_data_resampled_cpu[:, node_idx, coord_idx], 
                           '-', label='Resampled (Linear)', linewidth=1, alpha=0.8, color='red')
                    if node_idx == 0:
                        ax.set_title(f'{coord_names[coord_idx]}-coordinate', fontsize=12)
                    if coord_idx == 0:
                        ax.set_ylabel(f'Node {node_idx+1}', fontsize=11)
                    if node_idx == N-1:
                        ax.set_xlabel('Time', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    if node_idx == 0 and coord_idx == 0:
                        ax.legend(fontsize=9)
            
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(
                results_root,
                scene_label,
                f"sample_{n_samples}_noise_{noise}",
                timestamp,
            )
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'linear_vs_original.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Comparison plot saved to {os.path.join(save_dir, 'linear_vs_original.png')}")
            
            from scipy.interpolate import interp1d
            errors_per_node = []
            for node_idx in range(N):
                for coord_idx in range(state_dim):
                    interp_func = interp1d(t_data, x_data[:, node_idx, coord_idx], kind='cubic')
                    x_orig_interp = interp_func(t_data_resampled)
                    
                    error = np.abs(x_data_resampled_cpu[:, node_idx, coord_idx] - x_orig_interp)
                    errors_per_node.append(error)
            
            errors_all = np.concatenate(errors_per_node)
            print(f"\nLinear interpolation fitting error statistics:")
            print(f"  Mean absolute error: {errors_all.mean():.6f}")
            print(f"  Max absolute error: {errors_all.max():.6f}")
            print(f"  Std of error: {errors_all.std():.6f}")
            
            t_data = t_data_resampled
            x_data = x_data_resampled_cpu
            n_times = n_resampled
            print(f"\nReplaced data with resampled version: {x_data.shape}")
            print("="*80 + "\n")

        # Single-trajectory targets / smoothing
        x_target = x_data
        if is_discrete_social:
            print("Applying temporal smoothing for discrete contagion data (moving average, window=5)...")
            x_target = x_data.copy()
            window = 5
            pad = window // 2
            x_padded = np.pad(x_data, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            x_smooth = np.zeros_like(x_data)
            for t_idx in range(n_times):
                x_smooth[t_idx] = x_padded[t_idx:t_idx + window].mean(axis=0)
            x_data = x_smooth
        # Set up a default save_dir for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(
            results_root,
            scene_label,
            f"sample_{n_samples}_noise_{noise}",
            timestamp,
        )
        os.makedirs(save_dir, exist_ok=True)

    print("Transferring data to GPU...")
    t_data_gpu = torch.tensor(t_data, dtype=torch.float32, device=device)
    if is_discrete_social and n_trajectories > 1:
        x_data_gpu = torch.tensor(x_data, dtype=torch.float32, device=device)       # [K, T, N, D]
        x_target_gpu = torch.tensor(x_target, dtype=torch.float32, device=device)   # [K, T, N, D]
    else:
        x_data_gpu = torch.tensor(x_data, dtype=torch.float32, device=device)       # [T, N, D]
        x_target_gpu = torch.tensor(x_target, dtype=torch.float32, device=device)   # [T, N, D]
    print("Pre-computing basic dynamics f for all time points...")
    is_multi_trajectory = is_discrete_social and n_trajectories > 1
    f_all = precompute_basic_dynamics(
        x_data_gpu=x_data_gpu,
        t_data_gpu=t_data_gpu,
        n_times=n_times,
        n_nodes=N,
        state_dim=state_dim,
        device=device,
        is_multi_trajectory=is_multi_trajectory,
        n_trajectories=n_trajectories,
        show_progress=True,
    )
    dt_all = t_data_gpu[1:] - t_data_gpu[:-1]

    implicit_model = ImplicitHypergraphNet(
        n_nodes=N,
        max_order=max_order,
        embed_dim=embed_dim,
        head_hidden=head_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in implicit_model.parameters())
    print(f"Implicit model parameters: {n_params} (vs explicit candidates: {n_hyperedges})")

    optimizer = optim.Adam(implicit_model.parameters(), lr=lr)
    use_discrete_residual = is_discrete_social

    # Training loop (with progress bar)
    pbar = tqdm(range(n_epochs), desc="Training Progress", ncols=120)
    for epoch in pbar:
        all_possible_edges = build_training_candidate_pool(
            n_nodes=N,
            max_order=max_order,
            max_candidates_per_order=max_candidates_per_order,
            rng=rng,
        )
        all_possible_edges_gpu, n_hyperedges = candidate_pool_to_gpu(
            all_possible_edges=all_possible_edges,
            order_keys=order_keys,
            order_sizes=order_sizes,
            device=device,
        )
        Phi_all = precompute_phi_for_pool(
            x_data_gpu=x_data_gpu,
            all_possible_edges_gpu=all_possible_edges_gpu,
            n_times=n_times,
            n_nodes=N,
            state_dim=state_dim,
            n_hyperedges=n_hyperedges,
            device=device,
            is_multi_trajectory=is_multi_trajectory,
            n_trajectories=n_trajectories,
            show_progress=False,
        )
        if epoch == 0:
            print(f"Phi_all shape: {Phi_all.shape}, f_all shape: {f_all.shape}")

        optimizer.zero_grad()
        score_vec = implicit_model.predict_all_scores(all_possible_edges_gpu)  # [E]

        if use_discrete_residual:
            if is_discrete_social and n_trajectories > 1:
                # x_curr: [K, T-1, N, D]
                x_curr = x_data_gpu[:, :-1]
                y_next = x_target_gpu[:, 1:]
                dt_view = dt_all.view(1, -1, 1, 1)
                f_curr = f_all[:, :-1]              # [K, T-1, N, D]
                Phi_curr = Phi_all[:, :-1]          # [K, T-1, N, D, E]
                PhiS_curr = torch.einsum('ktnie,e->ktni', Phi_curr, score_vec)  # [K, T-1, N, D]
                incr_pred = dt_view * (f_curr + PhiS_curr)
                p_next = x_curr + incr_pred
            else:
                # Single trajectory (K=1) as before
                x_curr = x_data_gpu[:-1]          # [T-1, N, D]
                y_next = x_target_gpu[1:]         # [T-1, N, D]
                dt_view = dt_all.view(-1, 1, 1)   # [T-1, 1, 1]
                f_curr = f_all[:-1]               # [T-1, N, D]
                Phi_curr = Phi_all[:-1]           # [T-1, N, D, E]
                PhiS_curr = torch.einsum('tnie,e->tni', Phi_curr, score_vec)  # [T-1, N, D]
                incr_pred = dt_view * (f_curr + PhiS_curr)
                p_next = x_curr + incr_pred

            p_next = torch.clamp(p_next, 1e-4, 1.0 - 1e-4)
            # Binary cross-entropy: -[y log p + (1-y) log(1-p)]
            bce = -(y_next * torch.log(p_next) + (1.0 - y_next) * torch.log(1.0 - p_next))
            total_loss = bce.mean()
        else:
            # Original integral formulation over random time intervals
            losses = []
            for _ in range(batch_size):
                idx_i, idx_j = np.random.choice(n_times, size=2, replace=False)
                if idx_i > idx_j:
                    idx_i, idx_j = idx_j, idx_i

                x_i, x_j = x_data_gpu[idx_i], x_data_gpu[idx_j]  # [N, D]

                lhs = x_j - x_i  # [N, D]
                f_interval = f_all[idx_i:idx_j]  # [idx_j-idx_i, N, D]
                Phi_interval = Phi_all[idx_i:idx_j]  # [idx_j-idx_i, N, D, n_hyperedges]
                dt_interval = dt_all[idx_i:idx_j]  # [idx_j-idx_i]
                integral_f = torch.einsum('tni,t->ni', f_interval, dt_interval)  # [N, D]
                Phi_S_interval = torch.einsum('tnie,e->tni', Phi_interval, score_vec)  # [t, N, D]
                integral_phi_S = torch.einsum('tni,t->ni', Phi_S_interval, dt_interval)  # [N, D]

                residual = lhs - integral_f - integral_phi_S
                loss = torch.mean(residual ** 2)
                losses.append(loss)

            total_loss = torch.mean(torch.stack(losses))

        total_loss.backward()
        optimizer.step()

        pbar.set_postfix({
            'loss': f'{total_loss.item():.6f}'
        })
        
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            print(f"\n\n{'='*60}")
            print(f"Epoch {epoch+1}/{n_epochs} - AUC Evaluation")
            print('='*60)
            auc_scores, _, eval_stats = evaluate_full_auc_scores(
                implicit_model=implicit_model,
                edge_config=edge_config,
                N=N,
                max_order=max_order,
                device=device,
                score_batch_size=8192,
            )

            for order_label in ['2-edges', '3-edges', '4-edges', '5-edges', '6-edges', '7-edges']:
                if order_label not in auc_scores or order_label not in eval_stats:
                    continue
                st = eval_stats[order_label]
                auc_score = auc_scores[order_label]
                if auc_score is not None:
                    print(
                        f"  {order_label}: AUC = {auc_score:.4f} | "
                        f"Possible={st['possible']}, True={st['true']} | "
                        f"P range=[{st['p_min']:.3f}, {st['p_max']:.3f}]"
                    )
                else:
                    print(f"  {order_label}: N/A | Possible={st['possible']}, True={st['true']}")
            print('='*60 + '\n')
    
    return implicit_model, edge_config, save_dir, all_possible_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='neuronal',
                        choices=['ecological', 'neuronal', 'rossler', 'social'])
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--n_epochs', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--n_nodes', type=int, default=None)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--results_root', type=str, default='results/implicit')
    parser.add_argument('--n_trajectories', type=int, default=1)
    parser.add_argument('--max_order', type=int, default=None)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--head_hidden', type=int, default=64)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--max_candidates_per_order', type=int, default=4096)
    args = parser.parse_args()
    
    HypergraphModel, scene_spec = get_scene_model(args.scene)
    defaults = HypergraphModel.get_default_params()
    if args.scene in {"ecological", "neuronal", "social"}:
        N = defaults["n_nodes"]
        max_order = defaults["max_order"]
    else:
        N = args.n_nodes if args.n_nodes is not None else defaults["n_nodes"]
        max_order = args.max_order if args.max_order is not None else defaults["max_order"]
    
    implicit_model, edge_config, save_dir, all_possible_edges = train_implicit_model(
        N=N, 
        max_order=max_order, 
        n_epochs=args.n_epochs, 
        lr=args.lr, 
        batch_size=32,
        gpu_id=args.gpu_id,
        use_nn=True,
        n_samples=args.n_samples,
        noise=args.noise,
        n_trajectories=args.n_trajectories,
        embed_dim=args.embed_dim,
        head_hidden=args.head_hidden,
        eval_every=args.eval_every,
        max_candidates_per_order=args.max_candidates_per_order,
        results_root=args.results_root,
        scene_label=scene_spec.label,
    )
    
    print("\nComputing AUC scores...")
    model_device = next(implicit_model.parameters()).device
    auc_scores, roc_data, _ = evaluate_full_auc_scores(
        implicit_model=implicit_model,
        edge_config=edge_config,
        N=N,
        max_order=max_order,
        device=model_device,
        score_batch_size=8192,
    )
    
    print("\nAUC scores for each order:")
    for order_label, auc_score in auc_scores.items():
        if auc_score is not None:
            print(f"  {order_label}: {auc_score:.4f}")
        else:
            print(f"  {order_label}: N/A (no positive/negative samples)")
    
    with open(f"{save_dir}/auc_scores.txt", 'w') as f:
        f.write(f"scene={scene_spec.label}, lib={scene_spec.module}\n")
        f.write(f"N={N}, max_order={max_order}, n_samples={args.n_samples}, noise={args.noise}\n")
        f.write(f"embed_dim={args.embed_dim}, head_hidden={args.head_hidden}\n")
        f.write("\nAUC Scores:\n")
        for order_label, auc_score in auc_scores.items():
            if auc_score is not None:
                f.write(f"  {order_label}: {auc_score:.4f}\n")
            else:
                f.write(f"  {order_label}: N/A\n")
    
    print("\nPlotting ROC curves...")
    plot_roc_curves(roc_data, auc_scores, save_dir, max_order)

    write_standard_summary(
        save_dir=save_dir,
        method="implicit_hypergraph",
        scene=scene_spec.label,
        config={
            "n_nodes": N,
            "max_order": max_order,
            "n_samples": args.n_samples,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "noise": args.noise,
            "n_trajectories": args.n_trajectories,
            "embed_dim": args.embed_dim,
            "head_hidden": args.head_hidden,
            "eval_every": args.eval_every,
            "max_candidates_per_order": args.max_candidates_per_order,
        },
        auc_scores=auc_scores,
    )
    
    print(f"\nResults saved to {save_dir}/")
