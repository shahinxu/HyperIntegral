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


class OrderwiseSparseHypergraphTensors(nn.Module):
    def __init__(self, n_nodes: int, all_possible_edges: dict):
        super().__init__()
        self.n_nodes = n_nodes
        self.order_keys = ['edges', 'triangles', 'quads', 'quints', 'sexts', 'septs']
        self.order_dims = {
            'edges': 2,
            'triangles': 3,
            'quads': 4,
            'quints': 5,
            'sexts': 6,
            'septs': 7,
        }
        self.order_to_offset = {
            'edges': -2.0,
            'triangles': -3.0,
            'quads': -4.0,
            'quints': -4.0,
            'sexts': -4.0,
            'septs': -4.0,
        }
        self.params = nn.ParameterDict()
        for key in self.order_keys:
            edges = all_possible_edges.get(key, [])
            n_edges = len(edges)
            if n_edges == 0:
                self.register_buffer(f'{key}_indices', torch.empty((self.order_dims[key], 0), dtype=torch.long))
                continue
            index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous() - 1
            self.register_buffer(f'{key}_indices', index_tensor)
            values = torch.randn(n_edges, 1) + self.order_to_offset[key]
            self.params[key] = nn.Parameter(values)

    def flat_scores(self) -> torch.Tensor:
        parts = []
        device = next(self.parameters()).device
        for key in self.order_keys:
            param = self.params.get(key)
            if param is None or param.numel() == 0:
                continue
            parts.append(param)
        if not parts:
            return torch.empty((0, 1), device=device)
        return torch.cat(parts, dim=0)

    def numpy_by_order(self) -> dict:
        result = {}
        for key in self.order_keys:
            param = self.params.get(key)
            if param is None or param.numel() == 0:
                result[key] = np.empty(0, dtype=np.float32)
            else:
                result[key] = torch.sigmoid(param.detach()).cpu().numpy().reshape(-1)
        return result

    def probabilities_by_order(self) -> dict:
        result = {}
        for key in self.order_keys:
            param = self.params.get(key)
            if param is None or param.numel() == 0:
                result[key] = None
            else:
                result[key] = torch.sigmoid(param.view(-1))
        return result

    def sparse_tensors(self, apply_sigmoid: bool = False) -> dict:
        tensors = {}
        for key in self.order_keys:
            indices = getattr(self, f'{key}_indices')
            order = self.order_dims[key]
            shape = (self.n_nodes,) * order
            param = self.params.get(key)
            if param is None or param.numel() == 0:
                tensors[key] = torch.sparse_coo_tensor(
                    indices,
                    torch.empty(0, device=indices.device),
                    size=shape,
                    device=indices.device,
                ).coalesce()
                continue
            values = param.view(-1)
            if apply_sigmoid:
                values = torch.sigmoid(values)
            tensors[key] = torch.sparse_coo_tensor(
                indices,
                values,
                size=shape,
                device=values.device,
            ).coalesce()
        return tensors

    def clamp_(self, min_value: float, max_value: float):
        with torch.no_grad():
            for param in self.params.values():
                param.clamp_(min_value, max_value)
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



def compute_auc_scores(order_scores, edge_config, N, max_order, all_possible=None):
    # Candidate pool for evaluation: prefer the one used in training.
    if all_possible is None:
        all_possible = HypergraphModel.generate_all_possible_hyperedges(N, max_order)
    
    # Compute AUC for each order
    auc_scores = {}
    roc_data = {}
    
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
        y_pred = list(order_scores.get(order_name, np.empty(0, dtype=np.float32)))
        
        for edge in possible_edges:
            # Check if this hyperedge is in the true configuration
            is_true_edge = any(
                sorted(edge) == sorted(true_edge) 
                for true_edge in true_edges
            )
            y_true.append(1 if is_true_edge else 0)
        
        # Compute AUC
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Only compute AUC when both positive and negative samples exist
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc_score = auc(fpr, tpr)
            auc_scores[order_label] = auc_score
            roc_data[order_label] = (fpr, tpr)
        else:
            auc_scores[order_label] = None
            roc_data[order_label] = None
    
    return auc_scores, roc_data


def build_training_candidate_pool(
    n_nodes: int,
    max_order: int,
    edge_config: dict,
    max_candidates_per_order: int = 5000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
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
        true_set = {tuple(sorted(map(int, e))) for e in edge_config.get(key, [])}
        total = comb(n_nodes, order)

        if total <= max_candidates_per_order:
            pool[key] = [list(edge) for edge in combinations(range(1, n_nodes + 1), order)]
            continue

        target = max(max_candidates_per_order, len(true_set))
        selected = set(true_set)
        attempts = 0
        max_attempts = max(1000, target * 50)
        while len(selected) < target and attempts < max_attempts:
            cand = tuple(sorted((rng.choice(n_nodes, size=order, replace=False) + 1).tolist()))
            selected.add(cand)
            attempts += 1

        pool[key] = [list(edge) for edge in sorted(selected)]

    return pool


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
    
    # Plot random guess line
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


def _supports_tensorized_coupling(model_module: str) -> bool:
    return (
        "lib_ecological_dynamics" in model_module
        or "lib_social_contagion" in model_module
    )


def _tensorized_order_scale(order_key: str, model_module: str):
    params = HypergraphModel._cached_params
    if "lib_ecological_dynamics" in model_module:
        scale_map = {
            "edges": -params.w2,
            "triangles": -params.w3,
            "quads": -params.w4,
            "quints": -params.w5,
            "sexts": -params.w5,
            "septs": -params.w5,
        }
        return scale_map[order_key]
    if "lib_social_contagion" in model_module:
        scale_map = {
            "edges": params.beta,
            "triangles": params.beta_delta,
            "quads": params.beta_delta,
            "quints": params.beta_delta,
            "sexts": params.beta_delta,
            "septs": params.beta_delta,
        }
        return scale_map[order_key]
    raise NotImplementedError(f"Tensorized coupling not implemented for scene module: {model_module}")


def compute_tensorized_coupling_all_times(
    x_data_gpu: torch.Tensor,
    orderwise_params: OrderwiseSparseHypergraphTensors,
    model_module: str,
) -> torch.Tensor:
    if x_data_gpu.ndim == 4:
        leading_shape = x_data_gpu.shape[:-2]
        n_nodes = x_data_gpu.shape[-2]
        x_flat = x_data_gpu[..., 0].reshape(-1, n_nodes)
    else:
        leading_shape = x_data_gpu.shape[:-2]
        n_nodes = x_data_gpu.shape[-2]
        x_flat = x_data_gpu[..., 0].reshape(-1, n_nodes)

    batch_size = x_flat.shape[0]
    device = x_flat.device
    dtype = x_flat.dtype
    coupling = torch.zeros_like(x_flat)
    is_social = "lib_social_contagion" in model_module
    susceptible = 1.0 - x_flat if is_social else None
    order_probs = orderwise_params.probabilities_by_order()

    for order_key in orderwise_params.order_keys:
        edge_probs = order_probs.get(order_key)
        if edge_probs is None or edge_probs.numel() == 0:
            continue

        edge_indices = getattr(orderwise_params, f'{order_key}_indices').t()
        if edge_indices.numel() == 0:
            continue

        edge_indices = edge_indices.to(device)
        order = edge_indices.shape[1]
        scale = torch.as_tensor(_tensorized_order_scale(order_key, model_module), device=device, dtype=dtype)
        edge_probs = edge_probs.to(device=device, dtype=dtype)
        selected = x_flat[:, edge_indices]  # [B, E, order]

        if not is_social:
            edge_contrib = scale * selected.prod(dim=2) * edge_probs.view(1, -1)
            for local_pos in range(order):
                node_idx = edge_indices[:, local_pos].unsqueeze(0).expand(batch_size, -1)
                coupling.scatter_add_(1, node_idx, edge_contrib)
            continue

        for local_pos in range(order):
            other_prod = torch.ones((batch_size, edge_indices.shape[0]), device=device, dtype=dtype)
            for other_pos in range(order):
                if other_pos == local_pos:
                    continue
                other_prod = other_prod * selected[:, :, other_pos]
            node_idx = edge_indices[:, local_pos]
            susc = susceptible[:, node_idx]
            node_contrib = scale * edge_probs.view(1, -1) * susc * other_prod
            node_idx_expanded = node_idx.unsqueeze(0).expand(batch_size, -1)
            coupling.scatter_add_(1, node_idx_expanded, node_contrib)

    return coupling.reshape(*leading_shape, n_nodes, 1)


def train_integral_model(
    N=8, 
    max_order=7, 
    n_epochs=10000, 
    lr=0.001, 
    batch_size=32, 
    gpu_id=6,
    use_nn=True,
    n_samples=11,
    noise=0.0,
    n_trajectories: int = 1,
    max_candidates_per_order: int = 5000,
    results_root: str = "results/integral",
    scene_label: str = "unknown",
):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    edge_config = HypergraphModel.get_hyperedge_config(N, max_order)
    all_possible_edges = build_training_candidate_pool(
        n_nodes=N,
        max_order=max_order,
        edge_config=edge_config,
        max_candidates_per_order=max_candidates_per_order,
        seed=42,
    )
    order_keys = ['edges', 'triangles', 'quads', 'quints', 'sexts', 'septs']
    order_sizes = {
        'edges': 2,
        'triangles': 3,
        'quads': 4,
        'quints': 5,
        'sexts': 6,
        'septs': 7,
    }
    n_hyperedges = sum(len(all_possible_edges.get(key, [])) for key in order_keys)
    print(f"N={N}, max_order={max_order}")
    print(f"Total possible hyperedges={n_hyperedges}")
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

    # --- Data generation: single vs multi-trajectory ---
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
        # Assume all trajectories share the same time grid
        t_data = traj_t_list[0]
        x_data_multi = np.stack(traj_x_list, axis=0)  # [K, T, N, D]
        n_times = len(t_data)
        print(f"Multi-trajectory data shape: {x_data_multi.shape}, Time points: {n_times}")
        state_dim = x_data_multi.shape[3]
        if noise > 0:
            print(f"Added Gaussian noise with std={noise}")
        # For simplicity, in multi-trajectory mode we skip extra linear
        # resampling and work directly on the native ODE grid.
        save_dir = os.path.join(
            results_root,
            scene_label,
            f"sample_{n_samples}_noise_{noise}",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(save_dir, exist_ok=True)
        # Targets are the original 0/1 observations; inputs can be smoothed.
        x_target = x_data_multi.copy()
        # Temporal smoothing per trajectory to reduce Bernoulli noise.
        print("Applying temporal smoothing for discrete contagion data (multi-trajectory, moving average, window=5)...")
        window = 5
        pad = window // 2
        x_smooth = np.zeros_like(x_data_multi)
        for k in range(n_trajectories):
            x_padded = np.pad(x_data_multi[k], ((pad, pad), (0, 0), (0, 0)), mode="edge")
            for t_idx in range(n_times):
                x_smooth[k, t_idx] = x_padded[t_idx:t_idx + window].mean(axis=0)
        x_data = x_smooth  # [K, T, N, D]
    else:
        # Original single-trajectory path
        t_data, x_data = HypergraphModel.generate_training_data(N, edge_config, n_samples, noise=noise)
        n_times = len(t_data)
        print(f"Data shape: {x_data.shape}, Time points: {n_times}")
        state_dim = x_data.shape[2]
        if noise > 0:
            print(f"Added Gaussian noise with std={noise}")

        if use_nn and not is_discrete_social:
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
        elif use_nn and is_discrete_social:
            print("Skipping linear interpolation for discrete social contagion data; using native observation grid.")

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
    tensorized_supported = _supports_tensorized_coupling(model_module)

    all_possible_edges_gpu = {}
    Phi_all = None
    print("Pre-computing basic dynamics f for all time points...")
    if is_discrete_social and n_trajectories > 1:
        f_all = torch.zeros((n_trajectories, n_times, N, state_dim), device=device)
    else:
        f_all = torch.zeros((n_times, N, state_dim), device=device)
    dynamic_params = inspect.signature(HypergraphModel.dynamic_f).parameters
    dynamic_accepts_time = len(dynamic_params) >= 3

    if is_discrete_social and n_trajectories > 1:
        for k in range(n_trajectories):
            for t_idx in tqdm(range(n_times), desc=f"Precomputing f (traj {k+1}/{n_trajectories})", leave=False):
                x_t = x_data_gpu[k, t_idx]
                if dynamic_accepts_time:
                    f_all[k, t_idx] = HypergraphModel.dynamic_f(x_t, N, t_data_gpu[t_idx])
                else:
                    f_all[k, t_idx] = HypergraphModel.dynamic_f(x_t, N)
    else:
        for t_idx in tqdm(range(n_times), desc="Precomputing f", leave=False):
            x_t = x_data_gpu[t_idx]
            if dynamic_accepts_time:
                f_all[t_idx] = HypergraphModel.dynamic_f(x_t, N, t_data_gpu[t_idx])
            else:
                f_all[t_idx] = HypergraphModel.dynamic_f(x_t, N)

    if not tensorized_supported:
        print("Converting hyperedge indices to GPU tensors...")
        for key in order_keys:
            edges = all_possible_edges.get(key, [])
            if len(edges) > 0:
                all_possible_edges_gpu[key] = torch.tensor(edges, dtype=torch.long, device=device) - 1
            else:
                all_possible_edges_gpu[key] = torch.empty((0, order_sizes[key]), dtype=torch.long, device=device)

        print("Pre-computing coupling tensor Phi for all time points...")
        if is_discrete_social and n_trajectories > 1:
            Phi_all = torch.zeros((n_trajectories, n_times, N, state_dim, n_hyperedges), device=device)
            for k in range(n_trajectories):
                for t_idx in tqdm(range(n_times), desc=f"Precomputing Phi (traj {k+1}/{n_trajectories})", leave=False):
                    x_t = x_data_gpu[k, t_idx]
                    Phi_all[k, t_idx] = HypergraphModel.dynamic_phi(
                        x_t, all_possible_edges_gpu, N, device
                    )
        else:
            Phi_all = torch.zeros((n_times, N, state_dim, n_hyperedges), device=device)
            for t_idx in tqdm(range(n_times), desc="Precomputing Phi", leave=False):
                x_t = x_data_gpu[t_idx]
                Phi_all[t_idx] = HypergraphModel.dynamic_phi(
                    x_t, all_possible_edges_gpu, N, device
                )
    dt_all = t_data_gpu[1:] - t_data_gpu[:-1]
    if tensorized_supported:
        print(f"Tensorized mode active. f_all shape: {f_all.shape}")
    else:
        print(f"Phi_all shape: {Phi_all.shape}, f_all shape: {f_all.shape}")
    
    orderwise_params = OrderwiseSparseHypergraphTensors(N, all_possible_edges).to(device)
    n_param_tensors = len(orderwise_params.params)
    sparse_tensors = orderwise_params.sparse_tensors(apply_sigmoid=False)
    print(f"Using {n_param_tensors} order-specific sparse tensors for candidate scores")
    for key in order_keys:
        tensor = sparse_tensors[key]
        if tensor._nnz() == 0:
            continue
        print(f"  {key}: sparse shape={tuple(tensor.shape)}, nnz={tensor._nnz()}")
    optimizer = optim.Adam(orderwise_params.parameters(), lr=lr)

    # use_discrete_residual is true exactly for the social contagion
    # model, where observations are 0/1 and we use a Bernoulli-style
    # likelihood rather than an MSE on x.
    use_discrete_residual = is_discrete_social

    # Training loop (with progress bar)
    pbar = tqdm(range(n_epochs), desc="Training Progress", ncols=120)
    for epoch in pbar:
        optimizer.zero_grad()
        A = None
        if not tensorized_supported:
            A = orderwise_params.flat_scores()
        tensorized_coupling_all = None
        if tensorized_supported:
            tensorized_coupling_all = compute_tensorized_coupling_all_times(
                x_data_gpu=x_data_gpu,
                orderwise_params=orderwise_params,
                model_module=model_module,
            )

        if use_discrete_residual:
            # Discrete-time Bernoulli likelihood on Euler-step probabilities.
            # Multi-trajectory case: x_data_gpu/x_target_gpu and Phi_all/f_all
            # carry an explicit leading trajectory dimension.
            if is_discrete_social and n_trajectories > 1:
                # x_curr: [K, T-1, N, D]
                x_curr = x_data_gpu[:, :-1]
                y_next = x_target_gpu[:, 1:]
                dt_view = dt_all.view(1, -1, 1, 1)
                f_curr = f_all[:, :-1]              # [K, T-1, N, D]
                if tensorized_supported:
                    PhiA_curr = tensorized_coupling_all[:, :-1]
                else:
                    Phi_curr = Phi_all[:, :-1]          # [K, T-1, N, D, E]
                    PhiA_curr = torch.einsum('ktnie,el->ktni', Phi_curr, A)  # [K, T-1, N, D]
                incr_pred = dt_view * (f_curr + PhiA_curr)
                p_next = x_curr + incr_pred
            else:
                # Single trajectory (K=1) as before
                x_curr = x_data_gpu[:-1]          # [T-1, N, D]
                y_next = x_target_gpu[1:]         # [T-1, N, D]
                dt_view = dt_all.view(-1, 1, 1)   # [T-1, 1, 1]
                f_curr = f_all[:-1]               # [T-1, N, D]
                if tensorized_supported:
                    PhiA_curr = tensorized_coupling_all[:-1]
                else:
                    Phi_curr = Phi_all[:-1]           # [T-1, N, D, E]
                    PhiA_curr = torch.einsum('tnie,el->tni', Phi_curr, A)  # [T-1, N, D]
                incr_pred = dt_view * (f_curr + PhiA_curr)
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
                dt_interval = dt_all[idx_i:idx_j]  # [idx_j-idx_i]
                integral_f = torch.einsum('tni,t->ni', f_interval, dt_interval)  # [N, D]
                if tensorized_supported:
                    Phi_A_interval = tensorized_coupling_all[idx_i:idx_j]
                else:
                    Phi_interval = Phi_all[idx_i:idx_j]  # [idx_j-idx_i, N, D, n_hyperedges]
                    Phi_A_interval = torch.einsum('tnie,el->tni', Phi_interval, A)  # [t, N, D]
                integral_phi_A = torch.einsum('tni,t->ni', Phi_A_interval, dt_interval)  # [N, D]

                residual = lhs - integral_f - integral_phi_A
                loss = torch.mean(residual ** 2)
                losses.append(loss)

            total_loss = torch.mean(torch.stack(losses))

        total_loss.backward()
        optimizer.step()

        orderwise_params.clamp_(-2.0, 2.0)

        pbar.set_postfix({
            'loss': f'{total_loss.item():.6f}'
        })
        
        if (epoch + 1) % 500 == 0:
            print(f"\n\n{'='*60}")
            print(f"Epoch {epoch+1}/{n_epochs} - AUC Evaluation")
            print('='*60)
            order_scores = orderwise_params.numpy_by_order()
            auc_scores, _ = compute_auc_scores(order_scores, edge_config, N, max_order, all_possible_edges)
            
            all_possible = all_possible_edges
            
            for order_name, order_label in zip(['edges', 'triangles', 'quads', 'quints', 'sexts', 'septs'],
                                               ['2-edges', '3-edges', '4-edges', '5-edges', '6-edges', '7-edges']):
                if order_label not in auc_scores:
                    continue
                    
                possible_edges = all_possible.get(order_name, [])
                true_edges = edge_config.get(order_name, [])
                n_possible = len(possible_edges)
                n_true = len(true_edges)
                score_order = order_scores.get(order_name, np.empty(0, dtype=np.float32))
                
                auc_score = auc_scores[order_label]
                if auc_score is not None:
                    print(f"  {order_label}: AUC = {auc_score:.4f} | Possible={n_possible}, True={n_true} | score range=[{score_order.min():.3f}, {score_order.max():.3f}]")
                else:
                    print(f"  {order_label}: N/A | Possible={n_possible}, True={n_true}")
            print('='*60 + '\n')
    
    return orderwise_params.numpy_by_order(), edge_config, save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='neuronal',
                        choices=['ecological', 'neuronal', 'rossler', 'social'])
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--n_epochs', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--n_nodes', type=int, default=None)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--results_root', type=str, default='results/low_rank')
    parser.add_argument('--n_trajectories', type=int, default=1)
    parser.add_argument('--max_order', type=int, default=None)
    parser.add_argument('--rank', type=int, default=None)
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
    
    order_scores, edge_config, save_dir = train_integral_model(
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
        max_candidates_per_order=args.max_candidates_per_order,
        results_root=args.results_root,
        scene_label=scene_spec.label,
    )
    
    print("\nComputing AUC scores...")
    auc_scores, roc_data = compute_auc_scores(
        order_scores,
        edge_config,
        N,
        max_order,
        build_training_candidate_pool(
            n_nodes=N,
            max_order=max_order,
            edge_config=edge_config,
            max_candidates_per_order=args.max_candidates_per_order,
            seed=42,
        ),
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
        method="orderwise_sparse_tensors",
        scene=scene_spec.label,
        config={
            "n_nodes": N,
            "max_order": max_order,
            "n_samples": args.n_samples,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "noise": args.noise,
            "n_trajectories": args.n_trajectories,
            "max_candidates_per_order": args.max_candidates_per_order,
        },
        auc_scores=auc_scores,
    )
    
    print(f"\nResults saved to {save_dir}/")
