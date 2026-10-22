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


def set_global_seed(seed: int):
    """Seed every RNG that can influence a run (data gen, init, batching)."""
    import random
    # Scene libraries read this to seed the ODE initial condition / parameter
    # draws, so a seed sweep varies the trajectory, not just the optimizer.
    os.environ["HG_DATA_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_auc(auc_scores: dict):
    vals = [v for v in auc_scores.values() if v is not None]
    return float(np.mean(vals)) if vals else None


def build_order_blocks(orderwise_support: dict):
    """Return [(order_num, start_idx, end_idx), ...] over the flattened A vector."""
    blocks = []
    idx = 0
    for order_num, key in ((2, 'edges'), (3, 'triangles'), (4, 'quads'), (5, 'quints')):
        n = len(orderwise_support.get(key, []))
        if n > 0:
            blocks.append((order_num, idx, idx + n))
            idx += n
    return blocks


def staged_lambda_vector(epoch, n_epochs, blocks, n_hyperedges, device, schedule="none"):
    """Order-wise staged lambda: low orders are enabled first, higher orders
    are faded in over successive stages. schedule='none' -> all-ones."""
    lam = torch.ones(n_hyperedges, device=device)
    if schedule == "none" or len(blocks) <= 1:
        return lam
    n_stages = len(blocks)
    # Stage s owns the epoch window [s/n_stages, (s+1)/n_stages); within its
    # window the corresponding order ramps linearly from 0 to 1.
    prog = (epoch + 1) / max(1, n_epochs)
    for s, (order_num, lo, hi) in enumerate(blocks):
        start = s / n_stages
        end = (s + 1) / n_stages
        if prog <= start:
            w = 0.0
        elif prog >= end:
            w = 1.0
        else:
            w = (prog - start) / (end - start)
        lam[lo:hi] = w
    return lam
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



def build_orderwise_support(n_nodes: int, max_order: int) -> dict:
    order_to_key = {
        2: "edges",
        3: "triangles",
        4: "quads",
        5: "quints",
    }
    support = {k: [] for k in order_to_key.values()}
    max_order = min(max_order, 5)
    for order in range(2, max_order + 1):
        key = order_to_key[order]
        support[key] = [list(edge) for edge in combinations(range(1, n_nodes + 1), order)]
    return support


def build_orderwise_support_sampled(n_nodes: int, max_order: int, max_per_order: int,
                                    edge_config: dict, seed: int = 0) -> dict:
    """Candidate support with the negatives subsampled, per order.

    The full support is C(n, o) summed over orders, which is about 6e6 for
    n = 60 and order 5 -- far too large to materialize, let alone to hold a
    dense Phi for.  Every true hyperedge is kept, and the negatives are
    subsampled uniformly at random.  AUC computed against a uniform random
    sample of the negatives is an unbiased estimate of AUC against all of
    them, so the metric stays comparable to the full-support runs.
    """
    import random as _random
    order_to_key = {2: "edges", 3: "triangles", 4: "quads", 5: "quints"}
    rng = _random.Random(seed)
    support = {k: [] for k in order_to_key.values()}
    max_order = min(max_order, 5)

    for order in range(2, max_order + 1):
        key = order_to_key[order]
        total = comb(n_nodes, order)
        true_edges = {tuple(sorted(int(v) for v in e)) for e in edge_config.get(key, [])}

        if max_per_order <= 0 or total <= max_per_order:
            chosen = [tuple(e) for e in combinations(range(1, n_nodes + 1), order)]
        else:
            # Rejection-sample distinct sorted tuples; cheap because the number
            # drawn is many orders of magnitude below the total.
            seen = set(true_edges)
            chosen = list(true_edges)
            n_neg = max(0, max_per_order - len(true_edges))
            while len(chosen) < len(true_edges) + n_neg:
                cand = tuple(sorted(rng.sample(range(1, n_nodes + 1), order)))
                if cand in seen:
                    continue
                seen.add(cand)
                chosen.append(cand)
            chosen.sort()
        support[key] = [list(e) for e in chosen]
    return support


def infer_n_nodes_from_edge_config(edge_config: dict) -> int:
    max_node = 0
    for key in ['edges', 'triangles', 'quads', 'quints']:
        for edge in edge_config.get(key, []):
            if edge:
                max_node = max(max_node, *(int(v) for v in edge))
    return max_node


def infer_max_order_from_edge_config(edge_config: dict) -> int:
    for order, key in ((5, 'quints'), (4, 'quads'), (3, 'triangles'), (2, 'edges')):
        if edge_config.get(key):
            return order
    return 0


def compute_auc_scores(A_learned, edge_config, N, max_order, orderwise_support=None):
    if orderwise_support is None:
        orderwise_support = build_orderwise_support(N, max_order)
    
    # Flatten A
    A_flat = A_learned.flatten()
    
    # Compute AUC for each order
    auc_scores = {}
    roc_data = {}
    A_idx = 0  # Index in A
    
    order_names = ['edges', 'triangles', 'quads', 'quints']
    order_labels = ['2-edges', '3-edges', '4-edges', '5-edges']
    
    for order_idx, (order_name, order_label) in enumerate(zip(order_names, order_labels)):
        order_num = order_idx + 2
        
        if order_num > max_order:
            break
        
        possible_edges = orderwise_support.get(order_name, [])
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
            
            if A_idx < len(A_flat):
                pred_score = 1 / (1 + np.exp(-A_flat[A_idx]))
                y_pred.append(pred_score)
                A_idx += 1
            else:
                y_pred.append(0.0)
        
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


def save_roc_curve_data(A_learned, edge_config, N, max_order, save_dir, orderwise_support=None):
    if orderwise_support is None:
        orderwise_support = build_orderwise_support(N, max_order)

    roc_dir = os.path.join(save_dir, "roc_data")
    os.makedirs(roc_dir, exist_ok=True)

    A_flat = A_learned.flatten()
    A_idx = 0
    manifest = []

    order_names = ['edges', 'triangles', 'quads', 'quints']
    order_labels = ['2-edges', '3-edges', '4-edges', '5-edges']

    for order_idx, (order_name, order_label) in enumerate(zip(order_names, order_labels)):
        order_num = order_idx + 2
        if order_num > max_order:
            break

        possible_edges = orderwise_support.get(order_name, [])
        true_edges = edge_config.get(order_name, [])
        if len(possible_edges) == 0:
            continue

        y_true = []
        y_score = []
        rows = []

        for edge in possible_edges:
            is_true_edge = any(sorted(edge) == sorted(true_edge) for true_edge in true_edges)
            score = 1 / (1 + np.exp(-A_flat[A_idx])) if A_idx < len(A_flat) else 0.0
            y_true.append(1 if is_true_edge else 0)
            y_score.append(score)
            rows.append({
                'edge': '-'.join(str(v) for v in edge),
                'y_true': int(is_true_edge),
                'y_score': float(score),
            })
            A_idx += 1

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        csv_path = os.path.join(roc_dir, f"{order_label.replace('-', '_')}_scores.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('edge,y_true,y_score\n')
            for row in rows:
                f.write(f"{row['edge']},{row['y_true']},{row['y_score']:.12f}\n")

        roc_curve_path = None
        auc_score = None
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc_score = auc(fpr, tpr)
            roc_curve_path = os.path.join(roc_dir, f"{order_label.replace('-', '_')}_roc_curve.csv")
            with open(roc_curve_path, 'w', encoding='utf-8') as f:
                f.write('fpr,tpr,threshold\n')
                for fpr_val, tpr_val, thr_val in zip(fpr, tpr, thresholds):
                    f.write(f"{float(fpr_val):.12f},{float(tpr_val):.12f},{float(thr_val):.12f}\n")

        manifest.append({
            'order_label': order_label,
            'n_support': len(possible_edges),
            'n_true': int(np.sum(y_true)),
            'auc': None if auc_score is None else float(auc_score),
            'score_file': csv_path,
            'roc_curve_file': roc_curve_path,
        })

    manifest_path = os.path.join(roc_dir, 'manifest.json')
    import json
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"ROC source data saved to {roc_dir}")
    return roc_dir

def plot_roc_curves(roc_data, auc_scores, save_dir, max_order):
    plt.figure(figsize=(8, 6))
    colors = {
        '2-edges': 'blue',
        '3-edges': 'green',
        '4-edges': 'red',
        '5-edges': 'purple',
    }
    
    for order_label in ['2-edges', '3-edges', '4-edges', '5-edges']:
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
    results_root: str = "results/integral",
    scene_label: str = "unknown",
    seed: int = 42,
    loss_mode: str = "integral",
    bounded: bool = True,
    lambda_schedule: str = "none",
    n_intervals: int | None = None,
    window_frac: float = 1.0,
    l1_weight: float = 0.0,
    run_tag: str | None = None,
    oracle_regressor: bool = False,
    no_resample: bool = False,
    max_candidates_per_order: int = 0,
):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_global_seed(seed)
    if n_intervals is None:
        n_intervals = batch_size
    print(
        f"[config] seed={seed} loss_mode={loss_mode} bounded={bounded} "
        f"lambda_schedule={lambda_schedule} n_intervals={n_intervals} "
        f"window_frac={window_frac} l1={l1_weight} "
        f"oracle_regressor={oracle_regressor} no_resample={no_resample}"
    )

    def _make_save_dir():
        tag = run_tag if run_tag else datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(
            results_root,
            scene_label,
            f"sample_{n_samples}_noise_{noise}",
            tag,
        )
        os.makedirs(d, exist_ok=True)
        return d
    edge_config = HypergraphModel.get_hyperedge_config(N, max_order)
    if max_candidates_per_order and max_candidates_per_order > 0:
        orderwise_support = build_orderwise_support_sampled(
            N, max_order, max_candidates_per_order, edge_config, seed=seed)
        print(f"[config] candidate negatives subsampled to <= {max_candidates_per_order} per order")
    else:
        orderwise_support = build_orderwise_support(N, max_order)
    order_keys = ['edges', 'triangles', 'quads', 'quints']
    order_sizes = {
        'edges': 2,
        'triangles': 3,
        'quads': 4,
        'quints': 5,
    }
    n_hyperedges = sum(len(orderwise_support.get(key, [])) for key in order_keys)
    print(f"N={N}, max_order={max_order}")
    print(f"Total hyperedge parameters={n_hyperedges}")
    print(f"  2-edges: {len(orderwise_support.get('edges', []))}")
    print(f"  3-edges: {len(orderwise_support.get('triangles', []))}")
    print(f"  4-edges: {len(orderwise_support.get('quads', []))}")
    print(f"  5-edges: {len(orderwise_support.get('quints', []))}")
    print(f"\nTrue hyperedges:")
    print(f"  2-edges: {len(edge_config.get('edges', []))}")
    print(f"  3-edges: {len(edge_config.get('triangles', []))}")
    print(f"  4-edges: {len(edge_config.get('quads', []))}")
    print(f"  5-edges: {len(edge_config.get('quints', []))}")
    
    print("\nGenerating training data...")
    try:
        model_module = HypergraphModel.__module__
    except Exception:
        model_module = ""
    is_discrete_social = "lib_social_contagion" in model_module

    # --- Data generation: single vs multi-trajectory ---
    x_clean = None
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
        save_dir = _make_save_dir()
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
        gen_kwargs = {"noise": noise}
        if "seed" in inspect.signature(HypergraphModel.generate_training_data).parameters:
            gen_kwargs["seed"] = seed
        t_data, x_data = HypergraphModel.generate_training_data(N, edge_config, n_samples, **gen_kwargs)
        # Oracle study: the same trajectory without observation noise.  Used to
        # build Phi and f while the residual left-hand side still sees the noisy
        # observations, isolating errors-in-variables in the regressor from
        # noise in the derivative/increment being matched.
        x_clean = None
        if oracle_regressor:
            clean_kwargs = dict(gen_kwargs)
            clean_kwargs["noise"] = 0.0
            _, x_clean = HypergraphModel.generate_training_data(N, edge_config, n_samples, **clean_kwargs)
        n_times = len(t_data)
        print(f"Data shape: {x_data.shape}, Time points: {n_times}")
        state_dim = x_data.shape[2]
        if noise > 0:
            print(f"Added Gaussian noise with std={noise}")

        if use_nn and not is_discrete_social and not no_resample:
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
            
            if x_clean is not None:
                x_clean_resampled = np.zeros((n_resampled, N, state_dim))
                for node_idx in range(N):
                    for coord_idx in range(state_dim):
                        fc = interp1d(t_data, x_clean[:, node_idx, coord_idx],
                                      kind='linear', fill_value='extrapolate')
                        x_clean_resampled[:, node_idx, coord_idx] = fc(t_data_resampled)
                x_clean = x_clean_resampled

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
            save_dir = _make_save_dir()
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
        elif use_nn and no_resample:
            print("no_resample=True: staying on the native observation grid "
                  f"({n_times} points), so the finite-difference step is the raw dt.")
        elif use_nn and is_discrete_social:
            print("Skipping linear interpolation for discrete social contagion data; using native observation grid.")

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
        save_dir = _make_save_dir()

    print("Transferring data to GPU...")
    t_data_gpu = torch.tensor(t_data, dtype=torch.float32, device=device)
    # x_reg feeds Phi and f; x_data feeds the residual's left-hand side.
    # They differ only under oracle_regressor.
    x_reg = x_data
    if oracle_regressor:
        if x_clean is None:
            raise RuntimeError("oracle_regressor requires a clean trajectory; "
                               "it is only supported in single-trajectory mode.")
        x_reg = x_clean
        print("ORACLE REGRESSOR: Phi and f are built from the noise-free trajectory.")
    if is_discrete_social and n_trajectories > 1:
        x_data_gpu = torch.tensor(x_data, dtype=torch.float32, device=device)       # [K, T, N, D]
        x_target_gpu = torch.tensor(x_target, dtype=torch.float32, device=device)   # [K, T, N, D]
    else:
        x_data_gpu = torch.tensor(x_data, dtype=torch.float32, device=device)       # [T, N, D]
        x_target_gpu = torch.tensor(x_target, dtype=torch.float32, device=device)   # [T, N, D]
    x_reg_gpu = (x_data_gpu if not oracle_regressor
                 else torch.tensor(x_reg, dtype=torch.float32, device=device))
    print("Converting hyperedge indices to GPU tensors...")
    all_possible_edges_gpu = {}
    for key in order_keys:
        edges = orderwise_support.get(key, [])
        if len(edges) > 0:
            all_possible_edges_gpu[key] = torch.tensor(edges, dtype=torch.long, device=device) - 1
        else:
            all_possible_edges_gpu[key] = torch.empty((0, order_sizes[key]), dtype=torch.long, device=device)
    
    print("Pre-computing coupling tensor Phi and basic dynamics f for all time points...")
    if is_discrete_social and n_trajectories > 1:
        Phi_all = torch.zeros((n_trajectories, n_times, N, state_dim, n_hyperedges), device=device)
        f_all = torch.zeros((n_trajectories, n_times, N, state_dim), device=device)
    else:
        Phi_all = torch.zeros((n_times, N, state_dim, n_hyperedges), device=device)
        f_all = torch.zeros((n_times, N, state_dim), device=device)
    dynamic_params = inspect.signature(HypergraphModel.dynamic_f).parameters
    dynamic_accepts_time = len(dynamic_params) >= 3

    if is_discrete_social and n_trajectories > 1:
        for k in range(n_trajectories):
            for t_idx in tqdm(range(n_times), desc=f"Precomputing Phi (traj {k+1}/{n_trajectories})", leave=False):
                x_t = x_data_gpu[k, t_idx]
                Phi_all[k, t_idx] = HypergraphModel.dynamic_phi(
                    x_t, all_possible_edges_gpu, N, device
                )
                if dynamic_accepts_time:
                    f_all[k, t_idx] = HypergraphModel.dynamic_f(x_t, N, t_data_gpu[t_idx])
                else:
                    f_all[k, t_idx] = HypergraphModel.dynamic_f(x_t, N)
    else:
        for t_idx in tqdm(range(n_times), desc="Precomputing Phi", leave=False):
            x_t = x_reg_gpu[t_idx]
            Phi_all[t_idx] = HypergraphModel.dynamic_phi(
                x_t, all_possible_edges_gpu, N, device
            )
            if dynamic_accepts_time:
                f_all[t_idx] = HypergraphModel.dynamic_f(x_t, N, t_data_gpu[t_idx])
            else:
                f_all[t_idx] = HypergraphModel.dynamic_f(x_t, N)
    dt_all = t_data_gpu[1:] - t_data_gpu[:-1]
    print(f"Phi_all shape: {Phi_all.shape}, f_all shape: {f_all.shape}")
    
    A = torch.randn(n_hyperedges, 1, device=device)
    A_idx = 0
    n_edges = len(orderwise_support.get('edges', []))
    n_triangles = len(orderwise_support.get('triangles', []))
    n_quads = len(orderwise_support.get('quads', []))
    n_quints = len(orderwise_support.get('quints', []))
    
    if n_edges > 0:
        A[A_idx:A_idx+n_edges] -= 2.0
        A_idx += n_edges
    if n_triangles > 0:
        A[A_idx:A_idx+n_triangles] -= 3.0
        A_idx += n_triangles
    remaining = n_quads + n_quints
    if remaining > 0:
        A[A_idx:A_idx+remaining] -= 4.0
    
    A.requires_grad_(True)
    
    optimizer = optim.Adam([A], lr=lr)

    order_blocks = build_order_blocks(orderwise_support)
    # Maximum interval length used by the integral residual, as a fraction of
    # the full observation window. window_frac=1.0 reproduces the original
    # "any random pair of time points" behaviour.
    max_window = max(1, int(round(window_frac * (n_times - 1))))
    if loss_mode == "integral":
        print(f"[config] max integral window = {max_window} steps of {n_times - 1}")

    # use_discrete_residual is true exactly for the social contagion
    # model, where observations are 0/1 and we use a Bernoulli-style
    # likelihood rather than an MSE on x.
    use_discrete_residual = is_discrete_social

    # Training loop (with progress bar)
    pbar = tqdm(range(n_epochs), desc="Training Progress", ncols=120)
    for epoch in pbar:
        optimizer.zero_grad()

        # Order-wise staged lambda schedule (ablation). lam is all-ones when
        # lambda_schedule == "none", which recovers the un-staged objective.
        lam = staged_lambda_vector(
            epoch, n_epochs, order_blocks, n_hyperedges, device, lambda_schedule
        )
        A_eff = A.squeeze(-1) * lam                                      # [E]

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
                Phi_curr = Phi_all[:, :-1]          # [K, T-1, N, D, E]
                PhiA_curr = torch.einsum('ktnie,e->ktni', Phi_curr, A_eff)  # [K, T-1, N, D]
                incr_pred = dt_view * (f_curr + PhiA_curr)
                p_next = x_curr + incr_pred
            else:
                # Single trajectory (K=1) as before
                x_curr = x_data_gpu[:-1]          # [T-1, N, D]
                y_next = x_target_gpu[1:]         # [T-1, N, D]
                dt_view = dt_all.view(-1, 1, 1)   # [T-1, 1, 1]
                f_curr = f_all[:-1]               # [T-1, N, D]
                Phi_curr = Phi_all[:-1]           # [T-1, N, D, E]
                PhiA_curr = torch.einsum('tnie,e->tni', Phi_curr, A_eff)  # [T-1, N, D]
                incr_pred = dt_view * (f_curr + PhiA_curr)
                p_next = x_curr + incr_pred

            p_next = torch.clamp(p_next, 1e-4, 1.0 - 1e-4)
            # Binary cross-entropy: -[y log p + (1-y) log(1-p)]
            bce = -(y_next * torch.log(p_next) + (1.0 - y_next) * torch.log(1.0 - p_next))
            total_loss = bce.mean()
        elif loss_mode == "derivative":
            # Ablation: collocation-style derivative residual
            #   dx/dt - f(x) - Phi(x) A = 0
            # with dx/dt from central finite differences, evaluated on
            # n_intervals randomly sampled interior collocation points.
            # Same A parametrization and same architecture as the integral
            # variant, so the only thing that changes is the residual operator.
            idx = np.random.choice(np.arange(1, n_times - 1), size=min(n_intervals, n_times - 2), replace=False)
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
            dx = x_data_gpu[idx_t + 1] - x_data_gpu[idx_t - 1]            # [B, N, D]
            dt2 = (t_data_gpu[idx_t + 1] - t_data_gpu[idx_t - 1]).view(-1, 1, 1)
            dxdt = dx / dt2                                               # [B, N, D]
            f_c = f_all[idx_t]                                            # [B, N, D]
            Phi_c = Phi_all[idx_t]                                        # [B, N, D, E]
            PhiA_c = torch.einsum('bnie,e->bni', Phi_c, A_eff)
            residual = dxdt - f_c - PhiA_c
            total_loss = torch.mean(residual ** 2)
        else:
            # Integral formulation over random time intervals (ours).
            # Vectorized over n_intervals sampled windows; window length is
            # capped by max_window (ablation on integral window length).
            starts = np.random.randint(0, n_times - 1, size=n_intervals)
            lengths = np.random.randint(1, max_window + 1, size=n_intervals)
            ends = np.minimum(starts + lengths, n_times - 1)
            keep = ends > starts
            starts, ends = starts[keep], ends[keep]

            # Cumulative integrals let every window be evaluated in O(1).
            w = dt_all.view(-1, 1, 1)                                     # [T-1, 1, 1]
            cum_f = torch.cumsum(f_all[:-1] * w, dim=0)                   # [T-1, N, D]
            PhiA_all = torch.einsum('tnie,e->tni', Phi_all[:-1], A_eff)   # [T-1, N, D]
            cum_phiA = torch.cumsum(PhiA_all * w, dim=0)                  # [T-1, N, D]
            zero = torch.zeros(1, N, state_dim, device=device, dtype=cum_f.dtype)
            cum_f = torch.cat([zero, cum_f], dim=0)                       # [T, N, D]
            cum_phiA = torch.cat([zero, cum_phiA], dim=0)                 # [T, N, D]

            si = torch.as_tensor(starts, dtype=torch.long, device=device)
            ei = torch.as_tensor(ends, dtype=torch.long, device=device)
            lhs = x_data_gpu[ei] - x_data_gpu[si]                         # [B, N, D]
            integral_f = cum_f[ei] - cum_f[si]
            integral_phi_A = cum_phiA[ei] - cum_phiA[si]

            residual = lhs - integral_f - integral_phi_A
            total_loss = torch.mean(residual ** 2)

        if l1_weight > 0:
            total_loss = total_loss + l1_weight * A.abs().mean()

        total_loss.backward()
        optimizer.step()

        if bounded:
            # Bounded tensor parametrization: hard box constraint on the
            # interaction tensor entries. Ablating this removes the box.
            with torch.no_grad():
                A.clamp_(-2.0, 2.0)

        pbar.set_postfix({
            'loss': f'{total_loss.item():.6f}'
        })
        
        if (epoch + 1) % 500 == 0:
            print(f"\n\n{'='*60}")
            print(f"Epoch {epoch+1}/{n_epochs} - AUC Evaluation")
            print('='*60)
            A_current = A.detach().cpu().numpy()
            auc_scores, _ = compute_auc_scores(A_current, edge_config, N, max_order, orderwise_support)
            
            orderwise_edges = orderwise_support
            A_flat = A_current.flatten()
            A_idx = 0
            
            for order_name, order_label in zip(['edges', 'triangles', 'quads', 'quints'],
                                               ['2-edges', '3-edges', '4-edges', '5-edges']):
                if order_label not in auc_scores:
                    continue
                    
                possible_edges = orderwise_edges.get(order_name, [])
                true_edges = edge_config.get(order_name, [])
                n_support = len(possible_edges)
                n_true = len(true_edges)
                
                A_order = A_flat[A_idx:A_idx+n_support]
                A_idx += n_support
                
                auc_score = auc_scores[order_label]
                if auc_score is not None:
                    print(f"  {order_label}: AUC = {auc_score:.4f} | Support={n_support}, True={n_true} | A range=[{A_order.min():.3f}, {A_order.max():.3f}]")
                else:
                    print(f"  {order_label}: N/A | Support={n_support}, True={n_true}")
            print('='*60 + '\n')
    
    return A.detach().cpu().numpy(), edge_config, save_dir, orderwise_support


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
    parser.add_argument('--results_root', type=str, default='results/integral')
    parser.add_argument('--n_trajectories', type=int, default=1)
    parser.add_argument('--max_order', type=int, default=None)
    # --- reproducibility / ablation switches (rebuttal) ---
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data generation, init and batch sampling.')
    parser.add_argument('--loss_mode', type=str, default='integral',
                        choices=['integral', 'derivative'],
                        help='integral: windowed integral residual (ours); '
                             'derivative: finite-difference collocation residual.')
    parser.add_argument('--bounded', type=int, default=1,
                        help='1: box-constrained (bounded) tensor parametrization; 0: unconstrained.')
    parser.add_argument('--lambda_schedule', type=str, default='none',
                        choices=['none', 'staged'],
                        help='staged: order-wise staged lambda schedule.')
    parser.add_argument('--n_intervals', type=int, default=32,
                        help='Number of sampled integral windows (or collocation points) per step.')
    parser.add_argument('--window_frac', type=float, default=1.0,
                        help='Max integral window length as a fraction of the observation horizon.')
    parser.add_argument('--l1', type=float, default=0.0,
                        help='L1 sparsity weight on the interaction tensor.')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Output sub-directory name; defaults to a timestamp.')
    parser.add_argument('--oracle_regressor', type=int, default=0,
                        help='1: build Phi and f from the noise-free trajectory while the '
                             'residual LHS stays noisy. Isolates errors-in-variables in the '
                             'regressor from noise in the matched increment/derivative.')
    parser.add_argument('--no_resample', type=int, default=0,
                        help='1: skip the dense linear-interpolation resampling and stay on '
                             'the native observation grid (coarse dt regime).')
    parser.add_argument('--max_candidates_per_order', type=int, default=0,
                        help='Subsample negative candidates to this many per order '
                             '(0 = use the full support). Needed for N=60 scenes where '
                             'the full support is ~6e6 hyperedges.')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    set_global_seed(args.seed)
    HypergraphModel, scene_spec = get_scene_model(args.scene)
    defaults = HypergraphModel.get_default_params()
    if args.scene in {"ecological", "neuronal", "social"}:
        effective_edge_config = HypergraphModel.get_hyperedge_config(defaults["n_nodes"], defaults["max_order"])
        N = infer_n_nodes_from_edge_config(effective_edge_config) or defaults["n_nodes"]
        max_order = infer_max_order_from_edge_config(effective_edge_config) or defaults["max_order"]
    else:
        N = args.n_nodes if args.n_nodes is not None else defaults["n_nodes"]
        max_order = args.max_order if args.max_order is not None else defaults["max_order"]
    
    A_learned, edge_config, save_dir, orderwise_support = train_integral_model(
        N=N, 
        max_order=max_order, 
        n_epochs=args.n_epochs, 
        lr=args.lr, 
        batch_size=args.batch_size,
        gpu_id=args.gpu_id,
        use_nn=True,
        n_samples=args.n_samples,
        noise=args.noise,
        n_trajectories=args.n_trajectories,
        results_root=args.results_root,
        scene_label=scene_spec.label,
        seed=args.seed,
        loss_mode=args.loss_mode,
        bounded=bool(args.bounded),
        lambda_schedule=args.lambda_schedule,
        n_intervals=args.n_intervals,
        window_frac=args.window_frac,
        l1_weight=args.l1,
        run_tag=args.run_tag,
        oracle_regressor=bool(args.oracle_regressor),
        no_resample=bool(args.no_resample),
        max_candidates_per_order=args.max_candidates_per_order,
    )
    
    print("\nComputing AUC scores...")
    auc_scores, roc_data = compute_auc_scores(
        A_learned, edge_config, N, max_order, orderwise_support,
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
    save_roc_curve_data(
        A_learned, edge_config, N, max_order, save_dir, orderwise_support,
    )

    write_standard_summary(
        save_dir=save_dir,
        method=f"integral_pinn_linear[{args.loss_mode}]",
        scene=scene_spec.label,
        config={
            "n_nodes": N,
            "max_order": max_order,
            "n_samples": args.n_samples,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "noise": args.noise,
            "n_trajectories": args.n_trajectories,
            "seed": args.seed,
            "loss_mode": args.loss_mode,
            "bounded": bool(args.bounded),
            "lambda_schedule": args.lambda_schedule,
            "n_intervals": args.n_intervals,
            "window_frac": args.window_frac,
            "l1": args.l1,
            "oracle_regressor": bool(args.oracle_regressor),
            "no_resample": bool(args.no_resample),
            "max_candidates_per_order": args.max_candidates_per_order,
        },
        auc_scores=auc_scores,
        extra_metrics={"macro_auc": macro_auc(auc_scores)},
    )
    
    print(f"\nResults saved to {save_dir}/")
