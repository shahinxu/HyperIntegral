"""
Hypergraph Topology Learning Based on Integral Formulation
Does not use PINN, but directly uses integral equations:
x_i(t_{k+1}) - x_i(t_k) = ∫_{t_k}^{t_{k+1}} f(x_i(t)) dt + ∫_{t_k}^{t_{k+1}} Φ_i(t) A_i dt

Left side: Direct difference of raw data
Right side: 
  - f(x_i(t)): Integral of Rossler dynamics
  - Φ_i(t) A_i: Integral of hyperedge coupling term
  
Uses rectangular integration (simplest integration method)
Learnable parameters: A = [N_possible_hyperedge, 1]

DIFFERENCE FROM Rossler_Integral.py:
  - Trains TimeResNet to fit data first
  - Resamples densely from nn (10x more points)
  - Then proceeds with same precomputation approach as original
"""

import torch
from torch import optim, nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_curve, auc
import os
from datetime import datetime
from tqdm import tqdm
import argparse

# ============================================================================
# NEURAL NETWORK COMPONENTS (Same architecture as HyperPINNTopology.py)
# ============================================================================

class ResidualBlock(nn.Module):
    """ResidualBlock with LayerNorm and GELU (from HyperPINNTopology.py)"""
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
    """ResNet for time series prediction (EXACTLY same as HyperPINNTopology.py)"""
    def __init__(self, output_dim, hidden_dim=64, num_layers=8, dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(1, hidden_dim)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) 
            for _ in range(num_layers - 2)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Data normalization parameters (will be set during training)
        self.register_buffer('t_mean', torch.tensor(0.0))
        self.register_buffer('t_std', torch.tensor(1.0))
        self.register_buffer('x_mean', torch.zeros(output_dim))
        self.register_buffer('x_std', torch.ones(output_dim))

    def forward(self, t, normalize=False):
        # t: [B, 1]
        if normalize:
            t = (t - self.t_mean) / (self.t_std + 1e-8)
        # IMPORTANT: HyperPINNTopology uses tanh here, not gelu!
        h = torch.tanh(self.input_layer(t))
        for block in self.res_blocks:
            h = block(h)
        out = self.output_layer(h)
        if normalize:
            out = out * self.x_std + self.x_mean
        return out


# ============================================================================
# IDENTICAL TO Rossler_Integral.py FROM HERE
# ============================================================================

def roessler_dynamics(x, N):
    ar, br, cr = 0.2, 0.2, 0.7
    
    xold = x[:, 0]  # [N]
    yold = x[:, 1]  # [N]
    zold = x[:, 2]  # [N]
    
    dxdt = -yold - zold  # Only basic dynamics, no coupling
    dydt = xold + ar * yold
    dzdt = br + zold * (xold - cr)
    
    return torch.stack([dxdt, dydt, dzdt], dim=1)  # [N, 3]


def compute_hyperedge_coupling_tensor(x, all_possible_edges, N, device):
    xold = x[:, 0]  # [N]
    yold = x[:, 1]  # [N]
    zold = x[:, 2]  # [N]
    
    k, kD = 0.4, 0.3
    
    # Count all hyperedges
    n_edges = all_possible_edges['edges'].shape[0]
    n_triangles = all_possible_edges['triangles'].shape[0]
    n_quads = all_possible_edges['quads'].shape[0]
    n_quints = all_possible_edges['quints'].shape[0]
    n_sexts = all_possible_edges['sexts'].shape[0]
    n_septs = all_possible_edges['septs'].shape[0]
    n_total = n_edges + n_triangles + n_quads + n_quints + n_sexts + n_septs
    
    # Pre-allocate entire tensor
    Phi = torch.zeros((N, 3, n_total), device=device)
    edge_idx = 0
    
    # 2-edges - FULLY Vectorized (no loops!)
    if n_edges > 0:
        edges_tensor = all_possible_edges['edges']  # [n_edges, 2]
        i_indices = edges_tensor[:, 0]  # [n_edges]
        j_indices = edges_tensor[:, 1]  # [n_edges]
        
        # Vectorized computation
        diff = k * (xold[j_indices] - xold[i_indices])  # [n_edges]
        Phi[i_indices, 0, edge_idx + torch.arange(n_edges, device=device)] = diff
        Phi[j_indices, 0, edge_idx + torch.arange(n_edges, device=device)] = -diff
        edge_idx += n_edges
    
    # 3-edges - FULLY Vectorized
    if n_triangles > 0:
        triangles_tensor = all_possible_edges['triangles']  # [n_triangles, 3]
        i_idx = triangles_tensor[:, 0]
        j_idx = triangles_tensor[:, 1]
        k_idx = triangles_tensor[:, 2]
        
        xi, xj, xk = xold[i_idx], xold[j_idx], xold[k_idx]
        edge_range = edge_idx + torch.arange(n_triangles, device=device)
        
        Phi[i_idx, 0, edge_range] = kD * (xj**2 * xk - xi**3 + xj * xk**2 - xi**3)
        Phi[j_idx, 0, edge_range] = kD * (xi**2 * xk - xj**3 + xi * xk**2 - xj**3)
        Phi[k_idx, 0, edge_range] = kD * (xi**2 * xj - xk**3 + xi * xj**2 - xk**3)
        edge_idx += n_triangles
    
    # 4-edges - FULLY Vectorized
    if n_quads > 0:
        quads_tensor = all_possible_edges['quads']  # [n_quads, 4]
        i_idx = quads_tensor[:, 0]
        j_idx = quads_tensor[:, 1]
        k_idx = quads_tensor[:, 2]
        l_idx = quads_tensor[:, 3]
        
        xi, xj, xk, xl = xold[i_idx], xold[j_idx], xold[k_idx], xold[l_idx]
        edge_range = edge_idx + torch.arange(n_quads, device=device)
        
        Phi[i_idx, 0, edge_range] = kD * (xj**2 * xk * xl - xi**3)
        Phi[j_idx, 0, edge_range] = kD * (xi**2 * xk * xl - xj**3)
        Phi[k_idx, 0, edge_range] = kD * (xi**2 * xj * xl - xk**3)
        Phi[l_idx, 0, edge_range] = kD * (xi**2 * xj * xk - xl**3)
        edge_idx += n_quads
    
    # 5-edges - FULLY Vectorized
    if n_quints > 0:
        quints_tensor = all_possible_edges['quints']  # [n_quints, 5]
        i_idx = quints_tensor[:, 0]
        j_idx = quints_tensor[:, 1]
        k_idx = quints_tensor[:, 2]
        l_idx = quints_tensor[:, 3]
        m_idx = quints_tensor[:, 4]
        
        yi, yj, yk, yl, ym = yold[i_idx], yold[j_idx], yold[k_idx], yold[l_idx], yold[m_idx]
        edge_range = edge_idx + torch.arange(n_quints, device=device)
        
        Phi[i_idx, 1, edge_range] = kD * (yj**2 * yk * yl * ym - yi**3)
        Phi[j_idx, 1, edge_range] = kD * (yi**2 * yk * yl * ym - yj**3)
        Phi[k_idx, 1, edge_range] = kD * (yi**2 * yj * yl * ym - yk**3)
        Phi[l_idx, 1, edge_range] = kD * (yi**2 * yj * yk * ym - yl**3)
        Phi[m_idx, 1, edge_range] = kD * (yi**2 * yj * yk * yl - ym**3)
        edge_idx += n_quints
    
    # 6-edges - FULLY Vectorized
    if n_sexts > 0:
        sexts_tensor = all_possible_edges['sexts']  # [n_sexts, 6]
        i_idx = sexts_tensor[:, 0]
        j_idx = sexts_tensor[:, 1]
        k_idx = sexts_tensor[:, 2]
        l_idx = sexts_tensor[:, 3]
        m_idx = sexts_tensor[:, 4]
        n_idx = sexts_tensor[:, 5]
        
        yi, yj, yk, yl, ym, yn = yold[i_idx], yold[j_idx], yold[k_idx], yold[l_idx], yold[m_idx], yold[n_idx]
        edge_range = edge_idx + torch.arange(n_sexts, device=device)
        
        Phi[i_idx, 1, edge_range] = kD * (yj**2 * yk * yl * ym * yn - yi**3)
        Phi[j_idx, 1, edge_range] = kD * (yi**2 * yk * yl * ym * yn - yj**3)
        Phi[k_idx, 1, edge_range] = kD * (yi**2 * yj * yl * ym * yn - yk**3)
        Phi[l_idx, 1, edge_range] = kD * (yi**2 * yj * yk * ym * yn - yl**3)
        Phi[m_idx, 1, edge_range] = kD * (yi**2 * yj * yk * yl * yn - ym**3)
        Phi[n_idx, 1, edge_range] = kD * (yi**2 * yj * yk * yl * ym - yn**3)
        edge_idx += n_sexts
    
    # 7-edges - FULLY Vectorized
    if n_septs > 0:
        septs_tensor = all_possible_edges['septs']  # [n_septs, 7]
        i_idx = septs_tensor[:, 0]
        j_idx = septs_tensor[:, 1]
        k_idx = septs_tensor[:, 2]
        l_idx = septs_tensor[:, 3]
        m_idx = septs_tensor[:, 4]
        n_idx = septs_tensor[:, 5]
        o_idx = septs_tensor[:, 6]
        
        zi, zj, zk, zl, zm, zn, zo = zold[i_idx], zold[j_idx], zold[k_idx], zold[l_idx], zold[m_idx], zold[n_idx], zold[o_idx]
        edge_range = edge_idx + torch.arange(n_septs, device=device)
        
        Phi[i_idx, 2, edge_range] = kD * (zj**2 * zk * zl * zm * zn * zo - zi**3)
        Phi[j_idx, 2, edge_range] = kD * (zi**2 * zk * zl * zm * zn * zo - zj**3)
        Phi[k_idx, 2, edge_range] = kD * (zi**2 * zj * zl * zm * zn * zo - zk**3)
        Phi[l_idx, 2, edge_range] = kD * (zi**2 * zj * zk * zm * zn * zo - zl**3)
        Phi[m_idx, 2, edge_range] = kD * (zi**2 * zj * zk * zl * zn * zo - zm**3)
        Phi[n_idx, 2, edge_range] = kD * (zi**2 * zj * zk * zl * zm * zo - zn**3)
        Phi[o_idx, 2, edge_range] = kD * (zi**2 * zj * zk * zl * zm * zn - zo**3)
    
    return Phi  # [N, 3, N_total_hyperedges]


def get_hyperedge_config(N, max_order=7):
    """Get hyperedge configuration for N nodes"""
    configs = {
        8: {
            'edges': [[1, 2],[2, 3],[3, 4],[5, 6],[6, 7],[7, 8]],
            'triangles': [[1, 2, 3],[2, 4, 5],[5, 6, 7],[6, 7, 8]],
            'quads': [[1, 2, 3, 4]],
            'quints': [[4, 5, 6, 7, 8]],
            'sexts': [[1, 2, 3, 4, 5, 6]],
            'septs': [[3, 2, 4, 5, 6, 7, 8]]
        },
        10: {
            'edges': [[1, 2], [2, 3], [4, 5], [6, 7], [8, 9]],
            'triangles': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'quads': [[1, 2, 3, 4], [7, 8, 9, 10]],
            'quints': [[2, 3, 4, 5, 6]],
            'sexts': [[3, 4, 5, 6, 7, 8]],
            'septs': [[1, 2, 4, 6, 7, 9, 10]]
        },
        12: {
            'edges': [[1, 2], [2, 3], [4, 5], [6, 7], [9, 10], [11, 12]],
            'triangles': [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
            'quads': [[1, 2, 3, 4], [9, 10, 11, 12]],
            'quints': [[3, 4, 5, 6, 7]],
            'sexts': [[5, 6, 7, 8, 9, 10]],
            'septs': [[1, 3, 5, 7, 9, 11, 12]]
        },
        14: {
            'edges': [[1, 2], [2, 3], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13]],
            'triangles': [[1, 2, 3], [5, 6, 7], [10, 11, 12]],
            'quads': [[1, 2, 3, 4], [11, 12, 13, 14]],
            'quints': [[4, 5, 6, 7, 8]],
            'sexts': [[7, 8, 9, 10, 11, 12]],
            'septs': [[1, 3, 5, 8, 10, 12, 14]]
        },
        16: {
            'edges': [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [9, 10], [11, 12], [13, 14]],
            'triangles': [[1, 2, 3], [5, 6, 7], [10, 11, 12]],
            'quads': [[1, 2, 3, 4], [13, 14, 15, 16]],
            'quints': [[5, 6, 7, 8, 9]],
            'sexts': [[9, 10, 11, 12, 13, 14]],
            'septs': [[1, 4, 7, 10, 13, 15, 16]]
        }
    }
    
    if N not in configs:
        raise ValueError(f"N must be one of {list(configs.keys())}, got {N}")
    
    config = configs[N]
    
    # Filter by max_order
    filtered_config = {}
    if max_order >= 2:
        filtered_config['edges'] = config['edges']
    else:
        filtered_config['edges'] = []
        
    if max_order >= 3:
        filtered_config['triangles'] = config['triangles']
    else:
        filtered_config['triangles'] = []
        
    if max_order >= 4:
        filtered_config['quads'] = config['quads']
    else:
        filtered_config['quads'] = []
        
    if max_order >= 5:
        filtered_config['quints'] = config['quints']
    else:
        filtered_config['quints'] = []
        
    if max_order >= 6:
        filtered_config['sexts'] = config['sexts']
    else:
        filtered_config['sexts'] = []
        
    if max_order >= 7:
        filtered_config['septs'] = config['septs']
    else:
        filtered_config['septs'] = []
    
    return filtered_config


def generate_all_possible_hyperedges(N, max_order):
    all_possible = {}
    
    if max_order >= 2:
        all_possible['edges'] = [list(edge) for edge in combinations(range(1, N+1), 2)]
    else:
        all_possible['edges'] = []
    
    if max_order >= 3:
        all_possible['triangles'] = [list(edge) for edge in combinations(range(1, N+1), 3)]
    else:
        all_possible['triangles'] = []
    
    if max_order >= 4:
        all_possible['quads'] = [list(edge) for edge in combinations(range(1, N+1), 4)]
    else:
        all_possible['quads'] = []
    
    if max_order >= 5:
        all_possible['quints'] = [list(edge) for edge in combinations(range(1, N+1), 5)]
    else:
        all_possible['quints'] = []
    
    if max_order >= 6:
        all_possible['sexts'] = [list(edge) for edge in combinations(range(1, N+1), 6)]
    else:
        all_possible['sexts'] = []
    
    if max_order >= 7:
        all_possible['septs'] = [list(edge) for edge in combinations(range(1, N+1), 7)]
    else:
        all_possible['septs'] = []
    
    return all_possible


def compute_auc_scores(A_learned, edge_config, N, max_order):
    """
    Compute AUC scores for each order
    
    Args:
        A_learned: [n_hyperedges, 1] Learned weights
        edge_config: True hyperedge configuration
        N: Number of nodes
        max_order: Maximum hyperedge order
    
    Returns:
        auc_scores: Dictionary of AUC values for each order
        roc_data: Dictionary of ROC curve data (fpr, tpr) for each order
    """
    # Generate all possible hyperedges
    all_possible = generate_all_possible_hyperedges(N, max_order)
    
    # Flatten A
    A_flat = A_learned.flatten()
    
    # Compute AUC for each order
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
        possible_edges = all_possible[order_name]
        true_edges = edge_config[order_name]
        
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
                # If A length is not enough, means these hyperedges are not in training config
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


def plot_roc_curves(roc_data, auc_scores, save_dir):
    """
    Plot ROC curves
    
    Args:
        roc_data: Dictionary of ROC curve data (fpr, tpr) for each order
        auc_scores: Dictionary of AUC values for each order
        save_dir: Save directory
    """
    plt.figure(figsize=(8, 6))
    
    # Define colors
    colors = {
        '2-edges': 'blue',
        '3-edges': 'green',
        '4-edges': 'red',
        '5-edges': 'purple',
        '6-edges': 'orange',
        '7-edges': 'brown'
    }
    
    # Plot ROC curve for each order
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
    
    plt.savefig(os.path.join(save_dir, 'roc_curves_7_order.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"ROC curves saved to {os.path.join(save_dir, 'roc_curves_7_order.png')}")


def generate_training_data(N, edge_config, n_samples=11):
    """Generate true Rossler system data for training
    
    Args:
        N: Number of nodes
        edge_config: Hyperedge configuration
        n_samples: Number of time samples (default=11)
    """
    # Convert configuration to numpy arrays
    EdgeList = np.array(edge_config['edges']) if edge_config['edges'] else np.empty((0, 2), dtype=int)
    TriangleList = np.array(edge_config['triangles']) if edge_config['triangles'] else np.empty((0, 3), dtype=int)
    QuadList = np.array(edge_config['quads']) if edge_config['quads'] else np.empty((0, 4), dtype=int)
    QuintList = np.array(edge_config['quints']) if edge_config['quints'] else np.empty((0, 5), dtype=int)
    SextList = np.array(edge_config['sexts']) if edge_config['sexts'] else np.empty((0, 6), dtype=int)
    SeptList = np.array(edge_config['septs']) if edge_config['septs'] else np.empty((0, 7), dtype=int)
    
    def roessler_hoi(t, x):
        """Complete Rossler HOI system"""
        m1 = len(x)
        N_local = m1 // 3
        xold = x[0:N_local]
        yold = x[N_local:2*N_local]
        zold = x[2*N_local:3*N_local]
        ar, br, cr = 0.2, 0.2, 0.7
        k, kD = 0.4, 0.3

        coup_rete = np.zeros(N_local)
        coup_simplicial = np.zeros(N_local)
        coup_quads = np.zeros(N_local)
        coup_quints = np.zeros(N_local)
        coup_sexts = np.zeros(N_local)
        coup_septs = np.zeros(N_local)
        
        for ii in range(len(EdgeList)):
            i1 = EdgeList[ii, 0] - 1
            i2 = EdgeList[ii, 1] - 1
            coup_rete[i1] += xold[i2] - xold[i1]
            coup_rete[i2] += xold[i1] - xold[i2]
        
        for ii in range(len(TriangleList)):
            i1 = TriangleList[ii, 0] - 1
            i2 = TriangleList[ii, 1] - 1
            i3 = TriangleList[ii, 2] - 1
            coup_simplicial[i1] += xold[i2]**2 * xold[i3] - xold[i1]**3 + xold[i2] * xold[i3]**2 - xold[i1]**3
            coup_simplicial[i2] += xold[i1]**2 * xold[i3] - xold[i2]**3 + xold[i1] * xold[i3]**2 - xold[i2]**3
            coup_simplicial[i3] += xold[i1]**2 * xold[i2] - xold[i3]**3 + xold[i1] * xold[i2]**2 - xold[i3]**3
        
        for ii in range(len(QuadList)):
            i1 = QuadList[ii, 0] - 1
            i2 = QuadList[ii, 1] - 1
            i3 = QuadList[ii, 2] - 1
            i4 = QuadList[ii, 3] - 1
            coup_quads[i1] += xold[i2]**2 * xold[i3] * xold[i4] - xold[i1]**3
            coup_quads[i2] += xold[i1]**2 * xold[i3] * xold[i4] - xold[i2]**3
            coup_quads[i3] += xold[i1]**2 * xold[i2] * xold[i4] - xold[i3]**3
            coup_quads[i4] += xold[i1]**2 * xold[i2] * xold[i3] - xold[i4]**3
        
        for ii in range(len(QuintList)):
            i1 = QuintList[ii, 0] - 1
            i2 = QuintList[ii, 1] - 1
            i3 = QuintList[ii, 2] - 1
            i4 = QuintList[ii, 3] - 1
            i5 = QuintList[ii, 4] - 1
            coup_quints[i1] += yold[i2]**2 * yold[i3] * yold[i4] * yold[i5] - yold[i1]**3
            coup_quints[i2] += yold[i1]**2 * yold[i3] * yold[i4] * yold[i5] - yold[i2]**3
            coup_quints[i3] += yold[i1]**2 * yold[i2] * yold[i4] * yold[i5] - yold[i3]**3
            coup_quints[i4] += yold[i1]**2 * yold[i2] * yold[i3] * yold[i5] - yold[i4]**3
            coup_quints[i5] += yold[i1]**2 * yold[i2] * yold[i3] * yold[i4] - yold[i5]**3
        
        for ii in range(len(SextList)):
            i1 = SextList[ii, 0] - 1
            i2 = SextList[ii, 1] - 1
            i3 = SextList[ii, 2] - 1
            i4 = SextList[ii, 3] - 1
            i5 = SextList[ii, 4] - 1
            i6 = SextList[ii, 5] - 1
            coup_sexts[i1] += yold[i2]**2 * yold[i3] * yold[i4] * yold[i5] * yold[i6] - yold[i1]**3
            coup_sexts[i2] += yold[i1]**2 * yold[i3] * yold[i4] * yold[i5] * yold[i6] - yold[i2]**3
            coup_sexts[i3] += yold[i1]**2 * yold[i2] * yold[i4] * yold[i5] * yold[i6] - yold[i3]**3
            coup_sexts[i4] += yold[i1]**2 * yold[i2] * yold[i3] * yold[i5] * yold[i6] - yold[i4]**3
            coup_sexts[i5] += yold[i1]**2 * yold[i2] * yold[i3] * yold[i4] * yold[i6] - yold[i5]**3
            coup_sexts[i6] += yold[i1]**2 * yold[i2] * yold[i3] * yold[i4] * yold[i5] - yold[i6]**3
        
        for ii in range(len(SeptList)):
            i1 = SeptList[ii, 0] - 1
            i2 = SeptList[ii, 1] - 1
            i3 = SeptList[ii, 2] - 1
            i4 = SeptList[ii, 3] - 1
            i5 = SeptList[ii, 4] - 1
            i6 = SeptList[ii, 5] - 1
            i7 = SeptList[ii, 6] - 1
            coup_septs[i1] += zold[i2]**2 * zold[i3] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i1]**3
            coup_septs[i2] += zold[i1]**2 * zold[i3] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i2]**3
            coup_septs[i3] += zold[i1]**2 * zold[i2] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i3]**3
            coup_septs[i4] += zold[i1]**2 * zold[i2] * zold[i3] * zold[i5] * zold[i6] * zold[i7] - zold[i4]**3
            coup_septs[i5] += zold[i1]**2 * zold[i2] * zold[i3] * zold[i4] * zold[i6] * zold[i7] - zold[i5]**3
            coup_septs[i6] += zold[i1]**2 * zold[i2] * zold[i3] * zold[i4] * zold[i5] * zold[i7] - zold[i6]**3
            coup_septs[i7] += zold[i1]**2 * zold[i2] * zold[i3] * zold[i4] * zold[i5] * zold[i6] - zold[i7]**3
        
        dxdt1 = -yold - zold + k * coup_rete + kD * coup_simplicial + kD * coup_quads
        dydt1 = xold + ar * yold + kD * coup_quints + kD * coup_sexts
        dzdt1 = br + zold * (xold - cr) + kD * coup_septs
        dxdt = np.concatenate((dxdt1, dydt1, dzdt1))
        return dxdt
    
    # Initialize
    np.random.seed(42)
    x0 = np.random.randn(3 * N) * 0.1
    
    # Solve ODE
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, n_samples)
    sol = solve_ivp(roessler_hoi, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    
    x_data_flat = sol.y.T  # [T, 3N]
    T = x_data_flat.shape[0]
    x_data = np.zeros((T, N, 3))
    x_data[:, :, 0] = x_data_flat[:, 0:N]        # x coordinates
    x_data[:, :, 1] = x_data_flat[:, N:2*N]      # y coordinates
    x_data[:, :, 2] = x_data_flat[:, 2*N:3*N]    # z coordinates
    
    return sol.t, x_data  # [T], [T, N, 3]


def train_integral_model(
    N=8, 
    max_order=7, 
    n_epochs=10000, 
    lr=0.001, 
    batch_size=32, 
    gpu_id=6,
    use_nn=True,
    nn_hidden_dim=128,
    nn_layers=4,
    stage1_epochs=2500,
    resample_factor=10,
    n_samples=11
):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    edge_config = get_hyperedge_config(N, max_order)
    all_possible_edges = generate_all_possible_hyperedges(N, max_order)
    n_hyperedges = (len(all_possible_edges['edges']) + 
                    len(all_possible_edges['triangles']) + 
                    len(all_possible_edges['quads']) + 
                    len(all_possible_edges['quints']) + 
                    len(all_possible_edges['sexts']) + 
                    len(all_possible_edges['septs']))
    print(f"N={N}, max_order={max_order}")
    print(f"Total possible hyperedges={n_hyperedges}")
    print(f"  2-edges: {len(all_possible_edges['edges'])}")
    print(f"  3-edges: {len(all_possible_edges['triangles'])}")
    print(f"  4-edges: {len(all_possible_edges['quads'])}")
    print(f"  5-edges: {len(all_possible_edges['quints'])}")
    print(f"  6-edges: {len(all_possible_edges['sexts'])}")
    print(f"  7-edges: {len(all_possible_edges['septs'])}")
    print(f"\nTrue hyperedges:")
    print(f"  2-edges: {len(edge_config['edges'])}")
    print(f"  3-edges: {len(edge_config['triangles'])}")
    print(f"  4-edges: {len(edge_config['quads'])}")
    print(f"  5-edges: {len(edge_config['quints'])}")
    print(f"  6-edges: {len(edge_config['sexts'])}")
    print(f"  7-edges: {len(edge_config['septs'])}")
    
    print("\nGenerating training data...")
    t_data, x_data = generate_training_data(N, edge_config, n_samples)
    n_times = len(t_data)
    print(f"Data shape: {x_data.shape}, Time points: {n_times}")

    
    # ============================================================================
    # NEURAL NETWORK TRAINING + RESAMPLING (ONLY DIFFERENCE FROM Rossler_Integral.py)
    # ============================================================================
    if use_nn:
        print("\n" + "="*80)
        print("LINEAR INTERPOLATION MODE: Using piecewise linear curves")
        print("="*80)
        
        # Flatten x_data to [T, 3N]
        x_data_flat = x_data.reshape(n_times, 3 * N)
        
        print(f"\nUsing LINEAR INTERPOLATION (no neural network)")
        print(f"  Original {n_times} points will be connected by straight lines")
        print(f"  This GUARANTEES passing through all observation points!")
        
        # Dense resampling using LINEAR interpolation
        print(f"\nDense resampling using linear interpolation...")
        n_resampled = (n_times - 1) * resample_factor + 1
        t_data_resampled = np.linspace(t_data[0], t_data[-1], n_resampled)
        
        print(f"Original data points: {n_times}")
        print(f"Resampled data points: {n_resampled} ({resample_factor}x denser)")
        
        # Use scipy's interp1d for linear interpolation
        from scipy.interpolate import interp1d
        x_data_resampled = np.zeros((n_resampled, N, 3))
        
        for node_idx in range(N):
            for coord_idx in range(3):
                # Linear interpolation for each node-coordinate pair
                interp_func = interp1d(t_data, x_data[:, node_idx, coord_idx], 
                                      kind='linear', fill_value='extrapolate')
                x_data_resampled[:, node_idx, coord_idx] = interp_func(t_data_resampled)
        
        print(f"Linear interpolation completed!")
        
        # For visualization consistency
        x_data_resampled_cpu = x_data_resampled
        
        fig, axes = plt.subplots(N, 3, figsize=(15, 2.5*N))
        coord_names = ['x', 'y', 'z']
        
        for node_idx in range(N):
            for coord_idx in range(3):
                ax = axes[node_idx, coord_idx]
                
                # Plot original data
                ax.plot(t_data, x_data[:, node_idx, coord_idx], 'o', 
                       label='Original (ODE)', markersize=4, alpha=0.7, color='blue')
                
                # Plot resampled data
                ax.plot(t_data_resampled, x_data_resampled_cpu[:, node_idx, coord_idx], 
                       '-', label='Resampled (ResNet)', linewidth=1, alpha=0.8, color='red')
                
                # Labels and formatting
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
        save_dir = f"results_integral/{n_samples}/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'resnet_vs_original.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {os.path.join(save_dir, 'resnet_vs_original.png')}")
        
        from scipy.interpolate import interp1d
        errors_per_node = []
        for node_idx in range(N):
            for coord_idx in range(3):
                interp_func = interp1d(t_data, x_data[:, node_idx, coord_idx], kind='cubic')
                x_orig_interp = interp_func(t_data_resampled)
                
                error = np.abs(x_data_resampled_cpu[:, node_idx, coord_idx] - x_orig_interp)
                errors_per_node.append(error)
        
        errors_all = np.concatenate(errors_per_node)
        print(f"\nResNet fitting error statistics:")
        print(f"  Mean absolute error: {errors_all.mean():.6f}")
        print(f"  Max absolute error: {errors_all.max():.6f}")
        print(f"  Std of error: {errors_all.std():.6f}")
        
        t_data = t_data_resampled
        x_data = x_data_resampled_cpu
        n_times = n_resampled
        print(f"\nReplaced data with resampled version: {x_data.shape}")
        print("="*80 + "\n")
    
    # ============================================================================
    # IDENTICAL TO Rossler_Integral.py FROM HERE
    # ============================================================================
    
    print("Transferring data to GPU...")
    t_data_gpu = torch.tensor(t_data, dtype=torch.float32, device=device)
    x_data_gpu = torch.tensor(x_data, dtype=torch.float32, device=device)  # [T, N, 3]
    
    print("Converting hyperedge indices to GPU tensors...")
    # Pre-convert all hyperedge indices to GPU tensors (HUGE speedup!)
    all_possible_edges_gpu = {}
    for key in ['edges', 'triangles', 'quads', 'quints', 'sexts', 'septs']:
        if len(all_possible_edges[key]) > 0:
            all_possible_edges_gpu[key] = torch.tensor(all_possible_edges[key], dtype=torch.long, device=device) - 1
        else:
            all_possible_edges_gpu[key] = torch.empty((0, len(all_possible_edges[key][0]) if all_possible_edges[key] else 2), dtype=torch.long, device=device)
    
    print("Pre-computing coupling tensor Phi and basic dynamics f for all time points...")
    Phi_all = torch.zeros((n_times, N, 3, n_hyperedges), device=device)  # [T, N, 3, n_hyperedges]
    f_all = torch.zeros((n_times, N, 3), device=device)  # [T, N, 3]
    
    for t_idx in tqdm(range(n_times), desc="Precomputing Phi", leave=False):
        x_t = x_data_gpu[t_idx]  # [N, 3]
        Phi_all[t_idx] = compute_hyperedge_coupling_tensor(x_t, all_possible_edges_gpu, N, device)  # [N, 3, n_hyperedges]
        f_all[t_idx] = roessler_dynamics(x_t, N)  # [N, 3]
    
    dt_all = t_data_gpu[1:] - t_data_gpu[:-1]  # [T-1]
    print(f"Phi_all shape: {Phi_all.shape}, f_all shape: {f_all.shape}")
    
    A = torch.randn(n_hyperedges, 1, device=device)
    A_idx = 0
    n_edges = len(all_possible_edges['edges'])
    n_triangles = len(all_possible_edges['triangles'])
    n_quads = len(all_possible_edges['quads'])
    n_quints = len(all_possible_edges['quints'])
    n_sexts = len(all_possible_edges['sexts'])
    n_septs = len(all_possible_edges['septs'])
    
    if n_edges > 0:
        A[A_idx:A_idx+n_edges] -= 2.0
        A_idx += n_edges
    # 3-edges: bias = -3
    if n_triangles > 0:
        A[A_idx:A_idx+n_triangles] -= 3.0
        A_idx += n_triangles
    # 4/5/6/7-edges: bias = -4
    remaining = n_quads + n_quints + n_sexts + n_septs
    if remaining > 0:
        A[A_idx:A_idx+remaining] -= 4.0
    
    A.requires_grad_(True)
    
    optimizer = optim.Adam([A], lr=lr)
    
    # Training loop (with progress bar)
    pbar = tqdm(range(n_epochs), desc="Training Progress", ncols=120)
    for epoch in pbar:
        optimizer.zero_grad()
        losses = []
        
        for _ in range(batch_size):
            idx_i, idx_j = np.random.choice(n_times, size=2, replace=False)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            
            x_i, x_j = x_data_gpu[idx_i], x_data_gpu[idx_j]  # [N, 3]
            
            lhs = x_j - x_i  # [N, 3]
            f_interval = f_all[idx_i:idx_j]  # [idx_j-idx_i, N, 3]
            Phi_interval = Phi_all[idx_i:idx_j]  # [idx_j-idx_i, N, 3, n_hyperedges]
            dt_interval = dt_all[idx_i:idx_j]  # [idx_j-idx_i]
            integral_f = torch.einsum('tni,t->ni', f_interval, dt_interval)  # [N, 3]
            Phi_A_interval = torch.einsum('tnie,el->tni', Phi_interval, A)  # [t, N, 3]
            integral_phi_A = torch.einsum('tni,t->ni', Phi_A_interval, dt_interval)  # [N, 3]
            
            residual = lhs - integral_f - integral_phi_A
            loss = torch.mean(residual ** 2)
            losses.append(loss)
        
        total_loss = torch.mean(torch.stack(losses))
        
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            A.clamp_(-2.0, 2.0)
        
        A_np = A.detach().cpu().numpy().flatten()
        pbar.set_postfix({
            'loss': f'{total_loss.item():.6f}',
            'A_min': f'{A_np.min():.3f}',
            'A_max': f'{A_np.max():.3f}',
            'A_mean': f'{A_np.mean():.3f}'
        })
        
        if (epoch + 1) % 500 == 0:
            print(f"\n\n{'='*60}")
            print(f"Epoch {epoch+1}/{n_epochs} - AUC Evaluation")
            print('='*60)
            A_current = A.detach().cpu().numpy()
            auc_scores, _ = compute_auc_scores(A_current, edge_config, N, max_order)
            
            all_possible = generate_all_possible_hyperedges(N, max_order)
            A_flat = A_current.flatten()
            A_idx = 0
            
            for order_name, order_label in zip(['edges', 'triangles', 'quads', 'quints', 'sexts', 'septs'],
                                               ['2-edges', '3-edges', '4-edges', '5-edges', '6-edges', '7-edges']):
                if order_label not in auc_scores:
                    continue
                    
                possible_edges = all_possible[order_name]
                true_edges = edge_config[order_name]
                n_possible = len(possible_edges)
                n_true = len(true_edges)
                
                A_order = A_flat[A_idx:A_idx+n_possible]
                A_idx += n_possible
                
                auc_score = auc_scores[order_label]
                if auc_score is not None:
                    print(f"  {order_label}: AUC = {auc_score:.4f} | Possible={n_possible}, True={n_true} | A range=[{A_order.min():.3f}, {A_order.max():.3f}]")
                else:
                    print(f"  {order_label}: N/A | Possible={n_possible}, True={n_true}")
            print('='*60 + '\n')
    
    return A.detach().cpu().numpy(), edge_config, save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rossler Integral Model with Hypergraph Topology Learning')
    parser.add_argument('--n_samples', type=int, default=11, 
                        help='Number of time samples from ODE solver (default: 11)')
    parser.add_argument('--n_epochs', type=int, default=20000,
                        help='Number of training epochs (default: 20000)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--gpu_id', type=int, default=6,
                        help='GPU device ID (default: 6)')
    args = parser.parse_args()
    
    N = 8
    max_order = 7
    
    A_learned, edge_config, save_dir = train_integral_model(
        N=N, 
        max_order=max_order, 
        n_epochs=args.n_epochs, 
        lr=args.lr, 
        batch_size=32,
        gpu_id=args.gpu_id,
        use_nn=True,
        stage1_epochs=10000,
        resample_factor=10,
        n_samples=args.n_samples
    )
    
    print("\nComputing AUC scores...")
    auc_scores, roc_data = compute_auc_scores(A_learned, edge_config, N, max_order)
    
    print("\nAUC scores for each order:")
    for order_label, auc_score in auc_scores.items():
        if auc_score is not None:
            print(f"  {order_label}: {auc_score:.4f}")
        else:
            print(f"  {order_label}: N/A (no positive/negative samples)")
    
    np.save(f"{save_dir}/A_learned.npy", A_learned)
    
    with open(f"{save_dir}/auc_scores.txt", 'w') as f:
        f.write(f"N={N}, max_order={max_order}, n_samples={args.n_samples}\n")
        f.write("\nAUC Scores:\n")
        for order_label, auc_score in auc_scores.items():
            if auc_score is not None:
                f.write(f"  {order_label}: {auc_score:.4f}\n")
            else:
                f.write(f"  {order_label}: N/A\n")
    
    print("\nPlotting ROC curves...")
    plot_roc_curves(roc_data, auc_scores, save_dir)
    
    print(f"\nResults saved to {save_dir}/")
