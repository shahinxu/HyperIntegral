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
"""

import torch
from torch import optim
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_curve, auc
import os
from datetime import datetime
from tqdm import tqdm
import argparse

def roessler_dynamics(x, N):
    """
    Rossler dynamics basic equations (without coupling terms) - GPU version
    x: [N, 3] torch tensor [[x1,y1,z1], [x2,y2,z2], ...]
    Returns: [N, 3] torch tensor of derivatives
    """
    ar, br, cr = 0.2, 0.2, 0.7
    
    xold = x[:, 0]  # [N]
    yold = x[:, 1]  # [N]
    zold = x[:, 2]  # [N]
    
    dxdt = -yold - zold  # Only basic dynamics, no coupling
    dydt = xold + ar * yold
    dzdt = br + zold * (xold - cr)
    
    return torch.stack([dxdt, dydt, dzdt], dim=1)  # [N, 3]


def compute_hyperedge_coupling_tensor(x, all_possible_edges, N, device):
    """
    Compute coupling tensor for all possible hyperedges - Vectorized GPU version
    x: [N, 3] torch tensor [[x1,y1,z1], [x2,y2,z2], ...]
    all_possible_edges: Dictionary containing all possible hyperedge lists (not just true ones)
    Returns: Φ tensor [N, 3, N_total_hyperedges]
    """
    xold = x[:, 0]  # [N]
    yold = x[:, 1]  # [N]
    zold = x[:, 2]  # [N]
    
    k, kD = 0.4, 0.3
    
    # Count all hyperedges
    n_edges = len(all_possible_edges['edges'])
    n_triangles = len(all_possible_edges['triangles'])
    n_quads = len(all_possible_edges['quads'])
    n_quints = len(all_possible_edges['quints'])
    n_sexts = len(all_possible_edges['sexts'])
    n_septs = len(all_possible_edges['septs'])
    n_total = n_edges + n_triangles + n_quads + n_quints + n_sexts + n_septs
    
    # Pre-allocate entire tensor
    Phi = torch.zeros((N, 3, n_total), device=device)
    edge_idx = 0
    
    # 2-edges - Vectorized processing
    if n_edges > 0:
        edges_tensor = torch.tensor(all_possible_edges['edges'], device=device) - 1  # [n_edges, 2], 0-indexed
        i_indices = edges_tensor[:, 0]  # [n_edges]
        j_indices = edges_tensor[:, 1]  # [n_edges]
        
        # Get x coordinates of relevant nodes
        xi = xold[i_indices]  # [n_edges]
        xj = xold[j_indices]  # [n_edges]
        
        # Batch compute coupling
        for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            Phi[i, 0, edge_idx + idx] = k * (xold[j] - xold[i])
            Phi[j, 0, edge_idx + idx] = k * (xold[i] - xold[j])
        edge_idx += n_edges
    
    # 3-edges - Vectorized processing
    if n_triangles > 0:
        triangles_tensor = torch.tensor(all_possible_edges['triangles'], device=device) - 1  # [n_triangles, 3]
        for idx, triangle in enumerate(triangles_tensor):
            i, j, k_idx = triangle[0], triangle[1], triangle[2]
            xi, xj, xk = xold[i], xold[j], xold[k_idx]
            
            Phi[i, 0, edge_idx + idx] = kD * (xj**2 * xk - xi**3 + xj * xk**2 - xi**3)
            Phi[j, 0, edge_idx + idx] = kD * (xi**2 * xk - xj**3 + xi * xk**2 - xj**3)
            Phi[k_idx, 0, edge_idx + idx] = kD * (xi**2 * xj - xk**3 + xi * xj**2 - xk**3)
        edge_idx += n_triangles
    
    # 4-edges - Vectorized processing
    if n_quads > 0:
        quads_tensor = torch.tensor(all_possible_edges['quads'], device=device) - 1  # [n_quads, 4]
        for idx, quad in enumerate(quads_tensor):
            i, j, k_idx, l = quad[0], quad[1], quad[2], quad[3]
            xi, xj, xk, xl = xold[i], xold[j], xold[k_idx], xold[l]
            
            Phi[i, 0, edge_idx + idx] = kD * (xj**2 * xk * xl - xi**3)
            Phi[j, 0, edge_idx + idx] = kD * (xi**2 * xk * xl - xj**3)
            Phi[k_idx, 0, edge_idx + idx] = kD * (xi**2 * xj * xl - xk**3)
            Phi[l, 0, edge_idx + idx] = kD * (xi**2 * xj * xk - xl**3)
        edge_idx += n_quads
    
    # 5-edges - Vectorized processing
    if n_quints > 0:
        quints_tensor = torch.tensor(all_possible_edges['quints'], device=device) - 1  # [n_quints, 5]
        for idx, quint in enumerate(quints_tensor):
            i, j, k_idx, l, m = quint[0], quint[1], quint[2], quint[3], quint[4]
            yi, yj, yk, yl, ym = yold[i], yold[j], yold[k_idx], yold[l], yold[m]
            
            Phi[i, 1, edge_idx + idx] = kD * (yj**2 * yk * yl * ym - yi**3)
            Phi[j, 1, edge_idx + idx] = kD * (yi**2 * yk * yl * ym - yj**3)
            Phi[k_idx, 1, edge_idx + idx] = kD * (yi**2 * yj * yl * ym - yk**3)
            Phi[l, 1, edge_idx + idx] = kD * (yi**2 * yj * yk * ym - yl**3)
            Phi[m, 1, edge_idx + idx] = kD * (yi**2 * yj * yk * yl - ym**3)
        edge_idx += n_quints
    
    # 6-edges - Vectorized processing
    if n_sexts > 0:
        sexts_tensor = torch.tensor(all_possible_edges['sexts'], device=device) - 1  # [n_sexts, 6]
        for idx, sext in enumerate(sexts_tensor):
            i, j, k_idx, l, m, n = sext[0], sext[1], sext[2], sext[3], sext[4], sext[5]
            yi, yj, yk, yl, ym, yn = yold[i], yold[j], yold[k_idx], yold[l], yold[m], yold[n]
            
            Phi[i, 1, edge_idx + idx] = kD * (yj**2 * yk * yl * ym * yn - yi**3)
            Phi[j, 1, edge_idx + idx] = kD * (yi**2 * yk * yl * ym * yn - yj**3)
            Phi[k_idx, 1, edge_idx + idx] = kD * (yi**2 * yj * yl * ym * yn - yk**3)
            Phi[l, 1, edge_idx + idx] = kD * (yi**2 * yj * yk * ym * yn - yl**3)
            Phi[m, 1, edge_idx + idx] = kD * (yi**2 * yj * yk * yl * yn - ym**3)
            Phi[n, 1, edge_idx + idx] = kD * (yi**2 * yj * yk * yl * ym - yn**3)
        edge_idx += n_sexts
    
    # 7-edges - Vectorized processing
    if n_septs > 0:
        septs_tensor = torch.tensor(all_possible_edges['septs'], device=device) - 1  # [n_septs, 7]
        for idx, sept in enumerate(septs_tensor):
            i, j, k_idx, l, m, n, o = sept[0], sept[1], sept[2], sept[3], sept[4], sept[5], sept[6]
            zi, zj, zk, zl, zm, zn, zo = zold[i], zold[j], zold[k_idx], zold[l], zold[m], zold[n], zold[o]
            
            Phi[i, 2, edge_idx + idx] = kD * (zj**2 * zk * zl * zm * zn * zo - zi**3)
            Phi[j, 2, edge_idx + idx] = kD * (zi**2 * zk * zl * zm * zn * zo - zj**3)
            Phi[k_idx, 2, edge_idx + idx] = kD * (zi**2 * zj * zl * zm * zn * zo - zk**3)
            Phi[l, 2, edge_idx + idx] = kD * (zi**2 * zj * zk * zm * zn * zo - zl**3)
            Phi[m, 2, edge_idx + idx] = kD * (zi**2 * zj * zk * zl * zn * zo - zm**3)
            Phi[n, 2, edge_idx + idx] = kD * (zi**2 * zj * zk * zl * zm * zo - zn**3)
            Phi[o, 2, edge_idx + idx] = kD * (zi**2 * zj * zk * zl * zm * zn - zo**3)
    
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
    """
    Generate all possible hyperedges (for AUC computation)
    Returns: Dictionary containing lists of all possible hyperedges for each order
    """
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


def generate_training_data(N, edge_config):
    """Generate true Rossler system data for training"""
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
    t_eval = np.linspace(0, 20, 301)
    sol = solve_ivp(roessler_hoi, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    
    x_data_flat = sol.y.T  # [T, 3N]
    T = x_data_flat.shape[0]
    x_data = np.zeros((T, N, 3))
    x_data[:, :, 0] = x_data_flat[:, 0:N]        # x coordinates
    x_data[:, :, 1] = x_data_flat[:, N:2*N]      # y coordinates
    x_data[:, :, 2] = x_data_flat[:, 2*N:3*N]    # z coordinates
    
    return sol.t, x_data  # [T], [T, N, 3]


def train_integral_model(N=8, max_order=7, n_epochs=10000, lr=0.001, batch_size=32, gpu_id=0, noise=0):
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
    t_data, x_data = generate_training_data(N, edge_config)
    n_times = len(t_data)
    print(f"Data shape: {x_data.shape}, Time points: {n_times}")
    
    print("Transferring data to GPU...")
    t_data_gpu = torch.tensor(t_data, dtype=torch.float32, device=device)
    x_data_gpu = torch.tensor(x_data, dtype=torch.float32, device=device)  # [T, N, 3]
    
    if noise > 0:
        noise_std = noise * torch.std(x_data_gpu)
        noise_tensor = torch.randn_like(x_data_gpu) * noise_std
        x_data_gpu = x_data_gpu + noise_tensor
        print(f"Added {noise*100:.1f}% noise to data (std={noise_std:.6f})")
    
    print("Pre-computing coupling tensor Phi and basic dynamics f for all time points...")
    Phi_all = torch.zeros((n_times, N, 3, n_hyperedges), device=device)  # [T, N, 3, n_hyperedges]
    f_all = torch.zeros((n_times, N, 3), device=device)  # [T, N, 3]
    
    for t_idx in tqdm(range(n_times), desc="Precomputing Phi", leave=False):
        x_t = x_data_gpu[t_idx]  # [N, 3]
        Phi_all[t_idx] = compute_hyperedge_coupling_tensor(x_t, all_possible_edges, N, device)  # [N, 3, n_hyperedges]
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
    
    # 2-edges: bias = -2
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
        
        # if epoch >= n_epochs - 10000:
        #     progress = (epoch - (n_epochs - 10000)) / 10000  # 0 to 1
        #     # sparsity_weight = 0.00001 * progress
        #     sparsity_weight = 0
        #     l1_loss = sparsity_weight * torch.sum(torch.abs(A))
        #     total_loss = total_loss + l1_loss
        
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
            
            # Detailed diagnostic info
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
                
                # Get A values for this order
                A_order = A_flat[A_idx:A_idx+n_possible]
                A_idx += n_possible
                
                auc_score = auc_scores[order_label]
                if auc_score is not None:
                    print(f"  {order_label}: AUC = {auc_score:.4f} | Possible={n_possible}, True={n_true} | A range=[{A_order.min():.3f}, {A_order.max():.3f}]")
                else:
                    print(f"  {order_label}: N/A | Possible={n_possible}, True={n_true}")
            print('='*60 + '\n')
    
    return A.detach().cpu().numpy(), edge_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rossler Integral Model')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level (default: 0.0)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')
    args = parser.parse_args()
    
    N = 8
    max_order = 7
    
    A_learned, edge_config = train_integral_model(
        N=N, 
        max_order=max_order, 
        n_epochs=20000, 
        lr=0.01, 
        batch_size=32,
        gpu_id=args.gpu_id,
        noise=args.noise
    )
    
    print("\nComputing AUC scores...")
    auc_scores, roc_data = compute_auc_scores(A_learned, edge_config, N, max_order)
    
    print("\nAUC scores for each order:")
    for order_label, auc_score in auc_scores.items():
        if auc_score is not None:
            print(f"  {order_label}: {auc_score:.4f}")
        else:
            print(f"  {order_label}: N/A (no positive/negative samples)")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        save_dir = f"{args.output_dir}/{timestamp}"
    else:
        save_dir = f"results/integral_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f"{save_dir}/A_learned.npy", A_learned)
    
    with open(f"{save_dir}/auc_scores.txt", 'w') as f:
        f.write(f"N={N}, max_order={max_order}\n")
        f.write("\nAUC Scores:\n")
        for order_label, auc_score in auc_scores.items():
            if auc_score is not None:
                f.write(f"  {order_label}: {auc_score:.4f}\n")
            else:
                f.write(f"  {order_label}: N/A\n")
    
    print("\nPlotting ROC curves...")
    plot_roc_curves(roc_data, auc_scores, save_dir)
    
    print(f"\nResults saved to {save_dir}/")
