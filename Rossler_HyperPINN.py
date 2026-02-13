from HyperPINNTopology import HyperPINNTopology
import torch
from torch import optim as optim
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from datetime import datetime
import networkx as nx
import argparse

def roessler_hoi(t, x, EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList):
    m1 = len(x)
    N = m1 // 3
    xold = x[0:N]
    yold = x[N:2*N]
    zold = x[2*N:3*N]
    ar, br, cr = 0.2, 0.2, 0.7
    k, kD = 0.4, 0.3

    coup_rete = np.zeros(N)
    coup_simplicial = np.zeros(N)
    coup_quads = np.zeros(N)
    coup_quints = np.zeros(N)
    coup_sexts = np.zeros(N)
    coup_septs = np.zeros(N)
    
    for ii in range(len(EdgeList)):
        i1 = EdgeList[ii, 0] - 1
        i2 = EdgeList[ii, 1] - 1
        coup_rete[i1] += xold[i2] - xold[i1]
        coup_rete[i2] += xold[i1] - xold[i2]
    
    mtrianglelist, _ = TriangleList.shape
    for ii in range(mtrianglelist):
        i1 = TriangleList[ii, 0] - 1
        i2 = TriangleList[ii, 1] - 1
        i3 = TriangleList[ii, 2] - 1
        coup_simplicial[i1] += xold[i2]**2 * xold[i3] - xold[i1]**3 + xold[i2] * xold[i3]**2 - xold[i1]**3
        coup_simplicial[i2] += xold[i1]**2 * xold[i3] - xold[i2]**3 + xold[i1] * xold[i3]**2 - xold[i2]**3
        coup_simplicial[i3] += xold[i1]**2 * xold[i2] - xold[i3]**3 + xold[i1] * xold[i2]**2 - xold[i3]**3
    
    mquadlist, nquadlist = QuadList.shape
    for ii in range(mquadlist):
        i1 = QuadList[ii, 0] - 1
        i2 = QuadList[ii, 1] - 1
        i3 = QuadList[ii, 2] - 1
        i4 = QuadList[ii, 3] - 1
        coup_quads[i1] += xold[i2]**2 * xold[i3] * xold[i4] - xold[i1]**3
        coup_quads[i2] += xold[i1]**2 * xold[i3] * xold[i4] - xold[i2]**3
        coup_quads[i3] += xold[i1]**2 * xold[i2] * xold[i4] - xold[i3]**3
        coup_quads[i4] += xold[i1]**2 * xold[i2] * xold[i3] - xold[i4]**3
    
    mquintlist, nquintlist = QuintList.shape
    for ii in range(mquintlist):
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
    
    msextlist, nsextlist = SextList.shape
    for ii in range(msextlist):
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
    
    mseptlist, nseptlist = SeptList.shape
    for ii in range(mseptlist):
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Rossler Oscillators with HyperPINN')
parser.add_argument('--M', type=int, default=300)
parser.add_argument('--tmax', type=float, default=20)
parser.add_argument('--N', type=int, default=8)
parser.add_argument('--max_order', type=int, default=7)
parser.add_argument('--gpu_id', type=int, default=4)
parser.add_argument('--noise', type=float, default=0.0)
args = parser.parse_args()

N = args.N
max_order = args.max_order
gpu_id = args.gpu_id
noise = args.noise

def get_hyperedge_config(N):
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
    return (
        np.array(config['edges']),
        np.array(config['triangles']),
        np.array(config['quads']),
        np.array(config['quints']),
        np.array(config['sexts']),
        np.array(config['septs'])
    )

EdgeList, TriangleList_full, QuadList_full, QuintList_full, SextList_full, SeptList_full = get_hyperedge_config(N)

# Automatically adjust ground truth based on max_order
TriangleList = TriangleList_full if max_order >= 3 else np.array([]).reshape(0, 3)
QuadList = QuadList_full if max_order >= 4 else np.array([]).reshape(0, 4)
QuintList = QuintList_full if max_order >= 5 else np.array([]).reshape(0, 5)
SextList = SextList_full if max_order >= 6 else np.array([]).reshape(0, 6)
SeptList = SeptList_full if max_order >= 7 else np.array([]).reshape(0, 7)

all_2edges = list(combinations(range(1, N+1), 2))
all_3edges = list(combinations(range(1, N+1), 3))
all_4edges = list(combinations(range(1, N+1), 4))
all_5edges = list(combinations(range(1, N+1), 5))
all_6edges = list(combinations(range(1, N+1), 6))
all_7edges = list(combinations(range(1, N+1), 7))

true_2edges = set(tuple(sorted(edge)) for edge in EdgeList)
true_3edges = set(tuple(sorted(triangle)) for triangle in TriangleList)
true_4edges = set(tuple(sorted(quad)) for quad in QuadList)
true_5edges = set(tuple(sorted(quint)) for quint in QuintList)
true_6edges = set(tuple(sorted(sext)) for sext in SextList)
true_7edges = set(tuple(sorted(sept)) for sept in SeptList)

M = args.M
tmax = args.tmax
dt = tmax / M
t_eval = np.linspace(0, tmax, M+1)
t_data = torch.linspace(0, tmax, M+1, requires_grad=True).unsqueeze(1)

np.random.seed(42)
x0 = np.random.uniform(-1, 1, size=(3 * N,))
sol = solve_ivp(
    roessler_hoi,
    (0, tmax),
    x0,
    t_eval=t_eval,
    args=(EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList),
)
X = sol.y.T
nt = len(t_eval)
dxdt = np.array(
    [
        roessler_hoi(t, sol.y[:, i], EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList)
        for i, t in enumerate(sol.t)
    ]
)

if noise > 0:
    np.random.seed()
    X_noisy = X + np.random.randn(*X.shape) * noise
    print(f"Added Gaussian noise with std={noise:.6f} (absolute)")
else:
    X_noisy = X

x_data = torch.tensor(X_noisy, dtype=torch.float64)
architectures = [
    ("ResNet", True, False, False),
    ("Attention", False, True, False),
    ("Pirate", False, False, True),
]

n_cols = 4
n_rows = int(np.ceil(N / n_cols))
plt.figure(figsize=(4*n_cols, 3*n_rows))
X_plot = x_data.cpu().numpy() if noise > 0 else X
for i in range(N):
    plt.subplot(n_rows, n_cols, i+1)
    plt.plot(t_eval, X_plot[:, i], 'b-', label=f'x_{i+1}', alpha=0.7)
    plt.plot(t_eval, X_plot[:, i+N], 'r-', label=f'y_{i+1}', alpha=0.7)
    plt.plot(t_eval, X_plot[:, i+2*N], 'g-', label=f'z_{i+1}', alpha=0.7)
    if noise > 0:
        plt.plot(t_eval, X[:, i], 'b--', alpha=0.3, linewidth=0.5)
        plt.plot(t_eval, X[:, i+N], 'r--', alpha=0.3, linewidth=0.5)
        plt.plot(t_eval, X[:, i+2*N], 'g--', alpha=0.3, linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title(f'Node {i+1}')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join(
    'results_hyperpinn',
    f"sample_{M}_noise_{noise}",
    timestamp,
)
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "rossler_oscillators.png"))
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
    output_dim=3*N,
    use_resnet=use_resnet,
    use_attention=use_attention,
    use_pirate=use_pirate,
    max_order=max_order,
)
model = model.to(device)

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

for epoch in range(epochs):
    optimizer.zero_grad()
    x_pred = model.forward(t_data)
    physics_loss = model.physics_loss(t_data)
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
        with torch.no_grad():
            X_pred = model.forward(t_data).cpu().numpy()

        X_train = x_data.detach().cpu().numpy()  # training data (possibly noisy)

        fig, axes = plt.subplots(N, 3, figsize=(15, 2.5 * N))
        coord_names = ['x', 'y', 'z']

        for node_idx in range(N):
            for coord_idx in range(3):
                ax = axes[node_idx, coord_idx]
                ax.plot(
                    t_eval,
                    X_train[:, node_idx + coord_idx * N],
                    'o',
                    label='Data',
                    markersize=4,
                    alpha=0.7,
                    color='blue',
                )

                # NN prediction (line) -- analogous to "Resampled (Linear)"
                ax.plot(
                    t_eval,
                    X_pred[:, node_idx + coord_idx * N],
                    '-',
                    label='NN prediction',
                    linewidth=1,
                    alpha=0.8,
                    color='red',
                )

                if node_idx == 0:
                    ax.set_title(f'{coord_names[coord_idx]}-coordinate', fontsize=12)
                if coord_idx == 0:
                    ax.set_ylabel(f'Node {node_idx+1}', fontsize=11)
                if node_idx == N - 1:
                    ax.set_xlabel('Time', fontsize=10)
                ax.grid(True, alpha=0.3)
                if node_idx == 0 and coord_idx == 0:
                    ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"nn_prediction_epoch{epoch}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
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
plt.savefig(os.path.join(results_dir, 'roc_curves_7_order.png'), bbox_inches='tight')