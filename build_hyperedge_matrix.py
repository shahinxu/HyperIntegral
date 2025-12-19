"""
构建超边耦合三维张量
维度: (时间点, 超边索引, 节点索引)
张量元素表示：在某时间点，某超边对某节点的耦合贡献值
"""

import numpy as np
from scipy.integrate import solve_ivp
from itertools import combinations
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.linalg import svd
from sklearn.decomposition import PCA

def roessler_hoi(t, x, EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList):
    """Rossler 高阶交互动力学方程"""
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
    
    mtrianglelist, ntrianglelist = TriangleList.shape
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


def compute_edge_coupling_per_node(x, edge, order, N):
    """
    计算单个超边对所有N个节点的耦合贡献
    返回: 长度为N的数组，每个元素是该超边对对应节点的贡献
    """
    xold = x[0:N]
    yold = x[N:2*N]
    zold = x[2*N:3*N]
    
    # 初始化所有节点的贡献为0
    contributions = np.zeros(N)
    
    # 转换为 0-indexed
    nodes = [n - 1 for n in edge]
    
    if order == 2:
        # 2-edge: 使用 x 坐标
        i, j = nodes
        contributions[i] = xold[j] - xold[i]
        contributions[j] = xold[i] - xold[j]
    
    elif order == 3:
        # 3-edge (triangle): 使用 x 坐标
        i, j, k = nodes
        contributions[i] = xold[j]**2 * xold[k] - xold[i]**3 + xold[j] * xold[k]**2 - xold[i]**3
        contributions[j] = xold[i]**2 * xold[k] - xold[j]**3 + xold[i] * xold[k]**2 - xold[j]**3
        contributions[k] = xold[i]**2 * xold[j] - xold[k]**3 + xold[i] * xold[j]**2 - xold[k]**3
    
    elif order == 4:
        # 4-edge (quad): 使用 x 坐标
        i, j, k, l = nodes
        contributions[i] = xold[j]**2 * xold[k] * xold[l] - xold[i]**3
        contributions[j] = xold[i]**2 * xold[k] * xold[l] - xold[j]**3
        contributions[k] = xold[i]**2 * xold[j] * xold[l] - xold[k]**3
        contributions[l] = xold[i]**2 * xold[j] * xold[k] - xold[l]**3
    
    elif order == 5:
        # 5-edge (quint): 使用 y 坐标
        i, j, k, l, m = nodes
        contributions[i] = yold[j]**2 * yold[k] * yold[l] * yold[m] - yold[i]**3
        contributions[j] = yold[i]**2 * yold[k] * yold[l] * yold[m] - yold[j]**3
        contributions[k] = yold[i]**2 * yold[j] * yold[l] * yold[m] - yold[k]**3
        contributions[l] = yold[i]**2 * yold[j] * yold[k] * yold[m] - yold[l]**3
        contributions[m] = yold[i]**2 * yold[j] * yold[k] * yold[l] - yold[m]**3
    
    elif order == 6:
        # 6-edge (sext): 使用 y 坐标
        i, j, k, l, m, n = nodes
        contributions[i] = yold[j]**2 * yold[k] * yold[l] * yold[m] * yold[n] - yold[i]**3
        contributions[j] = yold[i]**2 * yold[k] * yold[l] * yold[m] * yold[n] - yold[j]**3
        contributions[k] = yold[i]**2 * yold[j] * yold[l] * yold[m] * yold[n] - yold[k]**3
        contributions[l] = yold[i]**2 * yold[j] * yold[k] * yold[m] * yold[n] - yold[l]**3
        contributions[m] = yold[i]**2 * yold[j] * yold[k] * yold[l] * yold[n] - yold[m]**3
        contributions[n] = yold[i]**2 * yold[j] * yold[k] * yold[l] * yold[m] - yold[n]**3
    
    elif order == 7:
        # 7-edge (sept): 使用 z 坐标
        i, j, k, l, m, n, o = nodes
        contributions[i] = zold[j]**2 * zold[k] * zold[l] * zold[m] * zold[n] * zold[o] - zold[i]**3
        contributions[j] = zold[i]**2 * zold[k] * zold[l] * zold[m] * zold[n] * zold[o] - zold[j]**3
        contributions[k] = zold[i]**2 * zold[j] * zold[l] * zold[m] * zold[n] * zold[o] - zold[k]**3
        contributions[l] = zold[i]**2 * zold[j] * zold[k] * zold[m] * zold[n] * zold[o] - zold[l]**3
        contributions[m] = zold[i]**2 * zold[j] * zold[k] * zold[l] * zold[n] * zold[o] - zold[m]**3
        contributions[n] = zold[i]**2 * zold[j] * zold[k] * zold[l] * zold[m] * zold[o] - zold[n]**3
        contributions[o] = zold[i]**2 * zold[j] * zold[k] * zold[l] * zold[m] * zold[n] - zold[o]**3
    
    return contributions


def build_hyperedge_tensor(N=8, max_order=7, M=300, tmax=20):
    """
    构建超边耦合三维张量
    
    参数:
        N: 节点数量
        max_order: 最大超边阶数
        M: 时间步数
        tmax: 最大时间
    
    返回:
        tensor: (M+1, total_edges, N) 三维张量
               - 第1维: 时间点
               - 第2维: 超边索引
               - 第3维: 节点索引
        edge_info: 超边信息列表 [(order, edge), ...]
        t_eval: 时间点数组
        X: 轨迹数据
    """
    
    # 真实存在的超边配置（用于生成数据）
    true_config = {
        'edges': [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]],
        'triangles': [[1, 2, 3], [2, 4, 5], [5, 6, 7], [6, 7, 8]],
        'quads': [[1, 2, 3, 4]],
        'quints': [[4, 5, 6, 7, 8]],
        'sexts': [[1, 2, 3, 4, 5, 6]],
        'septs': [[1, 2, 4, 5, 6, 7, 8]]
    }
    
    EdgeList = np.array(true_config['edges'])
    TriangleList = np.array(true_config['triangles']) if max_order >= 3 else np.array([]).reshape(0, 3)
    QuadList = np.array(true_config['quads']) if max_order >= 4 else np.array([]).reshape(0, 4)
    QuintList = np.array(true_config['quints']) if max_order >= 5 else np.array([]).reshape(0, 5)
    SextList = np.array(true_config['sexts']) if max_order >= 6 else np.array([]).reshape(0, 6)
    SeptList = np.array(true_config['septs']) if max_order >= 7 else np.array([]).reshape(0, 7)
    
    # 生成时间序列数据
    t_eval = np.linspace(0, tmax, M + 1)
    x0 = np.random.uniform(-1, 1, size=(3 * N,))
    
    print("求解 ODE...")
    sol = solve_ivp(
        roessler_hoi, 
        (0, tmax), 
        x0, 
        t_eval=t_eval, 
        args=(EdgeList, TriangleList, QuadList, QuintList, SextList, SeptList)
    )
    X = sol.y.T  # shape: (M+1, 3*N)
    
    # 生成所有可能的超边
    all_edges = []
    edge_info = []
    
    for order in range(2, max_order + 1):
        edges_of_order = list(combinations(range(1, N + 1), order))
        for edge in edges_of_order:
            all_edges.append(edge)
            edge_info.append((order, edge))
    
    total_edges = len(all_edges)
    print(f"\n节点数: {N}")
    print(f"最大阶数: {max_order}")
    print(f"时间步数: {M + 1}")
    print(f"可能的超边总数: {total_edges}")
    
    # 各阶超边数量统计
    for order in range(2, max_order + 1):
        count = len([e for o, e in edge_info if o == order])
        print(f"  {order}-edges: {count}")
    
    # 构建三维张量
    print(f"\n构建超边耦合张量: ({M + 1}, {total_edges}, {N})...")
    tensor = np.zeros((M + 1, total_edges, N))
    
    for t_idx in range(M + 1):
        if t_idx % 50 == 0:
            print(f"  处理时间步 {t_idx}/{M + 1}...")
        
        x_t = X[t_idx, :]
        
        for edge_idx, (order, edge) in enumerate(edge_info):
            contributions = compute_edge_coupling_per_node(x_t, edge, order, N)
            tensor[t_idx, edge_idx, :] = contributions
    
    print("完成！")
    return tensor, edge_info, t_eval, X


def analyze_hyperedge_indistinguishability(tensor, edge_info, results_dir=None):
    """
    分析超边之间的不可分辨性
    
    方法:
    1. 将每个超边的时空动力学展平为向量
    2. 计算超边空间的有效维数（秩）
    3. 识别线性相关的超边组合
    4. 找到可以被其他超边线性表示的冗余超边
    
    参数:
        tensor: (time, edge, node) 三维张量
        edge_info: 超边信息列表
        results_dir: 结果保存目录
    
    返回:
        analysis: 包含分析结果的字典
    """
    T, E, N = tensor.shape
    print(f"\n=== 超边不可分辨性分析 ===")
    print(f"张量形状: ({T} 时间点, {E} 超边, {N} 节点)")
    
    # 1. 将每个超边展平为向量: (E, T*N)
    edge_vectors = tensor.transpose(1, 0, 2).reshape(E, T * N)
    print(f"超边向量矩阵形状: {edge_vectors.shape}")
    
    # 2. 计算超边之间的Gram矩阵 (相关性矩阵)
    print("\n计算超边相关性矩阵...")
    # 标准化每个超边向量
    edge_vectors_normalized = edge_vectors / (np.linalg.norm(edge_vectors, axis=1, keepdims=True) + 1e-10)
    gram_matrix = edge_vectors_normalized @ edge_vectors_normalized.T
    
    # 3. SVD分解找有效秩
    print("\n进行奇异值分解...")
    U, S, Vt = svd(edge_vectors, full_matrices=False)
    
    # 计算有效秩（奇异值大于阈值）
    sv_threshold = 1e-6 * S[0]  # 相对于最大奇异值的阈值
    effective_rank = np.sum(S > sv_threshold)
    
    print(f"\n奇异值统计:")
    print(f"  总超边数: {E}")
    print(f"  有效秩: {effective_rank}")
    print(f"  冗余超边数: {E - effective_rank}")
    print(f"  最大奇异值: {S[0]:.6e}")
    print(f"  最小奇异值: {S[-1]:.6e}")
    print(f"  条件数: {S[0] / (S[-1] + 1e-10):.2e}")
    
    # 显示前10个奇异值
    print(f"\n前10个奇异值:")
    for i in range(min(10, len(S))):
        print(f"  σ_{i+1} = {S[i]:.6e} ({100*S[i]**2/np.sum(S**2):.2f}% 能量)")
    
    # 4. 识别高度相关的超边对（可能不可分辨）
    print("\n寻找高度相关的超边对 (|相关系数| > 0.95)...")
    highly_correlated_pairs = []
    for i in range(E):
        for j in range(i + 1, E):
            corr = gram_matrix[i, j]
            if abs(corr) > 0.95:
                highly_correlated_pairs.append((i, j, corr))
    
    highly_correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"发现 {len(highly_correlated_pairs)} 对高度相关的超边:")
    for i, j, corr in highly_correlated_pairs[:20]:  # 显示前20对
        order_i, edge_i = edge_info[i]
        order_j, edge_j = edge_info[j]
        print(f"  Edge {i} (order={order_i}, {edge_i}) <-> Edge {j} (order={order_j}, {edge_j}): ρ={corr:.4f}")
    
    # 5. 识别可以被其他超边线性表示的超边（通过最小二乘）
    print("\n寻找冗余超边（可被其他超边线性表示，R² > 0.95）...")
    redundant_edges = []
    
    for target_idx in range(E):
        # 用其他所有超边拟合目标超边
        other_indices = [i for i in range(E) if i != target_idx]
        X_fit = edge_vectors[other_indices, :].T
        y_target = edge_vectors[target_idx, :]
        
        # 最小二乘求解
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_fit, y_target, rcond=None)
            # 计算 R²
            y_pred = X_fit @ coeffs
            ss_res = np.sum((y_target - y_pred) ** 2)
            ss_tot = np.sum((y_target - y_target.mean()) ** 2)
            r_squared = 1 - ss_res / (ss_tot + 1e-10)
            
            if r_squared > 0.95:
                # 找到主要贡献的超边（系数最大的）
                top_contributors = np.argsort(np.abs(coeffs))[-5:][::-1]
                contributor_info = [(other_indices[idx], coeffs[idx]) for idx in top_contributors if abs(coeffs[idx]) > 0.01]
                redundant_edges.append((target_idx, r_squared, contributor_info))
        except:
            pass
    
    redundant_edges.sort(key=lambda x: x[1], reverse=True)
    
    print(f"发现 {len(redundant_edges)} 个冗余超边:")
    for target_idx, r_sq, contributors in redundant_edges[:20]:  # 显示前20个
        order_t, edge_t = edge_info[target_idx]
        print(f"\n  Edge {target_idx} (order={order_t}, {edge_t}): R²={r_sq:.4f}")
        print(f"    可表示为以下超边的线性组合:")
        for contrib_idx, coeff in contributors[:3]:
            order_c, edge_c = edge_info[contrib_idx]
            print(f"      {coeff:+.3f} × Edge {contrib_idx} (order={order_c}, {edge_c})")
    
    # 6. 可视化
    if results_dir is not None:
        print("\n生成可视化...")
        
        # 6.1 奇异值谱
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax = axes[0]
        ax.semilogy(range(1, len(S) + 1), S, 'o-', markersize=4)
        ax.axhline(y=sv_threshold, color='r', linestyle='--', label=f'Threshold ({sv_threshold:.2e})')
        ax.axvline(x=effective_rank, color='g', linestyle='--', label=f'Effective Rank = {effective_rank}')
        ax.set_xlabel('Singular Value Index', fontsize=12)
        ax.set_ylabel('Singular Value', fontsize=12)
        ax.set_title('Singular Value Spectrum', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        ax = axes[1]
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2) * 100
        ax.plot(range(1, len(S) + 1), cumulative_energy, 'o-', markersize=4)
        ax.axhline(y=95, color='r', linestyle='--', label='95% Energy')
        ax.axhline(y=99, color='orange', linestyle='--', label='99% Energy')
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
        ax.set_title('Cumulative Energy Explained', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'singular_value_analysis.png'), dpi=200)
        plt.close()
        
        # 6.2 相关性矩阵热图
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        im = ax.imshow(gram_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('Hyperedge Index', fontsize=12)
        ax.set_ylabel('Hyperedge Index', fontsize=12)
        ax.set_title('Hyperedge Correlation Matrix', fontsize=13)
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'hyperedge_correlation_matrix.png'), dpi=200)
        plt.close()
        
        # 6.3 按阶数分组的相关性分析
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for order in range(2, 8):
            ax = axes[order - 2]
            indices = [idx for idx, (o, e) in enumerate(edge_info) if o == order]
            if len(indices) > 1:
                sub_gram = gram_matrix[np.ix_(indices, indices)]
                im = ax.imshow(sub_gram, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax.set_title(f'Order {order} Correlation ({len(indices)} edges)', fontsize=11)
                ax.set_xlabel('Edge Index (within order)')
                ax.set_ylabel('Edge Index (within order)')
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_by_order.png'), dpi=200)
        plt.close()
    
    # 返回分析结果
    analysis = {
        'singular_values': S,
        'effective_rank': effective_rank,
        'total_edges': E,
        'redundant_count': E - effective_rank,
        'gram_matrix': gram_matrix,
        'highly_correlated_pairs': highly_correlated_pairs,
        'redundant_edges': redundant_edges,
        'edge_vectors_normalized': edge_vectors_normalized
    }
    
    return analysis


if __name__ == "__main__":
    # 参数设置
    N = 8
    max_order = 7
    M = 300
    tmax = 20
    
    # 构建三维张量
    tensor, edge_info, t_eval, X = build_hyperedge_tensor(N, max_order, M, tmax)
    
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', f'hyperedge_tensor_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存数据
    print(f"\n保存数据到 {results_dir}...")
    np.save(os.path.join(results_dir, 'hyperedge_tensor.npy'), tensor)
    np.save(os.path.join(results_dir, 'time_points.npy'), t_eval)
    np.save(os.path.join(results_dir, 'trajectories.npy'), X)
    
    # 保存超边信息
    with open(os.path.join(results_dir, 'edge_info.txt'), 'w') as f:
        f.write("Edge_Index\tOrder\tEdge\n")
        for idx, (order, edge) in enumerate(edge_info):
            f.write(f"{idx}\t{order}\t{edge}\n")
    
    # 可视化：对每个节点，展示所有超边的耦合强度
    print("生成可视化...")
    
    # 1. 针对每个节点的热图
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    
    for node_idx in range(N):
        ax = axes[node_idx]
        # tensor[:, :, node_idx] 是 (时间, 超边) 的矩阵
        node_matrix = tensor[:, :, node_idx]
        im = ax.imshow(node_matrix.T, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                      vmin=-np.abs(tensor).max(), vmax=np.abs(tensor).max())
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Hyperedge Index', fontsize=10)
        ax.set_title(f'Node {node_idx + 1} Coupling', fontsize=11)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hyperedge_tensor_by_node.png'), dpi=200)
    plt.close()
    
    # 2. 按超边阶数分组，展示所有节点的耦合强度总和
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for order in range(2, max_order + 1):
        ax = axes[order - 2]
        # 获取该阶的所有超边索引
        indices = [idx for idx, (o, e) in enumerate(edge_info) if o == order]
        if len(indices) > 0:
            # 对所有节点求和: shape (time, edges)
            sub_tensor = tensor[:, indices, :].sum(axis=2)
            im = ax.imshow(sub_tensor.T, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'{order}-edge Index')
            ax.set_title(f'Order {order} Total Coupling ({len(indices)} edges)')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hyperedge_tensor_by_order.png'), dpi=200)
    plt.close()
    
    # 统计信息
    print("\n张量统计信息:")
    print(f"  形状: {tensor.shape} (时间, 超边, 节点)")
    print(f"  最小值: {tensor.min():.6f}")
    print(f"  最大值: {tensor.max():.6f}")
    print(f"  均值: {tensor.mean():.6f}")
    print(f"  标准差: {tensor.std():.6f}")
    
    # 计算每个超边的时空平均强度（对时间和节点取平均）
    mean_coupling = np.abs(tensor).mean(axis=(0, 2))
    top_k = 20
    top_indices = np.argsort(mean_coupling)[-top_k:][::-1]
    
    print(f"\n平均耦合强度最大的前 {top_k} 个超边:")
    for rank, idx in enumerate(top_indices, 1):
        order, edge = edge_info[idx]
        print(f"  {rank}. Order={order}, Edge={edge}, Avg |Coupling|={mean_coupling[idx]:.6f}")
    
    # 输出张量使用示例
    print("\n\n=== 张量使用示例 ===")
    print(f"tensor[t, e, n] 表示:")
    print(f"  - 在时间点 t={t_eval[0]:.2f}s")
    print(f"  - 超边 e={edge_info[0]}")
    print(f"  - 对节点 n=1 的耦合贡献")
    print(f"  - 值为: {tensor[0, 0, 0]:.6f}")
    print(f"\n访问示例:")
    print(f"  tensor[100, 5, 3]  # 第100个时间点，第5个超边对第3个节点的贡献")
    print(f"  tensor[:, 10, :]   # 第10个超边对所有节点在所有时间的贡献矩阵 (时间×节点)")
    print(f"  tensor[50, :, 2]   # 第50个时间点，所有超边对第2个节点的贡献向量")
    
    # === 不可分辨性分析 ===
    print("\n\n" + "="*80)
    analysis = analyze_hyperedge_indistinguishability(tensor, edge_info, results_dir)
    
    # 保存分析结果
    np.save(os.path.join(results_dir, 'singular_values.npy'), analysis['singular_values'])
    np.save(os.path.join(results_dir, 'gram_matrix.npy'), analysis['gram_matrix'])
    
    with open(os.path.join(results_dir, 'indistinguishability_analysis.txt'), 'w') as f:
        f.write("=== 超边不可分辨性分析报告 ===\n\n")
        f.write(f"总超边数: {analysis['total_edges']}\n")
        f.write(f"有效秩: {analysis['effective_rank']}\n")
        f.write(f"冗余超边数: {analysis['redundant_count']}\n")
        f.write(f"信息压缩率: {100*(1 - analysis['effective_rank']/analysis['total_edges']):.2f}%\n\n")
        
        f.write("\n高度相关的超边对 (前30):\n")
        for i, j, corr in analysis['highly_correlated_pairs'][:30]:
            order_i, edge_i = edge_info[i]
            order_j, edge_j = edge_info[j]
            f.write(f"  Edge {i} (order={order_i}, {edge_i}) <-> Edge {j} (order={order_j}, {edge_j}): ρ={corr:.4f}\n")
        
        f.write("\n\n冗余超边 (可被线性表示，前30):\n")
        for target_idx, r_sq, contributors in analysis['redundant_edges'][:30]:
            order_t, edge_t = edge_info[target_idx]
            f.write(f"\nEdge {target_idx} (order={order_t}, {edge_t}): R²={r_sq:.4f}\n")
            f.write(f"  线性组合:\n")
            for contrib_idx, coeff in contributors[:5]:
                order_c, edge_c = edge_info[contrib_idx]
                f.write(f"    {coeff:+.3f} × Edge {contrib_idx} (order={order_c}, {edge_c})\n")
    
    print(f"\n所有结果已保存到: {results_dir}")
