"""
绘制 AUC 随 max_order 变化的曲线
对比 N=8 和 N=16 两种情况
"""

import matplotlib.pyplot as plt
import numpy as np

# 数据整理
# max_order = 2, 3, 4, 5, 6, 7
# 每个 max_order 下各阶超边的 AUC

# N=8 数据
data_N8 = {
    2: {
        2: 1.0000
    },
    3: {
        2: 0.9924,
        3: 0.9471
    },
    4: {
        2: 1.0000,
        3: 0.8558,
        4: 0.3043
    },
    5: {
        2: 1.0000,
        3: 0.7019,
        4: 0.9130,
        5: 1.0000
    },
    6: {
        2: 1.0000,
        3: 0.9087,
        4: 0.5362,
        5: 0.9455,
        6: 0.8889
    },
    7: {
        2: 0.9545,
        3: 0.8173,
        4: 0.4203,
        5: 0.8364,
        6: 1.0000,
        7: 0.7143
    }
}

# N=16 数据
data_N16 = {
    2: {
        2: 0.9978
    },
    3: {
        2: 0.9955,
        3: 0.9952
    },
    4: {
        2: 0.9911,
        3: 0.9785,
        4: 0.7668
    },
    5: {
        2: 0.9955,
        3: 0.9108,
        4: 0.3358,
        5: 0.9938
    },
    6: {
        2: 0.9699,
        3: 0.6990,
        4: 0.2016,
        5: 0.8496,
        6: 0.7161
    },
    7: {
        2: 0.9888,
        3: 0.9988,
        4: 0.2792,
        5: 0.7264,
        6: 0.8443,
        7: 0.6668
    }
}

# N=10 数据
data_N10 = {
    2: {
        2: 1.0000
    },
    3: {
        2: 1.0000,
        3: 0.6752
    },
    4: {
        2: 1.0000,
        3: 0.9915,
        4: 0.3798
    },
    5: {
        2: 1.0000,
        3: 0.9858,
        4: 0.5337,
        5: 1.0000
    },
    6: {
        2: 0.9900,
        3: 0.9174,
        4: 0.3894,
        5: 0.9841,
        6: 1.0000
    },
    7: {
        2: 0.9950,
        3: 0.8319,
        4: 0.4279,
        5: 0.9960,
        6: 0.9952,
        7: 0.8487
    }
}

# N=12 数据
data_N12 = {
    2: {
        2: 1.0000
    },
    3: {
        2: 1.0000,
        3: 0.9954
    },
    4: {
        2: 0.9833,
        3: 1.0000,
        4: 0.9645
    },
    5: {
        2: 1.0000,
        3: 0.9739,
        4: 0.5629,
        5: 0.9987
    },
    6: {
        2: 0.9972,
        3: 0.9339,
        4: 0.4777,
        5: 0.7623,
        6: 0.9393
    },
    7: {
        2: 1.0000,
        3: 0.7711,
        4: 0.5761,
        5: 0.9343,
        6: 0.5179,
        7: 0.9292
    }
}

# N=14 数据
data_N14 = {
    2: {
        2: 1.0000
    },
    3: {
        2: 0.9201,
        3: 0.9861
    },
    4: {
        2: 0.9779,
        3: 0.9935,
        4: 0.5405
    },
    5: {
        2: 0.9745,
        3: 0.8366,
        4: 0.4289,
        5: 0.9595
    },
    6: {
        2: 0.9660,
        3: 0.8079,
        4: 0.5546,
        5: 0.3888,
        6: 0.3921
    },
    7: {
        2: 0.9966,
        3: 0.7802,
        4: 0.5115,
        5: 0.9305,
        6: 0.6779,
        7: 0.7665
    }
}

max_orders = [2, 3, 4, 5, 6, 7]
edge_orders = [2, 3, 4, 5, 6, 7]

# 为每个 edge_order 构建曲线
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'v', 'D', 'p']

# 为每个 N 单独生成图
for data, N_val in [(data_N8, 8), (data_N10, 10), (data_N12, 12), (data_N14, 14), (data_N16, 16)]:
    plt.figure(figsize=(10, 7))
    
    for i, edge_order in enumerate(edge_orders):
        x_vals = []
        y_vals = []
        
        for max_order in max_orders:
            if max_order >= edge_order and edge_order in data[max_order]:
                x_vals.append(max_order)
                y_vals.append(data[max_order][edge_order])
        
        if len(x_vals) > 0:
            plt.plot(x_vals, y_vals, 
                    marker=markers[i], 
                    color=colors[i],
                    linewidth=2.5,
                    markersize=14,
                    label=f'{edge_order}-edges',
                    alpha=0.8)
    
    plt.xlabel('Max Order', fontsize=18)
    plt.ylabel('AUC Score', fontsize=18)
    plt.title(f'AUC Performance vs Max Order for N={N_val} Rossler System', fontsize=18)
    plt.xticks(max_orders, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([0, 1.05])
    plt.xlim([1.8, 7.2])
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=16, loc='best', framealpha=0.9)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f'auc_vs_maxorder_N{N_val}.png', dpi=300, bbox_inches='tight')
    print(f"N={N_val} 图已保存为 auc_vs_maxorder_N{N_val}.png")
    plt.close()

print("\n所有图片生成完成！")