# Neuronal Synchronization: Simplicial Complex Modeling

该目录实现了你给出的 `Neuronal Synchronization.pdf` 中 2-simplex / 1-simplex 双层模型（Eq. (1), Eq. (2)）的可运行建模。

## 模型结构（对应图示）

- 上层 `theta`：**全局 2-simplex**（三体相互作用）
  - 0-simplex: 节点（振子）
  - 1-simplex: 完全图边
  - 2-simplex: 所有三角形（complete 2-complex）
- 下层 `phi`：**全局 1-simplex**（Kuramoto）
- 跨层驱动：节点一一对应，`d * sin(theta_i - phi_i)`

动力学方程：

- `theta_dot_i = omega_i + (K/N^2) * sum_{j,k} sin(theta_j + theta_k - 2*theta_i)`
- `phi_dot_i = nu_i + (kappa/N) * sum_j sin(phi_j - phi_i) + d*sin(theta_i - phi_i)`

脚本中也使用了等价的序参量形式：

- `z1 = (1/N) * sum_j exp(i*theta_j)`
- `(K/N^2) * sum_{j,k} sin(theta_j + theta_k - 2*theta_i) = K * Im(z1^2 * exp(-2i*theta_i))`

## 文件

- `simplicial_model.py`：建模、仿真、绘图一体化脚本。

## 运行

在仓库根目录执行：

```bash
python Exp_Neuronal_Synchronization/simplicial_model.py
```

输出：

- `simplicial_complex_structure.png`：单纯复形示意图
- `neuronal_sync_dynamics.png`：同步序参量随时间变化
