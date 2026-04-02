# Rossler HyperPINN 的归纳式张量补全超图动力学

本文档固定 `models/hyperpinn_ITC` 下 ITC 分支所期望采用的数学定义。

关键点在于，归纳式张量补全模型不能为每一个候选超边都分配一个独立的自由参数。如果这么做，它就只是一个稠密的 transductive 超图模型。

ITC 分支应该改为：

- 从已观测到的节点层面信息中，将每个节点编码为一个潜在表示
- 由这些节点表示生成完整的二阶和三阶超图权重
- 通过共享的解码器参数而不是边特定的 logit 来推断未观测到的张量条目
- 将这一归纳式补全模型与 Rossler 物理损失耦合起来

对于第一版正确的 ITC 实现，推荐范围是：

- 二阶相互作用
- 三阶相互作用
- Rossler 动力学

这与 CP、Tucker 和 TT 分支当前已经采用的范围是一致的。

## 1. 原始 Rossler 动力学

对于节点 $i$，

$$
\dot{x}_i = -y_i - z_i + k \, C_i^{(2)} + k_D \, C_i^{(3)},
$$

$$
\dot{y}_i = x_i + a_r y_i,
\qquad
\dot{z}_i = b_r + z_i (x_i - c_r).
$$

在当前代码库中，

$$
a_r = 0.2,
\qquad
b_r = 0.2,
\qquad
c_r = 0.7,
\qquad
k = 0.4,
\qquad
k_D = 0.3.
$$

ITC 改变的唯一部分，是未知超图权重的表示方式。

## 2. 为什么当前的 dense scaffold 不是 ITC

假设我们为每一个无序的 pair、每一个无序的 triangle，乃至更高阶组合，都存储一个可学习标量。那么模型实际上就是在直接记忆超图张量的各个条目：

$$
W^{(2)}_{ij},
\qquad
W^{(3)}_{ijk},
\qquad \dots
$$

这是 transductive 的。它只能为训练时出现的那一组精确节点学习权重。

这并不是归纳式张量补全。

归纳式模型必须改为定义

$$
W^{(2)}_{ij} = f_2(c_i, c_j),
\qquad
W^{(3)}_{ijk} = f_3(c_i, c_j, c_k),
$$

其中 $c_i$ 是节点侧的观测信息，$f_2, f_3$ 是共享函数。这样，缺失的张量条目就由节点信息来预测，而不是被单独记忆成独立参数。

## 3. 节点描述符与编码器

对于每个节点 $i$，令

$$
c_i \in \mathbb{R}^{d_c}
$$

表示它的观测描述符。

在这个项目中，自然的选择是从节点 $i$ 的观测 Rossler 轨迹构造 $c_i$，例如来自：

- 采样得到的时间序列 $x_i(t), y_i(t), z_i(t)$
- 该轨迹的一些简单统计量
- 或者一个作用于整条轨迹上的小型可学习编码器

随后我们使用共享编码器

$$
h_i = E_\theta(c_i) \in \mathbb{R}^{d_h}
$$

将每个节点映射到一个潜在嵌入。

由于同一个编码器 $E_\theta$ 会被复用于所有节点，只要给出其描述符，模型就能对新节点进行评分。

这正是归纳性的来源。

## 4. 完整超图的归纳式 CP 风格参数化

对于这个代码库，最简单且正确的 ITC 模型是一个归纳式的 CP 风格解码器。

它不再直接学习自由的节点因子，而是从节点潜在嵌入中生成这些因子。

### 4.1 二阶因子

对于秩 $R_2$，定义

$$
u_i^{(2)} = \operatorname{softplus}(A^{(2)} h_i + b^{(2)}) \in \mathbb{R}_{\ge 0}^{R_2}
$$

以及分量权重

$$
\lambda^{(2)} = \operatorname{softplus}(\widetilde{\lambda}^{(2)}) \in \mathbb{R}_{\ge 0}^{R_2}.
$$

于是，完整的 pairwise 超图权重为

$$
W^{(2)}_{ij}
=
\sum_{r=1}^{R_2}
\lambda_r^{(2)} u_{ir}^{(2)} u_{jr}^{(2)}.
$$

### 4.2 三阶因子

对于秩 $R_3$，定义

$$
u_i^{(3)} = \operatorname{softplus}(A^{(3)} h_i + b^{(3)}) \in \mathbb{R}_{\ge 0}^{R_3}
$$

以及

$$
\lambda^{(3)} = \operatorname{softplus}(\widetilde{\lambda}^{(3)}) \in \mathbb{R}_{\ge 0}^{R_3}.
$$

于是，完整的三阶超图张量为

$$
W^{(3)}_{ijk}
=
\sum_{r=1}^{R_3}
\lambda_r^{(3)} u_{ir}^{(3)} u_{jr}^{(3)} u_{kr}^{(3)}.
$$

该张量在 $(i,j,k)$ 上会自动保持对称。

### 4.3 为什么这是 completion 而不是显式枚举

模型参数包括：

- 编码器参数 $\theta$
- 投影头 $(A^{(2)}, b^{(2)})$、$(A^{(3)}, b^{(3)})$
- 低秩分量权重 $\lambda^{(2)}, \lambda^{(3)}$

模型参数并不是“每条超边一个标量”。

因此，完整张量是通过共享的、节点条件化的因子被隐式表示出来的。

## 5. 完整超图的解释

解码器为所有节点元组定义了完整的 pairwise 与 third-order 张量：

$$
W^{(2)} \in \mathbb{R}^{N \times N},
\qquad
W^{(3)} \in \mathbb{R}^{N \times N \times N}.
$$

这些张量与 CP、Tucker、TT 分支中的“完整”含义一致：

- 它们表示所有候选的 2-edge 和 3-edge
- 它们不是在 physics term 内部基于采样候选超边来定义的
- 对任何未观测到的 pair 或 triangle，它们都可以被查询

唯一的区别只是，这里的张量条目是由节点描述符生成的。

## 6. 归纳式 CP 下的精确二阶收缩

pairwise Rossler 耦合项仍然是

$$
C_i^{(2)} = \sum_{j \ne i} W^{(2)}_{ij}(x_j - x_i).
$$

代入归纳式 CP 形式：

$$
C_i^{(2)}
=
\sum_{j \ne i}
\sum_{r=1}^{R_2}
\lambda_r^{(2)} u_{ir}^{(2)} u_{jr}^{(2)} (x_j - x_i).
$$

交换求和顺序：

$$
C_i^{(2)}
=
\sum_{r=1}^{R_2}
\lambda_r^{(2)} u_{ir}^{(2)}
\sum_{j \ne i} u_{jr}^{(2)} (x_j - x_i).
$$

由于 $j=i$ 时那一项本来就是零，因此可写成

$$
C_i^{(2)}
=
\sum_{r=1}^{R_2}
\lambda_r^{(2)} u_{ir}^{(2)}
\left(
\sum_{j=1}^{N} u_{jr}^{(2)} x_j
-
x_i \sum_{j=1}^{N} u_{jr}^{(2)}
\right).
$$

定义

$$
S_r^{(x)} = \sum_{j=1}^{N} u_{jr}^{(2)} x_j,
\qquad
S_r^{(1)} = \sum_{j=1}^{N} u_{jr}^{(2)}.
$$

则有

$$
C_i^{(2)}
=
\sum_{r=1}^{R_2}
\lambda_r^{(2)} u_{ir}^{(2)}
\left(S_r^{(x)} - x_i S_r^{(1)}\right).
$$

因此，pairwise 项可以在不显式构造稠密 $N \times N$ 张量的情况下被精确计算出来。

## 7. 归纳式 CP 下的精确三阶收缩

third-order Rossler 耦合项仍然是

$$
C_i^{(3)}
=
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
W^{(3)}_{ijk}
\left(x_j^2 x_k - x_i^3\right).
$$

代入归纳式 CP 形式：

$$
C_i^{(3)}
=
\sum_{r=1}^{R_3}
\lambda_r^{(3)} u_{ir}^{(3)}
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
u_{jr}^{(3)} u_{kr}^{(3)}
\left(x_j^2 x_k - x_i^3\right).
$$

对于某个固定秩 $r$，定义

$$
A_r = \sum_{j=1}^{N} u_{jr}^{(3)} x_j^2,
\qquad
B_r = \sum_{j=1}^{N} u_{jr}^{(3)} x_j,
\qquad
C_r = \sum_{j=1}^{N} u_{jr}^{(3)},
$$

$$
D_r = \sum_{j=1}^{N} (u_{jr}^{(3)})^2 x_j^3,
\qquad
E_r = \sum_{j=1}^{N} (u_{jr}^{(3)})^2.
$$

把目标节点 $i$ 排除后：

$$
\widetilde{A}_{ir} = A_r - u_{ir}^{(3)} x_i^2,
\qquad
\widetilde{B}_{ir} = B_r - u_{ir}^{(3)} x_i,
\qquad
\widetilde{C}_{ir} = C_r - u_{ir}^{(3)},
$$

$$
\widetilde{D}_{ir} = D_r - (u_{ir}^{(3)})^2 x_i^3,
\qquad
\widetilde{E}_{ir} = E_r - (u_{ir}^{(3)})^2.
$$

则有

$$
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
u_{jr}^{(3)}u_{kr}^{(3)}x_j^2x_k
=
\widetilde{A}_{ir}\widetilde{B}_{ir} - \widetilde{D}_{ir},
$$

并且

$$
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
u_{jr}^{(3)}u_{kr}^{(3)}
=
\widetilde{C}_{ir}^2 - \widetilde{E}_{ir}.
$$

因此

$$
C_i^{(3)}
=
\sum_{r=1}^{R_3}
\lambda_r^{(3)} u_{ir}^{(3)}
\left[
\widetilde{A}_{ir}\widetilde{B}_{ir}
-
\widetilde{D}_{ir}
-
x_i^3\left(\widetilde{C}_{ir}^2 - \widetilde{E}_{ir}\right)
\right].
$$

这与非归纳式 CP 分支中的解析收缩是完全相同的。唯一的区别只是，这里的因子 $u_i^{(2)}, u_i^{(3)}$ 是由编码器生成的，而不是自由的节点参数。

## 8. Completion 目标

为了让这个分支真正成为一个张量补全模型，我们必须在一个“部分观测”的超图上训练它。

令

$$
\Omega^{(2)} \subseteq \{(i,j): i < j\},
\qquad
\Omega^{(3)} \subseteq \{(i,j,k): i < j < k\}
$$

分别表示 2-edge 张量和 3-edge 张量中被观测到的条目。

对于观测条目 $e$，令 $y_e \in \{0,1\}$ 表示该超边在真实图中是否存在。

定义 logit

$$
s_e^{(2)} = \operatorname{logit}\bigl(p_e^{(2)}\bigr),
\qquad
p_e^{(2)} = \sigma\bigl(W_e^{(2)}\bigr),
$$

$$
s_e^{(3)} = \operatorname{logit}\bigl(p_e^{(3)}\bigr),
\qquad
p_e^{(3)} = \sigma\bigl(W_e^{(3)}\bigr).
$$

那么，一个自然的补全损失就是对观测条目做二元交叉熵：

$$
\mathcal{L}_{\mathrm{comp}}^{(2)}
=
\frac{1}{|\Omega^{(2)}|}
\sum_{e \in \Omega^{(2)}}
\operatorname{BCE}\bigl(s_e^{(2)}, y_e\bigr),
$$

$$
\mathcal{L}_{\mathrm{comp}}^{(3)}
=
\frac{1}{|\Omega^{(3)}|}
\sum_{e \in \Omega^{(3)}}
\operatorname{BCE}\bigl(s_e^{(3)}, y_e\bigr).
$$

总的补全损失为

$$
\mathcal{L}_{\mathrm{comp}}
=
\alpha_2 \mathcal{L}_{\mathrm{comp}}^{(2)}
+
\alpha_3 \mathcal{L}_{\mathrm{comp}}^{(3)}.
$$

在实际实现中，$\Omega^{(2)}$ 和 $\Omega^{(3)}$ 应同时包含观测到的正样本和采样得到的负样本。

## 9. 数据损失与物理损失

与其他 HyperPINN 分支一样，动力学网络根据时间来预测节点轨迹：

$$
\widehat{X}(t) = F_\phi(t).
$$

如果 $X(t)$ 是观测到的轨迹，则数据损失为

$$
\mathcal{L}_{\mathrm{data}}
=
\frac{1}{T}
\sum_t \|\widehat{X}(t) - X(t)\|_2^2.
$$

物理损失是在将 ITC 诱导出的耦合项 $C_i^{(2)}$ 和 $C_i^{(3)}$ 代入之后，对 Rossler 残差进行计算得到的。

令残差为

$$
R_{x,i}(t),
\qquad
R_{y,i}(t),
\qquad
R_{z,i}(t).
$$

则有

$$
\mathcal{L}_{\mathrm{phys}}
=
\frac{1}{TN}
\sum_{t,i}
\left(
R_{x,i}(t)^2 + R_{y,i}(t)^2 + R_{z,i}(t)^2
\right).
$$

## 10. 稀疏性与低秩正则化

ITC 分支也应该偏好稀疏的有效超图。

一个简单做法，是惩罚所有候选超边上的平均预测占用率：

$$
\mathcal{L}_{\mathrm{sparse}}^{(2)}
=
\frac{1}{|\mathcal{E}_2|}
\sum_{i<j} \sigma\bigl(W^{(2)}_{ij}\bigr),
$$

$$
\mathcal{L}_{\mathrm{sparse}}^{(3)}
=
\frac{1}{|\mathcal{E}_3|}
\sum_{i<j<k} \sigma\bigl(W^{(3)}_{ijk}\bigr).
$$

我们也可以对解码器和编码器加入标准的小型 L2 正则。

因此

$$
\mathcal{L}_{\mathrm{reg}}
=
\beta_2 \mathcal{L}_{\mathrm{sparse}}^{(2)}
+
\beta_3 \mathcal{L}_{\mathrm{sparse}}^{(3)}
+
\gamma \left(\|\theta\|_2^2 + \|A^{(2)}\|_2^2 + \|A^{(3)}\|_2^2\right).
$$

## 11. 最终训练目标

推荐的总目标函数是

$$
\mathcal{L}
=
\lambda_{\mathrm{data}} \mathcal{L}_{\mathrm{data}}
+
\lambda_{\mathrm{phys}} \mathcal{L}_{\mathrm{phys}}
+
\lambda_{\mathrm{comp}} \mathcal{L}_{\mathrm{comp}}
+
\lambda_{\mathrm{reg}} \mathcal{L}_{\mathrm{reg}}.
$$

它明确地区分了四个角色：

- 拟合观测到的轨迹
- 满足 Rossler 动力学
- 补全缺失的超图条目
- 避免退化为稠密的平凡解

## 12. 推荐的分阶段训练日程

对于这个 ITC 分支，最稳妥的训练日程是：

1. stage 1：只拟合观测到的轨迹
2. stage 2：打开 completion loss，同时让 physics 保持较弱或关闭
3. stage 3：打开完整的 physics loss 和 sparsity regularization

用符号表示就是：

- early：$\lambda_{\mathrm{data}} > 0$，其他项接近零
- middle：逐步增大 $\lambda_{\mathrm{comp}}$
- late：逐步增大 $\lambda_{\mathrm{phys}}$ 和 $\lambda_{\mathrm{reg}}$

这比一开始就强行压 physics 更稳定，因为节点编码器必须先学到有用的潜在表示。

## 13. 什么使这个模型具有归纳性

这个模型是归纳式的，因为：

- 超边权重是通过共享函数由节点描述符生成的
- 相同的编码器与解码器会被复用于每个节点和每个元组
- 缺失的张量条目是由节点信息推断出来的，而不是被直接记忆

如果之后观测到了新节点，并给出了它们的描述符 $c_i$，那么模型可以构造新的嵌入 $h_i$，并对这些节点的 pairwise 与 third-order 相互作用进行评分，而不需要再引入新的边特定参数。

这正是 ITC 与 dense baseline 之间的定义性区别。

## 14. 总结

对于这个仓库，推荐的 ITC 形式是：

- 从观测到的轨迹侧信息中编码每个节点
- 生成节点条件化的低秩因子
- 通过归纳式 CP 风格解码器表示完整的 2-edge 和 3-edge 张量
- 用精确的解析收缩来计算 Rossler 耦合项
- 联合使用 completion、data、physics 和 sparsity 损失进行训练

因此，ITC 分支不应该被理解为“稠密超图权重换了一个名字”。

它应该被理解为一个嵌入在 HyperPINN 物理目标中的、归纳式的、节点条件化的、低秩张量补全模型。