# Tucker-Parameterized Hypergraph Dynamics for Rossler HyperPINN

This note explains the mathematical logic behind the true Tucker version under `models/hyperpinn_tucker`.

The current implementation only supports:

- second-order interactions
- third-order interactions
- Rossler dynamics

The goal is the same as in the CP branch:

- do not sample candidate hyperedges inside the physics term
- let a low-rank model represent the complete second-order and third-order hypergraph weights
- rewrite the original dynamical couplings as exact analytic contractions

## 1. Original Rossler dynamics

For node $i$,

$$
\dot{x}_i = -y_i - z_i + k \, C_i^{(2)} + k_D \, C_i^{(3)},
$$

$$
\dot{y}_i = x_i + a_r y_i,
\qquad
\dot{z}_i = b_r + z_i (x_i - c_r).
$$

In the current code,

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

## 2. Dense hypergraph form before Tucker

### 2.1 Second-order term

Let $W^{(2)}_{ij}$ be the pairwise hypergraph weight. Then

$$
C_i^{(2)} = \sum_{j \ne i} W^{(2)}_{ij} (x_j - x_i).
$$

### 2.2 Third-order term

Let $W^{(3)}_{ijk}$ be the third-order hypergraph tensor. The dense baseline uses

$$
C_i^{(3)}
=
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
W^{(3)}_{ijk}
\left(x_j^2 x_k - x_i^3\right).
$$

Because both ordered pairs $(j,k)$ and $(k,j)$ appear, one unordered triangle $\{i,j,k\}$ contributes

$$
W^{(3)}_{ijk}
\left(x_j^2 x_k + x_j x_k^2 - 2x_i^3\right).
$$

## 3. Tucker parameterization of the complete hypergraph

The Tucker version does not store explicit edge-by-edge or triangle-by-triangle logits. Instead, it parameterizes the complete weight tensors through node factors and a small core tensor.

## 4. Second-order Tucker form

Let $U^{(2)} \in \mathbb{R}_{\ge 0}^{N \times R}$ and $G^{(2)} \in \mathbb{R}_{\ge 0}^{R \times R}$. Then

$$
W^{(2)}_{ij}
=
\sum_{a=1}^{R}\sum_{b=1}^{R}
G^{(2)}_{ab} U^{(2)}_{ia} U^{(2)}_{jb}.
$$

In the implementation, the core is symmetrized so that the pairwise weight is symmetric:

$$
G^{(2)} \leftarrow \frac{1}{2}\left(G^{(2)} + (G^{(2)})^\top\right).
$$

## 5. Third-order Tucker form

Let $U^{(3)} \in \mathbb{R}_{\ge 0}^{N \times R}$ and $G^{(3)} \in \mathbb{R}_{\ge 0}^{R \times R \times R}$. Then

$$
W^{(3)}_{ijk}
=
\sum_{a=1}^{R}\sum_{b=1}^{R}\sum_{c=1}^{R}
G^{(3)}_{abc} U^{(3)}_{ia} U^{(3)}_{jb} U^{(3)}_{kc}.
$$

Because the hypergraph is unordered, the core is symmetrized over all permutations:

$$
G^{(3)}_{\mathrm{sym}}
=
\frac{1}{6}
\sum_{\pi \in S_3}
\pi\left(G^{(3)}\right).
$$

This makes the induced triangle weights permutation-invariant.

## 6. Nonnegativity in code

As in the CP branch, the implementation stores raw unconstrained parameters and applies `softplus`:

$$
U^{(q)} = \operatorname{softplus}(A^{(q)}),
\qquad
G^{(q)} = \operatorname{softplus}(\widetilde{G}^{(q)}).
$$

So the represented hypergraph weights are nonnegative.

## 7. Exact second-order contraction under Tucker

Start from

$$
C_i^{(2)} = \sum_{j \ne i} W^{(2)}_{ij}(x_j - x_i).
$$

Substitute the Tucker form:

$$
C_i^{(2)}
=
\sum_{j \ne i}
\sum_{a,b}
G^{(2)}_{ab} U^{(2)}_{ia} U^{(2)}_{jb}(x_j - x_i).
$$

Rearrange:

$$
C_i^{(2)}
=
\sum_{a,b}
U^{(2)}_{ia} G^{(2)}_{ab}
\sum_{j \ne i} U^{(2)}_{jb}(x_j - x_i).
$$

Since the $j=i$ term is zero anyway,

$$
C_i^{(2)}
=
\sum_{a,b}
U^{(2)}_{ia} G^{(2)}_{ab}
\left(
\sum_j U^{(2)}_{jb}x_j
-
x_i \sum_j U^{(2)}_{jb}
\right).
$$

Define

$$
S_b^{(x)} = \sum_j U^{(2)}_{jb}x_j,
\qquad
S_b^{(1)} = \sum_j U^{(2)}_{jb}.
$$

Then

$$
C_i^{(2)}
=
\sum_{a,b}
U^{(2)}_{ia} G^{(2)}_{ab}
\left(S_b^{(x)} - x_i S_b^{(1)}\right).
$$

This is exact. No explicit $N \times N$ tensor is formed.

## 8. Exact third-order contraction under Tucker

Start from

$$
C_i^{(3)}
=
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
W^{(3)}_{ijk}
\left(x_j^2 x_k - x_i^3\right).
$$

Insert the Tucker form:

$$
C_i^{(3)}
=
\sum_{a,b,c}
U^{(3)}_{ia} G^{(3)}_{abc}
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
U^{(3)}_{jb} U^{(3)}_{kc}
\left(x_j^2 x_k - x_i^3\right).
$$

For one fixed target node $i$, define global statistics

$$
A_b = \sum_j U^{(3)}_{jb}x_j^2,
\qquad
B_c = \sum_j U^{(3)}_{jc}x_j,
\qquad
P_b = \sum_j U^{(3)}_{jb},
$$

$$
D_{bc} = \sum_j U^{(3)}_{jb}U^{(3)}_{jc}x_j^3,
\qquad
Q_{bc} = \sum_j U^{(3)}_{jb}U^{(3)}_{jc}.
$$

Exclude node $i$:

$$
\widetilde{A}_{ib} = A_b - U^{(3)}_{ib}x_i^2,
\qquad
\widetilde{B}_{ic} = B_c - U^{(3)}_{ic}x_i,
\qquad
\widetilde{P}_{ib} = P_b - U^{(3)}_{ib},
$$

$$
\widetilde{D}_{i,bc} = D_{bc} - U^{(3)}_{ib}U^{(3)}_{ic}x_i^3,
\qquad
\widetilde{Q}_{i,bc} = Q_{bc} - U^{(3)}_{ib}U^{(3)}_{ic}.
$$

Then the ordered distinct sum is

$$
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
U^{(3)}_{jb}U^{(3)}_{kc}x_j^2x_k
=
\widetilde{A}_{ib}\widetilde{B}_{ic} - \widetilde{D}_{i,bc},
$$

and the counting term is

$$
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
U^{(3)}_{jb}U^{(3)}_{kc}
=
\widetilde{P}_{ib}\widetilde{P}_{ic} - \widetilde{Q}_{i,bc}.
$$

Therefore,

$$
C_i^{(3)}
=
\sum_{a,b,c}
U^{(3)}_{ia} G^{(3)}_{abc}
\left[
\widetilde{A}_{ib}\widetilde{B}_{ic}
-
\widetilde{D}_{i,bc}
-
 x_i^3\left(\widetilde{P}_{ib}\widetilde{P}_{ic} - \widetilde{Q}_{i,bc}\right)
\right].
$$

This is again exact.

## 9. Final coupled Rossler dynamics under Tucker

After substitution, the physical model becomes

$$
\dot{x}_i
=
-y_i - z_i
+
k \sum_{a,b}
U^{(2)}_{ia} G^{(2)}_{ab}
\left(S_b^{(x)} - x_i S_b^{(1)}\right)
+
k_D \sum_{a,b,c}
U^{(3)}_{ia} G^{(3)}_{abc}
\left[
\widetilde{A}_{ib}\widetilde{B}_{ic}
-
\widetilde{D}_{i,bc}
-
 x_i^3\left(\widetilde{P}_{ib}\widetilde{P}_{ic} - \widetilde{Q}_{i,bc}\right)
\right],
$$

$$
\dot{y}_i = x_i + a_r y_i,
\qquad
\dot{z}_i = b_r + z_i (x_i - c_r).
$$

The meaning of the dynamics has not changed. Only the representation of the unknown second-order and third-order hypergraph tensors has changed.

## 10. Physics loss

The neural network predicts trajectories

$$
\hat{x}_i(t), \hat{y}_i(t), \hat{z}_i(t),
$$

and automatic differentiation provides the time derivatives. The physics loss is

$$
\mathcal{L}_{\mathrm{phys}}
=
\frac{1}{T}
\sum_t
\left\|
\partial_t \hat{\mathbf{s}}(t) - \mathbf{f}_{\mathrm{Tucker}}(\hat{\mathbf{s}}(t))
\right\|_2^2.
$$

## 11. Regularization in the Tucker branch

The current Tucker branch regularizes the total induced mass of the complete second-order and third-order hypergraphs, plus a small L2 penalty on the raw Tucker parameters.

### 11.1 Second-order total mass

$$
\sum_{i<j} W^{(2)}_{ij}
=
\frac{1}{2}
\sum_{a,b}
G^{(2)}_{ab}
\left[
\left(\sum_i U^{(2)}_{ia}\right)
\left(\sum_i U^{(2)}_{ib}\right)
-
\sum_i U^{(2)}_{ia}U^{(2)}_{ib}
\right].
$$

### 11.2 Third-order total mass

$$
\sum_{i<j<k} W^{(3)}_{ijk}
=
\frac{1}{6}
\sum_{a,b,c}
G^{(3)}_{abc}
\left[
S_a S_b S_c
-
S_{ab}S_c
-
S_{ac}S_b
-
S_{bc}S_a
+
2S_{abc}
\right],
$$

where

$$
S_a = \sum_i U^{(3)}_{ia},
\qquad
S_{ab} = \sum_i U^{(3)}_{ia}U^{(3)}_{ib},
\qquad
S_{abc} = \sum_i U^{(3)}_{ia}U^{(3)}_{ib}U^{(3)}_{ic}.
$$

### 11.3 Small factor penalty

$$
\mathcal{L}_{\mathrm{factor}}
=
\lambda_{\mathrm{factor}}
\left(
\|A^{(2)}\|_F^2 + \|\widetilde{G}^{(2)}\|_F^2
+
\|A^{(3)}\|_F^2 + \|\widetilde{G}^{(3)}\|_F^2
\right).
$$

## 12. Tucker versus CP

The CP branch writes the full tensor as a sum of rank-1 components:

$$
W^{(3)}_{ijk} = \sum_r \lambda_r u_{ir}u_{jr}u_{kr}.
$$

The Tucker branch writes the full tensor as latent projections plus a core interaction tensor:

$$
W^{(3)}_{ijk} = \sum_{a,b,c} G_{abc}U_{ia}U_{jb}U_{kc}.
$$

So compared with CP:

- Tucker is more flexible because the core can mix latent directions
- Tucker is more expensive because the core has size $R^2$ for order 2 and $R^3$ for order 3
- Tucker still supports exact analytic contractions, so it remains faithful to the original dynamics

That is the mathematical logic behind the current true Tucker implementation.
