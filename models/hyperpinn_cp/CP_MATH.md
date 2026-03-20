# CP-Parameterized Hypergraph Dynamics for Rossler HyperPINN

This note explains what changes mathematically after replacing the explicit hypergraph tensors with a CP decomposition in the CP branch under `models/hyperpinn_cp`.

The current implementation only supports:

- second-order interactions
- third-order interactions
- Rossler dynamics

The goal is not to score sampled candidate hyperedges. The goal is to let the CP factors represent the complete second-order and third-order hypergraph weights directly, and then rewrite the dynamical coupling terms as exact analytic contractions.

## 1. Original Rossler dynamics

For node $i$, the uncoupled Rossler system is

$$
\dot{x}_i = -y_i - z_i,
\qquad
\dot{y}_i = x_i + a_r y_i,
\qquad
\dot{z}_i = b_r + z_i (x_i - c_r).
$$

In this project, the coupled dynamics used by the dense HyperPINN baseline are

$$
\dot{x}_i = -y_i - z_i + k \, C_i^{(2)} + k_D \, C_i^{(3)},
$$

$$
\dot{y}_i = x_i + a_r y_i,
\qquad
\dot{z}_i = b_r + z_i (x_i - c_r).
$$

In the current code, the constants are

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

## 2. Dense hypergraph form before CP

### 2.1 Second-order term

Let $W^{(2)} \in \mathbb{R}^{N \times N}$ denote the pairwise hypergraph weight matrix. The dense pairwise coupling is

$$
C_i^{(2)} = \sum_{j \ne i} W^{(2)}_{ij} (x_j - x_i).
$$

This is the standard diffusive pairwise term.

### 2.2 Third-order term

Let $W^{(3)} \in \mathbb{R}^{N \times N \times N}$ denote the third-order hypergraph tensor. In the dense baseline, each unordered triangle $\{i,j,k\}$ contributes a symmetrized nonlinear term.

For node $i$, the third-order coupling can be written as

$$
C_i^{(3)}
=
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
W^{(3)}_{ijk}
\left(x_j^2 x_k - x_i^3\right).
$$

Because the sum runs over ordered pairs $(j,k)$ with $j \ne k$, the two permutations $(j,k)$ and $(k,j)$ are both included. Therefore, for one unordered triangle $\{i,j,k\}$ the contribution to node $i$ is

$$
W^{(3)}_{ijk}
\left(x_j^2 x_k + x_j x_k^2 - 2 x_i^3\right),
$$

which matches the dense implementation.

## 3. Why the dense form is expensive

If we represent the full hypergraph explicitly, then:

- second-order needs $O(N^2)$ weights
- third-order needs $O(N^3)$ weights

This becomes the bottleneck. The CP branch replaces these explicit tensors by low-rank factors while keeping the dynamical formula exact.

## 4. CP parameterization of the complete hypergraph

For each order, we introduce a separate nonnegative CP decomposition.

### 4.1 Second-order CP form

For rank $R$, define node factors $u^{(2)} \in \mathbb{R}_{\ge 0}^{N \times R}$ and component weights $\lambda^{(2)} \in \mathbb{R}_{\ge 0}^{R}$. Then

$$
W^{(2)}_{ij}
=
\sum_{r=1}^{R}
\lambda_r^{(2)} u_{ir}^{(2)} u_{jr}^{(2)}.
$$

### 4.2 Third-order CP form

Similarly, with factors $u^{(3)} \in \mathbb{R}_{\ge 0}^{N \times R}$ and weights $\lambda^{(3)} \in \mathbb{R}_{\ge 0}^{R}$,

$$
W^{(3)}_{ijk}
=
\sum_{r=1}^{R}
\lambda_r^{(3)} u_{ir}^{(3)} u_{jr}^{(3)} u_{kr}^{(3)}.
$$

### 4.3 Nonnegativity in code

The implementation stores unconstrained raw parameters and applies `softplus`:

$$
u_{ir}^{(q)} = \operatorname{softplus}(\alpha_{ir}^{(q)}),
\qquad
\lambda_r^{(q)} = \operatorname{softplus}(\beta_r^{(q)}).
$$

This ensures the represented hypergraph weights are nonnegative.

## 5. Exact second-order contraction after CP

Start from

$$
C_i^{(2)} = \sum_{j \ne i} W^{(2)}_{ij}(x_j - x_i).
$$

Substitute the CP form:

$$
C_i^{(2)}
=
\sum_{j \ne i}
\sum_{r=1}^{R}
\lambda_r^{(2)} u_{ir}^{(2)} u_{jr}^{(2)} (x_j - x_i).
$$

Swap the sums:

$$
C_i^{(2)}
=
\sum_{r=1}^{R}
\lambda_r^{(2)} u_{ir}^{(2)}
\sum_{j \ne i} u_{jr}^{(2)} (x_j - x_i).
$$

Since the $j=i$ term is zero anyway, this can be written as

$$
C_i^{(2)}
=
\sum_{r=1}^{R}
\lambda_r^{(2)} u_{ir}^{(2)}
\left(
\sum_{j=1}^{N} u_{jr}^{(2)} x_j
-
x_i \sum_{j=1}^{N} u_{jr}^{(2)}
\right).
$$

Define the rankwise statistics

$$
S_r^{(x)} = \sum_{j=1}^{N} u_{jr}^{(2)} x_j,
\qquad
S_r^{(1)} = \sum_{j=1}^{N} u_{jr}^{(2)}.
$$

Then

$$
C_i^{(2)}
=
\sum_{r=1}^{R}
\lambda_r^{(2)} u_{ir}^{(2)}
\left(S_r^{(x)} - x_i S_r^{(1)}\right).
$$

This is exactly the dense pairwise interaction, but evaluated without constructing the full $N \times N$ tensor.

## 6. Exact third-order contraction after CP

Start from the dense third-order term

$$
C_i^{(3)}
=
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
W^{(3)}_{ijk}
\left(x_j^2 x_k - x_i^3\right).
$$

Substitute the CP form:

$$
C_i^{(3)}
=
\sum_{r=1}^{R}
\lambda_r^{(3)} u_{ir}^{(3)}
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
u_{jr}^{(3)} u_{kr}^{(3)}
\left(x_j^2 x_k - x_i^3\right).
$$

For one fixed rank $r$, define

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

For a target node $i$, exclude node $i$ from these sums:

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

Then the exact ordered-pair sum over $j \ne k$, both different from $i$, is

$$
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
u_{jr}^{(3)} u_{kr}^{(3)} x_j^2 x_k
=
\widetilde{A}_{ir} \widetilde{B}_{ir} - \widetilde{D}_{ir},
$$

and

$$
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
u_{jr}^{(3)} u_{kr}^{(3)}
=
\widetilde{C}_{ir}^2 - \widetilde{E}_{ir}.
$$

Therefore the exact third-order contraction becomes

$$
C_i^{(3)}
=
\sum_{r=1}^{R}
\lambda_r^{(3)} u_{ir}^{(3)}
\left[
\widetilde{A}_{ir} \widetilde{B}_{ir}
- \widetilde{D}_{ir}
- x_i^3 \left(\widetilde{C}_{ir}^2 - \widetilde{E}_{ir}\right)
\right].
$$

This is again exact. No candidate-edge sampling is used.

Because the ordered pairs $(j,k)$ and $(k,j)$ are both included, this formula reproduces the dense symmetrized triangle contribution

$$
x_j^2 x_k + x_j x_k^2 - 2x_i^3.
$$

## 7. Resulting coupled Rossler dynamics after CP

After substituting the exact CP contractions, the dynamical system used in training is

$$
\dot{x}_i
=
-y_i - z_i
+ k
\sum_{r=1}^{R}
\lambda_r^{(2)} u_{ir}^{(2)}
\left(S_r^{(x)} - x_i S_r^{(1)}\right)
+ k_D
\sum_{r=1}^{R}
\lambda_r^{(3)} u_{ir}^{(3)}
\left[
\widetilde{A}_{ir} \widetilde{B}_{ir}
- \widetilde{D}_{ir}
- x_i^3 \left(\widetilde{C}_{ir}^2 - \widetilde{E}_{ir}\right)
\right],
$$

$$
\dot{y}_i = x_i + a_r y_i,
\qquad
\dot{z}_i = b_r + z_i (x_i - c_r).
$$

So the physical model itself does not change in meaning. What changes is only the representation of the unknown second-order and third-order hypergraph weights.

## 8. Physics loss used by HyperPINN

The neural network predicts trajectories

$$
\hat{x}_i(t), \hat{y}_i(t), \hat{z}_i(t).
$$

Automatic differentiation gives time derivatives

$$
\partial_t \hat{x}_i,
\qquad
\partial_t \hat{y}_i,
\qquad
\partial_t \hat{z}_i.
$$

The physics loss is the mean squared residual between these derivatives and the CP-parameterized Rossler right-hand side:

$$
\mathcal{L}_{\mathrm{phys}}
=
\frac{1}{T}
\sum_t
\left\|
\partial_t \hat{\mathbf{s}}(t) - \mathbf{f}_{\mathrm{CP}}(\hat{\mathbf{s}}(t))
\right\|_2^2,
$$

where

$$
\hat{\mathbf{s}}(t)
=
\bigl[
\hat{x}_1,\dots,\hat{x}_N,
\hat{y}_1,\dots,\hat{y}_N,
\hat{z}_1,\dots,\hat{z}_N
\bigr].
$$

## 9. What is regularized in the CP branch

The current CP branch does not regularize sampled edge logits, because there are no sampled edge logits anymore.

Instead, it regularizes:

### 9.1 Total second-order mass

$$
\sum_{i < j} W^{(2)}_{ij}
=
\frac{1}{2}
\sum_{r=1}^{R}
\lambda_r^{(2)}
\left[
\left(\sum_i u_{ir}^{(2)}\right)^2
- \sum_i (u_{ir}^{(2)})^2
\right].
$$

### 9.2 Total third-order mass

$$
\sum_{i < j < k} W^{(3)}_{ijk}
=
\sum_{r=1}^{R}
\lambda_r^{(3)}
\frac{
\left(\sum_i u_{ir}^{(3)}\right)^3
- 3 \left(\sum_i u_{ir}^{(3)}\right) \left(\sum_i (u_{ir}^{(3)})^2\right)
+ 2 \sum_i (u_{ir}^{(3)})^3
}{6}.
$$

### 9.3 Small L2 penalty on the raw factors

$$
\mathcal{L}_{\mathrm{factor}}
=
\lambda_{\mathrm{factor}}
\left(
\|\alpha^{(2)}\|_F^2 + \|\beta^{(2)}\|_2^2
+ \|\alpha^{(3)}\|_F^2 + \|\beta^{(3)}\|_2^2
\right).
$$

## 10. Main conceptual change

Before CP:

- the model explicitly stored many individual hyperedge weights
- training operated on a dense edge or tensor representation

After CP:

- the complete second-order and third-order hypergraphs are represented implicitly by low-rank factors
- the dynamical couplings are evaluated by exact analytic contractions
- no candidate-edge sampling is needed in the physics term

So the core idea is:

$$
\text{explicit huge tensor}
\quad \longrightarrow \quad
\text{CP factors} + \text{exact contraction}.
$$

This is the mathematical meaning of the current `hyperpinn_cp` implementation.