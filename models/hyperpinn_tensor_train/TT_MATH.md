# Tensor-Train Hypergraph Dynamics for Rossler HyperPINN

This note explains the mathematical logic behind the true TT version under `models/hyperpinn_tensor_train`.

The current implementation only supports:

- second-order interactions
- third-order interactions
- Rossler dynamics

The goal is the same as in the CP and Tucker branches:

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

## 2. Dense hypergraph form before TT

### 2.1 Second-order term

$$
C_i^{(2)} = \sum_{j \ne i} W^{(2)}_{ij}(x_j - x_i).
$$

### 2.2 Third-order term

$$
C_i^{(3)}
=
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
W^{(3)}_{ijk}
\left(x_j^2 x_k - x_i^3\right).
$$

Because the dense baseline sums over ordered pairs $(j,k)$, an unordered triangle contributes the symmetrized term

$$
W^{(3)}_{ijk}
\left(x_j^2 x_k + x_j x_k^2 - 2x_i^3\right).
$$

## 3. Why TT is subtler than CP or Tucker

TT is naturally ordered, but hyperedges are unordered. So the TT branch must explicitly symmetrize the induced third-order weight over all six permutations.

That is the critical step.

## 4. Second-order TT form

For order 2 we use a rank-$R$ factorization

$$
\widetilde{W}^{(2)}_{ij} = \sum_{r=1}^{R} L_{ir} R_{jr}.
$$

Because pairwise hyperedges are unordered, we symmetrize it:

$$
W^{(2)}_{ij} = \frac{1}{2}\left(\widetilde{W}^{(2)}_{ij} + \widetilde{W}^{(2)}_{ji}\right).
$$

So

$$
W^{(2)}_{ij}
=
\frac{1}{2}
\sum_{r=1}^{R}
\left(L_{ir}R_{jr} + L_{jr}R_{ir}\right).
$$

## 5. Third-order TT form

Let the TT cores be

- $A \in \mathbb{R}_{\ge 0}^{N \times R}$
- $B \in \mathbb{R}_{\ge 0}^{N \times R \times R}$
- $C \in \mathbb{R}_{\ge 0}^{N \times R}$

The ordered TT score is

$$
\widetilde{W}^{(3)}(i,j,k)
=
\sum_{a=1}^{R}\sum_{b=1}^{R}
A_{ia} B_{jab} C_{kb}.
$$

Because the hypergraph triangle is unordered, the actual weight is the average over all six permutations:

$$
W^{(3)}_{ijk}
=
\frac{1}{6}
\sum_{\pi \in S_3}
\widetilde{W}^{(3)}\bigl(\pi(i,j,k)\bigr).
$$

This is the TT analogue of the symmetry handling that is automatic or easier in CP and Tucker.

## 6. Nonnegativity in code

All TT cores are stored as raw parameters and mapped through `softplus`:

$$
L = \operatorname{softplus}(\widetilde{L}),
\qquad
R = \operatorname{softplus}(\widetilde{R}),
$$

$$
A = \operatorname{softplus}(\widetilde{A}),
\qquad
B = \operatorname{softplus}(\widetilde{B}),
\qquad
C = \operatorname{softplus}(\widetilde{C}).
$$

## 7. Exact second-order contraction under TT

Start from

$$
C_i^{(2)} = \sum_{j \ne i} W^{(2)}_{ij}(x_j - x_i).
$$

Insert the symmetrized TT form:

$$
C_i^{(2)}
=
\frac{1}{2}
\sum_{r}
L_{ir}
\sum_{j \ne i} R_{jr}(x_j - x_i)
+
\frac{1}{2}
\sum_{r}
R_{ir}
\sum_{j \ne i} L_{jr}(x_j - x_i).
$$

Since the $j=i$ term is zero anyway,

$$
C_i^{(2)}
=
\frac{1}{2}
\sum_r
L_{ir}\left(\sum_j R_{jr}x_j - x_i\sum_j R_{jr}\right)
+
\frac{1}{2}
\sum_r
R_{ir}\left(\sum_j L_{jr}x_j - x_i\sum_j L_{jr}\right).
$$

This is exact.

## 8. Exact third-order contraction under TT

For the ordered TT score

$$
\widetilde{W}^{(3)}(i,j,k)
=
\sum_{a,b} A_{ia}B_{jab}C_{kb},
$$

the unsymmetrized contribution to node $i$ is

$$
\widetilde{C}_i^{(3)}
=
\sum_{a,b} A_{ia}
\sum_{\substack{j \ne i,\; k \ne i,\\ j \ne k}}
B_{jab}C_{kb}(x_j^2x_k - x_i^3).
$$

Define the rankwise and TT-middle statistics

$$
S^{(2)}_{i,ab} = \sum_{j \ne i} B_{jab}x_j^2,
\qquad
S^{(1)}_{i,b} = \sum_{k \ne i} C_{kb}x_k,
$$

$$
N^{(B)}_{i,ab} = \sum_{j \ne i} B_{jab},
\qquad
N^{(C)}_{i,b} = \sum_{k \ne i} C_{kb},
$$

$$
D_{i,ab} = \sum_{j \ne i} B_{jab}C_{jb}x_j^3,
\qquad
E_{i,ab} = \sum_{j \ne i} B_{jab}C_{jb}.
$$

Then

$$
\widetilde{C}_i^{(3)}
=
\sum_{a,b} A_{ia}
\left[
S^{(2)}_{i,ab}S^{(1)}_{i,b}
-
D_{i,ab}
-
x_i^3\left(N^{(B)}_{i,ab}N^{(C)}_{i,b} - E_{i,ab}\right)
\right].
$$

Because the actual TT hypergraph weight is symmetrized over six permutations, the true third-order coupling is the average of six such ordered contractions:

$$
C_i^{(3)}
=
\frac{1}{6}
\left(
\widetilde{C}^{(3)}_{i|(i,j,k)}
+
\widetilde{C}^{(3)}_{i|(i,k,j)}
+
\widetilde{C}^{(3)}_{i|(j,i,k)}
+
\widetilde{C}^{(3)}_{i|(j,k,i)}
+
\widetilde{C}^{(3)}_{i|(k,i,j)}
+
\widetilde{C}^{(3)}_{i|(k,j,i)}
\right).
$$

The implementation computes all six terms explicitly with the correct tensor contraction order and then averages them.

That ordering detail is essential, because TT cores do not commute.

## 9. Final coupled Rossler dynamics under TT

The final dynamics are still

$$
\dot{x}_i = -y_i - z_i + k \, C_i^{(2)} + k_D \, C_i^{(3)},
$$

$$
\dot{y}_i = x_i + a_r y_i,
\qquad
\dot{z}_i = b_r + z_i (x_i - c_r).
$$

So the physical model has not changed in meaning. Only the representation of the unknown second-order and third-order hypergraph tensors has changed.

## 10. Regularization

The TT branch regularizes:

- the total induced pairwise mass
- the total induced triangle mass
- a small L2 penalty on the raw TT cores

For triangles, the total mass is computed by summing ordered distinct TT contributions and then dividing by 6, which matches the permutation-averaged symmetric definition.

## 11. Summary

The TT branch replaces the complete hypergraph by TT cores and evaluates the dynamics by exact contractions.

The key difference from CP and Tucker is:

- TT is ordered
- hyperedges are unordered
- therefore TT must be explicitly symmetrized across permutations

That is the main mathematical and implementation constraint for a correct TT hypergraph model.
