# Social Contagion Hypergraph Dynamics Scene

This scene models social contagion (e.g. information, behavior, or disease spreading) on a hypergraph, where both pairwise contacts and higher–order group interactions can drive infection. The library `lib_social_contagion.hypergraph` provides:

- A random simplicial complex generator for edges/triangles (and optional quads for small toy graphs),
- A continuous–time mean–field ODE for infection probabilities,
- A simulator that produces both latent probabilities and binarized 0/1 infection observations,
- A decomposition of the ODE into a drift term `dynamic_f` and a feature map `dynamic_phi` for hyperedge–based interactions.

---

## 1. Node set and hypergraph structure

- **Nodes**  
  We consider a population of `n_nodes` individuals, indexed by
  $$V = \{1, 2, \dots, n_\text{nodes}\}.$$
  For each node $i$, the continuous state $x_i(t)\in[0,1]$ denotes the probability that individual $i$ is infected at time $t$.

- **Hyperedges (2- and 3-body, optional 4-body)**  
  For general `n_nodes` we generate a random simplicial complex using parameters `RSCParams`:

  - `k_mean`: target mean degree (expected number of edges per node),
  - `k_delta`: expected "excess" participation in triangles,
  - `enforce_closure`: if True, every triangle implies its three constituent edges (simplicial closure).

  The generator draws:

  - A set of undirected edges $\mathcal{E}_2$ (order–2 hyperedges),
  - A set of undirected triangles $\mathcal{E}_3$ (order–3 hyperedges).

  For the small toy case `n_nodes = 8`, we instead use a **hand–crafted** structure:

  - A fixed set of edges and triangles forming two overlapping "communities";
  - If `max_order >= 4`, we additionally include a small set of 4–body hyperedges $\mathcal{E}_4$ so that the model has genuine positive samples at order–4.

- **Ground–truth vs candidate hyperedges**  
  - `get_hyperedge_config(n_nodes, max_order)` returns the **ground–truth** edges/triangles/quads in 1-based indexing (for evaluation and visualization).  
  - `generate_all_possible_hyperedges(n_nodes, max_order)` lists **all** possible 2/3/4–body combinations over $V$, which serve as the candidate set for structure inference (each candidate gets a learnable weight / gate).

---

## 2. Social contagion dynamics (SCM)

The time evolution of the infection probabilities $x_i(t)$ is governed by a continuous–time mean–field ODE of susceptible–infected–susceptible type.

Parameters (`SCMParams`):

- `beta` ($\beta$): pairwise infection rate,
- `beta_delta` ($\beta_\Delta$): higher–order infection rate for triangles/quads,
- `mu` ($\mu$): recovery rate,
- `t_max`: simulation horizon.

Let $\lambda_i(x)$ denote the total "infection pressure" on node $i$, aggregated from neighbors and higher–order groups. The ODE is
$$
\frac{dx_i}{dt}
= -\mu\, x_i
  + (1 - x_i)\,\lambda_i(x),\quad i=1,\dots,n_\text{nodes}.
$$

### 2.1 Pairwise (edge–based) infection

For each edge $(i,j)\in \mathcal{E}_2$:
$$
\lambda_i^\text{(2)}(x) \;{+}{=}\; \beta\, x_j,\qquad
\lambda_j^\text{(2)}(x) \;{+}{=}\; \beta\, x_i.
$$

### 2.2 3-body (triangle–based) infection

For each triangle $(i,j,k)\in \mathcal{E}_3$:
$$
\begin{aligned}
\lambda_i^\text{(3)}(x) &\;{+}{=}\; \beta_\Delta\, x_j x_k, \\
\lambda_j^\text{(3)}(x) &\;{+}{=}\; \beta_\Delta\, x_i x_k, \\
\lambda_k^\text{(3)}(x) &\;{+}{=}\; \beta_\Delta\, x_i x_j.
\end{aligned}
$$

### 2.3 4-body (optional quad–based) infection

For each quad $(i,j,k,l)\in \mathcal{E}_4$:
$$
\begin{aligned}
\lambda_i^\text{(4)}(x) &\;{+}{=}\; \beta_\Delta\, x_j x_k x_l,\\
\lambda_j^\text{(4)}(x) &\;{+}{=}\; \beta_\Delta\, x_i x_k x_l,\\
\lambda_k^\text{(4)}(x) &\;{+}{=}\; \beta_\Delta\, x_i x_j x_l,\\
\lambda_l^\text{(4)}(x) &\;{+}{=}\; \beta_\Delta\, x_i x_j x_k.
\end{aligned}
$$

The total infection pressure is
$$
\lambda_i(x)
= \lambda_i^\text{(2)}(x)
+ \lambda_i^\text{(3)}(x)
+ \lambda_i^\text{(4)}(x),
$$
and the ODE reads
$$
\frac{dx_i}{dt}
= -\mu\, x_i + (1 - x_i)\,\lambda_i(x).
$$

This is exactly what `_rhs` implements in `lib_social_contagion.hypergraph`.

---

## 3. Simulation and observations

### 3.1 Initial condition

At $t=0$, each node’s infection probability $x_i(0)$ is drawn from a small range (e.g. uniform in $[0, 0.1]$), and a random subset of "seed" nodes is given a slightly elevated infection probability (e.g. in $[0.2, 0.5]$), ensuring the contagion process actually starts.

### 3.2 Numerical integration

We integrate the ODE on $[0, t_\text{max}]$ using an adaptive RK45 solver, with a fixed evaluation grid of `n_steps` points, producing
$$
X_\text{cont}(t_m) \in [0,1]^{n_\text{nodes}}, \quad m=1,\dots,n_\text{steps},
$$
where $X_\text{cont}$ are the latent infection probabilities.

### 3.3 Binary observations

To mimic discrete 0/1 infection data, we sample Bernoulli observations independently for each node and time:
$$
X_\text{obs}(i, m) \sim \text{Bernoulli}\!\big(X_\text{cont}(i, m)\big).
$$

The simulator returns both:

- `X_continuous`: the true probabilities (latent continuous state),
- `X_observed`: the binarized infection observations (what we typically feed into downstream models as data).

### 3.4 Data interface

`generate_training_data(n_nodes, edge_config, n_samples, noise=0, seed=None)`:

- Uses `edge_config` (ground–truth edges/triangles/quads) to build the simplicial complex.
- Runs the SCM simulator with a given random seed.
- Returns resampled time grid `t` and `x_observed` with shape `[T, N, 1]` (with `T` close to `n_samples`), optionally adding small Gaussian noise on top of the 0/1 values for robustness.
- The `seed` argument lets us generate **multiple independent trajectories** on the same underlying hypergraph by varying SCM randomness while keeping `edge_config` fixed.

---

## 4. Decomposition for PINN / Integral models

For structure inference, the ODE is decomposed as
$$
\frac{dx}{dt} = f(x) + \Phi(x)\, A,
$$
where:

- **Drift term `dynamic_f(x, n_nodes)`**  
  Implements the recovery part only:
  $$
  f_i(x) = -\mu\, x_i.
  $$
  This part is independent of any hypergraph structure.

- **Feature map `dynamic_phi(x, all_possible_edges, n_nodes, device)`**  
  For each candidate edge/triangle/quad $e$ in the full candidate set (not just ground truth), builds a feature $\phi_e(x)$ so that the interaction term can be written as a linear combination:
  $$
  (\Phi(x) A)_i = \sum_{e\ni i} A_e\, \phi_{e,i}(x).
  $$

  Concretely (for a scalar state per node):

  - For a candidate edge $(i,j)$:
    $$
    \phi_{(i,j), i}(x) = (1-x_i)\,\beta\,x_j,
    \quad
    \phi_{(i,j), j}(x) = (1-x_j)\,\beta\,x_i.
    $$

  - For a candidate triangle $(i,j,k)$:
    $$
    \begin{aligned}
    \phi_{(i,j,k), i}(x) &= (1-x_i)\,\beta_\Delta\,x_j x_k,\\
    \phi_{(i,j,k), j}(x) &= (1-x_j)\,\beta_\Delta\,x_i x_k,\\
    \phi_{(i,j,k), k}(x) &= (1-x_k)\,\beta_\Delta\,x_i x_j.
    \end{aligned}
    $$

  - For a candidate quad $(i,j,k,l)$:
    $$
    \phi_{(i,j,k,l), i}(x) = (1-x_i)\,\beta_\Delta\,x_j x_k x_l, \text{ etc.}
    $$

  The function returns a tensor `Phi` of shape `[N, 1, E_total]`, where `E_total` is the total number of candidate hyperedges, stacked in the order (edges, triangles, quads).

- **Ground–truth vs learnable weights**  
  In the simulator, all hyperedges in `edge_config` have implicit weight 1 in `_rhs`.  
  In Integral PINN / HyperPINN, we introduce explicit learnable parameters $A_e$ (one per candidate hyperedge), and use `dynamic_f` + `dynamic_phi` to ensure the model has the same functional form as SCM, but with unknown structure.

---

## 5. Summary for experiments

- **State:**  
  Node–wise infection probability $x_i(t)\in[0,1]$ (latent) and 0/1 observations from Bernoulli sampling.

- **Structure:**  
  Random simplicial complex or hand–crafted small hypergraph, with 2–3–(optional 4) body interactions, represented as hyperedges $\mathcal{E}_2,\mathcal{E}_3,\mathcal{E}_4$.

- **Dynamics:**  
  Continuous–time SCM ODE
  $$
  \frac{dx_i}{dt} = -\mu x_i + (1-x_i)\lambda_i(x),
  $$
  with $\lambda_i(x)$ built from pairwise and higher–order infection terms.

- **Data generation:**  
  ODE solved by RK45; latent probabilities thresholded via Bernoulli sampling to generate discrete infection time series; multiple trajectories possible via different random seeds.

- **For structure inference (Integral PINN / HyperPINN):**  
  Use `dynamic_f` and `dynamic_phi` to express SCM as a linear function of unknown hyperedge weights $A$, and learn $A$ from time series by combining data loss and physics–based residuals (integral residual for continuous case, Euler/Bernoulli likelihood for discrete observations).
