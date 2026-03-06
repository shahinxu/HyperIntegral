# Neuronal Synchronization Hypergraph Dynamics Scene

This scene models neuronal (phase) synchronization on a hypergraph, where both pairwise and higher–order group interactions couple oscillatory units. The library `lib_neuronal_synchronization.hypergraph` provides:

- A sparse, multi–order hypergraph generator for neuronal populations,
- A Kuramoto–style continuous–time ODE on two coupled phase variables per node,
- A simulator that produces time series of phases over all nodes,
- A decomposition of the ODE into a drift term `dynamic_f` and a feature map `dynamic_phi` for hyperedge–based interactions (including batched versions for PINNs).

---

## 1. Node set, state variables, and hypergraph structure

### 1.1 Nodes and state

We consider a population of `n_oscillators` phase oscillators, indexed by
$$
V = \{1, 2, \dots, n_\text{osc}\}.
$$

Each node $i$ carries **two** phase variables:

- $\theta_i(t)$: a "slow" or base phase (e.g. neuronal membrane potential phase),
- $\phi_i(t)$: a "fast" or driven phase (e.g. an auxiliary or synaptic phase),

so the full state at time $t$ is
$$
X(t) = \big((\theta_1, \phi_1),\dots,(\theta_N, \phi_N)\big) \in (\mathbb{R}/2\pi\mathbb{Z})^{2N}.
$$
Internally the simulator keeps phases in $[-\pi, \pi)$ by wrapping.

### 1.2 Hyperedges: pairwise and simplicial interactions

The structural connectivity is represented by an order–2 and order–3 hypergraph:

- A set of undirected **edges** $\mathcal{E}_2$ (pairwise interactions),
- A set of undirected **triangles** $\mathcal{E}_3$ (three–body/simplicial interactions).

`get_hyperedge_config(n_nodes, max_order)` samples a sparse hypergraph via
`EcologicalHypergraphParams`–like settings (here `SimplicialParams`):

- The number of edges/triangles is controlled by per–node incidence targets,
- Edges and triangles are returned in **1-based indexing** for convenience,
- Internally, 0-based versions are cached for efficient simulation and PINN use.

For the neuronal scene, we typically restrict to

- Order–2: pairwise couplings on the $\phi$ phases,
- Order–3: simplicial couplings on the $\theta$ phases,

and do not include higher orders (4+); however, the PINN side can consider
all possible 2- and 3-body candidate hyperedges when performing structure
inference.

---

## 2. Neuronal synchronization dynamics

The evolution of $(\theta_i, \phi_i)$ is governed by a Kuramoto–style ODE with
multi–order coupling:

Parameters are stored in `SimplicialParams`:

- `K`: strength of simplicial (triangle–based) coupling on $\theta$,
- `kappa`: strength of pairwise (edge–based) coupling on $\phi$,
- `d`: strength of cross–coupling (drive) between $\theta$ and $\phi$,
- `omega_mean`, `omega_std`: distribution of intrinsic frequencies for $\theta$,
- `nu_mean`, `nu_std`: distribution of intrinsic frequencies for $\phi$,
- `t_span`, `n_steps`, `seed`, etc. for simulation.

Let $\omega_i$ and $\nu_i$ be the intrinsic frequencies of oscillator $i$ for
$\theta$ and $\phi$ respectively. The ODE implemented in `_rhs` is:

### 2.1 Pairwise coupling on $\phi$ (edges)

For each edge $(i,j)\in \mathcal{E}_2$, the pairwise Kuramoto–type term on
$\phi$ is:
$$
\begin{aligned}
\text{pairwise}_i(\phi)
&\;{+}{=}\; \frac{\kappa}{N}\, \sin(\phi_j - \phi_i), \\
\text{pairwise}_j(\phi)
&\;{+}{=}\; \frac{\kappa}{N}\, \sin(\phi_i - \phi_j).
\end{aligned}
$$
This term tends to synchronize the $\phi$ phases along edges.

### 2.2 Simplicial (triangle) coupling on $\theta$

For each triangle $(i,j,k)\in \mathcal{E}_3$, we use a higher–order Kuramoto–
like interaction on $\theta$:
$$
\begin{aligned}
\text{simplex}_i(\theta)
&\;{+}{=}\; \frac{K}{N^2}\, \sin(\theta_j + \theta_k - 2\theta_i), \\
\text{simplex}_j(\theta)
&\;{+}{=}\; \frac{K}{N^2}\, \sin(\theta_i + \theta_k - 2\theta_j), \\
\text{simplex}_k(\theta)
&\;{+}{=}\; \frac{K}{N^2}\, \sin(\theta_i + \theta_j - 2\theta_k).
\end{aligned}
$$
These terms favor configurations where the phases in a triangle align in a
nontrivial higher–order synchronized pattern.

### 2.3 Cross–layer drive between $\theta$ and $\phi$

Each node has an additional local drive term linking its two phases:
$$
\text{drive}_i = d\, \sin(\theta_i - \phi_i),
$$
which appears in the $\phi$ dynamics and tends to pull $\phi_i$ toward $\theta_i$.

### 2.4 Full ODE

Collecting all contributions, the ODE for each node $i$ is
$$
\begin{aligned}
\dot{\theta}_i
&= \omega_i + \text{simplex}_i(\theta), \\
\dot{\phi}_i
&= \nu_i + \text{pairwise}_i(\phi) + d\, \sin(\theta_i - \phi_i).
\end{aligned}
$$
The system is integrated numerically to produce coupled phase trajectories.

---

## 3. Simulation and data generation

### 3.1 Intrinsic frequencies and initial conditions

In `_simulate`, for a given `SimplicialParams` instance:

- Intrinsic frequencies
  - $\omega_i$ are drawn i.i.d. from $\mathcal{N}(\text{omega\_mean}, \text{omega\_std}^2)$,
  - $\nu_i$ are drawn i.i.d. from $\mathcal{N}(\text{nu\_mean}, \text{nu\_std}^2)$.
- Initial phases
  - $\theta_i(0)$ and $\phi_i(0)$ are drawn uniformly from $[-\pi, \pi)$.

### 3.2 Numerical integration

We solve the ODE on a time interval `t_span = (t0, t1)` using `solve_ivp`
with method `RK45`, evaluating at `n_steps` points `t_eval`.

After integration:

- The phases $\theta_i(t)$ and $\phi_i(t)$ are wrapped back into $[-\pi, \pi)$,
- Global order parameters (e.g. $r_\theta$, $r^2_\theta$, $r_\phi$) are computed
  as mean phase coherence measures, but these are primarily for analysis,
  not directly used in PINN training.

### 3.3 Returned data

`generate_training_data(n_nodes, edge_config, n_samples, noise=0.0)`:

- Constructs a `SimplicialParams` with `n_steps = n_samples`,
- Converts the 1-based `edge_config` into 0-based edges/triangles,
- Calls `_simulate` to obtain
  - `t`: time grid `(T,)`,
  - `theta, phi`: arrays of shape `(n_nodes, T)` each,
- Stacks phases into a single tensor `x_data` of shape `(T, N, 2)`:
  $$
  x_\text{data}[m, i, 0] = \theta_i(t_m),\quad
  x_\text{data}[m, i, 1] = \phi_i(t_m).
  $$
- Optionally adds Gaussian noise if `noise > 0`.

The hypergraph and frequencies used for simulation are cached so that
`dynamic_f` / `dynamic_phi` can later use exactly the same parameters.

---

## 4. Decomposition for PINN / HyperPINN

For structure inference (Integral PINN, HyperPINN), we factor the dynamics as
$$
\dot{X}(t) \approx f(X(t)) + \Phi(X(t)) A,
$$
where $X(t)$ stacks all $(\theta_i,\phi_i)$ and $A$ collects learnable weights
for candidate hyperedges across orders.

### 4.1 Drift term `dynamic_f`

The function `dynamic_f(x, n_nodes)` implements the baseline ODE using the
**cached** hypergraph and frequencies:

- Input: `x` of shape `[N, 2]` (or `[N, 2]`–like in batched interfacing),
- Output: tensor `[N, 2]` with components
  $$
  \dot{\theta}_i = \omega_i + \text{simplex}_i(\theta),\quad
  \dot{\phi}_i = \nu_i + \text{pairwise}_i(\phi) + d\, \sin(\theta_i - \phi_i),
  $$
  where the actual edges/triangles (and thus which nodes contribute) are
  determined by the cached ground–truth hypergraph.

We also provide `dynamic_f_batch(x, n_nodes)` which computes the same drift
for all time steps in a batch at once:

- Input: `x` of shape `[T, N, 2]`,
- Output: `[T, N, 2]` containing $\dot{\theta}, \dot{\phi}$ for each time.

### 4.2 Feature map `dynamic_phi`

The function `dynamic_phi(x, all_possible_edges, n_nodes, device)` constructs
feature tensors corresponding to all **candidate** edges/triangles (not just the
true ones):

- Input:
  - `x`: `[N, 2]` with $(\theta,\phi)$ per node,
  - `all_possible_edges`: a dict with keys `"edges"` and `"triangles"`, whose
    values are 0-based index arrays of candidate hyperedges.
- Output:
  - `Phi`: tensor of shape `[N, 2, E_total]`, where `E_total` is the total
    number of candidates across orders.

Semantically:

- For candidate edges, `dynamic_phi` fills only the $\phi$ component (dimension 1)
  with pairwise Kuramoto terms (the same structure as in `_rhs` but decoupled
  from the unknown weights).
- For candidate triangles, it fills only the $\theta$ component (dimension 0)
  with the simplicial terms.

We also provide `dynamic_phi_batch(x, all_possible_edges, n_nodes, device)` for
batched use in PINN physics loss:

- Input: `x` of shape `[T, N, 2]`,
- Output: `Phi_all` of shape `[T, N, 2, E_total]`.

### 4.3 Ground–truth vs learnable weights

In the simulator, the ground–truth hyperedges are already "baked into" the
ODE via a fixed choice of which pairs/triangles enter `_rhs`. In PINN/HyperPINN
experiments we instead:

- Enumerate all candidate edges/triangles over nodes,
- Attach a learnable weight/gate $A_e$ to each candidate,
- Use `dynamic_f` for the intrinsic + cached hypergraph part, and
  `dynamic_phi` / `dynamic_phi_batch` to express additional coupling as
  \( \Phi(X(t)) A \), where the nonzero $A_e$ should ideally match the
  true connectivity.

---

## 5. Summary for experiments

- **State:**  
  Two phase variables $(\theta_i, \phi_i)$ per node, wrapped to $[-\pi, \pi)$,
  forming time series of shape `[T, N, 2]`.

- **Structure:**  
  Sparse 2- and 3-body hypergraph (edges and triangles) over the oscillator
  population.

- **Dynamics:**  
  Kuramoto–style ODE with
  - pairwise coupling on $\phi$ along edges,
  - simplicial (triangle) coupling on $\theta$,
  - local cross–layer drive between $\theta$ and $\phi$.

- **Data generation:**  
  ODE solved by RK45; phases wrapped to $[-\pi, \pi)$; time series returned as
  continuous phase trajectories (no binarization).

- **For structure inference (Integral PINN / HyperPINN):**  
  Use `dynamic_f` / `dynamic_f_batch` and
  `dynamic_phi` / `dynamic_phi_batch` to express the ODE as a linear function
  of unknown hyperedge weights over a candidate set, and learn these weights
  from observed phase trajectories using physics–informed losses and sparsity
  regularization.
