"""
Sparse ecological hypergraph dynamics with trophic predation.

Requested modeling choices implemented:
- sparse (not fully connected)
- undirected competition hyperedges of order-2/3/4
- directed predation only on order-2 edges
- trophic groups: producers -> primary consumers -> secondary consumers
- producers have autonomous growth to avoid being eaten out completely
- predation decreases prey and increases predator with trophic efficiency decay
- single output figure
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class EcologicalHypergraphParams:
    n_species: int = 9
    n_producers: int = 3
    n_primary_consumers: int = 3
    n_secondary_consumers: int = 3

    t_span: tuple = (0.0, 420.0)
    n_steps: int = 4200
    seed: int = 42
    rtol: float = 1e-8
    atol: float = 1e-10

    target_incidence_per_node_order2: int = 5
    target_incidence_per_node_order3: int = 5
    target_incidence_per_node_order4: int = 5

    producer_growth_mean: float = 1.20
    producer_growth_std: float = 0.05
    producer_carrying_capacity: float = 0.80
    producer_replenish: float = 0.0
    season_amplitude: float = 0.25
    season_period: float = 42.0

    primary_mortality: float = 0.24
    secondary_mortality: float = 0.20
    primary_immigration: float = 0.0
    secondary_immigration: float = 0.0

    # Competition weights (undirected hypergraph)
    w2: float = 0.0006
    w3: float = 0.0008
    w4: float = 0.0010

    # Predation parameters (directed order-2 only)
    target_prey_per_primary: int = 2
    target_prey_per_secondary: int = 2
    attack_p_to_c1: float = 2.2
    attack_c1_to_c2: float = 1.6
    half_sat: float = 0.01
    eta_p_to_c1: float = 0.55
    eta_c1_to_c2: float = 0.35


def _validate_params(params: EcologicalHypergraphParams):
    total = params.n_producers + params.n_primary_consumers + params.n_secondary_consumers
    if total != params.n_species:
        raise ValueError("n_species must equal n_producers + n_primary_consumers + n_secondary_consumers")


def _sample_sparse_hyperedges(
    n: int,
    order: int,
    target_incidence_per_node: int,
    rng: np.random.Generator,
):
    candidates = list(combinations(range(n), order))
    if not candidates:
        return []

    m_target = max(n // 2, int(round(n * target_incidence_per_node / order)))
    m_target = min(m_target, len(candidates))

    picked_idx = set(rng.choice(len(candidates), size=m_target, replace=False).tolist())

    node_cover = {i: 0 for i in range(n)}
    for idx in picked_idx:
        for node in candidates[idx]:
            node_cover[node] += 1

    for node in range(n):
        if node_cover[node] > 0:
            continue
        contain_node = [i for i, e in enumerate(candidates) if node in e]
        if not contain_node:
            continue
        add_idx = int(rng.choice(contain_node))
        picked_idx.add(add_idx)
        for u in candidates[add_idx]:
            node_cover[u] += 1

    return [candidates[i] for i in sorted(picked_idx)]


def _build_trophic_groups(params: EcologicalHypergraphParams):
    p = np.arange(0, params.n_producers, dtype=int)
    c1_start = params.n_producers
    c1_end = c1_start + params.n_primary_consumers
    c1 = np.arange(c1_start, c1_end, dtype=int)
    c2 = np.arange(c1_end, params.n_species, dtype=int)
    return p, c1, c2


def _build_predation_edges(params: EcologicalHypergraphParams, rng: np.random.Generator):
    """
    Directed order-2 predation edges represented as tuples:
      (prey_idx, predator_idx, attack_rate, conversion_efficiency)
    """
    p, c1, c2 = _build_trophic_groups(params)

    pred_edges = []

    # Fixed ring food web: producers -> primary consumers.
    # Each predator connects to k nearest prey positions on a circular index.
    n_prey_p = len(p)
    k1 = max(1, min(params.target_prey_per_primary, n_prey_p))
    for pred_pos, predator in enumerate(c1):
        for shift in range(k1):
            prey = p[(pred_pos + shift) % n_prey_p]
            pred_edges.append((int(prey), int(predator), params.attack_p_to_c1, params.eta_p_to_c1))

    # Fixed ring food web: primary consumers -> secondary consumers.
    n_prey_c1 = len(c1)
    k2 = max(1, min(params.target_prey_per_secondary, n_prey_c1))
    for pred_pos, predator in enumerate(c2):
        for shift in range(k2):
            prey = c1[(pred_pos + shift) % n_prey_c1]
            pred_edges.append((int(prey), int(predator), params.attack_c1_to_c2, params.eta_c1_to_c2))

    return pred_edges


def build_sparse_hypergraph_and_foodchain(params: EcologicalHypergraphParams, rng: np.random.Generator):
    nodes = np.arange(params.n_species, dtype=int)

    edges2 = _sample_sparse_hyperedges(
        n=params.n_species,
        order=2,
        target_incidence_per_node=params.target_incidence_per_node_order2,
        rng=rng,
    )
    edges3 = _sample_sparse_hyperedges(
        n=params.n_species,
        order=3,
        target_incidence_per_node=params.target_incidence_per_node_order3,
        rng=rng,
    )
    edges4 = _sample_sparse_hyperedges(
        n=params.n_species,
        order=4,
        target_incidence_per_node=params.target_incidence_per_node_order4,
        rng=rng,
    )

    pred_edges = _build_predation_edges(params, rng)
    producers, primary, secondary = _build_trophic_groups(params)

    return {
        "nodes": nodes,
        "edges2": edges2,
        "edges3": edges3,
        "edges4": edges4,
        "pred_edges": pred_edges,
        "producers": producers,
        "primary": primary,
        "secondary": secondary,
    }


def _hyperedge_competition_term(x: np.ndarray, hyperedges: list[tuple[int, ...]], weight: float) -> np.ndarray:
    term = np.zeros_like(x)
    for e in hyperedges:
        vals = x[list(e)]
        for local_idx, node in enumerate(e):
            others_prod = np.prod(np.delete(vals, local_idx))
            term[node] -= weight * x[node] * others_prod
    return term


def _predation_term(
    x: np.ndarray,
    pred_edges: list[tuple[int, int, float, float]],
    half_sat: float,
) -> np.ndarray:
    """
        For each directed predation edge prey -> predator:
            flux = attack * predator * prey
      prey    -= flux
      predator += eta * flux
    """
    term = np.zeros_like(x)
    for prey, predator, attack, eta in pred_edges:
        prey_val = x[prey]
        pred_val = x[predator]
        flux = attack * pred_val * prey_val

        term[prey] -= flux
        term[predator] += eta * flux
    return term


def rhs_ecological_hypergraph(
    t: float,
    x: np.ndarray,
    hypergraph: dict,
    producer_growth_rates: np.ndarray,
    params: EcologicalHypergraphParams,
) -> np.ndarray:
    x = np.clip(x, 1e-12, None)

    producers = hypergraph["producers"]
    primary = hypergraph["primary"]
    secondary = hypergraph["secondary"]

    dx = np.zeros_like(x)

    # Producers: autonomous growth + replenishment, with carrying cap on producer biomass.
    seasonal = 1.0 + params.season_amplitude * np.sin(2.0 * np.pi * t / params.season_period)
    for local_idx, node in enumerate(producers):
        r = producer_growth_rates[local_idx]
        dx[node] += seasonal * r * x[node] * (1.0 - x[node] / params.producer_carrying_capacity)
        dx[node] += params.producer_replenish * (1.0 - x[node])

    # Consumer baseline mortality.
    dx[primary] -= params.primary_mortality * x[primary]
    dx[secondary] -= params.secondary_mortality * x[secondary]
    dx[primary] += params.primary_immigration
    dx[secondary] += params.secondary_immigration

    # Undirected higher-order competition.
    dx += _hyperedge_competition_term(x, hypergraph["edges2"], params.w2)
    dx += _hyperedge_competition_term(x, hypergraph["edges3"], params.w3)
    dx += _hyperedge_competition_term(x, hypergraph["edges4"], params.w4)

    # Directed predation on order-2 edges only.
    dx += _predation_term(x, hypergraph["pred_edges"], params.half_sat)

    return dx


def simulate(params: EcologicalHypergraphParams):
    _validate_params(params)
    rng = np.random.default_rng(params.seed)

    hypergraph = build_sparse_hypergraph_and_foodchain(params, rng)

    producer_growth_rates = rng.normal(params.producer_growth_mean, params.producer_growth_std, size=params.n_producers)
    producer_growth_rates = np.clip(producer_growth_rates, 0.55, 0.95)

    x0 = rng.uniform(0.04, 0.15, size=params.n_species)
    x0[hypergraph["producers"]] += 0.04
    x0 = np.clip(x0, 1e-6, None)

    t_eval = np.linspace(params.t_span[0], params.t_span[1], params.n_steps)

    sol = solve_ivp(
        fun=lambda t, x: rhs_ecological_hypergraph(t, x, hypergraph, producer_growth_rates, params),
        t_span=params.t_span,
        y0=x0,
        t_eval=t_eval,
        method="RK45",
        rtol=params.rtol,
        atol=params.atol,
    )

    X = np.clip(sol.y, 0.0, None)

    return {
        "t": sol.t,
        "X": X,
        "x0": x0,
        **hypergraph,
    }


def plot_nodes_timeseries(t: np.ndarray, X: np.ndarray, hypergraph: dict, save_path: Path):
    n = X.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.8))

    producers = set(hypergraph["producers"].tolist())
    primary = set(hypergraph["primary"].tolist())

    for i in range(n):
        if i in producers:
            color = "tab:green"
        elif i in primary:
            color = "tab:blue"
        else:
            color = "tab:red"
        ax.plot(t, X[i], lw=1.2, alpha=0.9, color=color)

    ax.set_title(f"Sparse ecological hypergraph dynamics (N={n})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Abundance")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    params = EcologicalHypergraphParams()
    result = simulate(params)

    out = BASE_DIR / "ecology_nodes_timeseries.png"
    plot_nodes_timeseries(result["t"], result["X"], result, out)

    print("=" * 78)
    print("Sparse ecological hypergraph summary (competition + directed predation)")
    print("=" * 78)
    print(f"Nodes: {len(result['nodes'])}")
    print(f"Groups: producers={list(result['producers'])}, primary={list(result['primary'])}, secondary={list(result['secondary'])}")
    print(f"Undirected order-2 competition edges: {len(result['edges2'])}")
    print(f"Undirected order-3 competition edges: {len(result['edges3'])}")
    print(f"Undirected order-4 competition edges: {len(result['edges4'])}")
    print(f"Directed order-2 predation edges: {len(result['pred_edges'])}")
    print(f"Predation edge examples (prey, predator, attack, eta): {result['pred_edges'][:8]}")
    print(f"Saved: {out}")
