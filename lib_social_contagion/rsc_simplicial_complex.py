"""
Random simplicial complex (RSC) construction for social contagion.

Reference:
- Iacopini et al., "Simplicial models of social contagion", Nat. Commun. 2019.

This builds a simplicial complex of dimension D=2 with:
- 1-simplices (edges) sampled with probability p1
- 2-simplices (full triangles) sampled with probability p_delta

For a desired average node degree <k> and average number of triangles per node
<k_delta>, the paper gives (for small p1, p_delta):
    p1 = (<k> - 2<k_delta>) / ((N - 1) - 2<k_delta>)
    p_delta = 2<k_delta> / ((N - 1)(N - 2))
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class RSCParams:
    n_nodes: int = 2000
    k_mean: float = 20.0
    k_delta: float = 6.0
    seed: int = 42
    enforce_closure: bool = True


@dataclass
class SCMParams:
    beta: float = 0.08
    beta_delta: float = 0.18
    mu: float = 0.1
    rho0: float = 0.05
    t_max: int = 500
    seed: int = 123


def rsc_probabilities(n_nodes: int, k_mean: float, k_delta: float) -> tuple[float, float]:
    if n_nodes < 3:
        raise ValueError("n_nodes must be >= 3 for 2-simplices")
    if k_delta < 0:
        raise ValueError("k_delta must be >= 0")

    denom = (n_nodes - 1) - 2.0 * k_delta
    if denom <= 0:
        raise ValueError("k_mean and k_delta are inconsistent with n_nodes")

    p1 = (k_mean - 2.0 * k_delta) / denom
    p_delta = 2.0 * k_delta / ((n_nodes - 1) * (n_nodes - 2))

    if not (0.0 <= p1 <= 1.0):
        raise ValueError("p1 is outside [0, 1]; check k_mean and k_delta")
    if not (0.0 <= p_delta <= 1.0):
        raise ValueError("p_delta is outside [0, 1]; check k_delta and n_nodes")

    return p1, p_delta


def _unrank_pair(n_nodes: int, rank: int) -> tuple[int, int]:
    for i in range(n_nodes - 1):
        count = n_nodes - i - 1
        if rank < count:
            return i, i + 1 + rank
        rank -= count
    raise ValueError("Pair rank out of range")


def _unrank_triplet(n_nodes: int, rank: int) -> tuple[int, int, int]:
    for i in range(n_nodes - 2):
        count_i = comb(n_nodes - i - 1, 2)
        if rank < count_i:
            for j in range(i + 1, n_nodes - 1):
                count_j = n_nodes - j - 1
                if rank < count_j:
                    return i, j, j + 1 + rank
                rank -= count_j
        rank -= count_i
    raise ValueError("Triplet rank out of range")


def _sample_unique_combinations(
    n_nodes: int,
    order: int,
    p: float,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    total = comb(n_nodes, order)
    m = int(rng.binomial(total, p))
    if m == 0:
        return []

    ranks = rng.choice(total, size=m, replace=False)
    ranks.sort()

    if order == 2:
        return [_unrank_pair(n_nodes, int(r)) for r in ranks]
    if order == 3:
        return [_unrank_triplet(n_nodes, int(r)) for r in ranks]

    raise ValueError("Only order=2 or order=3 are supported")


def build_rsc_simplicial_complex(params: RSCParams) -> dict:
    rng = np.random.default_rng(params.seed)
    p1, p_delta = rsc_probabilities(params.n_nodes, params.k_mean, params.k_delta)

    edges = _sample_unique_combinations(params.n_nodes, 2, p1, rng)
    triangles = _sample_unique_combinations(params.n_nodes, 3, p_delta, rng)

    edge_set = {tuple(sorted(e)) for e in edges}
    if params.enforce_closure:
        for i, j, k in triangles:
            edge_set.add(tuple(sorted((i, j))))
            edge_set.add(tuple(sorted((i, k))))
            edge_set.add(tuple(sorted((j, k))))

    edges = sorted(edge_set)
    triangles = sorted({tuple(sorted(t)) for t in triangles})

    return {
        "nodes": np.arange(params.n_nodes, dtype=int),
        "edges": edges,
        "triangles": triangles,
        "p1": p1,
        "p_delta": p_delta,
        "k_mean_target": params.k_mean,
        "k_delta_target": params.k_delta,
        "seed": params.seed,
        "enforce_closure": params.enforce_closure,
    }


def _build_edge_adjacency(edges: Iterable[tuple[int, int]], n_nodes: int) -> list[list[int]]:
    adjacency = [[] for _ in range(n_nodes)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    return adjacency


def _build_triangle_pairs(triangles: Iterable[tuple[int, int, int]], n_nodes: int) -> list[list[tuple[int, int]]]:
    pairs = [[] for _ in range(n_nodes)]
    for i, j, k in triangles:
        pairs[i].append((j, k))
        pairs[j].append((i, k))
        pairs[k].append((i, j))
    return pairs


def simulate_scm(complex_dict: dict, params: SCMParams) -> dict:
    n_nodes = len(complex_dict["nodes"])
    rng = np.random.default_rng(params.seed)

    edges = complex_dict["edges"]
    triangles = complex_dict["triangles"]
    adjacency = _build_edge_adjacency(edges, n_nodes)
    triangle_pairs = _build_triangle_pairs(triangles, n_nodes)

    infected = rng.random(n_nodes) < params.rho0
    states = np.zeros((params.t_max + 1, n_nodes), dtype=np.uint8)
    states[0] = infected.astype(np.uint8)

    for t in range(1, params.t_max + 1):
        next_infected = infected.copy()

        for i in range(n_nodes):
            if infected[i]:
                if rng.random() < params.mu:
                    next_infected[i] = False
                continue

            p_not = 1.0
            for j in adjacency[i]:
                if infected[j]:
                    p_not *= 1.0 - params.beta
                    if p_not <= 0.0:
                        break

            if p_not > 0.0:
                for j, k in triangle_pairs[i]:
                    if infected[j] and infected[k]:
                        p_not *= 1.0 - params.beta_delta
                        if p_not <= 0.0:
                            break

            if rng.random() < 1.0 - p_not:
                next_infected[i] = True

        infected = next_infected
        states[t] = infected.astype(np.uint8)

    return {
        "states": states,
        "infected": infected,
        "params": params,
    }


def summarize_complex(complex_dict: dict) -> str:
    n_nodes = len(complex_dict["nodes"])
    n_edges = len(complex_dict["edges"])
    n_triangles = len(complex_dict["triangles"])

    return (
        "RSC simplicial complex summary\n"
        f"- nodes: {n_nodes}\n"
        f"- edges: {n_edges}\n"
        f"- triangles: {n_triangles}\n"
        f"- p1: {complex_dict['p1']:.6f}\n"
        f"- p_delta: {complex_dict['p_delta']:.6e}\n"
    )


def save_npz(complex_dict: dict, out_path: Path) -> None:
    edges = np.array(complex_dict["edges"], dtype=int)
    triangles = np.array(complex_dict["triangles"], dtype=int)
    np.savez_compressed(
        out_path,
        nodes=complex_dict["nodes"],
        edges=edges,
        triangles=triangles,
        p1=complex_dict["p1"],
        p_delta=complex_dict["p_delta"],
        k_mean_target=complex_dict["k_mean_target"],
        k_delta_target=complex_dict["k_delta_target"],
        seed=complex_dict["seed"],
        enforce_closure=complex_dict["enforce_closure"],
    )


def save_node_timeseries(states: np.ndarray, out_path: Path, max_nodes: int = 10) -> None:
    t = np.arange(states.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=(12, 7.2))
    n_nodes = states.shape[1]
    plot_nodes = min(max_nodes, n_nodes)
    for i in range(plot_nodes):
        ax.plot(t, states[:, i], lw=1.2, alpha=0.9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Node state")
    ax.set_title(f"SCM node trajectories (first {plot_nodes} nodes)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    params = RSCParams()
    complex_dict = build_rsc_simplicial_complex(params)
    print(summarize_complex(complex_dict))

    sim_params = SCMParams()
    sim = simulate_scm(complex_dict, sim_params)
    print("SCM dynamics summary")
    print(f"- t_max: {sim_params.t_max}")
    print(f"- rho0: {sim_params.rho0:.3f}")
    print(f"- rho_final: {sim['states'][-1].mean():.3f}")

    fig_path = Path(__file__).resolve().parent / "scm_node_timeseries.png"
    save_node_timeseries(sim["states"], fig_path, max_nodes=1)
    print(f"Saved: {fig_path}")
