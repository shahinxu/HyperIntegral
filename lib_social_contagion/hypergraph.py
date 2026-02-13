"""
Hypergraph-style interface for social contagion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch


class HypergraphModel:
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

    @staticmethod
    def _rsc_probabilities(n_nodes: int, k_mean: float, k_delta: float) -> tuple[float, float]:
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

    @staticmethod
    def _unrank_pair(n_nodes: int, rank: int) -> tuple[int, int]:
        for i in range(n_nodes - 1):
            count = n_nodes - i - 1
            if rank < count:
                return i, i + 1 + rank
            rank -= count
        raise ValueError("Pair rank out of range")

    @staticmethod
    def _unrank_triplet(n_nodes: int, rank: int) -> tuple[int, int, int]:
        for i in range(n_nodes - 2):
            count_i = (n_nodes - i - 1) * (n_nodes - i - 2) // 2
            if rank < count_i:
                for j in range(i + 1, n_nodes - 1):
                    count_j = n_nodes - j - 1
                    if rank < count_j:
                        return i, j, j + 1 + rank
                    rank -= count_j
            rank -= count_i
        raise ValueError("Triplet rank out of range")

    @staticmethod
    def _sample_unique_combinations(
        n_nodes: int,
        order: int,
        p: float,
        rng: np.random.Generator,
    ) -> list[tuple[int, ...]]:
        total = 1
        for i in range(order):
            total = total * (n_nodes - i) // (i + 1)
        m = int(rng.binomial(total, p))
        if m == 0:
            return []

        ranks = rng.choice(total, size=m, replace=False)
        ranks.sort()

        if order == 2:
            return [HypergraphModel._unrank_pair(n_nodes, int(r)) for r in ranks]
        if order == 3:
            return [HypergraphModel._unrank_triplet(n_nodes, int(r)) for r in ranks]

        raise ValueError("Only order=2 or order=3 are supported")

    @staticmethod
    def _build_rsc_simplicial_complex(params: "HypergraphModel.RSCParams") -> dict:
        rng = np.random.default_rng(params.seed)
        p1, p_delta = HypergraphModel._rsc_probabilities(params.n_nodes, params.k_mean, params.k_delta)

        edges = HypergraphModel._sample_unique_combinations(params.n_nodes, 2, p1, rng)
        triangles = HypergraphModel._sample_unique_combinations(params.n_nodes, 3, p_delta, rng)

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

    @staticmethod
    def _build_edge_adjacency(edges: list[tuple[int, int]], n_nodes: int) -> list[list[int]]:
        adjacency = [[] for _ in range(n_nodes)]
        for i, j in edges:
            adjacency[i].append(j)
            adjacency[j].append(i)
        return adjacency

    @staticmethod
    def _build_triangle_pairs(triangles: list[tuple[int, int, int]], n_nodes: int) -> list[list[tuple[int, int]]]:
        pairs = [[] for _ in range(n_nodes)]
        for i, j, k in triangles:
            pairs[i].append((j, k))
            pairs[j].append((i, k))
            pairs[k].append((i, j))
        return pairs

    @staticmethod
    def _simulate_scm(complex_dict: dict, params: "HypergraphModel.SCMParams") -> dict:
        n_nodes = len(complex_dict["nodes"])
        rng = np.random.default_rng(params.seed)

        edges = complex_dict["edges"]
        triangles = complex_dict["triangles"]
        adjacency = HypergraphModel._build_edge_adjacency(edges, n_nodes)
        triangle_pairs = HypergraphModel._build_triangle_pairs(triangles, n_nodes)

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

    @staticmethod
    def dynamic(x: torch.Tensor, n_nodes: int) -> torch.Tensor:
        return torch.zeros_like(x)

    @staticmethod
    def compute_hyperedge_coupling_tensor(
        x: torch.Tensor,
        all_possible_edges: dict,
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        n_total = 0
        for key in ["edges", "triangles", "quads", "quints", "sexts", "septs"]:
            n_total += len(all_possible_edges.get(key, []))
        if x.ndim == 1:
            x_vec = x
        else:
            x_vec = x[:, 0]

        phi = torch.zeros((n_nodes, 1, n_total), device=device)
        edge_idx = 0

        edges = all_possible_edges.get("edges", [])
        if len(edges) > 0:
            edges_t = torch.as_tensor(edges, dtype=torch.long, device=device)
            i_idx = edges_t[:, 0]
            j_idx = edges_t[:, 1]
            edge_range = edge_idx + torch.arange(edges_t.shape[0], device=device)
            phi[i_idx, 0, edge_range] = x_vec[j_idx]
            phi[j_idx, 0, edge_range] = x_vec[i_idx]
            edge_idx += edges_t.shape[0]

        triangles = all_possible_edges.get("triangles", [])
        if len(triangles) > 0:
            tri_t = torch.as_tensor(triangles, dtype=torch.long, device=device)
            i_idx = tri_t[:, 0]
            j_idx = tri_t[:, 1]
            k_idx = tri_t[:, 2]
            edge_range = edge_idx + torch.arange(tri_t.shape[0], device=device)
            phi[i_idx, 0, edge_range] = x_vec[j_idx] * x_vec[k_idx]
            phi[j_idx, 0, edge_range] = x_vec[i_idx] * x_vec[k_idx]
            phi[k_idx, 0, edge_range] = x_vec[i_idx] * x_vec[j_idx]
            edge_idx += tri_t.shape[0]

        return phi

    @staticmethod
    def get_hyperedge_config(
        n_nodes: int,
        max_order: int = 3,
        k_mean: float = 20.0,
        k_delta: float = 6.0,
        seed: int = 42,
        enforce_closure: bool = True,
    ) -> dict:
        params = HypergraphModel.RSCParams(
            n_nodes=n_nodes,
            k_mean=k_mean,
            k_delta=k_delta,
            seed=seed,
            enforce_closure=enforce_closure,
        )
        complex_dict = HypergraphModel._build_rsc_simplicial_complex(params)

        edges_1b = [[i + 1, j + 1] for i, j in complex_dict["edges"]]
        triangles_1b = [[i + 1, j + 1, k + 1] for i, j, k in complex_dict["triangles"]]

        return {
            "edges": edges_1b if max_order >= 2 else [],
            "triangles": triangles_1b if max_order >= 3 else [],
            "quads": [],
            "quints": [],
            "sexts": [],
            "septs": [],
        }

    @staticmethod
    def generate_all_possible_hyperedges(n_nodes: int, max_order: int) -> dict:
        all_possible = {}

        if max_order >= 2:
            all_possible["edges"] = [list(edge) for edge in combinations(range(1, n_nodes + 1), 2)]
        else:
            all_possible["edges"] = []

        if max_order >= 3:
            all_possible["triangles"] = [list(edge) for edge in combinations(range(1, n_nodes + 1), 3)]
        else:
            all_possible["triangles"] = []

        all_possible["quads"] = []
        all_possible["quints"] = []
        all_possible["sexts"] = []
        all_possible["septs"] = []

        return all_possible

    @staticmethod
    def generate_training_data(n_nodes: int, edge_config: dict, n_samples: int = 11, noise: float = 0.0):
        edges = [[i - 1, j - 1] for i, j in edge_config.get("edges", [])]
        triangles = [[i - 1, j - 1, k - 1] for i, j, k in edge_config.get("triangles", [])]

        complex_dict = {
            "nodes": np.arange(n_nodes, dtype=int),
            "edges": edges,
            "triangles": triangles,
        }

        params = HypergraphModel.SCMParams(t_max=max(1, n_samples - 1))
        sim = HypergraphModel._simulate_scm(complex_dict, params)
        states = sim["states"].astype(float)
        t = np.arange(states.shape[0], dtype=float)

        if noise > 0:
            states = states + np.random.randn(*states.shape) * noise

        x_data = states[:, :, None]
        return t, x_data

    @staticmethod
    def get_default_params() -> dict:
        return {
            "n_nodes": 2000,
            "max_order": 3,
        }
