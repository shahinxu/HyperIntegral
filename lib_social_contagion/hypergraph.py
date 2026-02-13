"""
Hypergraph-style interface for social contagion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from scipy.integrate import solve_ivp


class HypergraphModel:
    _cached_edges = None
    _cached_triangles = None

    @dataclass
    class RSCParams:
        n_nodes: int = 2000
        k_mean: float = 20.0
        k_delta: float = 6.0
        seed: int = 42
        enforce_closure: bool = True

    @dataclass
    class SCMParams:
        n_nodes: int = 2000
        t_max: float = 50.0
        beta: float = 0.5
        beta_delta: float = 2.0
        mu: float = 0.4
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
    def _cache_hyperedges(edges: list[list[int]], triangles: list[list[int]]) -> None:
        HypergraphModel._cached_edges = [list(e) for e in edges]
        HypergraphModel._cached_triangles = [list(t) for t in triangles]

    @staticmethod
    def _edge_config_to_zero_based(edge_config: dict) -> tuple[list[list[int]], list[list[int]]]:
        edges = [[i - 1, j - 1] for i, j in edge_config.get("edges", [])]
        triangles = [[i - 1, j - 1, k - 1] for i, j, k in edge_config.get("triangles", [])]
        return edges, triangles

    @staticmethod
    def generate_training_data(n_nodes: int, edge_config: dict, n_samples: int = 100, noise: float = 0.0):
        params = HypergraphModel.SCMParams(n_nodes=n_nodes, t_max=50)
        complex_dict = {
            "edges": [],
            "triangles": [],
            "nodes": np.arange(n_nodes)
        }
        
        if "edges" in edge_config:
            e_list = edge_config["edges"]
            if len(e_list) > 0:
                e_arr = np.array(e_list)
                if e_arr.min() >= 1:
                    complex_dict["edges"] = (e_arr - 1).tolist()
                else:
                    complex_dict["edges"] = e_list
                    
        if "triangles" in edge_config:
            t_list = edge_config["triangles"]
            if len(t_list) > 0:
                t_arr = np.array(t_list)
                if t_arr.min() >= 1:
                    complex_dict["triangles"] = (t_arr - 1).tolist()
                else:
                    complex_dict["triangles"] = t_list

        result = HypergraphModel._simulate(params, complex_dict, n_steps=n_samples)
        HypergraphModel._cached_edges = complex_dict["edges"]
        HypergraphModel._cached_triangles = complex_dict["triangles"]

        t = result["t"]
        x_observed = result["X_observed"].T[:, :, None]
        if x_observed.shape[0] != n_samples:
             indices = np.linspace(0, x_observed.shape[0]-1, n_samples).astype(int)
             x_observed = x_observed[indices]
             t = np.linspace(0, params.t_max, n_samples)
        if noise > 0:
             x_observed = x_observed + np.random.randn(*x_observed.shape) * noise

        return t, x_observed

    @staticmethod
    def _rhs(
        t: float,
        x: np.ndarray,
        edges: list[list[int]],
        triangles: list[list[int]],
        params: "HypergraphModel.SCMParams",
    ) -> np.ndarray:
        
        n_nodes = x.shape[0]
        infection_force = np.zeros(n_nodes)

        # Pairwise force
        if edges:
            for i, j in edges:
                infection_force[i] += params.beta * x[j]
                infection_force[j] += params.beta * x[i]

        if triangles:
            for i, j, k in triangles:
                infection_force[i] += params.beta_delta * (x[j] * x[k])
                infection_force[j] += params.beta_delta * (x[i] * x[k])
                infection_force[k] += params.beta_delta * (x[i] * x[j])

        dx = -params.mu * x + (1.0 - x) * infection_force
        return dx

    @staticmethod
    def _simulate(params: "HypergraphModel.SCMParams", complex_dict: dict, n_steps: int = 100):
        rng = np.random.default_rng(params.seed)
        
        # Initial condition: Fraction of infected people (continuous probability [0,1])
        # Start with a small random seed of infection
        x0 = rng.uniform(0.0, 0.1, size=params.n_nodes)
        
        # Ensure at least some infection to start dynamics
        seeds = rng.choice(params.n_nodes, size=max(1, int(0.1 * params.n_nodes)), replace=False)
        x0[seeds] = rng.uniform(0.2, 0.5, size=len(seeds))

        t_eval = np.linspace(0, params.t_max, n_steps) # n_steps for integration

        edges = complex_dict["edges"]
        triangles = complex_dict["triangles"]

        sol = solve_ivp(
            fun=lambda t, x: HypergraphModel._rhs(t, x, edges, triangles, params),
            t_span=(0, params.t_max),
            y0=x0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        
        X_continuous = np.clip(sol.y, 0.0, 1.0) # Probability x in [0,1]
        X_observed = rng.binomial(1, X_continuous).astype(float)

        return {
            "t": sol.t,
            "X_continuous": X_continuous, # Truth (Probability)
            "X_observed": X_observed,     # Observation (Binary)
            "x0": x0,
            **complex_dict
        }

    @staticmethod
    def dynamic_f(x: torch.Tensor, n_nodes: int) -> torch.Tensor:
        # The drift part of the ODE: f(x) = -mu * x
        # This is the part INDEPENDENT of the unknown structure
        params = HypergraphModel.SCMParams()
        
        if x.ndim == 1:
            x_vec = x
            out_shape = None
        else:
            x_vec = x[:, 0]
            out_shape = x.shape

        dx = -params.mu * x_vec

        if out_shape is not None:
            return dx.view(out_shape)
        return dx

    @staticmethod
    def dynamic_phi(
        x: torch.Tensor,
        all_possible_edges: dict,
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        # The interaction part: Phi(x) * A
        # For each candidate edge/triangle, compute its contribution term
        # Term = (1 - x_i) * force
        
        params = HypergraphModel.SCMParams()
        n_total = 0
        for key in ["edges", "triangles"]:
            n_total += len(all_possible_edges.get(key, []))

        if x.ndim == 1:
            x_vec = x
            d = 1
        else:
            x_vec = x[:, 0]
            d = x.shape[1]

        # Precompute (1-x)
        susceptible = 1.0 - x_vec
        
        # Phi shape: (N_nodes, Dimension_of_states, N_candidates)
        # Here Dimension is 1 (infection probability)
        Phi = torch.zeros((n_nodes, 1, n_total), device=device, dtype=x.dtype)
        edge_idx = 0

        # 1. Candidate Edges (Pairwise)
        edges = all_possible_edges.get("edges", [])
        if len(edges) > 0:
            edges_t = torch.as_tensor(edges, dtype=torch.long, device=device)
            num_edges = edges_t.shape[0]
            i_idx = edges_t[:, 0]
            j_idx = edges_t[:, 1]
            feature_range = edge_idx + torch.arange(num_edges, device=device)

            # Interaction: (1-x_i) * beta * x_j
            term_i = params.beta * susceptible[i_idx] * x_vec[j_idx]
            term_j = params.beta * susceptible[j_idx] * x_vec[i_idx]
            
            Phi[i_idx, 0, feature_range] = term_i
            Phi[j_idx, 0, feature_range] = term_j
            
            edge_idx += num_edges

        # 2. Candidate Triangles (Simplicial)
        triangles = all_possible_edges.get("triangles", [])
        if len(triangles) > 0:
            tri_t = torch.as_tensor(triangles, dtype=torch.long, device=device)
            num_tris = tri_t.shape[0]
            i_idx = tri_t[:, 0]
            j_idx = tri_t[:, 1]
            k_idx = tri_t[:, 2]
            feature_range = edge_idx + torch.arange(num_tris, device=device)

            term_i = params.beta_delta * susceptible[i_idx] * (x_vec[j_idx] * x_vec[k_idx])
            term_j = params.beta_delta * susceptible[j_idx] * (x_vec[i_idx] * x_vec[k_idx])
            term_k = params.beta_delta * susceptible[k_idx] * (x_vec[i_idx] * x_vec[j_idx])

            Phi[i_idx, 0, feature_range] = term_i
            Phi[j_idx, 0, feature_range] = term_j
            Phi[k_idx, 0, feature_range] = term_k
            
            edge_idx += num_tris

        return Phi

    @staticmethod
    def get_hyperedge_config(
        n_nodes: int,
        max_order: int = 3,
        k_mean: float = 20.0,
        k_delta: float = 6.0,
        seed: int = 42,
        enforce_closure: bool = True,
    ) -> dict:
        if n_nodes == 8:
            config = {
                "edges": [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]],
                "triangles": [[1, 2, 3], [2, 4, 5], [5, 6, 7], [6, 7, 8]],
            }
        else:
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

            config = {
                "edges": edges_1b,
                "triangles": triangles_1b,
            }

        edges_0b, triangles_0b = HypergraphModel._edge_config_to_zero_based(config)
        HypergraphModel._cache_hyperedges(edges_0b, triangles_0b)

        return {
            "edges": config["edges"] if max_order >= 2 else [],
            "triangles": config["triangles"] if max_order >= 3 else [],
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

        return all_possible


    @staticmethod
    def get_default_params() -> dict:
        return {
            "n_nodes": 8,
            "max_order": 3,
        }
