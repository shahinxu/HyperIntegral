"""
Hypergraph-style interface for social contagion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch


class HypergraphModel:
    _cached_edges = None
    _cached_triangles = None
    _state_exposure = None
    _state_recovery = None
    _state_device = None
    _state_dtype = None

    @dataclass
    class RSCParams:
        n_nodes: int = 2000
        k_mean: float = 20.0
        k_delta: float = 6.0
        seed: int = 42
        enforce_closure: bool = True

    @dataclass
    class SCMParams:
        beta: float = 0.05
        beta_delta: float = 0.2
        mu: float = 0.1
        rho0: float = 0.5
        infect_threshold: float = 1.0
        state_threshold: float = 0.5
        t_max: int = 500
        seed: int = 123

    @staticmethod
    def reset_internal_state() -> None:
        HypergraphModel._state_exposure = None
        HypergraphModel._state_recovery = None
        HypergraphModel._state_device = None
        HypergraphModel._state_dtype = None

    @staticmethod
    def _ensure_state(n_nodes: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            HypergraphModel._state_exposure is None
            or HypergraphModel._state_recovery is None
            or HypergraphModel._state_exposure.numel() != n_nodes
            or HypergraphModel._state_device != device
            or HypergraphModel._state_dtype != dtype
        ):
            HypergraphModel._state_exposure = torch.zeros(n_nodes, device=device, dtype=dtype)
            HypergraphModel._state_recovery = torch.zeros(n_nodes, device=device, dtype=dtype)
            HypergraphModel._state_device = device
            HypergraphModel._state_dtype = dtype

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
    def _build_complex_dict(n_nodes: int, edges: list[list[int]], triangles: list[list[int]]) -> dict:
        return {
            "nodes": np.arange(n_nodes, dtype=int),
            "edges": edges,
            "triangles": triangles,
        }

    @staticmethod
    def _as_infected_mask(x_vec: torch.Tensor, threshold: float) -> torch.Tensor:
        return x_vec >= threshold

    @staticmethod
    def _exposure_increment_numpy(
        infected: np.ndarray,
        edges: list[list[int]],
        triangles: list[list[int]],
        beta: float,
        beta_delta: float,
        n_nodes: int,
    ) -> np.ndarray:
        exposure_inc = np.zeros(n_nodes, dtype=float)

        if edges:
            for i, j in edges:
                if infected[j]:
                    exposure_inc[i] += 1.0
                if infected[i]:
                    exposure_inc[j] += 1.0
            exposure_inc *= beta

        if triangles:
            tri_inc = np.zeros(n_nodes, dtype=float)
            for i, j, k in triangles:
                if infected[j] and infected[k]:
                    tri_inc[i] += 1.0
                if infected[i] and infected[k]:
                    tri_inc[j] += 1.0
                if infected[i] and infected[j]:
                    tri_inc[k] += 1.0
            exposure_inc += tri_inc * beta_delta

        return exposure_inc

    @staticmethod
    def _exposure_increment_torch(
        infected: torch.Tensor,
        edges: list | torch.Tensor,
        triangles: list | torch.Tensor,
        beta: float,
        beta_delta: float,
        n_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        exposure_inc = torch.zeros(n_nodes, device=device, dtype=dtype)

        if isinstance(edges, list):
            edges_t = torch.as_tensor(edges, dtype=torch.long, device=device)
        else:
            edges_t = edges

        if edges_t.numel() > 0:
            i_idx = edges_t[:, 0]
            j_idx = edges_t[:, 1]
            exposure_inc.index_add_(0, i_idx, infected[j_idx].to(dtype))
            exposure_inc.index_add_(0, j_idx, infected[i_idx].to(dtype))
            exposure_inc = exposure_inc * beta

        if isinstance(triangles, list):
            tri_t = torch.as_tensor(triangles, dtype=torch.long, device=device)
        else:
            tri_t = triangles

        if tri_t.numel() > 0:
            i_idx = tri_t[:, 0]
            j_idx = tri_t[:, 1]
            k_idx = tri_t[:, 2]
            tri_inc = torch.zeros(n_nodes, device=device, dtype=dtype)
            tri_inc.index_add_(0, i_idx, (infected[j_idx] & infected[k_idx]).to(dtype))
            tri_inc.index_add_(0, j_idx, (infected[i_idx] & infected[k_idx]).to(dtype))
            tri_inc.index_add_(0, k_idx, (infected[i_idx] & infected[j_idx]).to(dtype))
            exposure_inc = exposure_inc + tri_inc * beta_delta

        return exposure_inc

    @staticmethod
    def dynamic_f(x: torch.Tensor, n_nodes: int) -> torch.Tensor:
        params = HypergraphModel.SCMParams()
        if x.ndim == 1:
            x_vec = x
            out_shape = None
        else:
            x_vec = x[:, 0]
            out_shape = x.shape

        if x_vec.numel() != n_nodes:
            raise ValueError(f"n_nodes mismatch: got {n_nodes}, but x has {x_vec.numel()} entries")

        HypergraphModel._ensure_state(n_nodes, x_vec.device, x_vec.dtype)

        infected = x_vec >= params.state_threshold
        exposure = HypergraphModel._state_exposure
        recovery = HypergraphModel._state_recovery

        recovery[infected] = recovery[infected] + params.mu
        recovered = recovery >= 1.0

        next_infected = infected.clone()
        next_infected[recovered] = False
        recovery[recovered] = 0.0

        susceptible = ~infected
        new_infected = susceptible & (exposure >= params.infect_threshold)
        next_infected[new_infected] = True

        exposure[infected | new_infected] = 0.0

        dx = next_infected.to(x_vec.dtype) - infected.to(x_vec.dtype)

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
        params = HypergraphModel.SCMParams()
        n_total = 0
        for key in ["edges", "triangles"]:
            n_total += len(all_possible_edges.get(key, []))
        if x.ndim == 1:
            x_vec = x
        else:
            x_vec = x[:, 0]

        infected = HypergraphModel._as_infected_mask(x_vec, params.state_threshold)
        HypergraphModel._ensure_state(n_nodes, device, x_vec.dtype)

        phi = torch.zeros((n_nodes, 1, n_total), device=device)
        edge_idx = 0

        edges = all_possible_edges.get("edges", [])
        if len(edges) > 0:
            edges_t = torch.as_tensor(edges, dtype=torch.long, device=device)
            i_idx = edges_t[:, 0]
            j_idx = edges_t[:, 1]
            edge_range = edge_idx + torch.arange(edges_t.shape[0], device=device)
            phi[i_idx, 0, edge_range] = params.beta * infected[j_idx].to(x_vec.dtype)
            phi[j_idx, 0, edge_range] = params.beta * infected[i_idx].to(x_vec.dtype)
            edge_idx += edges_t.shape[0]

        triangles = all_possible_edges.get("triangles", [])
        if len(triangles) > 0:
            tri_t = torch.as_tensor(triangles, dtype=torch.long, device=device)
            i_idx = tri_t[:, 0]
            j_idx = tri_t[:, 1]
            k_idx = tri_t[:, 2]
            edge_range = edge_idx + torch.arange(tri_t.shape[0], device=device)
            tri_active = (infected[j_idx] & infected[k_idx]).to(x_vec.dtype)
            phi[i_idx, 0, edge_range] = params.beta_delta * tri_active
            tri_active = (infected[i_idx] & infected[k_idx]).to(x_vec.dtype)
            phi[j_idx, 0, edge_range] = params.beta_delta * tri_active
            tri_active = (infected[i_idx] & infected[j_idx]).to(x_vec.dtype)
            phi[k_idx, 0, edge_range] = params.beta_delta * tri_active
            edge_idx += tri_t.shape[0]

        edges_cached = HypergraphModel._cached_edges or []
        triangles_cached = HypergraphModel._cached_triangles or []
        if edges_cached or triangles_cached:
            susceptible = ~infected
            exposure_inc = HypergraphModel._exposure_increment_torch(
                infected,
                edges_cached,
                triangles_cached,
                params.beta,
                params.beta_delta,
                n_nodes,
                device,
                x_vec.dtype,
            )
            HypergraphModel._state_exposure[susceptible] = (
                HypergraphModel._state_exposure[susceptible] + exposure_inc[susceptible]
            )

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
    def generate_training_data(n_nodes: int, edge_config: dict, n_samples: int = 11, noise: float = 0.0):
        edges, triangles = HypergraphModel._edge_config_to_zero_based(edge_config)
        HypergraphModel._cache_hyperedges(edges, triangles)
        params = HypergraphModel.SCMParams(t_max=max(1, n_samples - 1))

        infected = np.random.default_rng(params.seed).random(n_nodes) < params.rho0
        exposure = np.zeros(n_nodes, dtype=float)
        recovery = np.zeros(n_nodes, dtype=float)

        states = np.zeros((params.t_max + 1, n_nodes), dtype=np.uint8)
        states[0] = infected.astype(np.uint8)

        for t_idx in range(1, params.t_max + 1):
            next_infected = infected.copy()

            for i in range(n_nodes):
                if infected[i]:
                    recovery[i] += params.mu
                    if recovery[i] >= 1.0:
                        next_infected[i] = False
                        recovery[i] = 0.0
                    exposure[i] = 0.0
                    continue

            exposure_inc = HypergraphModel._exposure_increment_numpy(
                infected,
                edges,
                triangles,
                params.beta,
                params.beta_delta,
                n_nodes,
            )
            exposure[~infected] = exposure[~infected] + exposure_inc[~infected]

            new_infected = (~infected) & (exposure >= params.infect_threshold)
            next_infected[new_infected] = True
            exposure[new_infected] = 0.0

            infected = next_infected
            states[t_idx] = infected.astype(np.uint8)

        states = states.astype(float)
        t = np.arange(states.shape[0], dtype=float)

        if noise > 0:
            states = states + np.random.randn(*states.shape) * noise

        x_data = states[:, :, None]
        return t, x_data

    @staticmethod
    def get_default_params() -> dict:
        return {
            "n_nodes": 8,
            "max_order": 3,
        }
