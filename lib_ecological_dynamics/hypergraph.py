"""
Hypergraph-style interface for ecological dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from scipy.integrate import solve_ivp


class HypergraphModel:
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

        w2: float = 0.0006
        w3: float = 0.0008
        w4: float = 0.0010

        target_prey_per_primary: int = 2
        target_prey_per_secondary: int = 2
        attack_p_to_c1: float = 2.2
        attack_c1_to_c2: float = 1.6
        half_sat: float = 0.01
        eta_p_to_c1: float = 0.55
        eta_c1_to_c2: float = 0.35

    @staticmethod
    def _validate_params(params: "HypergraphModel.EcologicalHypergraphParams"):
        total = params.n_producers + params.n_primary_consumers + params.n_secondary_consumers
        if total != params.n_species:
            raise ValueError("n_species must equal n_producers + n_primary_consumers + n_secondary_consumers")

    @staticmethod
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

    @staticmethod
    def _build_trophic_groups(params: "HypergraphModel.EcologicalHypergraphParams"):
        p = np.arange(0, params.n_producers, dtype=int)
        c1_start = params.n_producers
        c1_end = c1_start + params.n_primary_consumers
        c1 = np.arange(c1_start, c1_end, dtype=int)
        c2 = np.arange(c1_end, params.n_species, dtype=int)
        return p, c1, c2

    @staticmethod
    def _build_predation_edges(params: "HypergraphModel.EcologicalHypergraphParams", rng: np.random.Generator):
        p, c1, c2 = HypergraphModel._build_trophic_groups(params)

        pred_edges = []

        n_prey_p = len(p)
        k1 = max(1, min(params.target_prey_per_primary, n_prey_p))
        for pred_pos, predator in enumerate(c1):
            for shift in range(k1):
                prey = p[(pred_pos + shift) % n_prey_p]
                pred_edges.append((int(prey), int(predator), params.attack_p_to_c1, params.eta_p_to_c1))

        n_prey_c1 = len(c1)
        k2 = max(1, min(params.target_prey_per_secondary, n_prey_c1))
        for pred_pos, predator in enumerate(c2):
            for shift in range(k2):
                prey = c1[(pred_pos + shift) % n_prey_c1]
                pred_edges.append((int(prey), int(predator), params.attack_c1_to_c2, params.eta_c1_to_c2))

        return pred_edges

    @staticmethod
    def _build_sparse_hypergraph_and_foodchain(params: "HypergraphModel.EcologicalHypergraphParams", rng: np.random.Generator):
        nodes = np.arange(params.n_species, dtype=int)

        edges2 = HypergraphModel._sample_sparse_hyperedges(
            n=params.n_species,
            order=2,
            target_incidence_per_node=params.target_incidence_per_node_order2,
            rng=rng,
        )
        edges3 = HypergraphModel._sample_sparse_hyperedges(
            n=params.n_species,
            order=3,
            target_incidence_per_node=params.target_incidence_per_node_order3,
            rng=rng,
        )
        edges4 = HypergraphModel._sample_sparse_hyperedges(
            n=params.n_species,
            order=4,
            target_incidence_per_node=params.target_incidence_per_node_order4,
            rng=rng,
        )

        pred_edges = HypergraphModel._build_predation_edges(params, rng)
        producers, primary, secondary = HypergraphModel._build_trophic_groups(params)

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

    @staticmethod
    def _hyperedge_competition_term(x: np.ndarray, hyperedges: list[tuple[int, ...]], weight: float) -> np.ndarray:
        term = np.zeros_like(x)
        for e in hyperedges:
            vals = x[list(e)]
            for local_idx, node in enumerate(e):
                others_prod = np.prod(np.delete(vals, local_idx))
                term[node] -= weight * x[node] * others_prod
        return term

    @staticmethod
    def _predation_term(
        x: np.ndarray,
        pred_edges: list[tuple[int, int, float, float]],
        half_sat: float,
    ) -> np.ndarray:
        term = np.zeros_like(x)
        for prey, predator, attack, eta in pred_edges:
            prey_val = x[prey]
            pred_val = x[predator]
            flux = attack * pred_val * prey_val

            term[prey] -= flux
            term[predator] += eta * flux
        return term

    @staticmethod
    def _rhs_ecological_hypergraph(
        t: float,
        x: np.ndarray,
        hypergraph: dict,
        producer_growth_rates: np.ndarray,
        params: "HypergraphModel.EcologicalHypergraphParams",
    ) -> np.ndarray:
        x = np.clip(x, 1e-12, None)

        producers = hypergraph["producers"]
        primary = hypergraph["primary"]
        secondary = hypergraph["secondary"]

        dx = np.zeros_like(x)

        seasonal = 1.0 + params.season_amplitude * np.sin(2.0 * np.pi * t / params.season_period)
        for local_idx, node in enumerate(producers):
            r = producer_growth_rates[local_idx]
            dx[node] += seasonal * r * x[node] * (1.0 - x[node] / params.producer_carrying_capacity)
            dx[node] += params.producer_replenish * (1.0 - x[node])

        dx[primary] -= params.primary_mortality * x[primary]
        dx[secondary] -= params.secondary_mortality * x[secondary]
        dx[primary] += params.primary_immigration
        dx[secondary] += params.secondary_immigration

        dx += HypergraphModel._hyperedge_competition_term(x, hypergraph["edges2"], params.w2)
        dx += HypergraphModel._hyperedge_competition_term(x, hypergraph["edges3"], params.w3)
        dx += HypergraphModel._hyperedge_competition_term(x, hypergraph["edges4"], params.w4)

        dx += HypergraphModel._predation_term(x, hypergraph["pred_edges"], params.half_sat)

        return dx

    @staticmethod
    def _simulate(params: "HypergraphModel.EcologicalHypergraphParams"):
        HypergraphModel._validate_params(params)
        rng = np.random.default_rng(params.seed)

        hypergraph = HypergraphModel._build_sparse_hypergraph_and_foodchain(params, rng)

        producer_growth_rates = rng.normal(params.producer_growth_mean, params.producer_growth_std, size=params.n_producers)
        producer_growth_rates = np.clip(producer_growth_rates, 0.55, 0.95)

        x0 = rng.uniform(0.04, 0.15, size=params.n_species)
        x0[hypergraph["producers"]] += 0.04
        x0 = np.clip(x0, 1e-6, None)

        t_eval = np.linspace(params.t_span[0], params.t_span[1], params.n_steps)

        sol = solve_ivp(
            fun=lambda t, x: HypergraphModel._rhs_ecological_hypergraph(t, x, hypergraph, producer_growth_rates, params),
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
            d = 1
        else:
            d = x.shape[1]

        return torch.zeros((n_nodes, d, n_total), device=device)

    @staticmethod
    def get_hyperedge_config(n_nodes: int, max_order: int = 4, seed: int = 42) -> dict:
        n_producers = max(1, n_nodes // 3)
        n_primary = max(1, n_nodes // 3)
        n_secondary = max(1, n_nodes - n_producers - n_primary)

        params = HypergraphModel.EcologicalHypergraphParams(
            n_species=n_nodes,
            n_producers=n_producers,
            n_primary_consumers=n_primary,
            n_secondary_consumers=n_secondary,
            seed=seed,
        )

        result = HypergraphModel._simulate(params)
        edges2 = result["edges2"]
        edges3 = result["edges3"]
        edges4 = result["edges4"]

        edges_1b = [[i + 1, j + 1] for i, j in edges2]
        triangles_1b = [[i + 1, j + 1, k + 1] for i, j, k in edges3]
        quads_1b = [[i + 1, j + 1, k + 1, l + 1] for i, j, k, l in edges4]

        return {
            "edges": edges_1b if max_order >= 2 else [],
            "triangles": triangles_1b if max_order >= 3 else [],
            "quads": quads_1b if max_order >= 4 else [],
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

        if max_order >= 4:
            all_possible["quads"] = [list(edge) for edge in combinations(range(1, n_nodes + 1), 4)]
        else:
            all_possible["quads"] = []

        all_possible["quints"] = []
        all_possible["sexts"] = []
        all_possible["septs"] = []

        return all_possible

    @staticmethod
    def generate_training_data(n_nodes: int, edge_config: dict, n_samples: int = 11, noise: float = 0.0):
        n_producers = max(1, n_nodes // 3)
        n_primary = max(1, n_nodes // 3)
        n_secondary = max(1, n_nodes - n_producers - n_primary)

        params = HypergraphModel.EcologicalHypergraphParams(
            n_species=n_nodes,
            n_producers=n_producers,
            n_primary_consumers=n_primary,
            n_secondary_consumers=n_secondary,
            n_steps=max(10, n_samples),
        )

        result = HypergraphModel._simulate(params)
        t = result["t"]
        x_data = result["X"].T[:, :, None]

        if x_data.shape[0] != len(t):
            t = np.linspace(t[0], t[-1], x_data.shape[0])

        if noise > 0:
            x_data = x_data + np.random.randn(*x_data.shape) * noise

        return t, x_data

    @staticmethod
    def get_default_params() -> dict:
        return {
            "n_nodes": 9,
            "max_order": 4,
        }
