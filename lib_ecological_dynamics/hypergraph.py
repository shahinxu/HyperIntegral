from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
import torch
from scipy.integrate import solve_ivp


class HypergraphModel:
    _cached_params = None
    _cached_hypergraph = None

    @dataclass
    class EcologicalHypergraphParams:
        n_species: int = 60
        n_producers: int = 20
        n_primary_consumers: int = 20
        n_secondary_consumers: int = 20

        t_span: tuple = (0.0, 420.0)
        n_steps: int = 4200
        seed: int = 42
        rtol: float = 1e-8
        atol: float = 1e-10

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

        w2: float = 0.1
        w3: float = 0.2
        w4: float = 0.30
        w5: float = 0.40

        designed_hypergraph_file: str | None = None
        designed_hypergraph_preset: str | None = None

    @staticmethod
    def _preset_hypergraph_file(preset: str = "n60") -> Path:
        preset_name = (preset or "n60").strip().lower()
        preset_map = {
            "n60": "ecological_dynamics_n60_hypergraph.json",
            "n9": "ecological_dynamics_n9_hypergraph.json",
        }
        if preset_name not in preset_map:
            raise ValueError(f"Unsupported ecological hypergraph preset: {preset}")
        return Path(__file__).resolve().parent / "presets" / preset_map[preset_name]

    @staticmethod
    def _load_hypergraph_payload(file_path: Path) -> dict:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _payload_n_nodes(payload: dict) -> int:
        if "n_nodes" in payload:
            return int(payload["n_nodes"])

        index_base = int(payload.get("index_base", 0))
        max_node = 0
        for key in ("edges2", "edges3", "edges4", "edges5"):
            for edge in payload.get(key, []):
                if edge:
                    max_node = max(max_node, *(int(u) for u in edge))
        for item in payload.get("pred_edges", []):
            max_node = max(max_node, int(item[0]), int(item[1]))
        return max(0, max_node - index_base + 1)

    @staticmethod
    def _payload_max_order(payload: dict) -> int:
        for order, key in ((5, "edges5"), (4, "edges4"), (3, "edges3"), (2, "edges2")):
            if payload.get(key):
                return order
        return 0

    @staticmethod
    def _resolve_designed_hypergraph_file(params: "HypergraphModel.EcologicalHypergraphParams") -> Path | None:
        env_path = os.getenv("ECOLOGY_HYPERGRAPH_FILE", "").strip()
        if env_path:
            return Path(env_path)

        env_preset = os.getenv("ECOLOGY_HYPERGRAPH_PRESET", "").strip()
        if env_preset:
            return HypergraphModel._preset_hypergraph_file(env_preset)

        if params.designed_hypergraph_file:
            return Path(params.designed_hypergraph_file)

        if params.designed_hypergraph_preset:
            return HypergraphModel._preset_hypergraph_file(params.designed_hypergraph_preset)

        return HypergraphModel._preset_hypergraph_file()

    @staticmethod
    def _normalize_edges(raw_edges, index_base: int) -> list[tuple[int, ...]]:
        normalized = []
        seen = set()
        for edge in raw_edges:
            edge0 = tuple(sorted(int(u) - index_base for u in edge))
            if edge0 in seen:
                continue
            seen.add(edge0)
            normalized.append(edge0)
        return normalized

    @staticmethod
    def _load_designed_hyperedges(file_path: Path) -> dict:
        payload = HypergraphModel._load_hypergraph_payload(file_path)
        index_base = int(payload.get("index_base", 0))

        pred_edges_raw = payload.get("pred_edges", [])
        pred_edges = []
        for item in pred_edges_raw:
            prey = int(item[0]) - index_base
            predator = int(item[1]) - index_base
            attack = float(item[2])
            eta = float(item[3])
            pred_edges.append((prey, predator, attack, eta))

        return {
            "edges2": HypergraphModel._normalize_edges(payload.get("edges2", []), index_base),
            "edges3": HypergraphModel._normalize_edges(payload.get("edges3", []), index_base),
            "edges4": HypergraphModel._normalize_edges(payload.get("edges4", []), index_base),
            "edges5": HypergraphModel._normalize_edges(payload.get("edges5", []), index_base),
            "pred_edges": pred_edges,
            "n_nodes": HypergraphModel._payload_n_nodes(payload),
            "max_order": HypergraphModel._payload_max_order(payload),
        }

    @staticmethod
    def _edge_config_n_nodes(edge_config: dict) -> int:
        max_node = 0
        for key in ("edges", "triangles", "quads", "quints", "sexts", "septs"):
            for edge in edge_config.get(key, []):
                if edge:
                    max_node = max(max_node, *(int(u) for u in edge))
        return max_node

    @staticmethod
    def _params_from_hypergraph_file(
        seed: int = 42,
        n_steps: int = 4200,
        hypergraph_file: str | None = None,
        hypergraph_preset: str | None = None,
    ) -> "HypergraphModel.EcologicalHypergraphParams":
        params = HypergraphModel.EcologicalHypergraphParams(
            seed=seed,
            n_steps=n_steps,
            designed_hypergraph_file=hypergraph_file,
            designed_hypergraph_preset=hypergraph_preset,
        )
        file_path = HypergraphModel._resolve_designed_hypergraph_file(params)
        designed = HypergraphModel._load_designed_hyperedges(file_path)
        n_species = designed["n_nodes"]
        n_producers = max(1, n_species // 3)
        n_primary = max(1, n_species // 3)
        n_secondary = max(1, n_species - n_producers - n_primary)
        return HypergraphModel.EcologicalHypergraphParams(
            n_species=n_species,
            n_producers=n_producers,
            n_primary_consumers=n_primary,
            n_secondary_consumers=n_secondary,
            seed=seed,
            n_steps=n_steps,
            designed_hypergraph_file=hypergraph_file,
            designed_hypergraph_preset=hypergraph_preset,
        )

    @staticmethod
    def _build_trophic_groups(params: "HypergraphModel.EcologicalHypergraphParams"):
        p = np.arange(0, params.n_producers, dtype=int)
        c1_start = params.n_producers
        c1_end = c1_start + params.n_primary_consumers
        c1 = np.arange(c1_start, c1_end, dtype=int)
        c2 = np.arange(c1_end, params.n_species, dtype=int)
        return p, c1, c2

    @staticmethod
    def _build_sparse_hypergraph_and_foodchain(params: "HypergraphModel.EcologicalHypergraphParams"):
        nodes = np.arange(params.n_species, dtype=int)

        custom_file = HypergraphModel._resolve_designed_hypergraph_file(params)
        designed = HypergraphModel._load_designed_hyperedges(custom_file)
        edges2 = designed["edges2"]
        edges3 = designed["edges3"]
        edges4 = designed["edges4"]
        edges5 = designed["edges5"]
        pred_edges = designed["pred_edges"]
        producers, primary, secondary = HypergraphModel._build_trophic_groups(params)

        return {
            "nodes": nodes,
            "edges2": edges2,
            "edges3": edges3,
            "edges4": edges4,
            "edges5": edges5,
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
        # 5th-order competition (if present)
        if "edges5" in hypergraph:
            dx += HypergraphModel._hyperedge_competition_term(x, hypergraph["edges5"], params.w5)

        dx += HypergraphModel._predation_term(x, hypergraph["pred_edges"])

        return dx

    @staticmethod
    def _simulate(params: "HypergraphModel.EcologicalHypergraphParams"):
        rng = np.random.default_rng(params.seed)

        hypergraph = HypergraphModel._build_sparse_hypergraph_and_foodchain(params)

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
            "growth_rates": producer_growth_rates,
            **hypergraph,
        }

    @staticmethod
    def dynamic_f(x: torch.Tensor, n_nodes: int, t: torch.Tensor | float | None = None) -> torch.Tensor:
        params = HypergraphModel._cached_params
        hypergraph = HypergraphModel._cached_hypergraph

        if x.ndim == 1:
            x_flat = x
            out_shape = None
        else:
            x_flat = x[:, 0]
            out_shape = x.shape

        device = x_flat.device
        dtype = x_flat.dtype

        producers = torch.as_tensor(hypergraph["producers"], device=device, dtype=torch.long)
        primary = torch.as_tensor(hypergraph["primary"], device=device, dtype=torch.long)
        secondary = torch.as_tensor(hypergraph["secondary"], device=device, dtype=torch.long)
        growth_rates = torch.as_tensor(hypergraph["growth_rates"], device=device, dtype=dtype)

        dx = torch.zeros_like(x_flat)

        seasonal = 1.0
        if params.season_amplitude != 0.0 and t is not None:
            t_val = t
            if not torch.is_tensor(t_val):
                t_val = torch.tensor(t_val, device=device, dtype=dtype)
            seasonal = 1.0 + params.season_amplitude * torch.sin(
                2.0 * torch.pi * t_val / params.season_period
            )

        if producers.numel() > 0:
            r = growth_rates
            x_p = x_flat[producers]
            dx[producers] += seasonal * r * x_p * (1.0 - x_p / params.producer_carrying_capacity)
            dx[producers] += params.producer_replenish * (1.0 - x_p)

        if primary.numel() > 0:
            dx[primary] -= params.primary_mortality * x_flat[primary]
            dx[primary] += params.primary_immigration
        if secondary.numel() > 0:
            dx[secondary] -= params.secondary_mortality * x_flat[secondary]
            dx[secondary] += params.secondary_immigration

        for prey, predator, attack, eta in hypergraph["pred_edges"]:
            prey_idx = int(prey)
            pred_idx = int(predator)
            flux = attack * x_flat[pred_idx] * x_flat[prey_idx]
            dx[prey_idx] -= flux
            dx[pred_idx] += eta * flux

        if out_shape is not None:
            return dx.view(out_shape)

        return dx

    @staticmethod
    def dynamic_f_batch(
        x: torch.Tensor,
        n_nodes: int,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized ecological baseline dynamics for a batch of time points.

        This matches dynamic_f but operates on x with shape [T, n_nodes]
        (or [T, n_nodes, 1]) and t with shape [T] or [T, 1].
        Competition hyperedges are intentionally excluded here and are
        handled separately via the topology-dependent competition term.
        """
        params = HypergraphModel._cached_params
        hypergraph = HypergraphModel._cached_hypergraph

        if x.ndim == 3:
            # [T, n_nodes, 1] -> [T, n_nodes]
            x_flat = x[:, :, 0]
        else:
            x_flat = x

        if t.ndim > 1:
            t_flat = t.squeeze(-1)
        else:
            t_flat = t

        device = x_flat.device
        dtype = x_flat.dtype

        producers = torch.as_tensor(hypergraph["producers"], device=device, dtype=torch.long)
        primary = torch.as_tensor(hypergraph["primary"], device=device, dtype=torch.long)
        secondary = torch.as_tensor(hypergraph["secondary"], device=device, dtype=torch.long)
        growth_rates = torch.as_tensor(hypergraph["growth_rates"], device=device, dtype=dtype)

        T = x_flat.shape[0]
        dx = torch.zeros_like(x_flat)

        # Seasonal modulation
        if params.season_amplitude != 0.0:
            seasonal = 1.0 + params.season_amplitude * torch.sin(
                2.0 * torch.pi * t_flat.to(dtype) / params.season_period
            )
        else:
            seasonal = torch.ones_like(t_flat, dtype=dtype, device=device)

        # Producers: logistic growth + replenish
        if producers.numel() > 0:
            x_p = x_flat[:, producers]  # [T, n_producers]
            r = growth_rates.view(1, -1)  # [1, n_producers]
            seasonal_p = seasonal.view(T, 1)
            dx[:, producers] += seasonal_p * r * x_p * (
                1.0 - x_p / params.producer_carrying_capacity
            )
            dx[:, producers] += params.producer_replenish * (1.0 - x_p)

        # Primary & secondary consumers: mortality + immigration
        if primary.numel() > 0:
            x_c1 = x_flat[:, primary]
            dx[:, primary] -= params.primary_mortality * x_c1
            dx[:, primary] += params.primary_immigration
        if secondary.numel() > 0:
            x_c2 = x_flat[:, secondary]
            dx[:, secondary] -= params.secondary_mortality * x_c2
            dx[:, secondary] += params.secondary_immigration

        # Predation term, vectorized over all predator-prey pairs
        pred_edges = hypergraph["pred_edges"]
        if len(pred_edges) > 0:
            prey_idx = torch.tensor([int(e[0]) for e in pred_edges], device=device, dtype=torch.long)
            pred_idx = torch.tensor([int(e[1]) for e in pred_edges], device=device, dtype=torch.long)
            attack = torch.tensor([float(e[2]) for e in pred_edges], device=device, dtype=dtype).view(1, -1)
            eta = torch.tensor([float(e[3]) for e in pred_edges], device=device, dtype=dtype).view(1, -1)

            x_prey = x_flat[:, prey_idx]      # [T, E]
            x_pred = x_flat[:, pred_idx]      # [T, E]
            flux = attack * x_pred * x_prey   # [T, E]

            prey_idx_exp = prey_idx.unsqueeze(0).expand(T, -1)  # [T, E]
            pred_idx_exp = pred_idx.unsqueeze(0).expand(T, -1)  # [T, E]

            dx.scatter_add_(1, prey_idx_exp, -flux)
            dx.scatter_add_(1, pred_idx_exp, eta * flux)

        return dx

    @staticmethod
    def dynamic_phi(
        x: torch.Tensor,
        all_possible_edges: dict,
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        params = HypergraphModel._cached_params

        n_total = 0
        for key in ["edges", "triangles", "quads", "quints", "sexts", "septs"]:
            n_total += len(all_possible_edges.get(key, []))

        if x.ndim == 1:
            x_flat = x
            d = 1
        else:
            x_flat = x[:, 0]
            d = x.shape[1]

        Phi = torch.zeros((n_nodes, d, n_total), device=device, dtype=x.dtype)
        edge_idx = 0

        edges = all_possible_edges.get("edges")
        if edges is not None and len(edges) > 0:
            i_idx = edges[:, 0]
            j_idx = edges[:, 1]
            prod = x_flat[i_idx] * x_flat[j_idx]
            edge_range = edge_idx + torch.arange(edges.shape[0], device=device)
            Phi[i_idx, 0, edge_range] = -params.w2 * prod
            Phi[j_idx, 0, edge_range] = -params.w2 * prod
            edge_idx += edges.shape[0]

        triangles = all_possible_edges.get("triangles")
        if triangles is not None and len(triangles) > 0:
            i_idx = triangles[:, 0]
            j_idx = triangles[:, 1]
            k_idx = triangles[:, 2]
            prod = x_flat[i_idx] * x_flat[j_idx] * x_flat[k_idx]
            edge_range = edge_idx + torch.arange(triangles.shape[0], device=device)
            Phi[i_idx, 0, edge_range] = -params.w3 * prod
            Phi[j_idx, 0, edge_range] = -params.w3 * prod
            Phi[k_idx, 0, edge_range] = -params.w3 * prod
            edge_idx += triangles.shape[0]

        quads = all_possible_edges.get("quads")
        if quads is not None and len(quads) > 0:
            i_idx = quads[:, 0]
            j_idx = quads[:, 1]
            k_idx = quads[:, 2]
            l_idx = quads[:, 3]
            prod = x_flat[i_idx] * x_flat[j_idx] * x_flat[k_idx] * x_flat[l_idx]
            edge_range = edge_idx + torch.arange(quads.shape[0], device=device)
            Phi[i_idx, 0, edge_range] = -params.w4 * prod
            Phi[j_idx, 0, edge_range] = -params.w4 * prod
            Phi[k_idx, 0, edge_range] = -params.w4 * prod
            Phi[l_idx, 0, edge_range] = -params.w4 * prod
            edge_idx += quads.shape[0]

        quints = all_possible_edges.get("quints")
        if quints is not None and len(quints) > 0:
            i_idx = quints[:, 0]
            j_idx = quints[:, 1]
            k_idx = quints[:, 2]
            l_idx = quints[:, 3]
            m_idx = quints[:, 4]
            prod = (
                x_flat[i_idx]
                * x_flat[j_idx]
                * x_flat[k_idx]
                * x_flat[l_idx]
                * x_flat[m_idx]
            )
            edge_range = edge_idx + torch.arange(quints.shape[0], device=device)
            Phi[i_idx, 0, edge_range] = -params.w5 * prod
            Phi[j_idx, 0, edge_range] = -params.w5 * prod
            Phi[k_idx, 0, edge_range] = -params.w5 * prod
            Phi[l_idx, 0, edge_range] = -params.w5 * prod
            Phi[m_idx, 0, edge_range] = -params.w5 * prod
            edge_idx += quints.shape[0]

        return Phi

    @staticmethod
    def get_hyperedge_config(
        n_nodes: int,
        max_order: int = 5,
        seed: int = 42,
        hypergraph_file: str | None = None,
        hypergraph_preset: str | None = None,
    ) -> dict:
        _ = (n_nodes, max_order)
        params = HypergraphModel._params_from_hypergraph_file(
            seed=seed,
            hypergraph_file=hypergraph_file,
            hypergraph_preset=hypergraph_preset,
        )

        result = HypergraphModel._simulate(params)
        HypergraphModel._cached_params = params
        HypergraphModel._cached_hypergraph = result
        edges2 = result["edges2"]
        edges3 = result["edges3"]
        edges4 = result["edges4"]
        edges5 = result.get("edges5", [])

        edges_1b = [[i + 1, j + 1] for i, j in edges2]
        triangles_1b = [[i + 1, j + 1, k + 1] for i, j, k in edges3]
        quads_1b = [[i + 1, j + 1, k + 1, l + 1] for i, j, k, l in edges4]
        quints_1b = [[i + 1, j + 1, k + 1, l + 1, m + 1] for i, j, k, l, m in edges5]

        return {
            "edges": edges_1b,
            "triangles": triangles_1b,
            "quads": quads_1b,
            "quints": quints_1b,
            "sexts": [],
            "septs": [],
        }

    @staticmethod
    def generate_all_possible_hyperedges(n_nodes: int, max_order: int) -> dict:
        return HypergraphModel.get_hyperedge_config(n_nodes, max_order)

    @staticmethod
    def generate_training_data(
        n_nodes: int,
        edge_config: dict,
        n_samples: int = 11,
        noise: float = 0.0,
        hypergraph_file: str | None = None,
        hypergraph_preset: str | None = None,
    ):
        inferred_n_nodes = HypergraphModel._edge_config_n_nodes(edge_config)
        if hypergraph_file is not None or hypergraph_preset is not None:
            params = HypergraphModel._params_from_hypergraph_file(
                n_steps=max(10, n_samples),
                hypergraph_file=hypergraph_file,
                hypergraph_preset=hypergraph_preset,
            )
        else:
            n_species = inferred_n_nodes or n_nodes
            n_producers = max(1, n_species // 3)
            n_primary = max(1, n_species // 3)
            n_secondary = max(1, n_species - n_producers - n_primary)
            params = HypergraphModel.EcologicalHypergraphParams(
                n_species=n_species,
                n_producers=n_producers,
                n_primary_consumers=n_primary,
                n_secondary_consumers=n_secondary,
                n_steps=max(10, n_samples),
                designed_hypergraph_file=hypergraph_file,
                designed_hypergraph_preset=hypergraph_preset,
            )

        result = HypergraphModel._simulate(params)
        HypergraphModel._cached_params = params
        HypergraphModel._cached_hypergraph = result
        t = result["t"]
        x_data = result["X"].T[:, :, None]

        if x_data.shape[0] != len(t):
            t = np.linspace(t[0], t[-1], x_data.shape[0])

        if noise > 0:
            x_data = x_data + np.random.randn(*x_data.shape) * noise

        return t, x_data

    @staticmethod
    def get_default_params(hypergraph_preset: str | None = None) -> dict:
        designed = HypergraphModel._load_designed_hyperedges(
            HypergraphModel._preset_hypergraph_file(hypergraph_preset or "n60")
        )
        return {
            "n_nodes": designed["n_nodes"],
            "max_order": designed["max_order"],
        }
