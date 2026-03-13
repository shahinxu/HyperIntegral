from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
import torch
from scipy.integrate import solve_ivp


class HypergraphModel:
    @dataclass
    class SCMParams:
        n_nodes: int = 60
        t_max: float = 50.0
        beta: float = 0.5
        beta_delta: float = 2.0
        mu: float = 0.4
        seed: int = 123

    @staticmethod
    def _preset_hypergraph_file() -> Path:
        return Path(__file__).resolve().parent / "presets" / "social_contagion_n60_hypergraph.json"

    @staticmethod
    def _load_hypergraph_payload(file_path: Path) -> dict:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _payload_n_nodes(payload: dict) -> int:
        if "n_nodes" in payload:
            return int(payload["n_nodes"])

        index_base = int(payload.get("index_base", 1))
        max_node = 0
        for key in ("edges", "triangles", "quads", "quints"):
            for edge in payload.get(key, []):
                if edge:
                    max_node = max(max_node, *(int(u) for u in edge))
        return max(0, max_node - index_base + 1)

    @staticmethod
    def _payload_max_order(payload: dict) -> int:
        for order, key in ((5, "quints"), (4, "quads"), (3, "triangles"), (2, "edges")):
            if payload.get(key):
                return order
        return 0

    @staticmethod
    def _resolve_hypergraph_file(hypergraph_file: str | None = None) -> Path:
        env_path = os.getenv("SOCIAL_CONTAGION_HYPERGRAPH_FILE", "").strip()
        if env_path:
            return Path(env_path)
        if hypergraph_file:
            return Path(hypergraph_file)
        return HypergraphModel._preset_hypergraph_file()

    @staticmethod
    def _load_hypergraph_config(hypergraph_file: str | None = None) -> dict:
        file_path = HypergraphModel._resolve_hypergraph_file(hypergraph_file)
        payload = HypergraphModel._load_hypergraph_payload(file_path)
        index_base = int(payload.get("index_base", 1))

        def normalize(key: str):
            return [
                [int(u) - index_base + 1 for u in edge]
                for edge in payload.get(key, [])
            ]

        return {
            "edges": normalize("edges"),
            "triangles": normalize("triangles"),
            "quads": normalize("quads"),
            "quints": normalize("quints"),
            "n_nodes": HypergraphModel._payload_n_nodes(payload),
            "max_order": HypergraphModel._payload_max_order(payload),
        }

    @staticmethod
    def _edge_config_n_nodes(edge_config: dict) -> int:
        max_node = 0
        for key in ("edges", "triangles", "quads", "quints"):
            for edge in edge_config.get(key, []):
                if edge:
                    max_node = max(max_node, *(int(u) for u in edge))
        return max_node

    @staticmethod
    def generate_training_data(
        n_nodes: int,
        edge_config: dict,
        n_samples: int = 100,
        noise: float = 0.0,
        seed: int | None = None,
    ):
        # Allow overriding the SCM seed so that multiple independent
        # trajectories can be generated on the same underlying
        # hypergraph by varying the random seed.
        scm_seed = 123 if seed is None else seed
        inferred_n_nodes = HypergraphModel._edge_config_n_nodes(edge_config)
        params = HypergraphModel.SCMParams(n_nodes=inferred_n_nodes or n_nodes, t_max=50, seed=scm_seed)
        complex_dict = {
            "edges": [],
            "triangles": [],
            "quads": [],
            "quints": [],
            "nodes": np.arange(params.n_nodes),
        }

        def _to_zero_based(items: list[list[int]]) -> list[list[int]]:
            if len(items) == 0:
                return []
            arr = np.array(items)
            if arr.min() >= 1:
                return (arr - 1).tolist()
            return items

        complex_dict["edges"] = _to_zero_based(edge_config.get("edges", []))
        complex_dict["triangles"] = _to_zero_based(edge_config.get("triangles", []))
        complex_dict["quads"] = _to_zero_based(edge_config.get("quads", []))
        complex_dict["quints"] = _to_zero_based(edge_config.get("quints", []))

        result = HypergraphModel._simulate(params, complex_dict, n_steps=n_samples)

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
        quads: list[list[int]],
        quints: list[list[int]],
        params: "HypergraphModel.SCMParams",
    ) -> np.ndarray:
        
        n_nodes = x.shape[0]
        infection_force = np.zeros(n_nodes)

        # Pairwise force
        if edges:
            for i, j in edges:
                infection_force[i] += params.beta * x[j]
                infection_force[j] += params.beta * x[i]

        # 3-body (triangle) force
        if triangles:
            for i, j, k in triangles:
                infection_force[i] += params.beta_delta * (x[j] * x[k])
                infection_force[j] += params.beta_delta * (x[i] * x[k])
                infection_force[k] += params.beta_delta * (x[i] * x[j])

        # 4-body (quad) force – symmetric in all four nodes, using
        # beta_delta as the scale (consistent with dynamic_phi).
        if quads:
            for i, j, k, l in quads:
                prod_jkl = x[j] * x[k] * x[l]
                prod_ikl = x[i] * x[k] * x[l]
                prod_ijl = x[i] * x[j] * x[l]
                prod_ijk = x[i] * x[j] * x[k]
                infection_force[i] += params.beta_delta * prod_jkl
                infection_force[j] += params.beta_delta * prod_ikl
                infection_force[k] += params.beta_delta * prod_ijl
                infection_force[l] += params.beta_delta * prod_ijk

        # 5-body (quint) force – symmetric in all five nodes, using beta_delta.
        if quints:
            for i, j, k, l, m in quints:
                infection_force[i] += params.beta_delta * (x[j] * x[k] * x[l] * x[m])
                infection_force[j] += params.beta_delta * (x[i] * x[k] * x[l] * x[m])
                infection_force[k] += params.beta_delta * (x[i] * x[j] * x[l] * x[m])
                infection_force[l] += params.beta_delta * (x[i] * x[j] * x[k] * x[m])
                infection_force[m] += params.beta_delta * (x[i] * x[j] * x[k] * x[l])

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
        quads = complex_dict.get("quads", [])
        quints = complex_dict.get("quints", [])

        sol = solve_ivp(
            fun=lambda t, x: HypergraphModel._rhs(t, x, edges, triangles, quads, quints, params),
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
        _ = n_nodes
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
        _ = n_nodes
        # The interaction part: Phi(x) * A
        # For each candidate edge/triangle, compute its contribution term
        # Term = (1 - x_i) * force
        
        params = HypergraphModel.SCMParams()
        n_total = 0
        for key in ["edges", "triangles", "quads", "quints"]:
            n_total += len(all_possible_edges.get(key, []))

        if x.ndim == 1:
            x_vec = x
        else:
            x_vec = x[:, 0]

        n = x_vec.shape[0]

        # Precompute (1-x)
        susceptible = 1.0 - x_vec
        
        # Phi shape: (N_nodes, Dimension_of_states, N_candidates)
        # Here Dimension is 1 (infection probability)
        Phi = torch.zeros((n, 1, n_total), device=device, dtype=x.dtype)
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

        # 3. Candidate 4-body interactions (quads)
        # These are included as additional basis functions for inference;
        # the current simulator _rhs does not use them, so in ground truth
        # their weights are effectively zero.
        quads = all_possible_edges.get("quads", [])
        if len(quads) > 0:
            quads_t = torch.as_tensor(quads, dtype=torch.long, device=device)
            num_quads = quads_t.shape[0]
            i_idx = quads_t[:, 0]
            j_idx = quads_t[:, 1]
            k_idx = quads_t[:, 2]
            l_idx = quads_t[:, 3]
            feature_range = edge_idx + torch.arange(num_quads, device=device)

            # Use beta_delta as scale for 4-body terms as well
            term_i = params.beta_delta * susceptible[i_idx] * (x_vec[j_idx] * x_vec[k_idx] * x_vec[l_idx])
            term_j = params.beta_delta * susceptible[j_idx] * (x_vec[i_idx] * x_vec[k_idx] * x_vec[l_idx])
            term_k = params.beta_delta * susceptible[k_idx] * (x_vec[i_idx] * x_vec[j_idx] * x_vec[l_idx])
            term_l = params.beta_delta * susceptible[l_idx] * (x_vec[i_idx] * x_vec[j_idx] * x_vec[k_idx])

            Phi[i_idx, 0, feature_range] = term_i
            Phi[j_idx, 0, feature_range] = term_j
            Phi[k_idx, 0, feature_range] = term_k
            Phi[l_idx, 0, feature_range] = term_l

            edge_idx += num_quads

        # 4. Candidate 5-body interactions (quints)
        quints = all_possible_edges.get("quints", [])
        if len(quints) > 0:
            quints_t = torch.as_tensor(quints, dtype=torch.long, device=device)
            num_quints = quints_t.shape[0]
            i_idx = quints_t[:, 0]
            j_idx = quints_t[:, 1]
            k_idx = quints_t[:, 2]
            l_idx = quints_t[:, 3]
            m_idx = quints_t[:, 4]
            feature_range = edge_idx + torch.arange(num_quints, device=device)

            term_i = params.beta_delta * susceptible[i_idx] * (x_vec[j_idx] * x_vec[k_idx] * x_vec[l_idx] * x_vec[m_idx])
            term_j = params.beta_delta * susceptible[j_idx] * (x_vec[i_idx] * x_vec[k_idx] * x_vec[l_idx] * x_vec[m_idx])
            term_k = params.beta_delta * susceptible[k_idx] * (x_vec[i_idx] * x_vec[j_idx] * x_vec[l_idx] * x_vec[m_idx])
            term_l = params.beta_delta * susceptible[l_idx] * (x_vec[i_idx] * x_vec[j_idx] * x_vec[k_idx] * x_vec[m_idx])
            term_m = params.beta_delta * susceptible[m_idx] * (x_vec[i_idx] * x_vec[j_idx] * x_vec[k_idx] * x_vec[l_idx])

            Phi[i_idx, 0, feature_range] = term_i
            Phi[j_idx, 0, feature_range] = term_j
            Phi[k_idx, 0, feature_range] = term_k
            Phi[l_idx, 0, feature_range] = term_l
            Phi[m_idx, 0, feature_range] = term_m

            edge_idx += num_quints

        return Phi

    @staticmethod
    def get_hyperedge_config(
        n_nodes: int,
        max_order: int = 3,
        k_mean: float = 20.0,
        k_delta: float = 6.0,
        seed: int = 42,
        enforce_closure: bool = True,
        hypergraph_file: str | None = None,
    ) -> dict:
        _ = (n_nodes, max_order, k_mean, k_delta, seed, enforce_closure)
        config = HypergraphModel._load_hypergraph_config(hypergraph_file)

        return {
            "edges": config["edges"],
            "triangles": config["triangles"],
            "quads": config.get("quads", []),
            "quints": config.get("quints", []),
        }

    @staticmethod
    def generate_all_possible_hyperedges(n_nodes: int, max_order: int) -> dict:
        return HypergraphModel.get_hyperedge_config(n_nodes, max_order)


    @staticmethod
    def get_default_params() -> dict:
        config = HypergraphModel._load_hypergraph_config()
        return {
            "n_nodes": config["n_nodes"],
            "max_order": config["max_order"],
        }
