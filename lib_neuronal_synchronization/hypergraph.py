"""
Hypergraph-style interface for neuronal synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from scipy.integrate import solve_ivp

class HypergraphModel:
    _cached_omega = None
    _cached_nu = None
    _cached_edges = None
    _cached_triangles = None

    @dataclass
    class SimplicialParams:
        n_oscillators: int = 30
        K: float = 4.0
        kappa: float = 1.5
        d: float = 2.0
        omega_mean: float = 0.0
        omega_std: float = 1.0
        nu_mean: float = 0.0
        nu_std: float = 1.0
        t_span: tuple = (0.0, 120.0)
        n_steps: int = 2000
        seed: int = 42
        target_incidence_per_node_order2: int = 2
        target_incidence_per_node_order3: int = 1

    @staticmethod
    def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _sample_sparse_hyperedges(
        n: int,
        order: int,
        target_incidence_per_node: int,
        rng: np.random.Generator,
    ):
        candidates = list(combinations(range(1, n + 1), order))
        if not candidates:
            return []

        m_target = max(n // 2, int(round(n * target_incidence_per_node / order)))
        m_target = min(m_target, len(candidates))

        picked_idx = set(rng.choice(len(candidates), size=m_target, replace=False).tolist())

        node_cover = {i: 0 for i in range(1, n + 1)}
        for idx in picked_idx:
            for node in candidates[idx]:
                node_cover[node] += 1

        for node in range(1, n + 1):
            if node_cover[node] > 0:
                continue
            contain_node = [i for i, e in enumerate(candidates) if node in e]
            if not contain_node:
                continue
            add_idx = int(rng.choice(contain_node))
            picked_idx.add(add_idx)
            for u in candidates[add_idx]:
                node_cover[u] += 1

        return [list(candidates[i]) for i in sorted(picked_idx)]

    @staticmethod
    def _kuramoto_pairwise_term(phi: np.ndarray) -> np.ndarray:
        diff = phi[None, :] - phi[:, None]
        return np.sin(diff).mean(axis=1)

    @staticmethod
    def _rhs(
        t: float,
        y: np.ndarray,
        omega: np.ndarray,
        nu: np.ndarray,
        p: "HypergraphModel.SimplicialParams",
        edges: np.ndarray,
        triangles: np.ndarray,
    ):
        n = p.n_oscillators
        theta = y[:n]
        phi = y[n:]

        pairwise_term = np.zeros(n)
        if edges.size > 0:
            for i_idx, j_idx in edges:
                pairwise_term[i_idx] += (p.kappa / n) * np.sin(phi[j_idx] - phi[i_idx])
                pairwise_term[j_idx] += (p.kappa / n) * np.sin(phi[i_idx] - phi[j_idx])

        simplex_term = np.zeros(n)
        if triangles.size > 0:
            for i_idx, j_idx, k_idx in triangles:
                simplex_term[i_idx] += (p.K / (n ** 2)) * np.sin(theta[j_idx] + theta[k_idx] - 2 * theta[i_idx])
                simplex_term[j_idx] += (p.K / (n ** 2)) * np.sin(theta[i_idx] + theta[k_idx] - 2 * theta[j_idx])
                simplex_term[k_idx] += (p.K / (n ** 2)) * np.sin(theta[i_idx] + theta[j_idx] - 2 * theta[k_idx])

        drive_term = p.d * np.sin(theta - phi)

        theta_dot = omega + simplex_term
        phi_dot = nu + pairwise_term + drive_term

        return np.concatenate([theta_dot, phi_dot])

    @staticmethod
    def _simulate(
        params: "HypergraphModel.SimplicialParams",
        edges: np.ndarray,
        triangles: np.ndarray,
    ):
        rng = np.random.default_rng(params.seed)

        omega = rng.normal(params.omega_mean, params.omega_std, params.n_oscillators)
        nu = rng.normal(params.nu_mean, params.nu_std, params.n_oscillators)

        theta0 = rng.uniform(-np.pi, np.pi, size=params.n_oscillators)
        phi0 = rng.uniform(-np.pi, np.pi, size=params.n_oscillators)
        y0 = np.concatenate([theta0, phi0])

        t_eval = np.linspace(params.t_span[0], params.t_span[1], params.n_steps)

        sol = solve_ivp(
            fun=lambda t, y: HypergraphModel._rhs(t, y, omega, nu, params, edges, triangles),
            t_span=params.t_span,
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        theta = HypergraphModel._wrap_to_pi(sol.y[: params.n_oscillators])
        phi = HypergraphModel._wrap_to_pi(sol.y[params.n_oscillators :])

        z_theta = np.exp(1j * theta).mean(axis=0)
        z2_theta = np.exp(2j * theta).mean(axis=0)
        z_phi = np.exp(1j * phi).mean(axis=0)

        out = {
            "t": sol.t,
            "theta": theta,
            "phi": phi,
            "r_theta": np.abs(z_theta),
            "r2_theta": np.abs(z2_theta),
            "r_phi": np.abs(z_phi),
            "omega": omega,
            "nu": nu,
        }
        return out

    @staticmethod
    def dynamic_f(x: torch.Tensor, n_nodes: int) -> torch.Tensor:
        params = HypergraphModel.SimplicialParams(n_oscillators=n_nodes)
        theta = x[:, 0]
        phi = x[:, 1]

        if HypergraphModel._cached_omega is None or HypergraphModel._cached_nu is None:
            omega = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
            nu = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
        else:
            omega = torch.as_tensor(HypergraphModel._cached_omega, device=x.device, dtype=x.dtype)
            nu = torch.as_tensor(HypergraphModel._cached_nu, device=x.device, dtype=x.dtype)

        pairwise_term = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
        edges = HypergraphModel._cached_edges
        if edges is not None and len(edges) > 0:
            edges_t = torch.as_tensor(edges, device=x.device, dtype=torch.long)
            i_idx = edges_t[:, 0]
            j_idx = edges_t[:, 1]
            term_ij = (params.kappa / n_nodes) * torch.sin(phi[j_idx] - phi[i_idx])
            term_ji = (params.kappa / n_nodes) * torch.sin(phi[i_idx] - phi[j_idx])
            pairwise_term.index_add_(0, i_idx, term_ij)
            pairwise_term.index_add_(0, j_idx, term_ji)

        simplex_term = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
        triangles = HypergraphModel._cached_triangles
        if triangles is not None and len(triangles) > 0:
            tri_t = torch.as_tensor(triangles, device=x.device, dtype=torch.long)
            i_idx = tri_t[:, 0]
            j_idx = tri_t[:, 1]
            k_idx = tri_t[:, 2]
            term_i = (params.K / (n_nodes ** 2)) * torch.sin(theta[j_idx] + theta[k_idx] - 2 * theta[i_idx])
            term_j = (params.K / (n_nodes ** 2)) * torch.sin(theta[i_idx] + theta[k_idx] - 2 * theta[j_idx])
            term_k = (params.K / (n_nodes ** 2)) * torch.sin(theta[i_idx] + theta[j_idx] - 2 * theta[k_idx])
            simplex_term.index_add_(0, i_idx, term_i)
            simplex_term.index_add_(0, j_idx, term_j)
            simplex_term.index_add_(0, k_idx, term_k)

        drive_term = params.d * torch.sin(theta - phi)

        theta_dot = omega + simplex_term
        phi_dot = nu + pairwise_term + drive_term

        return torch.stack([theta_dot, phi_dot], dim=1)

    @staticmethod
    def dynamic_phi(
        x: torch.Tensor,
        all_possible_edges: dict,
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        n_total = 0
        for key in ["edges", "triangles", "quads", "quints", "sexts", "septs"]:
            n_total += len(all_possible_edges.get(key, []))

        params = HypergraphModel.SimplicialParams(n_oscillators=n_nodes)

        theta = x[:, 0]
        phi = x[:, 1]

        Phi = torch.zeros((n_nodes, 2, n_total), device=device)
        edge_idx = 0

        edges = all_possible_edges.get("edges", [])
        if len(edges) > 0:
            edges_t = torch.as_tensor(edges, dtype=torch.long, device=device)
            i_idx = edges_t[:, 0]
            j_idx = edges_t[:, 1]
            edge_range = edge_idx + torch.arange(edges_t.shape[0], device=device)

            term_ij = (params.kappa / n_nodes) * torch.sin(phi[j_idx] - phi[i_idx])
            term_ji = (params.kappa / n_nodes) * torch.sin(phi[i_idx] - phi[j_idx])
            Phi[i_idx, 1, edge_range] = term_ij
            Phi[j_idx, 1, edge_range] = term_ji
            edge_idx += edges_t.shape[0]

        triangles = all_possible_edges.get("triangles", [])
        if len(triangles) > 0:
            tri_t = torch.as_tensor(triangles, dtype=torch.long, device=device)
            i_idx = tri_t[:, 0]
            j_idx = tri_t[:, 1]
            k_idx = tri_t[:, 2]
            edge_range = edge_idx + torch.arange(tri_t.shape[0], device=device)

            term_i = (params.K / (n_nodes ** 2)) * torch.sin(theta[j_idx] + theta[k_idx] - 2 * theta[i_idx])
            term_j = (params.K / (n_nodes ** 2)) * torch.sin(theta[i_idx] + theta[k_idx] - 2 * theta[j_idx])
            term_k = (params.K / (n_nodes ** 2)) * torch.sin(theta[i_idx] + theta[j_idx] - 2 * theta[k_idx])
            Phi[i_idx, 0, edge_range] = term_i
            Phi[j_idx, 0, edge_range] = term_j
            Phi[k_idx, 0, edge_range] = term_k
            edge_idx += tri_t.shape[0]

        return Phi

    @staticmethod
    def get_hyperedge_config(n_nodes: int, max_order: int = 3) -> dict:
        params = HypergraphModel.SimplicialParams(n_oscillators=n_nodes)
        rng = np.random.default_rng(params.seed)

        edges = HypergraphModel._sample_sparse_hyperedges(
            n_nodes,
            2,
            params.target_incidence_per_node_order2,
            rng,
        )
        triangles = HypergraphModel._sample_sparse_hyperedges(
            n_nodes,
            3,
            params.target_incidence_per_node_order3,
            rng,
        )

        return {
            "edges": edges if max_order >= 2 else [],
            "triangles": triangles if max_order >= 3 else [],
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
        params = HypergraphModel.SimplicialParams(n_oscillators=n_nodes, n_steps=n_samples)
        edges = np.array(edge_config.get("edges", []), dtype=int) - 1
        triangles = np.array(edge_config.get("triangles", []), dtype=int) - 1
        result = HypergraphModel._simulate(params, edges, triangles)

        HypergraphModel._cached_omega = result["omega"].copy()
        HypergraphModel._cached_nu = result["nu"].copy()
        HypergraphModel._cached_edges = edges.tolist() if edges.size > 0 else []
        HypergraphModel._cached_triangles = triangles.tolist() if triangles.size > 0 else []

        theta = result["theta"].T
        phi = result["phi"].T
        t = result["t"]

        x_data = np.stack([theta, phi], axis=2)
        if noise > 0:
            x_data = x_data + np.random.randn(*x_data.shape) * noise

        return t, x_data

    @staticmethod
    def get_default_params() -> dict:
        return {
            "n_nodes": 30,
            "max_order": 3,
        }
