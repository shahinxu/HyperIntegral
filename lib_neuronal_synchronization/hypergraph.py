"""
Hypergraph-style interface for neuronal synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from scipy.integrate import solve_ivp


_CACHED_OMEGA = None
_CACHED_NU = None


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


def build_global_simplicial_complex(n: int):
    nodes = np.arange(n)
    edges = list(combinations(range(n), 2))
    triangles = list(combinations(range(n), 3))
    return nodes, edges, triangles


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _kuramoto_pairwise_term(phi: np.ndarray) -> np.ndarray:
    diff = phi[None, :] - phi[:, None]
    return np.sin(diff).mean(axis=1)


def rhs(t: float, y: np.ndarray, omega: np.ndarray, nu: np.ndarray, p: SimplicialParams):
    n = p.n_oscillators
    theta = y[:n]
    phi = y[n:]

    z1 = np.exp(1j * theta).mean()
    simplex_term = p.K * np.imag((z1 ** 2) * np.exp(-2j * theta))

    pairwise_term = p.kappa * _kuramoto_pairwise_term(phi)
    drive_term = p.d * np.sin(theta - phi)

    theta_dot = omega + simplex_term
    phi_dot = nu + pairwise_term + drive_term

    return np.concatenate([theta_dot, phi_dot])


def simulate(params: SimplicialParams):
    rng = np.random.default_rng(params.seed)

    omega = rng.normal(params.omega_mean, params.omega_std, params.n_oscillators)
    nu = rng.normal(params.nu_mean, params.nu_std, params.n_oscillators)

    theta0 = rng.uniform(-np.pi, np.pi, size=params.n_oscillators)
    phi0 = rng.uniform(-np.pi, np.pi, size=params.n_oscillators)
    y0 = np.concatenate([theta0, phi0])

    t_eval = np.linspace(params.t_span[0], params.t_span[1], params.n_steps)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, omega, nu, params),
        t_span=params.t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    theta = _wrap_to_pi(sol.y[: params.n_oscillators])
    phi = _wrap_to_pi(sol.y[params.n_oscillators :])

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


def roessler_dynamics(x: torch.Tensor, n_nodes: int) -> torch.Tensor:
    params = SimplicialParams(n_oscillators=n_nodes)
    theta = x[:, 0]
    phi = x[:, 1]

    if _CACHED_OMEGA is None or _CACHED_NU is None:
        omega = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
        nu = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
    else:
        omega = torch.as_tensor(_CACHED_OMEGA, device=x.device, dtype=x.dtype)
        nu = torch.as_tensor(_CACHED_NU, device=x.device, dtype=x.dtype)

    z1 = torch.exp(1j * theta).mean()
    simplex_term = params.K * torch.imag((z1 ** 2) * torch.exp(-2j * theta))

    diff = phi[None, :] - phi[:, None]
    pairwise_term = params.kappa * torch.sin(diff).mean(dim=1)
    drive_term = params.d * torch.sin(theta - phi)

    theta_dot = omega + simplex_term
    phi_dot = nu + pairwise_term + drive_term

    return torch.stack([theta_dot, phi_dot], dim=1)


def _count_edges(all_possible_edges: dict) -> int:
    return sum(len(all_possible_edges.get(key, [])) for key in ["edges", "triangles", "quads", "quints", "sexts", "septs"])


def compute_hyperedge_coupling_tensor(
    x: torch.Tensor,
    all_possible_edges: dict,
    n_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    params = SimplicialParams(n_oscillators=n_nodes)
    n_total = _count_edges(all_possible_edges)

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


def get_hyperedge_config(n_nodes: int, max_order: int = 3) -> dict:
    edges = [list(edge) for edge in combinations(range(1, n_nodes + 1), 2)]
    triangles = [list(edge) for edge in combinations(range(1, n_nodes + 1), 3)]

    return {
        "edges": edges if max_order >= 2 else [],
        "triangles": triangles if max_order >= 3 else [],
        "quads": [],
        "quints": [],
        "sexts": [],
        "septs": [],
    }


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


def generate_training_data(n_nodes: int, edge_config: dict, n_samples: int = 11, noise: float = 0.0):
    params = SimplicialParams(n_oscillators=n_nodes, n_steps=n_samples)
    result = simulate(params)

    global _CACHED_OMEGA, _CACHED_NU
    _CACHED_OMEGA = result["omega"].copy()
    _CACHED_NU = result["nu"].copy()

    theta = result["theta"].T
    phi = result["phi"].T
    t = result["t"]

    x_data = np.stack([theta, phi], axis=2)
    if noise > 0:
        x_data = x_data + np.random.randn(*x_data.shape) * noise

    return t, x_data
