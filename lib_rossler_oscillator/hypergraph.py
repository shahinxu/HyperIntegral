from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import os

import numpy as np
import torch
from scipy.integrate import solve_ivp


class HypergraphModel:
    @staticmethod
    def _preset_hypergraph_file(preset: str = "n100") -> Path:
        preset_name = (preset or "n100").strip().lower()
        preset_map = {
            "n8": "rossler_oscillator_n8_hypergraph.json",
            "n16": "rossler_oscillator_n16_hypergraph.json",
            "n32": "rossler_oscillator_n32_hypergraph.json",
            "n64": "rossler_oscillator_n64_hypergraph.json",
            "n100": "rossler_oscillator_n100_hypergraph.json",
        }
        if preset_name not in preset_map:
            raise ValueError(f"Unsupported Rossler hypergraph preset: {preset}")
        return Path(__file__).resolve().parent / "presets" / preset_map[preset_name]

    @staticmethod
    def _load_hypergraph_payload(file_path: Path) -> dict:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Rossler hypergraph file not found: {file_path}. "
                "Provide a JSON file instead of relying on hardcoded topology."
            )
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _payload_n_nodes(payload: dict) -> int:
        if "n_nodes" not in payload:
            raise ValueError("Rossler hypergraph payload must define 'n_nodes'.")
        return int(payload["n_nodes"])

    @staticmethod
    def _payload_max_order(payload: dict) -> int:
        for order, key in ((5, "edges5"), (4, "edges4"), (3, "edges3"), (2, "edges2")):
            if payload.get(key):
                return order
        return 0

    @staticmethod
    def _normalize_edges(raw_edges, order: int, index_base: int) -> list[list[int]]:
        normalized = []
        seen = set()
        for edge in raw_edges:
            edge_list = sorted(int(u) - index_base + 1 for u in edge)
            edge_tuple = tuple(edge_list)
            if len(edge_tuple) != order or edge_tuple in seen:
                continue
            seen.add(edge_tuple)
            normalized.append(list(edge_tuple))
        return normalized

    @staticmethod
    def _resolve_hypergraph_file(
        hypergraph_file: str | None = None,
        hypergraph_preset: str | None = None,
    ) -> Path:
        env_path = os.getenv("ROSSLER_HYPERGRAPH_FILE", "").strip()
        if env_path:
            return Path(env_path)

        env_preset = os.getenv("ROSSLER_HYPERGRAPH_PRESET", "").strip()
        if env_preset:
            return HypergraphModel._preset_hypergraph_file(env_preset)

        if hypergraph_file:
            return Path(hypergraph_file)

        if hypergraph_preset:
            return HypergraphModel._preset_hypergraph_file(hypergraph_preset)

        return HypergraphModel._preset_hypergraph_file()

    @staticmethod
    def _load_hypergraph_config(
        hypergraph_file: str | None = None,
        hypergraph_preset: str | None = None,
    ) -> dict:
        file_path = HypergraphModel._resolve_hypergraph_file(hypergraph_file, hypergraph_preset)
        payload = HypergraphModel._load_hypergraph_payload(file_path)
        index_base = int(payload.get("index_base", 1))

        config = {
            "edges": HypergraphModel._normalize_edges(payload.get("edges2", []), 2, index_base),
            "triangles": HypergraphModel._normalize_edges(payload.get("edges3", []), 3, index_base),
            "quads": HypergraphModel._normalize_edges(payload.get("edges4", []), 4, index_base),
            "quints": HypergraphModel._normalize_edges(payload.get("edges5", []), 5, index_base),
            "n_nodes": HypergraphModel._payload_n_nodes(payload),
            "max_order": HypergraphModel._payload_max_order(payload),
        }

        return config

    @staticmethod
    def _edge_array(edge_config: dict, key: str, order: int) -> np.ndarray:
        if edge_config.get(key):
            return np.array(edge_config[key], dtype=int)
        return np.empty((0, order), dtype=int)

    @staticmethod
    def _roessler_hoi_rhs(
        t: float,
        x: np.ndarray,
        edge_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        del t

        edge_list, triangle_list, quad_list, quint_list = edge_arrays
        m1 = len(x)
        n_local = m1 // 3
        xold = x[0:n_local]
        yold = x[n_local : 2 * n_local]
        zold = x[2 * n_local : 3 * n_local]
        ar, br, cr = 0.2, 0.2, 0.7
        k, kD = 0.4, 0.3

        coup_rete = np.zeros(n_local)
        coup_simplicial = np.zeros(n_local)
        coup_quads = np.zeros(n_local)
        coup_quints = np.zeros(n_local)

        for ii in range(len(edge_list)):
            i1 = edge_list[ii, 0] - 1
            i2 = edge_list[ii, 1] - 1
            coup_rete[i1] += xold[i2] - xold[i1]
            coup_rete[i2] += xold[i1] - xold[i2]

        for ii in range(len(triangle_list)):
            i1 = triangle_list[ii, 0] - 1
            i2 = triangle_list[ii, 1] - 1
            i3 = triangle_list[ii, 2] - 1
            coup_simplicial[i1] += xold[i2] ** 2 * xold[i3] - xold[i1] ** 3 + xold[i2] * xold[i3] ** 2 - xold[i1] ** 3
            coup_simplicial[i2] += xold[i1] ** 2 * xold[i3] - xold[i2] ** 3 + xold[i1] * xold[i3] ** 2 - xold[i2] ** 3
            coup_simplicial[i3] += xold[i1] ** 2 * xold[i2] - xold[i3] ** 3 + xold[i1] * xold[i2] ** 2 - xold[i3] ** 3

        for ii in range(len(quad_list)):
            i1 = quad_list[ii, 0] - 1
            i2 = quad_list[ii, 1] - 1
            i3 = quad_list[ii, 2] - 1
            i4 = quad_list[ii, 3] - 1
            coup_quads[i1] += xold[i2] ** 2 * xold[i3] * xold[i4] - xold[i1] ** 3
            coup_quads[i2] += xold[i1] ** 2 * xold[i3] * xold[i4] - xold[i2] ** 3
            coup_quads[i3] += xold[i1] ** 2 * xold[i2] * xold[i4] - xold[i3] ** 3
            coup_quads[i4] += xold[i1] ** 2 * xold[i2] * xold[i3] - xold[i4] ** 3

        for ii in range(len(quint_list)):
            i1 = quint_list[ii, 0] - 1
            i2 = quint_list[ii, 1] - 1
            i3 = quint_list[ii, 2] - 1
            i4 = quint_list[ii, 3] - 1
            i5 = quint_list[ii, 4] - 1
            coup_quints[i1] += yold[i2] ** 2 * yold[i3] * yold[i4] * yold[i5] - yold[i1] ** 3
            coup_quints[i2] += yold[i1] ** 2 * yold[i3] * yold[i4] * yold[i5] - yold[i2] ** 3
            coup_quints[i3] += yold[i1] ** 2 * yold[i2] * yold[i4] * yold[i5] - yold[i3] ** 3
            coup_quints[i4] += yold[i1] ** 2 * yold[i2] * yold[i3] * yold[i5] - yold[i4] ** 3
            coup_quints[i5] += yold[i1] ** 2 * yold[i2] * yold[i3] * yold[i4] - yold[i5] ** 3

        dxdt1 = -yold - zold + k * coup_rete + kD * coup_simplicial + kD * coup_quads
        dydt1 = xold + ar * yold + kD * coup_quints
        dzdt1 = br + zold * (xold - cr)
        return np.concatenate((dxdt1, dydt1, dzdt1))

    @staticmethod
    def dynamic_f(x: torch.Tensor, n_nodes: int) -> torch.Tensor:
        ar, br, cr = 0.2, 0.2, 0.7

        xold = x[:, 0]
        yold = x[:, 1]
        zold = x[:, 2]

        dxdt = -yold - zold
        dydt = xold + ar * yold
        dzdt = br + zold * (xold - cr)

        return torch.stack([dxdt, dydt, dzdt], dim=1)

    @staticmethod
    def dynamic_phi(
        x: torch.Tensor,
        all_possible_edges: dict,
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        xold = x[:, 0]
        yold = x[:, 1]
        zold = x[:, 2]

        k, kD = 0.4, 0.3

        n_edges = all_possible_edges["edges"].shape[0]
        n_triangles = all_possible_edges["triangles"].shape[0]
        n_quads = all_possible_edges["quads"].shape[0]
        n_quints = all_possible_edges["quints"].shape[0]
        n_total = n_edges + n_triangles + n_quads + n_quints

        Phi = torch.zeros((n_nodes, 3, n_total), device=device)
        edge_idx = 0

        if n_edges > 0:
            edges_tensor = all_possible_edges["edges"]
            i_indices = edges_tensor[:, 0]
            j_indices = edges_tensor[:, 1]

            diff = k * (xold[j_indices] - xold[i_indices])
            Phi[i_indices, 0, edge_idx + torch.arange(n_edges, device=device)] = diff
            Phi[j_indices, 0, edge_idx + torch.arange(n_edges, device=device)] = -diff
            edge_idx += n_edges

        if n_triangles > 0:
            triangles_tensor = all_possible_edges["triangles"]
            i_idx = triangles_tensor[:, 0]
            j_idx = triangles_tensor[:, 1]
            k_idx = triangles_tensor[:, 2]

            xi, xj, xk = xold[i_idx], xold[j_idx], xold[k_idx]
            edge_range = edge_idx + torch.arange(n_triangles, device=device)

            Phi[i_idx, 0, edge_range] = kD * (xj**2 * xk - xi**3 + xj * xk**2 - xi**3)
            Phi[j_idx, 0, edge_range] = kD * (xi**2 * xk - xj**3 + xi * xk**2 - xj**3)
            Phi[k_idx, 0, edge_range] = kD * (xi**2 * xj - xk**3 + xi * xj**2 - xk**3)
            edge_idx += n_triangles

        if n_quads > 0:
            quads_tensor = all_possible_edges["quads"]
            i_idx = quads_tensor[:, 0]
            j_idx = quads_tensor[:, 1]
            k_idx = quads_tensor[:, 2]
            l_idx = quads_tensor[:, 3]

            xi, xj, xk, xl = xold[i_idx], xold[j_idx], xold[k_idx], xold[l_idx]
            edge_range = edge_idx + torch.arange(n_quads, device=device)

            Phi[i_idx, 0, edge_range] = kD * (xj**2 * xk * xl - xi**3)
            Phi[j_idx, 0, edge_range] = kD * (xi**2 * xk * xl - xj**3)
            Phi[k_idx, 0, edge_range] = kD * (xi**2 * xj * xl - xk**3)
            Phi[l_idx, 0, edge_range] = kD * (xi**2 * xj * xk - xl**3)
            edge_idx += n_quads

        if n_quints > 0:
            quints_tensor = all_possible_edges["quints"]
            i_idx = quints_tensor[:, 0]
            j_idx = quints_tensor[:, 1]
            k_idx = quints_tensor[:, 2]
            l_idx = quints_tensor[:, 3]
            m_idx = quints_tensor[:, 4]

            yi, yj, yk, yl, ym = yold[i_idx], yold[j_idx], yold[k_idx], yold[l_idx], yold[m_idx]
            edge_range = edge_idx + torch.arange(n_quints, device=device)

            Phi[i_idx, 1, edge_range] = kD * (yj**2 * yk * yl * ym - yi**3)
            Phi[j_idx, 1, edge_range] = kD * (yi**2 * yk * yl * ym - yj**3)
            Phi[k_idx, 1, edge_range] = kD * (yi**2 * yj * yl * ym - yk**3)
            Phi[l_idx, 1, edge_range] = kD * (yi**2 * yj * yk * ym - yl**3)
            Phi[m_idx, 1, edge_range] = kD * (yi**2 * yj * yk * yl - ym**3)
            edge_idx += n_quints

        return Phi

    @staticmethod
    def get_hyperedge_config(
        n_nodes: int,
        max_order: int = 5,
        hypergraph_file: str | None = None,
        hypergraph_preset: str | None = None,
    ) -> dict:
        if max_order > 5:
            raise ValueError(f"Rossler only supports max_order <= 5, got {max_order}.")

        config = HypergraphModel._load_hypergraph_config(hypergraph_file, hypergraph_preset)

        filtered_config = {}
        if max_order >= 2:
            filtered_config["edges"] = config["edges"]
        else:
            filtered_config["edges"] = []

        if max_order >= 3:
            filtered_config["triangles"] = config["triangles"]
        else:
            filtered_config["triangles"] = []

        if max_order >= 4:
            filtered_config["quads"] = config["quads"]
        else:
            filtered_config["quads"] = []

        if max_order >= 5:
            filtered_config["quints"] = config["quints"]
        else:
            filtered_config["quints"] = []

        return filtered_config

    @staticmethod
    def generate_all_possible_hyperedges(n_nodes: int, max_order: int) -> dict:
        if max_order > 5:
            raise ValueError(f"Rossler only supports max_order <= 5, got {max_order}.")

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

        if max_order >= 5:
            all_possible["quints"] = [list(edge) for edge in combinations(range(1, n_nodes + 1), 5)]
        else:
            all_possible["quints"] = []

        return all_possible

    @staticmethod
    def generate_training_data(
        n_nodes: int,
        edge_config: dict,
        n_samples: int = 11,
        noise: float = 0.0,
        tmax: float = 20.0,
        flatten: bool = False,
    ):
        edge_arrays = (
            HypergraphModel._edge_array(edge_config, "edges", 2),
            HypergraphModel._edge_array(edge_config, "triangles", 3),
            HypergraphModel._edge_array(edge_config, "quads", 4),
            HypergraphModel._edge_array(edge_config, "quints", 5),
        )
        np.random.seed(42)
        x0 = np.random.uniform(-1, 1, size=(3 * n_nodes,))

        t_span = (0, tmax)
        t_eval = np.linspace(0, tmax, n_samples)
        sol = solve_ivp(
            lambda t, x: HypergraphModel._roessler_hoi_rhs(t, x, edge_arrays),
            t_span,
            x0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )

        x_data_flat = sol.y.T
        if flatten:
            if noise > 0:
                np.random.seed()
                x_data_flat = x_data_flat + np.random.randn(*x_data_flat.shape) * noise
            return sol.t, x_data_flat

        t_count = x_data_flat.shape[0]
        x_data = np.zeros((t_count, n_nodes, 3))
        x_data[:, :, 0] = x_data_flat[:, 0:n_nodes]
        x_data[:, :, 1] = x_data_flat[:, n_nodes : 2 * n_nodes]
        x_data[:, :, 2] = x_data_flat[:, 2 * n_nodes : 3 * n_nodes]

        if noise > 0:
            np.random.seed()
            x_data += np.random.randn(*x_data.shape) * noise
        return sol.t, x_data

    @staticmethod
    def get_default_params(hypergraph_preset: str | None = None) -> dict:
        config = HypergraphModel._load_hypergraph_config(hypergraph_preset=hypergraph_preset or "n100")
        return {
            "n_nodes": config["n_nodes"],
            "max_order": config["max_order"],
        }