from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from scipy.integrate import solve_ivp


class HypergraphModel:
    @staticmethod
    def dynamic_f(x: torch.Tensor, _n_nodes: int) -> torch.Tensor:

        v = x[:, 0]
        theta = x[:, 3]

        v_desired = 30.0
        tau_accel = 2.0
        tau_steer = 1.5

        dvdt = (v_desired - v) / tau_accel
        dxdt = v * torch.cos(theta)
        dydt = v * torch.sin(theta)
        dsdt = -theta / tau_steer

        return torch.stack([dvdt, dxdt, dydt, dsdt], dim=1)

    @staticmethod
    def dynamic_phi(
        x: torch.Tensor,
        all_possible_edges: dict,
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:

        v = x[:, 0]
        theta = x[:, 3]

        n_total = 0
        for key in ["edges", "triangles", "quads", "quints", "sexts", "septs"]:
            edges_k = all_possible_edges.get(key, None)
            if edges_k is None:
                continue
            if isinstance(edges_k, torch.Tensor):
                n_total += edges_k.shape[0]
            else:
                n_total += len(edges_k)

        Phi = torch.zeros((n_nodes, 4, n_total), device=device, dtype=x.dtype)
        edge_idx = 0

        def _as_long_tensor(edges, order: int) -> torch.Tensor:
            if edges is None:
                return torch.empty((0, order), dtype=torch.long, device=device)
            if isinstance(edges, torch.Tensor):
                if edges.numel() == 0:
                    return edges.to(device=device, dtype=torch.long)
                return edges.to(device=device, dtype=torch.long)
            edges = np.asarray(edges, dtype=np.int64)
            if edges.size == 0:
                return torch.empty((0, order), dtype=torch.long, device=device)
            if edges.min() >= 1:
                edges = edges - 1
            return torch.as_tensor(edges, dtype=torch.long, device=device)

        for key, order in [
            ("edges", 2),
            ("triangles", 3),
            ("quads", 4),
            ("quints", 5),
            ("sexts", 6),
            ("septs", 7),
        ]:
            edges_k = _as_long_tensor(all_possible_edges.get(key, None), order)
            if edges_k.shape[0] == 0:
                continue

            for e in edges_k:
                node_idx = e.tolist()  # 0-based indices
                v_group = v[node_idx]
                theta_group = theta[node_idx]

                if len(node_idx) > 1:
                    mean_v = v_group.mean()
                    mean_theta = theta_group.mean()
                else:
                    mean_v = v_group[0]
                    mean_theta = theta_group[0]

                for n in node_idx:
                    # Velocity alignment term
                    Phi[n, 0, edge_idx] = mean_v - v[n]
                    # Steering alignment term (wrapped via sine)
                    Phi[n, 3, edge_idx] = torch.sin(mean_theta - theta[n])

                edge_idx += 1

        return Phi

    @staticmethod
    def get_hyperedge_config(n_nodes: int, max_order: int = 5) -> dict:

        if n_nodes == 8:
            config = {
                "edges": [
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [5, 6],
                    [6, 7],
                    [7, 8],
                    [1, 5],
                    [4, 8],
                ],
                "triangles": [
                    [1, 2, 5],
                    [3, 4, 7],
                    [2, 6, 8],
                ],
                "quads": [
                    [1, 3, 5, 7],
                    [2, 4, 6, 8],
                ],
                "quints": [
                    [1, 2, 3, 4, 5],
                    [4, 5, 6, 7, 8],
                ],
                "sexts": [],
                "septs": [],
            }
        else:
            edges = [[i, i + 1] for i in range(1, n_nodes)]
            config = {
                "edges": edges,
                "triangles": [],
                "quads": [],
                "quints": [],
                "sexts": [],
                "septs": [],
            }

        def _filter(key: str, order: int):
            if max_order >= order:
                return config[key]
            return []

        return {
            "edges": _filter("edges", 2),
            "triangles": _filter("triangles", 3),
            "quads": _filter("quads", 4),
            "quints": _filter("quints", 5),
            "sexts": _filter("sexts", 6),
            "septs": _filter("septs", 7),
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

        if max_order >= 5:
            all_possible["quints"] = [list(edge) for edge in combinations(range(1, n_nodes + 1), 5)]
        else:
            all_possible["quints"] = []

        if max_order >= 6:
            all_possible["sexts"] = [list(edge) for edge in combinations(range(1, n_nodes + 1), 6)]
        else:
            all_possible["sexts"] = []

        if max_order >= 7:
            all_possible["septs"] = [list(edge) for edge in combinations(range(1, n_nodes + 1), 7)]
        else:
            all_possible["septs"] = []

        return all_possible

    @staticmethod
    def generate_training_data(
        n_nodes: int,
        edge_config: dict,
        n_samples: int = 201,
        noise: float = 0.0,
    ):

        def _to_array(name: str, order: int) -> np.ndarray:
            arr = np.array(edge_config.get(name, []), dtype=int)
            if arr.size == 0:
                return np.empty((0, order), dtype=int)
            if arr.min() >= 1:
                arr = arr - 1
            return arr

        EdgeList = _to_array("edges", 2)
        TriangleList = _to_array("triangles", 3)
        QuadList = _to_array("quads", 4)
        QuintList = _to_array("quints", 5)

        v_desired = 30.0
        tau_accel = 2.0
        tau_steer = 1.5
        k_follow = 0.3
        k_lane = 0.2
        k_intersection = 0.4
        k_roundabout = 0.5
        k_complex = 0.6

        def driving_rhs(t, x):
            m1 = len(x)
            N = m1 // 4

            v = x[0:N]
            theta = x[3 * N : 4 * N]

            coupling_follow = np.zeros(N)
            coupling_lane = np.zeros(N)
            coupling_intersection = np.zeros(N)
            coupling_roundabout = np.zeros(N)
            coupling_complex = np.zeros(N)

            # 2nd-order: following and lane-change interactions
            for e in EdgeList:
                i1, i2 = int(e[0]), int(e[1])
                if v[i1] > v[i2]:
                    coupling_follow[i2] += (v[i1] - v[i2]) * 0.5
                    coupling_follow[i1] -= (v[i1] - v[i2]) * 0.1
                else:
                    coupling_follow[i1] += (v[i2] - v[i1]) * 0.5
                    coupling_follow[i2] -= (v[i2] - v[i1]) * 0.1

                coupling_lane[i1] += np.sin(theta[i2]) * 0.3
                coupling_lane[i2] += np.sin(theta[i1]) * 0.3

            # 3rd-order: intersections
            if TriangleList.size > 0:
                for tri in TriangleList:
                    i1, i2, i3 = [int(u) for u in tri]
                    speeds = np.array([v[i1], v[i2], v[i3]])
                    idxs = np.array([i1, i2, i3])
                    max_idx = np.argmax(speeds)
                    priority = idxs[max_idx]
                    for idx in idxs:
                        if idx == priority:
                            others = [j for j in idxs if j != idx]
                            coupling_intersection[idx] += 0.2 * np.mean(v[others])
                        else:
                            coupling_intersection[idx] -= 0.4 * (v[priority] - v[idx])

            # 4th-order: roundabouts
            if QuadList.size > 0:
                for quad in QuadList:
                    i1, i2, i3, i4 = [int(u) for u in quad]
                    idxs = [i1, i2, i3, i4]
                    for idx in idxs:
                        others = [j for j in idxs if j != idx]
                        right_priority = sum(1 for j in others if j < idx)
                        coupling_roundabout[idx] += 0.3 * right_priority - 0.2 * len(others)
                        coupling_roundabout[idx] += 0.1 * np.sin(theta[idx] + np.pi / 4.0)

            # 5th-order: complex hubs
            if QuintList.size > 0:
                for quint in QuintList:
                    i1, i2, i3, i4, i5 = [int(u) for u in quint]
                    idxs = [i1, i2, i3, i4, i5]
                    for idx in idxs:
                        others = [j for j in idxs if j != idx]
                        avg_speed = np.mean(v[others])
                        coupling_complex[idx] += 0.2 * (avg_speed - v[idx])
                        avg_theta = np.mean(theta[others])
                        coupling_complex[idx] += 0.1 * np.sin(avg_theta - theta[idx])

            dvdt = (1.0 / tau_accel) * (
                v_desired
                - v
                + k_follow * coupling_follow
                + k_lane * coupling_lane
                + k_intersection * coupling_intersection
                + k_roundabout * coupling_roundabout
            )
            dxdt = v * np.cos(theta)
            dydt = v * np.sin(theta)
            dsdt = -theta / tau_steer + k_complex * coupling_complex

            return np.concatenate([dvdt, dxdt, dydt, dsdt])

        # Initial condition similar to the original script
        rng = np.random.default_rng(42)
        x0 = np.zeros(4 * n_nodes, dtype=float)
        x0[0:n_nodes] = rng.uniform(20.0, 35.0, size=n_nodes)
        x0[n_nodes : 2 * n_nodes] = rng.uniform(0.0, 1000.0, size=n_nodes)
        x0[2 * n_nodes : 3 * n_nodes] = rng.uniform(0.0, 500.0, size=n_nodes)
        x0[3 * n_nodes : 4 * n_nodes] = rng.uniform(-0.2, 0.2, size=n_nodes)

        t_span = (0.0, 50.0)
        t_eval = np.linspace(t_span[0], t_span[1], n_samples)

        sol = solve_ivp(
            driving_rhs,
            t_span,
            x0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        X_flat = sol.y.T  # (n_samples, 4 * n_nodes)
        x_data = np.zeros((X_flat.shape[0], n_nodes, 4), dtype=float)
        x_data[:, :, 0] = X_flat[:, 0:n_nodes]
        x_data[:, :, 1] = X_flat[:, n_nodes : 2 * n_nodes]
        x_data[:, :, 2] = X_flat[:, 2 * n_nodes : 3 * n_nodes]
        x_data[:, :, 3] = X_flat[:, 3 * n_nodes : 4 * n_nodes]

        if noise > 0.0:
            rng = np.random.default_rng()
            x_data = x_data + rng.normal(scale=noise, size=x_data.shape)

        return sol.t, x_data

    @staticmethod
    def get_default_params() -> dict:
        return {
            "n_nodes": 8,
            "max_order": 5,
        }
