from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from scipy.integrate import solve_ivp


class HypergraphModel:
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
        n_sexts = all_possible_edges["sexts"].shape[0]
        n_septs = all_possible_edges["septs"].shape[0]
        n_total = n_edges + n_triangles + n_quads + n_quints + n_sexts + n_septs

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

        if n_sexts > 0:
            sexts_tensor = all_possible_edges["sexts"]
            i_idx = sexts_tensor[:, 0]
            j_idx = sexts_tensor[:, 1]
            k_idx = sexts_tensor[:, 2]
            l_idx = sexts_tensor[:, 3]
            m_idx = sexts_tensor[:, 4]
            n_idx = sexts_tensor[:, 5]

            yi, yj, yk, yl, ym, yn = (
                yold[i_idx],
                yold[j_idx],
                yold[k_idx],
                yold[l_idx],
                yold[m_idx],
                yold[n_idx],
            )
            edge_range = edge_idx + torch.arange(n_sexts, device=device)

            Phi[i_idx, 1, edge_range] = kD * (yj**2 * yk * yl * ym * yn - yi**3)
            Phi[j_idx, 1, edge_range] = kD * (yi**2 * yk * yl * ym * yn - yj**3)
            Phi[k_idx, 1, edge_range] = kD * (yi**2 * yj * yl * ym * yn - yk**3)
            Phi[l_idx, 1, edge_range] = kD * (yi**2 * yj * yk * ym * yn - yl**3)
            Phi[m_idx, 1, edge_range] = kD * (yi**2 * yj * yk * yl * yn - ym**3)
            Phi[n_idx, 1, edge_range] = kD * (yi**2 * yj * yk * yl * ym - yn**3)
            edge_idx += n_sexts

        if n_septs > 0:
            septs_tensor = all_possible_edges["septs"]
            i_idx = septs_tensor[:, 0]
            j_idx = septs_tensor[:, 1]
            k_idx = septs_tensor[:, 2]
            l_idx = septs_tensor[:, 3]
            m_idx = septs_tensor[:, 4]
            n_idx = septs_tensor[:, 5]
            o_idx = septs_tensor[:, 6]

            zi, zj, zk, zl, zm, zn, zo = (
                zold[i_idx],
                zold[j_idx],
                zold[k_idx],
                zold[l_idx],
                zold[m_idx],
                zold[n_idx],
                zold[o_idx],
            )
            edge_range = edge_idx + torch.arange(n_septs, device=device)

            Phi[i_idx, 2, edge_range] = kD * (zj**2 * zk * zl * zm * zn * zo - zi**3)
            Phi[j_idx, 2, edge_range] = kD * (zi**2 * zk * zl * zm * zn * zo - zj**3)
            Phi[k_idx, 2, edge_range] = kD * (zi**2 * zj * zl * zm * zn * zo - zk**3)
            Phi[l_idx, 2, edge_range] = kD * (zi**2 * zj * zk * zm * zn * zo - zl**3)
            Phi[m_idx, 2, edge_range] = kD * (zi**2 * zj * zk * zl * zn * zo - zm**3)
            Phi[n_idx, 2, edge_range] = kD * (zi**2 * zj * zk * zl * zm * zo - zn**3)
            Phi[o_idx, 2, edge_range] = kD * (zi**2 * zj * zk * zl * zm * zn - zo**3)

        return Phi

    @staticmethod
    def get_hyperedge_config(n_nodes: int, max_order: int = 7) -> dict:
        configs = {
            8: {
                "edges": [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]],
                "triangles": [[1, 2, 3], [2, 4, 5], [5, 6, 7], [6, 7, 8]],
                "quads": [[1, 2, 3, 4]],
                "quints": [[4, 5, 6, 7, 8]],
                "sexts": [[1, 2, 3, 4, 5, 6]],
                "septs": [[3, 2, 4, 5, 6, 7, 8]],
            },
            10: {
                "edges": [[1, 2], [2, 3], [4, 5], [6, 7], [8, 9]],
                "triangles": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "quads": [[1, 2, 3, 4], [7, 8, 9, 10]],
                "quints": [[2, 3, 4, 5, 6]],
                "sexts": [[3, 4, 5, 6, 7, 8]],
                "septs": [[1, 2, 4, 6, 7, 9, 10]],
            },
            12: {
                "edges": [[1, 2], [2, 3], [4, 5], [6, 7], [9, 10], [11, 12]],
                "triangles": [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
                "quads": [[1, 2, 3, 4], [9, 10, 11, 12]],
                "quints": [[3, 4, 5, 6, 7]],
                "sexts": [[5, 6, 7, 8, 9, 10]],
                "septs": [[1, 3, 5, 7, 9, 11, 12]],
            },
            14: {
                "edges": [[1, 2], [2, 3], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13]],
                "triangles": [[1, 2, 3], [5, 6, 7], [10, 11, 12]],
                "quads": [[1, 2, 3, 4], [11, 12, 13, 14]],
                "quints": [[4, 5, 6, 7, 8]],
                "sexts": [[7, 8, 9, 10, 11, 12]],
                "septs": [[1, 3, 5, 8, 10, 12, 14]],
            },
            16: {
                "edges": [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [9, 10], [11, 12], [13, 14]],
                "triangles": [[1, 2, 3], [5, 6, 7], [10, 11, 12]],
                "quads": [[1, 2, 3, 4], [13, 14, 15, 16]],
                "quints": [[5, 6, 7, 8, 9]],
                "sexts": [[9, 10, 11, 12, 13, 14]],
                "septs": [[1, 4, 7, 10, 13, 15, 16]],
            },
        }

        if n_nodes not in configs:
            raise ValueError(f"n_nodes must be one of {list(configs.keys())}, got {n_nodes}")

        config = configs[n_nodes]

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

        if max_order >= 6:
            filtered_config["sexts"] = config["sexts"]
        else:
            filtered_config["sexts"] = []

        if max_order >= 7:
            filtered_config["septs"] = config["septs"]
        else:
            filtered_config["septs"] = []

        return filtered_config

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
    def generate_training_data(n_nodes: int, edge_config: dict, n_samples: int = 11, noise: float = 0.0):
        EdgeList = np.array(edge_config["edges"]) if edge_config["edges"] else np.empty((0, 2), dtype=int)
        TriangleList = np.array(edge_config["triangles"]) if edge_config["triangles"] else np.empty((0, 3), dtype=int)
        QuadList = np.array(edge_config["quads"]) if edge_config["quads"] else np.empty((0, 4), dtype=int)
        QuintList = np.array(edge_config["quints"]) if edge_config["quints"] else np.empty((0, 5), dtype=int)
        SextList = np.array(edge_config["sexts"]) if edge_config["sexts"] else np.empty((0, 6), dtype=int)
        SeptList = np.array(edge_config["septs"]) if edge_config["septs"] else np.empty((0, 7), dtype=int)

        def roessler_hoi(t, x):
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
            coup_sexts = np.zeros(n_local)
            coup_septs = np.zeros(n_local)

            for ii in range(len(EdgeList)):
                i1 = EdgeList[ii, 0] - 1
                i2 = EdgeList[ii, 1] - 1
                coup_rete[i1] += xold[i2] - xold[i1]
                coup_rete[i2] += xold[i1] - xold[i2]

            for ii in range(len(TriangleList)):
                i1 = TriangleList[ii, 0] - 1
                i2 = TriangleList[ii, 1] - 1
                i3 = TriangleList[ii, 2] - 1
                coup_simplicial[i1] += xold[i2] ** 2 * xold[i3] - xold[i1] ** 3 + xold[i2] * xold[i3] ** 2 - xold[i1] ** 3
                coup_simplicial[i2] += xold[i1] ** 2 * xold[i3] - xold[i2] ** 3 + xold[i1] * xold[i3] ** 2 - xold[i2] ** 3
                coup_simplicial[i3] += xold[i1] ** 2 * xold[i2] - xold[i3] ** 3 + xold[i1] * xold[i2] ** 2 - xold[i3] ** 3

            for ii in range(len(QuadList)):
                i1 = QuadList[ii, 0] - 1
                i2 = QuadList[ii, 1] - 1
                i3 = QuadList[ii, 2] - 1
                i4 = QuadList[ii, 3] - 1
                coup_quads[i1] += xold[i2] ** 2 * xold[i3] * xold[i4] - xold[i1] ** 3
                coup_quads[i2] += xold[i1] ** 2 * xold[i3] * xold[i4] - xold[i2] ** 3
                coup_quads[i3] += xold[i1] ** 2 * xold[i2] * xold[i4] - xold[i3] ** 3
                coup_quads[i4] += xold[i1] ** 2 * xold[i2] * xold[i3] - xold[i4] ** 3

            for ii in range(len(QuintList)):
                i1 = QuintList[ii, 0] - 1
                i2 = QuintList[ii, 1] - 1
                i3 = QuintList[ii, 2] - 1
                i4 = QuintList[ii, 3] - 1
                i5 = QuintList[ii, 4] - 1
                coup_quints[i1] += yold[i2] ** 2 * yold[i3] * yold[i4] * yold[i5] - yold[i1] ** 3
                coup_quints[i2] += yold[i1] ** 2 * yold[i3] * yold[i4] * yold[i5] - yold[i2] ** 3
                coup_quints[i3] += yold[i1] ** 2 * yold[i2] * yold[i4] * yold[i5] - yold[i3] ** 3
                coup_quints[i4] += yold[i1] ** 2 * yold[i2] * yold[i3] * yold[i5] - yold[i4] ** 3
                coup_quints[i5] += yold[i1] ** 2 * yold[i2] * yold[i3] * yold[i4] - yold[i5] ** 3

            for ii in range(len(SextList)):
                i1 = SextList[ii, 0] - 1
                i2 = SextList[ii, 1] - 1
                i3 = SextList[ii, 2] - 1
                i4 = SextList[ii, 3] - 1
                i5 = SextList[ii, 4] - 1
                i6 = SextList[ii, 5] - 1
                coup_sexts[i1] += yold[i2] ** 2 * yold[i3] * yold[i4] * yold[i5] * yold[i6] - yold[i1] ** 3
                coup_sexts[i2] += yold[i1] ** 2 * yold[i3] * yold[i4] * yold[i5] * yold[i6] - yold[i2] ** 3
                coup_sexts[i3] += yold[i1] ** 2 * yold[i2] * yold[i4] * yold[i5] * yold[i6] - yold[i3] ** 3
                coup_sexts[i4] += yold[i1] ** 2 * yold[i2] * yold[i3] * yold[i5] * yold[i6] - yold[i4] ** 3
                coup_sexts[i5] += yold[i1] ** 2 * yold[i2] * yold[i3] * yold[i4] * yold[i6] - yold[i5] ** 3
                coup_sexts[i6] += yold[i1] ** 2 * yold[i2] ** 2 * yold[i3] * yold[i4] * yold[i5] - yold[i6] ** 3

            for ii in range(len(SeptList)):
                i1 = SeptList[ii, 0] - 1
                i2 = SeptList[ii, 1] - 1
                i3 = SeptList[ii, 2] - 1
                i4 = SeptList[ii, 3] - 1
                i5 = SeptList[ii, 4] - 1
                i6 = SeptList[ii, 5] - 1
                i7 = SeptList[ii, 6] - 1
                coup_septs[i1] += zold[i2] ** 2 * zold[i3] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i1] ** 3
                coup_septs[i2] += zold[i1] ** 2 * zold[i3] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i2] ** 3
                coup_septs[i3] += zold[i1] ** 2 * zold[i2] * zold[i4] * zold[i5] * zold[i6] * zold[i7] - zold[i3] ** 3
                coup_septs[i4] += zold[i1] ** 2 * zold[i2] * zold[i3] * zold[i5] * zold[i6] * zold[i7] - zold[i4] ** 3
                coup_septs[i5] += zold[i1] ** 2 * zold[i2] ** 2 * zold[i3] * zold[i4] * zold[i6] * zold[i7] - zold[i5] ** 3
                coup_septs[i6] += zold[i1] ** 2 * zold[i2] ** 2 * zold[i3] * zold[i4] * zold[i5] * zold[i7] - zold[i6] ** 3
                coup_septs[i7] += zold[i1] ** 2 * zold[i2] ** 2 * zold[i3] * zold[i4] * zold[i5] * zold[i6] - zold[i7] ** 3

            dxdt1 = -yold - zold + k * coup_rete + kD * coup_simplicial + kD * coup_quads
            dydt1 = xold + ar * yold + kD * coup_quints + kD * coup_sexts
            dzdt1 = br + zold * (xold - cr) + kD * coup_septs
            dxdt = np.concatenate((dxdt1, dydt1, dzdt1))
            return dxdt

        np.random.seed(42)
        x0 = np.random.uniform(-1, 1, size=(3 * n_nodes,))

        t_span = (0, 20)
        t_eval = np.linspace(0, 20, n_samples)
        sol = solve_ivp(roessler_hoi, t_span, x0, t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)

        x_data_flat = sol.y.T
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
    def get_default_params() -> dict:
        return {
            "n_nodes": 8,
            "max_order": 7,
        }