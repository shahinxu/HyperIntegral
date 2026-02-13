import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from itertools import combinations
from scipy.integrate import solve_ivp
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


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
    """
    Build the simplicial structure matching the paper's globally coupled setting.

    - 0-simplices: nodes
    - 1-simplices: all pairwise links (complete graph)
    - 2-simplices: all triangles (complete 2-complex)
    """
    nodes = np.arange(n)
    edges = list(combinations(range(n), 2))
    triangles = list(combinations(range(n), 3))
    return nodes, edges, triangles


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _kuramoto_pairwise_term(phi: np.ndarray) -> np.ndarray:
    """Compute (1/N) * sum_j sin(phi_j - phi_i) for each i."""
    diff = phi[None, :] - phi[:, None]
    return np.sin(diff).mean(axis=1)


def rhs(t: float, y: np.ndarray, omega: np.ndarray, nu: np.ndarray, p: SimplicialParams):
    """
    2-layer simplicial oscillator model (from the paper):

    theta_dot_i = omega_i + (K/N^2) * sum_{j,k} sin(theta_j + theta_k - 2 theta_i)
    phi_dot_i   = nu_i + (kappa/N) * sum_j sin(phi_j - phi_i) + d * sin(theta_i - phi_i)

    Using z1 = (1/N) * sum_j exp(i*theta_j), the 2-simplex term becomes:
    (K/N^2) * sum_{j,k} sin(theta_j + theta_k - 2 theta_i)
      = K * Im( z1^2 * exp(-2 i theta_i) ).
    """
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


def plot_simplicial_structure(n_visual: int = 10, save_path: Path = BASE_DIR / "simplicial_complex_structure.png"):
    """Draw a compact schematic of complete 2-complex: nodes + edges + filled triangles."""
    nodes, edges, triangles = build_global_simplicial_complex(n_visual)

    angles = np.linspace(0, 2 * np.pi, n_visual, endpoint=False)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, j, k in triangles:
        tri = np.array([xy[i], xy[j], xy[k]])
        poly = plt.Polygon(tri, color="#8ecae6", alpha=0.03, linewidth=0)
        ax.add_patch(poly)

    for i, j in edges:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="#6c757d", alpha=0.15, lw=0.8)

    ax.scatter(xy[:, 0], xy[:, 1], s=60, c="#1f77b4", zorder=3)
    for i, (x, y) in enumerate(xy):
        ax.text(x * 1.08, y * 1.08, str(i), fontsize=9, ha="center", va="center")

    ax.set_title("Complete 2-Simplicial Complex (Schematic)")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=220)
    plt.close(fig)


def plot_dynamics(result: dict, save_path: Path = BASE_DIR / "neuronal_sync_dynamics.png"):
    t = result["t"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, result["r_theta"], label=r"$r_\theta=|z_1|$", color="tab:blue")
    axes[0].plot(t, result["r2_theta"], label=r"$|z_2|$", color="tab:orange")
    axes[0].set_ylabel("2-simplex layer order")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, result["r_phi"], label=r"$r_\phi$", color="tab:green")
    axes[1].set_ylabel("1-simplex layer order")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Neuronal Synchronization on Simplicial Complex")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=220)
    plt.close(fig)


def plot_node_trajectories(result: dict):
    t = result["t"]
    theta = result["theta"]
    phi = result["phi"]
    n = theta.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(n):
        axes[0].plot(t, theta[i], lw=0.9, alpha=0.9)
        axes[1].plot(t, phi[i], lw=0.9, alpha=0.9)

    axes[0].set_title("Theta-layer node trajectories")
    axes[0].set_ylabel("theta_i(t)")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Phi-layer node trajectories")
    axes[1].set_ylabel("phi_i(t)")
    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Node-wise dynamics (N={n})")
    fig.tight_layout()
    out_path = BASE_DIR / "neuronal_nodes_timeseries.png"
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    params = SimplicialParams()

    nodes, edges, triangles = build_global_simplicial_complex(params.n_oscillators)
    print("=" * 72)
    print("Simplicial model summary")
    print("=" * 72)
    print(f"0-simplices (nodes): {len(nodes)}")
    print(f"1-simplices (edges): {len(edges)}")
    print(f"2-simplices (triangles): {len(triangles)}")
    print("Dynamics:")
    print("  theta_dot_i = omega_i + (K/N^2) * sum_{j,k} sin(theta_j + theta_k - 2 theta_i)")
    print("  phi_dot_i   = nu_i + (kappa/N) * sum_j sin(phi_j - phi_i) + d*sin(theta_i - phi_i)")
    print("=" * 72)

    res = simulate(params)
    out_path = plot_node_trajectories(res)

    print("Saved figure:")
    print(f"  - {out_path}")
