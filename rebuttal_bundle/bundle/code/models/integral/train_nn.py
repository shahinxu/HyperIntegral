"""IntegralPINN as described in the paper: a neural surrogate of the trajectory
trained jointly with the higher-order interaction tensors under an integral
physics constraint.

This is the faithful implementation of Section III:

  * state parameterization: a residual MLP surrogate xhat(t)  (Sec. III-A)
  * data loss:    L_data    = mean || xhat(t_l) - xtilde(t_l) ||^2
  * interaction parameterization: Bhat = sigmoid(clip(W, -c, c))  (Sec. III-A)
  * integral physics constraint over sampled windows, evaluated with
    Gauss-Legendre quadrature on the surrogate                   (Sec. III-B)
  * staged optimization: data only, then a linear ramp on lambda_physics
  * binary/non-smooth extension via a Bernoulli cross-entropy data term
                                                                 (Sec. III-C)

It is scene-generic: the dictionary terms f and f^(k) come from the scene
library through ``dynamic_f`` / ``dynamic_phi``.  Because ``dynamic_phi``
materializes an [N, d, E] tensor, and E is ~1.7e5 for the 30-oscillator
neuronal benchmark, the candidate set is processed in chunks and contracted
against the corresponding slice of the interaction vector, so the full tensor
is never formed.

The derivative-based counterpart (``--loss_mode derivative``) keeps the
surrogate, the parameterization and the optimizer identical and only swaps the
integral residual for a pointwise derivative residual, which is the ablation
the reviewers ask for.
"""

import argparse
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypergraph.outputs import write_standard_summary
from hypergraph.scene_registry import get_scene_model
from models.integral.train import (
    TimeResNet,
    build_orderwise_support,
    build_orderwise_support_sampled,
    compute_auc_scores,
    infer_max_order_from_edge_config,
    infer_n_nodes_from_edge_config,
    macro_auc,
    plot_roc_curves,
    save_roc_curve_data,
    set_global_seed,
)

ORDER_KEYS = ["edges", "triangles", "quads", "quints"]
ORDER_SIZES = {"edges": 2, "triangles": 3, "quads": 4, "quints": 5}


def legendre_quadrature(n_quad, device):
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    return (torch.tensor(nodes, dtype=torch.float32, device=device),
            torch.tensor(weights, dtype=torch.float32, device=device))


def chunk_edges(orderwise_support, device, chunk_size):
    """Split the candidate set into chunks of edge-index tensors.

    Each chunk is ``(edges_dict, lo, hi)`` where ``edges_dict`` is in the shape
    ``dynamic_phi`` expects and ``[lo, hi)`` is the slice of the flattened
    interaction vector it corresponds to.  Chunk boundaries never cross an
    order, so each call sees a single populated order.
    """
    chunks = []
    offset = 0
    for key in ORDER_KEYS:
        edges = orderwise_support.get(key, [])
        n = len(edges)
        if n == 0:
            continue
        arr = torch.tensor(edges, dtype=torch.long, device=device) - 1
        step = n if chunk_size <= 0 else min(chunk_size, n)
        for start in range(0, n, step):
            stop = min(start + step, n)
            d = {k: torch.empty((0, ORDER_SIZES[k]), dtype=torch.long, device=device)
                 for k in ORDER_KEYS}
            d[key] = arr[start:stop]
            chunks.append((d, offset + start, offset + stop))
        offset += n
    return chunks


def dynamics_with_A(HypergraphModel, x, A_eff, chunks, N, device, t=None,
                    accepts_time=False):
    """f(x) + sum_k Q^(k)(x, Bhat) for a batch of states, without ever
    materializing the [N, d, E] coupling tensor.

    x: [P, N, d] states at P time points.  Returns [P, N, d].
    """
    rows = []
    for p in range(x.shape[0]):
        x_p = x[p]
        if accepts_time and t is not None:
            acc = HypergraphModel.dynamic_f(x_p, N, t[p])
        else:
            acc = HypergraphModel.dynamic_f(x_p, N)
        for edges_d, lo, hi in chunks:
            phi = HypergraphModel.dynamic_phi(x_p, edges_d, N, device)  # [N, d, chunk]
            acc = acc + torch.einsum('nie,e->ni', phi, A_eff[lo:hi])
        rows.append(acc)
    return torch.stack(rows, dim=0)



# ---------------------------------------------------------------------------
# Batched, contraction-only dynamics
#
# ``dynamic_phi`` materializes an [N, d, E] tensor per time point, which is both
# slow (one call per point) and impossible at E ~ 1.7e5.  What the residual
# actually needs is Phi(x) @ A, so we compute the per-edge term directly and
# scatter term * A into the node rows.  Memory is [P, E] instead of [P, N, d, E].
#
# Each scene's rule below mirrors its ``dynamic_phi`` exactly; `verify_contract`
# checks that numerically, so the duplication cannot silently drift.
# ---------------------------------------------------------------------------

def _contract_ecological(HypergraphModel, x, edges, order, A_slice, out):
    """Every member of a k-edge receives -w_k * prod(x over members)."""
    params = HypergraphModel._cached_params
    w = {2: params.w2, 3: params.w3, 4: params.w4, 5: params.w5}[order]
    xv = x[:, :, 0]                                            # [P, N]
    cols = [xv[:, edges[:, m]] for m in range(order)]          # order x [P, Ek]
    prod = cols[0]
    for c in cols[1:]:
        prod = prod * c
    term = (-w) * prod * A_slice[None, :]                      # [P, Ek]
    for m in range(order):
        out[:, :, 0].index_add_(1, edges[:, m], term)
    return out


def _contract_social(HypergraphModel, x, edges, order, A_slice, out):
    """Member m receives coef * (1 - x_m) * prod(x over the other members)."""
    params = HypergraphModel.SCMParams()
    coef = params.beta if order == 2 else params.beta_delta
    xv = x[:, :, 0]
    cols = [xv[:, edges[:, m]] for m in range(order)]
    for m in range(order):
        others = None
        for q in range(order):
            if q == m:
                continue
            others = cols[q] if others is None else others * cols[q]
        term = coef * (1.0 - cols[m]) * others * A_slice[None, :]
        out[:, :, 0].index_add_(1, edges[:, m], term)
    return out


def _contract_neuronal(HypergraphModel, x, edges, order, A_slice, out):
    """Order 2 acts on phi (dim 1); orders >= 3 act on theta (dim 0)."""
    params = HypergraphModel.SimplicialParams()
    N = x.shape[1]
    if order == 2:
        phi = x[:, :, 1]
        pi, pj = phi[:, edges[:, 0]], phi[:, edges[:, 1]]
        coef = params.kappa / N
        out[:, :, 1].index_add_(1, edges[:, 0], coef * torch.sin(pj - pi) * A_slice[None, :])
        out[:, :, 1].index_add_(1, edges[:, 1], coef * torch.sin(pi - pj) * A_slice[None, :])
        return out
    gamma = {3: params.K, 4: params.K4, 5: params.K5}[order]
    coef = gamma / (N ** (order - 1))
    th = x[:, :, 0]
    cols = [th[:, edges[:, m]] for m in range(order)]
    total = cols[0]
    for c in cols[1:]:
        total = total + c
    for m in range(order):
        term = coef * torch.sin(total - cols[m] - (order - 1) * cols[m]) * A_slice[None, :]
        out[:, :, 0].index_add_(1, edges[:, m], term)
    return out


CONTRACT_RULES = {
    "lib_ecological_dynamics": _contract_ecological,
    "lib_social_contagion": _contract_social,
    "lib_neuronal_synchronization": _contract_neuronal,
}


def get_contract_rule(HypergraphModel):
    mod = getattr(HypergraphModel, "__module__", "")
    for key, fn in CONTRACT_RULES.items():
        if key in mod:
            return fn
    return None


def dynamics_with_A_fast(HypergraphModel, x, A_eff, order_blocks, N, device,
                         t=None, accepts_time=False, rule=None):
    """f(x) + Phi(x) @ A for a batch of states, contraction only.

    order_blocks: [(order, edges_tensor, lo, hi), ...] over the flattened A.
    """
    batch_f = getattr(HypergraphModel, "dynamic_f_batch", None)
    if batch_f is not None:
        base = batch_f(x, N, t) if accepts_time else batch_f(x, N)
        if base.shape != x.shape:
            base = base.view(x.shape)
    elif accepts_time and t is not None:
        base = torch.stack([HypergraphModel.dynamic_f(x[p], N, t[p])
                            for p in range(x.shape[0])], dim=0)
    else:
        base = torch.stack([HypergraphModel.dynamic_f(x[p], N)
                            for p in range(x.shape[0])], dim=0)
    out = torch.zeros_like(base)
    for order, edges, lo, hi in order_blocks:
        rule(HypergraphModel, x, edges, order, A_eff[lo:hi], out)
    return base + out


def build_order_blocks(orderwise_support, device):
    blocks, offset = [], 0
    for key in ORDER_KEYS:
        e = orderwise_support.get(key, [])
        if not e:
            continue
        arr = torch.tensor(e, dtype=torch.long, device=device) - 1
        blocks.append((ORDER_SIZES[key], arr, offset, offset + len(e)))
        offset += len(e)
    return blocks


def verify_contract(HypergraphModel, orderwise_support, N, state_dim, device, rule,
                    n_points=3, tol=1e-4):
    """Assert the contracted path equals dynamic_phi @ A on random states."""
    blocks = build_order_blocks(orderwise_support, device)
    E = sum(len(orderwise_support.get(k, [])) for k in ORDER_KEYS)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.rand((n_points, N, state_dim), generator=g).to(device) * 0.5 + 0.25
    A = torch.rand((E,), generator=g).to(device)
    edges_full = {k: (torch.tensor(orderwise_support[k], dtype=torch.long, device=device) - 1
                      if orderwise_support.get(k) else
                      torch.empty((0, ORDER_SIZES[k]), dtype=torch.long, device=device))
                  for k in ORDER_KEYS}
    ref = torch.stack([
        torch.einsum('nie,e->ni', HypergraphModel.dynamic_phi(x[p], edges_full, N, device), A)
        for p in range(n_points)], dim=0)
    got = torch.zeros_like(ref)
    for order, edges, lo, hi in blocks:
        rule(HypergraphModel, x, edges, order, A[lo:hi], got)
    err = (ref - got).abs().max().item()
    scale = ref.abs().max().item() + 1e-12
    assert err / scale < tol, f"contracted dynamics disagree with dynamic_phi: {err:.3e} (scale {scale:.3e})"
    return err / scale


def flat_to_state(flat, N, state_dim):
    """[P, N*d] -> [P, N, d] in the same layout used to flatten the data."""
    return flat.view(flat.shape[0], state_dim, N).permute(0, 2, 1).contiguous()


def state_to_flat(x, N, state_dim):
    return x.permute(0, 2, 1).reshape(x.shape[0], state_dim * N)


def train(
    HypergraphModel, scene_label, N, max_order, edge_config, orderwise_support,
    n_samples=300, noise=0.02, n_epochs=20000, lr=1e-3, seed=42, gpu_id=0,
    stage1_epochs=2500, physics_ramp_epochs=2000, lambda_physics=1.0,
    n_intervals=32, n_quad=5, window_frac=1.0, loss_mode="integral",
    bounded=True, clip_c=2.0, chunk_size=20000, w_init=-3.0,
    nn_hidden_dim=64, nn_layers=8, results_root="results/integral_nn",
    run_tag=None, eval_every=500,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed)
    print(f"[config] scene={scene_label} N={N} max_order={max_order} seed={seed} "
          f"loss_mode={loss_mode} bounded={bounded} n_intervals={n_intervals} "
          f"n_quad={n_quad} window_frac={window_frac} stage1={stage1_epochs} "
          f"ramp={physics_ramp_epochs}")

    is_binary = "lib_social_contagion" in getattr(HypergraphModel, "__module__", "")

    gen_kwargs = {"noise": noise}
    if "seed" in inspect.signature(HypergraphModel.generate_training_data).parameters:
        gen_kwargs["seed"] = seed
    t_data, x_data = HypergraphModel.generate_training_data(
        N, edge_config, n_samples, **gen_kwargs)
    n_times, _, state_dim = x_data.shape
    print(f"data: {x_data.shape}  binary_scene={is_binary}")

    t_gpu = torch.tensor(t_data, dtype=torch.float32, device=device).unsqueeze(1)   # [T,1]
    x_obs = torch.tensor(x_data, dtype=torch.float32, device=device)                # [T,N,d]
    x_obs_flat = state_to_flat(x_obs, N, state_dim)                                 # [T,N*d]

    net = TimeResNet(output_dim=N * state_dim, hidden_dim=nn_hidden_dim,
                     num_layers=nn_layers, dropout=0.0).to(device)
    net.t_mean = t_gpu.mean()
    net.t_std = t_gpu.std()
    net.x_mean = x_obs_flat.mean(dim=0)
    net.x_std = x_obs_flat.std(dim=0) + 1e-8

    n_hyperedges = sum(len(orderwise_support.get(k, [])) for k in ORDER_KEYS)
    rule = get_contract_rule(HypergraphModel)
    if rule is not None:
        order_blocks = build_order_blocks(orderwise_support, device)
        rel = verify_contract(HypergraphModel, orderwise_support, N, state_dim, device, rule)
        print(f"candidates E={n_hyperedges}; contracted dynamics verified "
              f"against dynamic_phi (rel. err {rel:.1e})")
        def dyn(xb, tb=None):
            return dynamics_with_A_fast(HypergraphModel, xb, _cell[0], order_blocks, N,
                                        device, t=tb, accepts_time=accepts_time, rule=rule)
    else:
        chunks = chunk_edges(orderwise_support, device, chunk_size)
        print(f"candidates E={n_hyperedges} in {len(chunks)} chunk(s) (slow path)")
        def dyn(xb, tb=None):
            return dynamics_with_A(HypergraphModel, xb, _cell[0], chunks, N, device,
                                   t=tb, accepts_time=accepts_time)

    # Bounded parameterization: Bhat = sigmoid(clip(W, -c, c)).  The clip keeps
    # the pre-activation inside the sensitive region of the sigmoid.
    # Weak-coupling init: W = 0 would mean Bhat = 0.5 for every candidate, i.e.
    # a dense hypergraph with 372+ simultaneous interactions at half strength.
    # Starting near-empty matches the sparse prior and the init used by the
    # other implementations in this repository.
    W = (w_init + 0.02 * torch.randn(n_hyperedges, device=device)).requires_grad_(True)
    opt = optim.Adam([{"params": net.parameters(), "lr": lr},
                      {"params": [W], "lr": lr}])

    quad_nodes, quad_weights = legendre_quadrature(n_quad, device)
    t0, t1_end = float(t_data[0]), float(t_data[-1])
    span = t1_end - t0
    max_window = max(1e-6, window_frac * span)
    accepts_time = len(inspect.signature(HypergraphModel.dynamic_f).parameters) >= 3
    _cell = [None]

    run_tag = run_tag or f"{scene_label}_seed{seed}"
    save_dir = os.path.join(results_root, scene_label,
                            f"sample_{n_samples}_noise_{noise}", run_tag)
    os.makedirs(save_dir, exist_ok=True)

    pbar = tqdm(range(n_epochs), desc=f"IntegralPINN[{scene_label}]", ncols=130)
    for epoch in pbar:
        opt.zero_grad()

        x_pred_flat = net(t_gpu, normalize=True)
        if is_binary:
            # Binary observations: Bernoulli cross-entropy instead of MSE.
            p = torch.clamp(x_pred_flat, 1e-4, 1.0 - 1e-4)
            loss_data = -(x_obs_flat * torch.log(p)
                          + (1.0 - x_obs_flat) * torch.log(1.0 - p)).mean()
        else:
            loss_data = torch.mean((x_pred_flat - x_obs_flat) ** 2)

        A_eff = torch.sigmoid(torch.clamp(W, -clip_c, clip_c) if bounded else W)
        _cell[0] = A_eff

        if epoch < stage1_epochs:
            loss_phys = torch.zeros((), device=device)
        elif loss_mode == "derivative":
            tc = t0 + span * torch.rand(n_intervals, device=device)
            tc = tc.unsqueeze(1).requires_grad_(True)
            xc_flat = net(tc, normalize=True)
            dxdt_flat = torch.stack([
                torch.autograd.grad(xc_flat[:, i].sum(), tc,
                                    create_graph=True, retain_graph=True)[0].squeeze(-1)
                for i in range(xc_flat.shape[1])], dim=1)
            xc = flat_to_state(xc_flat, N, state_dim)
            rhs = dyn(xc, tc.squeeze(-1))
            loss_phys = torch.mean((flat_to_state(dxdt_flat, N, state_dim) - rhs) ** 2)
        else:
            ta = t0 + (span - max_window) * torch.rand(n_intervals, device=device) \
                 if max_window < span else t0 + span * torch.rand(n_intervals, device=device)
            length = max_window * torch.rand(n_intervals, device=device)
            tb = torch.clamp(ta + length, max=t1_end)
            mid, half = 0.5 * (ta + tb), 0.5 * (tb - ta)

            t_eval = mid[:, None] + half[:, None] * quad_nodes[None, :]       # [B,Q]
            x_eval = flat_to_state(net(t_eval.reshape(-1, 1), normalize=True),
                                   N, state_dim)                              # [B*Q,N,d]
            dyn_vals = dyn(x_eval, t_eval.reshape(-1)).view(n_intervals, n_quad, N, state_dim)
            integral = torch.einsum('bqni,q->bni', dyn_vals, quad_weights) * half[:, None, None]

            lhs = flat_to_state(net(tb.unsqueeze(1), normalize=True)
                                - net(ta.unsqueeze(1), normalize=True), N, state_dim)
            loss_phys = torch.mean((lhs - integral) ** 2)

        if epoch < stage1_epochs:
            w_phys = 0.0
        elif physics_ramp_epochs <= 0:
            w_phys = lambda_physics
        else:
            w_phys = lambda_physics * min(
                1.0, (epoch - stage1_epochs) / float(physics_ramp_epochs))

        total = loss_data + w_phys * loss_phys
        total.backward()
        opt.step()

        pbar.set_postfix({"L": f"{total.item():.3e}", "Ld": f"{loss_data.item():.2e}",
                          "Lp": f"{float(loss_phys):.2e}", "w": f"{w_phys:.2f}"})

        if eval_every and (epoch + 1) % eval_every == 0 and epoch + 1 > stage1_epochs:
            s, _ = compute_auc_scores(W.detach().cpu().numpy().reshape(-1, 1),
                                      edge_config, N, max_order, orderwise_support)
            m = macro_auc(s)
            print(f"\n[epoch {epoch+1}] macro-AUC={m:.4f}  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in s.items() if v is not None))

    return W.detach().cpu().numpy().reshape(-1, 1), save_dir


def main():
    ap = argparse.ArgumentParser(description="IntegralPINN (neural surrogate, paper version)")
    ap.add_argument("--scene", required=True,
                    choices=["ecological", "neuronal", "rossler", "social"])
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--n_nodes", type=int, default=None)
    ap.add_argument("--max_order", type=int, default=None)
    ap.add_argument("--n_epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--stage1_epochs", type=int, default=2500)
    ap.add_argument("--physics_ramp_epochs", type=int, default=2000,
                    help="0 disables the staged lambda ramp (ablation).")
    ap.add_argument("--lambda_physics", type=float, default=1.0)
    ap.add_argument("--n_intervals", type=int, default=32)
    ap.add_argument("--n_quad", type=int, default=5)
    ap.add_argument("--window_frac", type=float, default=1.0)
    ap.add_argument("--loss_mode", type=str, default="integral",
                    choices=["integral", "derivative"])
    ap.add_argument("--bounded", type=int, default=1)
    ap.add_argument("--clip_c", type=float, default=2.0)
    ap.add_argument("--w_init", type=float, default=-3.0,
                    help="Weak-coupling init for the pre-activation; sigmoid(w_init) is the initial interaction strength of every candidate.")
    ap.add_argument("--chunk_size", type=int, default=20000)
    ap.add_argument("--nn_hidden_dim", type=int, default=64)
    ap.add_argument("--nn_layers", type=int, default=8)
    ap.add_argument("--max_candidates_per_order", type=int, default=0)
    ap.add_argument("--results_root", type=str, default="results/integral_nn")
    ap.add_argument("--run_tag", type=str, default=None)
    ap.add_argument("--eval_every", type=int, default=500)
    args = ap.parse_args()

    set_global_seed(args.seed)
    HypergraphModel, spec = get_scene_model(args.scene)
    defaults = HypergraphModel.get_default_params()
    if args.scene in {"ecological", "neuronal", "social"}:
        cfg = HypergraphModel.get_hyperedge_config(defaults["n_nodes"], defaults["max_order"])
        N = infer_n_nodes_from_edge_config(cfg) or defaults["n_nodes"]
        max_order = infer_max_order_from_edge_config(cfg) or defaults["max_order"]
    else:
        N = args.n_nodes or defaults["n_nodes"]
        max_order = args.max_order or defaults["max_order"]
    if args.max_order is not None:
        max_order = args.max_order

    edge_config = HypergraphModel.get_hyperedge_config(N, max_order)
    if args.max_candidates_per_order > 0:
        support = build_orderwise_support_sampled(
            N, max_order, args.max_candidates_per_order, edge_config, seed=args.seed)
    else:
        support = build_orderwise_support(N, max_order)

    W, save_dir = train(
        HypergraphModel, spec.label, N, max_order, edge_config, support,
        n_samples=args.n_samples, noise=args.noise, n_epochs=args.n_epochs,
        lr=args.lr, seed=args.seed, gpu_id=args.gpu_id,
        stage1_epochs=args.stage1_epochs,
        physics_ramp_epochs=args.physics_ramp_epochs,
        lambda_physics=args.lambda_physics, n_intervals=args.n_intervals,
        n_quad=args.n_quad, window_frac=args.window_frac,
        loss_mode=args.loss_mode, bounded=bool(args.bounded), clip_c=args.clip_c,
        chunk_size=args.chunk_size, w_init=args.w_init, nn_hidden_dim=args.nn_hidden_dim,
        nn_layers=args.nn_layers, results_root=args.results_root,
        run_tag=args.run_tag, eval_every=args.eval_every,
    )

    auc_scores, roc_data = compute_auc_scores(W, edge_config, N, max_order, support)
    print("\nFinal AUC:")
    for k, v in auc_scores.items():
        print(f"  {k}: {'N/A' if v is None else f'{v:.4f}'}")
    plot_roc_curves(roc_data, auc_scores, save_dir, max_order)
    save_roc_curve_data(W, edge_config, N, max_order, save_dir, support)
    np.save(os.path.join(save_dir, "W_learned.npy"), W)

    write_standard_summary(
        save_dir=save_dir, method=f"integral_pinn_nn[{args.loss_mode}]",
        scene=spec.label,
        config={"n_nodes": N, "max_order": max_order, "n_samples": args.n_samples,
                "noise": args.noise, "n_epochs": args.n_epochs, "lr": args.lr,
                "seed": args.seed, "loss_mode": args.loss_mode,
                "bounded": bool(args.bounded), "clip_c": args.clip_c, "w_init": args.w_init,
                "stage1_epochs": args.stage1_epochs,
                "physics_ramp_epochs": args.physics_ramp_epochs,
                "lambda_physics": args.lambda_physics,
                "n_intervals": args.n_intervals, "n_quad": args.n_quad,
                "window_frac": args.window_frac,
                "max_candidates_per_order": args.max_candidates_per_order},
        auc_scores=auc_scores,
        extra_metrics={"macro_auc": macro_auc(auc_scores), "library_size": int(W.shape[0])},
    )
    print(f"\nResults saved to {save_dir}/")


if __name__ == "__main__":
    main()
