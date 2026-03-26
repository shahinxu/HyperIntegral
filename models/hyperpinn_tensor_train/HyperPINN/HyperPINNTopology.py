import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return x + self.net(x)


class PirateBlock(nn.Module):
    def __init__(self, hidden_dim, activation="tanh", nonlinearity_init=0.0):
        super().__init__()
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation for PirateBlock: {activation}")

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(nonlinearity_init)))

    def forward(self, x, u, v):
        identity = x

        h = self.activation(self.fc1(x))
        h = h * u + (1.0 - h) * v

        h = self.activation(self.fc2(h))
        h = h * u + (1.0 - h) * v

        h = self.activation(self.fc3(h))

        return self.alpha * h + (1.0 - self.alpha) * identity


class HyperPINNTopology(nn.Module):
    def __init__(
        self,
        N,
        output_dim,
        hidden_dim=64,
        num_layers=8,
        use_resnet=True,
        use_attention=False,
        use_pirate=False,
        max_order=3,
        tt_rank=8,
    ):
        super().__init__()
        self.N = N
        self.use_resnet = use_resnet
        self.use_attention = use_attention
        self.use_pirate = use_pirate
        self.max_order = max_order
        self.tt_rank = tt_rank
        input_dim = 1

        if max_order not in (2, 3):
            raise ValueError(f"hyperpinn_tensor_train only supports max_order in {{2, 3}}, got {max_order}.")
        if tt_rank <= 0:
            raise ValueError(f"tt_rank must be positive, got {tt_rank}.")

        if use_attention:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        elif use_resnet:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.res_blocks = nn.ModuleList()
            for _ in range(num_layers - 2):
                self.res_blocks.append(ResidualBlock(hidden_dim))
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        elif use_pirate:
            if use_resnet or use_attention:
                raise ValueError("Only one of use_resnet, use_attention, use_pirate can be True")

            self.pirate_input = nn.Linear(input_dim, hidden_dim)
            self.pirate_u = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.pirate_v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.pirate_blocks = nn.ModuleList(
                [PirateBlock(hidden_dim=hidden_dim, activation="tanh", nonlinearity_init=0.0) for _ in range(num_layers)]
            )
            self.pirate_output = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError("Specify one of: use_resnet=True or use_attention=True")

        # Weak-coupling initialization keeps the early training phase focused on fitting trajectories.
        self.tt_raw_pair_left = nn.Parameter(-6.0 + 0.02 * torch.randn(N, tt_rank))
        self.tt_raw_pair_right = nn.Parameter(-6.0 + 0.02 * torch.randn(N, tt_rank))
        if max_order >= 3:
            self.tt_raw_core1 = nn.Parameter(-6.0 + 0.02 * torch.randn(N, tt_rank))
            self.tt_raw_core2 = nn.Parameter(-6.0 + 0.02 * torch.randn(N, tt_rank, tt_rank))
            self.tt_raw_core3 = nn.Parameter(-6.0 + 0.02 * torch.randn(N, tt_rank))
        else:
            self.tt_raw_core1 = None
            self.tt_raw_core2 = None
            self.tt_raw_core3 = None

        self.lambda_l1_edges = 0.01
        self.lambda_l1_triangles = 0.01
        self.factor_l2 = 1e-6

    def _pair_factors(self):
        return F.softplus(self.tt_raw_pair_left), F.softplus(self.tt_raw_pair_right)

    def _triple_factors(self):
        if self.max_order < 3:
            raise ValueError("Third-order TT factors requested with max_order < 3")
        return (
            F.softplus(self.tt_raw_core1),
            F.softplus(self.tt_raw_core2),
            F.softplus(self.tt_raw_core3),
        )

    def forward(self, t):
        t = t.float()
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if self.use_attention:
            h = self.input_proj(t).unsqueeze(1)
            attn_out, _ = self.attention(h, h, h)
            h = self.norm1(h + attn_out)
            ff_out = self.ff(h)
            h = self.norm2(h + ff_out)
            return self.output_proj(h.squeeze(1))
        if self.use_resnet:
            h = torch.tanh(self.input_layer(t))
            for block in self.res_blocks:
                h = block(h)
            return self.output_layer(h)
        if self.use_pirate:
            h = torch.tanh(self.pirate_input(t))
            u = self.pirate_u(h)
            v = self.pirate_v(h)
            for block in self.pirate_blocks:
                h = block(h, u, v)
            return self.pirate_output(h)
        raise RuntimeError("No network backbone configured")

    def _tt_pair_score(self, left_idx, right_idx):
        left, right = self._pair_factors()
        return torch.einsum("er,er->e", left[left_idx], right[right_idx])

    def _tt_triple_base_score(self, idx1, idx2, idx3):
        core1, core2, core3 = self._triple_factors()
        first = core1[idx1]
        middle = core2[idx2]
        third = core3[idx3]
        return torch.einsum("er,err,er->e", first, middle, third)

    def score_order_edges(self, order_key, edges, one_based=True):
        order = 2 if order_key == "edges" else 3 if order_key == "triangles" else None
        if order is None:
            raise ValueError(f"Unsupported order key: {order_key}")
        if order > self.max_order:
            return torch.empty(0, device=next(self.parameters()).device)

        edges_tensor = torch.as_tensor(edges, dtype=torch.long, device=next(self.parameters()).device)
        if edges_tensor.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=edges_tensor.device)
        if one_based:
            edges_tensor = edges_tensor - 1
        if edges_tensor.ndim != 2 or edges_tensor.shape[1] != order:
            raise ValueError(f"Expected shape [E, {order}] for {order_key}, got {tuple(edges_tensor.shape)}")

        if order == 2:
            i_idx = edges_tensor[:, 0]
            j_idx = edges_tensor[:, 1]
            return 0.5 * (self._tt_pair_score(i_idx, j_idx) + self._tt_pair_score(j_idx, i_idx))

        i_idx = edges_tensor[:, 0]
        j_idx = edges_tensor[:, 1]
        k_idx = edges_tensor[:, 2]
        scores = (
            self._tt_triple_base_score(i_idx, j_idx, k_idx)
            + self._tt_triple_base_score(i_idx, k_idx, j_idx)
            + self._tt_triple_base_score(j_idx, i_idx, k_idx)
            + self._tt_triple_base_score(j_idx, k_idx, i_idx)
            + self._tt_triple_base_score(k_idx, i_idx, j_idx)
            + self._tt_triple_base_score(k_idx, j_idx, i_idx)
        ) / 6.0
        return scores

    def sparsity_regularization(self):
        left, right = self._pair_factors()
        sum_left = torch.sum(left, dim=0)
        sum_right = torch.sum(right, dim=0)
        diag_lr = torch.sum(left * right, dim=0)
        edge_mass = 0.5 * torch.sum(sum_left * sum_right - diag_lr)

        triangle_mass = torch.tensor(0.0, device=left.device)
        if self.max_order >= 3:
            core1, core2, core3 = self._triple_factors()
            sum_a = torch.sum(core1, dim=0)
            sum_b = torch.sum(core2, dim=0)
            sum_c = torch.sum(core3, dim=0)
            ac_diag = torch.einsum("nr,ns->rs", core1, core3)
            ab_diag = torch.einsum("nr,nrs->rs", core1, core2)
            bc_diag = torch.einsum("nrs,ns->rs", core2, core3)
            abc_diag = torch.einsum("nr,nrs,ns->rs", core1, core2, core3)
            ordered_distinct_mass = torch.sum(
                sum_a[:, None] * sum_b * sum_c[None, :]
                - ab_diag * sum_c[None, :]
                - sum_a[:, None] * bc_diag
                - ac_diag * sum_b
                + 2.0 * abc_diag
            )
            triangle_mass = ordered_distinct_mass / 6.0
            factor_penalty = self.factor_l2 * (
                torch.sum(self.tt_raw_pair_left * self.tt_raw_pair_left)
                + torch.sum(self.tt_raw_pair_right * self.tt_raw_pair_right)
                + torch.sum(self.tt_raw_core1 * self.tt_raw_core1)
                + torch.sum(self.tt_raw_core2 * self.tt_raw_core2)
                + torch.sum(self.tt_raw_core3 * self.tt_raw_core3)
            )
        else:
            factor_penalty = self.factor_l2 * (
                torch.sum(self.tt_raw_pair_left * self.tt_raw_pair_left)
                + torch.sum(self.tt_raw_pair_right * self.tt_raw_pair_right)
            )

        sparsity_loss = self.lambda_l1_edges * edge_mass + self.lambda_l1_triangles * triangle_mass + factor_penalty
        return sparsity_loss, {
            "l1_edges": float(edge_mass.detach().cpu()),
            "l1_triangles": float(triangle_mass.detach().cpu()),
            "l1_tt_factor_penalty": float(factor_penalty.detach().cpu()),
        }

    def _tt_pairwise_coupling(self, x_old):
        left, right = self._pair_factors()
        sum_rx = x_old @ right
        sum_r = torch.sum(right, dim=0)
        sum_lx = x_old @ left
        sum_l = torch.sum(left, dim=0)
        term_lr = left[None, :, :] * (sum_rx[:, None, :] - x_old[:, :, None] * sum_r[None, None, :])
        term_rl = right[None, :, :] * (sum_lx[:, None, :] - x_old[:, :, None] * sum_l[None, None, :])
        return 0.5 * (torch.sum(term_lr, dim=2) + torch.sum(term_rl, dim=2))

    def _tt_triangle_coupling(self, x_old):
        if self.max_order < 3:
            return torch.zeros_like(x_old)

        core1, core2, core3 = self._triple_factors()
        x_sq = x_old * x_old
        x_cu = x_sq * x_old
        batch_size = x_old.shape[0]

        a1 = x_old @ core1
        a2 = x_sq @ core1
        n_a = torch.sum(core1, dim=0)

        b1 = torch.einsum("xn,nrs->xrs", x_old, core2)
        b2 = torch.einsum("xn,nrs->xrs", x_sq, core2)
        n_b = torch.sum(core2, dim=0)

        c1 = x_old @ core3
        c2 = x_sq @ core3
        n_c = torch.sum(core3, dim=0)

        d_bc = torch.einsum("xn,nrs,ns->xrs", x_cu, core2, core3)
        e_bc = torch.einsum("nrs,ns->rs", core2, core3)
        d_ac = torch.einsum("xn,nr,ns->xrs", x_cu, core1, core3)
        e_ac = torch.einsum("nr,ns->rs", core1, core3)
        d_ab = torch.einsum("xn,nr,nrs->xrs", x_cu, core1, core2)
        e_ab = torch.einsum("nr,nrs->rs", core1, core2)

        core1_i = core1[None, :, :]
        core2_i = core2[None, :, :, :]
        core3_i = core3[None, :, :]

        x_i = x_old[:, :, None]
        x_sq_i = x_sq[:, :, None]
        x_cu_i = x_cu[:, :, None, None]

        a1_ex = a1[:, None, :] - core1_i * x_i
        a2_ex = a2[:, None, :] - core1_i * x_sq_i
        n_a_ex = n_a[None, None, :] - core1_i

        b1_ex = b1[:, None, :, :] - core2_i * x_old[:, :, None, None]
        b2_ex = b2[:, None, :, :] - core2_i * x_sq[:, :, None, None]
        n_b_ex = n_b[None, None, :, :] - core2_i

        c1_ex = c1[:, None, :] - core3_i * x_i
        c2_ex = c2[:, None, :] - core3_i * x_sq_i
        n_c_ex = n_c[None, None, :] - core3_i

        bc_diag = core2_i * core3_i[:, :, None, :]
        ac_diag = core1_i[:, :, :, None] * core3_i[:, :, None, :]
        ab_diag = core1_i[:, :, :, None] * core2_i

        d_bc_ex = d_bc[:, None, :, :] - bc_diag * x_cu_i
        e_bc_ex = e_bc[None, None, :, :] - bc_diag
        d_ac_ex = d_ac[:, None, :, :] - ac_diag * x_cu_i
        e_ac_ex = e_ac[None, None, :, :] - ac_diag
        d_ab_ex = d_ab[:, None, :, :] - ab_diag * x_cu_i
        e_ab_ex = e_ab[None, None, :, :] - ab_diag

        x_cu_factor = x_cu[:, :, None, None]

        inner_123 = b2_ex * c1_ex[:, :, None, :] - d_bc_ex - x_cu_factor * (n_b_ex * n_c_ex[:, :, None, :] - e_bc_ex)
        inner_132 = b1_ex * c2_ex[:, :, None, :] - d_bc_ex - x_cu_factor * (n_b_ex * n_c_ex[:, :, None, :] - e_bc_ex)
        inner_213 = a2_ex[:, :, :, None] * c1_ex[:, :, None, :] - d_ac_ex - x_cu_factor * (n_a_ex[:, :, :, None] * n_c_ex[:, :, None, :] - e_ac_ex)
        inner_231 = a2_ex[:, :, :, None] * b1_ex - d_ab_ex - x_cu_factor * (n_a_ex[:, :, :, None] * n_b_ex - e_ab_ex)
        inner_312 = a1_ex[:, :, :, None] * c2_ex[:, :, None, :] - d_ac_ex - x_cu_factor * (n_a_ex[:, :, :, None] * n_c_ex[:, :, None, :] - e_ac_ex)
        inner_321 = a1_ex[:, :, :, None] * b2_ex - d_ab_ex - x_cu_factor * (n_a_ex[:, :, :, None] * n_b_ex - e_ab_ex)

        term_123 = torch.einsum("xna,xnab->xn", core1_i.expand(batch_size, -1, -1), inner_123)
        term_132 = torch.einsum("xna,xnab->xn", core1_i.expand(batch_size, -1, -1), inner_132)
        term_213 = torch.sum(core2_i.expand(batch_size, -1, -1, -1) * inner_213, dim=(2, 3))
        term_231 = torch.einsum("xnb,xnab->xn", core3_i.expand(batch_size, -1, -1), inner_231)
        term_312 = torch.sum(core2_i.expand(batch_size, -1, -1, -1) * inner_312, dim=(2, 3))
        term_321 = torch.einsum("xnb,xnab->xn", core3_i.expand(batch_size, -1, -1), inner_321)

        return (term_123 + term_132 + term_213 + term_231 + term_312 + term_321) / 6.0

    def physics_loss(self, t):
        x_pred = self.forward(t)
        dx_dt_pred = torch.zeros_like(x_pred)
        for i in range(x_pred.shape[1]):
            grad_i = torch.autograd.grad(x_pred[:, i].sum(), t, create_graph=True, retain_graph=True)[0]
            dx_dt_pred[:, i] = grad_i.squeeze(-1)

        N = self.N
        x_old = x_pred[:, 0:N]
        y_old = x_pred[:, N:2 * N]
        z_old = x_pred[:, 2 * N:3 * N]

        ar, br, cr = 0.2, 0.2, 0.7
        k, kD = 0.4, 0.3
        coup_rete = self._tt_pairwise_coupling(x_old)
        coup_simplicial = self._tt_triangle_coupling(x_old)

        dxdt_expected = -y_old - z_old + k * coup_rete + kD * coup_simplicial
        dydt_expected = x_old + ar * y_old
        dzdt_expected = br + z_old * (x_old - cr)

        expected = torch.cat([dxdt_expected, dydt_expected, dzdt_expected], dim=1)
        return torch.mean((dx_dt_pred - expected) ** 2)
