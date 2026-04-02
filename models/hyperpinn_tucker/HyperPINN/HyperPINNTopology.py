import torch
import torch.nn.functional as F
from torch import nn
try:
    from torch.func import jacrev, vmap
except ImportError:
    from functorch import jacrev, vmap


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
        tucker_rank=16,
    ):
        super().__init__()
        self.N = N
        self.use_resnet = use_resnet
        self.use_attention = use_attention
        self.use_pirate = use_pirate
        self.max_order = max_order
        self.tucker_rank = tucker_rank
        input_dim = 1

        if max_order not in (2, 3):
            raise ValueError(f"hyperpinn_tucker only supports max_order in {{2, 3}}, got {max_order}.")
        if tucker_rank <= 0:
            raise ValueError(f"tucker_rank must be positive, got {tucker_rank}.")

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

        # Weak-coupling initialization keeps early training focused on fitting trajectories.
        self.tucker_raw_u2 = nn.Parameter(-3.0 + 0.02 * torch.randn(N, tucker_rank))
        self.tucker_raw_core2 = nn.Parameter(-3.0 + 0.02 * torch.randn(tucker_rank, tucker_rank))
        if max_order >= 3:
            self.tucker_raw_u3 = nn.Parameter(-3.0 + 0.02 * torch.randn(N, tucker_rank))
            self.tucker_raw_core3 = nn.Parameter(-3.0 + 0.02 * torch.randn(tucker_rank, tucker_rank, tucker_rank))
        else:
            self.tucker_raw_u3 = None
            self.tucker_raw_core3 = None

        self.lambda_l1_edges = 0.01
        self.lambda_l1_triangles = 0.01
        self.factor_l2 = 1e-6

    def _symmetrize_core3(self, core):
        return (
            core
            + core.permute(0, 2, 1)
            + core.permute(1, 0, 2)
            + core.permute(1, 2, 0)
            + core.permute(2, 0, 1)
            + core.permute(2, 1, 0)
        ) / 6.0

    def _tucker_factors(self, order):
        if order == 2:
            u2 = F.softplus(self.tucker_raw_u2)
            core2 = F.softplus(self.tucker_raw_core2)
            core2 = 0.5 * (core2 + core2.transpose(0, 1))
            return u2, core2
        if order == 3 and self.max_order >= 3:
            u3 = F.softplus(self.tucker_raw_u3)
            core3 = F.softplus(self.tucker_raw_core3)
            core3 = self._symmetrize_core3(core3)
            return u3, core3
        raise ValueError(f"Unsupported order for Tucker factors: {order}")

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
            u2, core2 = self._tucker_factors(2)
            lhs = u2[edges_tensor[:, 0]]
            rhs = u2[edges_tensor[:, 1]]
            return torch.einsum("ea,ab,eb->e", lhs, core2, rhs)

        u3, core3 = self._tucker_factors(3)
        first = u3[edges_tensor[:, 0]]
        second = u3[edges_tensor[:, 1]]
        third = u3[edges_tensor[:, 2]]
        return torch.einsum("ea,abc,eb,ec->e", first, core3, second, third)

    def sparsity_regularization(self):
        u2, core2 = self._tucker_factors(2)
        s1_2 = torch.sum(u2, dim=0)
        s2_2 = torch.einsum("ia,ib->ab", u2, u2)
        edge_mass = 0.5 * (torch.einsum("a,b,ab->", s1_2, s1_2, core2) - torch.einsum("ab,ab->", s2_2, core2))

        triangle_mass = torch.tensor(0.0, device=u2.device)
        if self.max_order >= 3:
            u3, core3 = self._tucker_factors(3)
            s1_3 = torch.sum(u3, dim=0)
            s2_3 = torch.einsum("ia,ib->ab", u3, u3)
            s3_3 = torch.einsum("ia,ib,ic->abc", u3, u3, u3)
            triangle_mass = (
                torch.einsum("a,b,c,abc->", s1_3, s1_3, s1_3, core3)
                - torch.einsum("ab,c,abc->", s2_3, s1_3, core3)
                - torch.einsum("ac,b,abc->", s2_3, s1_3, core3)
                - torch.einsum("bc,a,abc->", s2_3, s1_3, core3)
                + 2.0 * torch.einsum("abc,abc->", s3_3, core3)
            ) / 6.0
            factor_penalty = self.factor_l2 * (
                torch.sum(self.tucker_raw_u2 * self.tucker_raw_u2)
                + torch.sum(self.tucker_raw_core2 * self.tucker_raw_core2)
                + torch.sum(self.tucker_raw_u3 * self.tucker_raw_u3)
                + torch.sum(self.tucker_raw_core3 * self.tucker_raw_core3)
            )
        else:
            factor_penalty = self.factor_l2 * (
                torch.sum(self.tucker_raw_u2 * self.tucker_raw_u2)
                + torch.sum(self.tucker_raw_core2 * self.tucker_raw_core2)
            )

        sparsity_loss = self.lambda_l1_edges * edge_mass + self.lambda_l1_triangles * triangle_mass + factor_penalty

        return sparsity_loss, {
            "l1_edges": float(edge_mass.detach().cpu()),
            "l1_triangles": float(triangle_mass.detach().cpu()),
            "l1_tucker_factor_penalty": float(factor_penalty.detach().cpu()),
        }

    def _tucker_pairwise_coupling(self, x_old):
        u2, core2 = self._tucker_factors(2)
        s_x = x_old @ u2
        s_1 = torch.sum(u2, dim=0)
        pair_kernel = u2 @ core2
        inner = s_x[:, None, :] - x_old[:, :, None] * s_1[None, None, :]
        return torch.sum(pair_kernel[None, :, :] * inner, dim=2)

    def _tucker_triangle_coupling(self, x_old):
        if self.max_order < 3:
            return torch.zeros_like(x_old)

        u3, core3 = self._tucker_factors(3)
        x_sq = x_old * x_old
        x_cu = x_sq * x_old

        a = x_sq @ u3
        b = x_old @ u3
        p = torch.sum(u3, dim=0)
        d = torch.einsum("xn,na,nb->xab", x_cu, u3, u3)
        q = torch.einsum("na,nb->ab", u3, u3)

        ui = u3[None, :, :]
        x_i = x_old[:, :, None]
        x_i_sq = x_sq[:, :, None]
        x_i_cu = x_cu[:, :, None]

        a_tilde = a[:, None, :] - ui * x_i_sq
        b_tilde = b[:, None, :] - ui * x_i
        p_tilde = p[None, None, :] - ui
        ui_outer = torch.einsum("bia,bic->biac", ui, ui)
        d_tilde = d[:, None, :, :] - ui_outer * x_cu[:, :, None, None]
        q_tilde = q[None, None, :, :] - ui_outer

        product_term = torch.einsum("bia,bic->biac", a_tilde, b_tilde)
        count_term = torch.einsum("bia,bic->biac", p_tilde, p_tilde)
        inner = product_term - d_tilde - x_cu[:, :, None, None] * (count_term - q_tilde)
        contracted = torch.einsum("abc,xibc->xia", core3, inner)
        return torch.einsum("ia,xia->xi", u3, contracted)

    def _time_derivative(self, t):
        t_flat = t.reshape(-1)

        def single_forward(t_scalar):
            return self.forward(t_scalar.reshape(1, 1)).squeeze(0)

        return vmap(jacrev(single_forward))(t_flat)

    def physics_loss(self, t):
        x_pred = self.forward(t)
        dx_dt_pred = self._time_derivative(t)

        N = self.N
        x_old = x_pred[:, 0:N]
        y_old = x_pred[:, N:2 * N]
        z_old = x_pred[:, 2 * N:3 * N]

        ar, br, cr = 0.2, 0.2, 0.7
        k, kD = 0.4, 0.3

        coup_rete = self._tucker_pairwise_coupling(x_old)
        coup_simplicial = self._tucker_triangle_coupling(x_old)

        dxdt_expected = -y_old - z_old + k * coup_rete + kD * coup_simplicial
        dydt_expected = x_old + ar * y_old
        dzdt_expected = br + z_old * (x_old - cr)

        expected = torch.cat([dxdt_expected, dydt_expected, dzdt_expected], dim=1)
        return torch.mean((dx_dt_pred - expected) ** 2)
