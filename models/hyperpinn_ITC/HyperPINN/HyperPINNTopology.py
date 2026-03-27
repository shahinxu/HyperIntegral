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
            nn.LayerNorm(hidden_dim)
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
        itc_rank=16,
    ):
        super().__init__()
        self.N = N
        self.use_resnet = use_resnet
        self.use_attention = use_attention
        self.use_pirate = use_pirate
        self.max_order = max_order
        self.itc_rank = itc_rank
        input_dim = 1

        if max_order not in (2, 3):
            raise ValueError(f"hyperpinn_ITC only supports max_order in {{2, 3}}, got {max_order}.")
        if itc_rank <= 0:
            raise ValueError(f"itc_rank must be positive, got {itc_rank}.")

        if use_attention:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
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

        descriptor_dim = 24
        self.node_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.factor_head2 = nn.Linear(hidden_dim, itc_rank)
        self.raw_lambda2 = nn.Parameter(-6.0 + 0.02 * torch.randn(itc_rank))
        if max_order >= 3:
            self.factor_head3 = nn.Linear(hidden_dim, itc_rank)
            self.raw_lambda3 = nn.Parameter(-6.0 + 0.02 * torch.randn(itc_rank))
        else:
            self.factor_head3 = None
            self.raw_lambda3 = None

        self.lambda_l1_edges = 0.01      
        self.lambda_l1_triangles = 0.01 
        self.factor_l2 = 1e-6

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
        elif self.use_resnet:
            h = torch.tanh(self.input_layer(t))
            for block in self.res_blocks:
                h = block(h)
            return self.output_layer(h)
        elif self.use_pirate:
            h = torch.tanh(self.pirate_input(t))
            u = self.pirate_u(h)
            v = self.pirate_v(h)
            for block in self.pirate_blocks:
                h = block(h, u, v)
            return self.pirate_output(h)

    def _build_node_descriptors(self, observed_trajectory):
        if observed_trajectory.ndim != 2 or observed_trajectory.shape[1] != 3 * self.N:
            raise ValueError(
                f"Expected observed trajectory with shape [T, {3 * self.N}], got {tuple(observed_trajectory.shape)}."
            )

        coords = torch.stack(
            [
                observed_trajectory[:, 0:self.N],
                observed_trajectory[:, self.N:2 * self.N],
                observed_trajectory[:, 2 * self.N:3 * self.N],
            ],
            dim=-1,
        )
        coords = coords.permute(1, 0, 2)
        if coords.shape[1] > 1:
            diffs = coords[:, 1:, :] - coords[:, :-1, :]
        else:
            diffs = torch.zeros(coords.shape[0], 1, coords.shape[2], device=coords.device, dtype=coords.dtype)

        features = [
            coords.mean(dim=1),
            coords.std(dim=1, unbiased=False),
            coords.min(dim=1).values,
            coords.max(dim=1).values,
            coords[:, 0, :],
            coords[:, -1, :],
            diffs.mean(dim=1),
            diffs.std(dim=1, unbiased=False),
        ]
        return torch.cat(features, dim=1)

    def encode_nodes(self, observed_trajectory):
        descriptors = self._build_node_descriptors(observed_trajectory)
        return self.node_encoder(descriptors)

    def _itc_factors(self, order, observed_trajectory):
        node_latents = self.encode_nodes(observed_trajectory)
        if order == 2:
            return F.softplus(self.factor_head2(node_latents)), F.softplus(self.raw_lambda2)
        if order == 3 and self.max_order >= 3:
            return F.softplus(self.factor_head3(node_latents)), F.softplus(self.raw_lambda3)
        raise ValueError(f"Unsupported order for ITC factors: {order}")

    def score_order_edges(self, order_key, edges, observed_trajectory, one_based=True):
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

        u, lam = self._itc_factors(order, observed_trajectory)
        selected = u[edges_tensor]
        rank_products = selected.prod(dim=1)
        return (rank_products * lam.unsqueeze(0)).sum(dim=1)

    def completion_loss(self, observed_trajectory, observed_samples):
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        stats = {
            "completion_edges": 0.0,
            "completion_triangles": 0.0,
        }

        edge_batch = observed_samples.get("edges")
        if edge_batch is not None and edge_batch["indices"].numel() > 0:
            edge_logits = self.score_order_edges("edges", edge_batch["indices"], observed_trajectory, one_based=True)
            edge_labels = edge_batch["labels"].to(edge_logits.device)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
            total_loss = total_loss + edge_loss
            stats["completion_edges"] = float(edge_loss.detach().cpu())

        if self.max_order >= 3:
            tri_batch = observed_samples.get("triangles")
            if tri_batch is not None and tri_batch["indices"].numel() > 0:
                tri_logits = self.score_order_edges("triangles", tri_batch["indices"], observed_trajectory, one_based=True)
                tri_labels = tri_batch["labels"].to(tri_logits.device)
                tri_loss = F.binary_cross_entropy_with_logits(tri_logits, tri_labels)
                total_loss = total_loss + tri_loss
                stats["completion_triangles"] = float(tri_loss.detach().cpu())

        return total_loss, stats

    def sparsity_regularization(self, observed_trajectory):
        u2, lam2 = self._itc_factors(2, observed_trajectory)
        s1_2 = torch.sum(u2, dim=0)
        s2_2 = torch.sum(u2 * u2, dim=0)
        edge_mass = 0.5 * torch.sum(lam2 * (s1_2 * s1_2 - s2_2))

        triangle_mass = torch.tensor(0.0, device=u2.device)
        if self.max_order >= 3:
            u3, lam3 = self._itc_factors(3, observed_trajectory)
            s1_3 = torch.sum(u3, dim=0)
            s2_3 = torch.sum(u3 * u3, dim=0)
            s3_3 = torch.sum(u3 * u3 * u3, dim=0)
            triangle_mass = torch.sum(lam3 * (s1_3 * s1_3 * s1_3 - 3.0 * s1_3 * s2_3 + 2.0 * s3_3) / 6.0)
        else:
            u3 = None

        factor_penalty = self.factor_l2 * (
            sum(torch.sum(param * param) for param in self.node_encoder.parameters())
            + sum(torch.sum(param * param) for param in self.factor_head2.parameters())
            + torch.sum(self.raw_lambda2 * self.raw_lambda2)
        )
        if self.max_order >= 3:
            factor_penalty = factor_penalty + self.factor_l2 * (
                sum(torch.sum(param * param) for param in self.factor_head3.parameters())
                + torch.sum(self.raw_lambda3 * self.raw_lambda3)
            )

        sparsity_loss = self.lambda_l1_edges * edge_mass + self.lambda_l1_triangles * triangle_mass + factor_penalty
        return sparsity_loss, {
            "l1_edges": float(edge_mass.detach().cpu()),
            "l1_triangles": float(triangle_mass.detach().cpu()),
            "l2_factor_penalty": float(factor_penalty.detach().cpu()),
        }

    def _pairwise_coupling(self, x_old, observed_trajectory):
        u2, lam2 = self._itc_factors(2, observed_trajectory)
        sum_ux = x_old @ u2
        sum_u = torch.sum(u2, dim=0)
        term = sum_ux[:, None, :] - x_old[:, :, None] * sum_u[None, None, :]
        return torch.sum(term * u2[None, :, :] * lam2[None, None, :], dim=2)

    def _triangle_coupling(self, x_old, observed_trajectory):
        if self.max_order < 3:
            return torch.zeros_like(x_old)

        u3, lam3 = self._itc_factors(3, observed_trajectory)
        u3_sq = u3 * u3
        x_sq = x_old * x_old
        x_cu = x_sq * x_old

        a = x_sq @ u3
        b = x_old @ u3
        c = torch.sum(u3, dim=0)
        d = x_cu @ u3_sq
        e = torch.sum(u3_sq, dim=0)

        ui = u3[None, :, :]
        ui_sq = u3_sq[None, :, :]
        x_i = x_old[:, :, None]
        x_i_sq = x_sq[:, :, None]
        x_i_cu = x_cu[:, :, None]

        a_tilde = a[:, None, :] - ui * x_i_sq
        b_tilde = b[:, None, :] - ui * x_i
        c_tilde = c[None, None, :] - ui
        d_tilde = d[:, None, :] - ui_sq * x_i_cu
        e_tilde = e[None, None, :] - ui_sq

        inner = a_tilde * b_tilde - d_tilde - x_i_cu * (c_tilde * c_tilde - e_tilde)
        return torch.sum(inner * ui * lam3[None, None, :], dim=2)

    def physics_loss(self, t, observed_trajectory):
        x_pred = self.forward(t)
        dx_dt_pred = torch.zeros_like(x_pred)
        for i in range(x_pred.shape[1]):
            grad_i = torch.autograd.grad(x_pred[:, i].sum(), t, create_graph=True, retain_graph=True)[0]
            dx_dt_pred[:, i] = grad_i.squeeze(-1)

        N = self.N
        x_old = x_pred[:, 0:N]
        y_old = x_pred[:, N:2*N]
        z_old = x_pred[:, 2*N:3*N]

        ar, br, cr = 0.2, 0.2, 0.7
        k, kD = 0.4, 0.3

        coup_rete = self._pairwise_coupling(x_old, observed_trajectory)
        coup_simplicial = self._triangle_coupling(x_old, observed_trajectory)

        dxdt_expected = -y_old - z_old + k * coup_rete + kD * coup_simplicial
        dydt_expected = x_old + ar * y_old
        dzdt_expected = br + z_old * (x_old - cr)

        expected = torch.cat([dxdt_expected, dydt_expected, dzdt_expected], dim=1)

        loss = torch.mean((dx_dt_pred - expected) ** 2)
        return loss
