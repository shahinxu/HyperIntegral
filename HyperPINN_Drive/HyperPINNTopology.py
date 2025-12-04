import torch
from torch import nn
from torch import optim as optim
import numpy as np
from itertools import combinations

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

class HyperPINNTopology(nn.Module):
    def __init__(self, N, output_dim, hidden_dim=64, num_layers=4, use_resnet=True, use_attention=False):
        super().__init__() 
        self.N = N
        self.use_resnet = use_resnet
        self.use_attention = use_attention
        input_dim = 1

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
            self.input_layer = nn.Linear(input_dim, hidden_dim * 2)
            self.res_blocks = nn.ModuleList()
            for _ in range(num_layers - 2):
                self.res_blocks.append(ResidualBlock(hidden_dim * 2))
            self.output_layer = nn.Linear(hidden_dim * 2, output_dim)
        else:
            raise ValueError("Specify one of: use_resnet=True or use_attention=True")
        
        num_edges = N * (N-1)//2
        num_triangles = N * (N-1) * (N-2) // 6
        num_quads = N * (N-1) * (N-2) * (N-3) // 24
        num_quints = N * (N-1) * (N-2) * (N-3) * (N-4) // 120
        num_sexts = N * (N-1) * (N-2) * (N-3) * (N-4) * (N-5) // 720
        num_septs = N * (N-1) * (N-2) * (N-3) * (N-4) * (N-5) * (N-6) // 5040
        
        self.edge_weights = nn.Parameter(torch.randn(num_edges) * 0.1)  
        self.triangle_weights = nn.Parameter(torch.randn(num_triangles) * 0.1)
        self.quad_weights = nn.Parameter(torch.randn(num_quads) * 0.1)
        self.quint_weights = nn.Parameter(torch.randn(num_quints) * 0.1)
        self.sext_weights = nn.Parameter(torch.randn(num_sexts) * 0.1)
        self.sept_weights = nn.Parameter(torch.randn(num_septs) * 0.1)
        
        self.lambda_l1_edges = 0.01      
        self.lambda_l1_triangles = 0.01 
        self.lambda_l1_quads = 0.01
        self.lambda_l1_quints = 0.01
        self.lambda_l1_sexts = 0.01
        self.lambda_l1_septs = 0.01
        self.lambda_l0_edges = 0.001     
        self.lambda_l0_triangles = 0.001 
        self.lambda_l0_quads = 0.001
        self.lambda_l0_quints = 0.001
        self.lambda_l0_sexts = 0.001
        self.lambda_l0_septs = 0.001
        self.temperature = 1.0           
        self.edge_indices = list(combinations(range(N), 2))
        self.triangle_indices = list(combinations(range(N), 3))
        self.quad_indices = list(combinations(range(N), 4))
        self.quint_indices = list(combinations(range(N), 5))
        self.sext_indices = list(combinations(range(N), 6))
        self.sept_indices = list(combinations(range(N), 7))
    
    def initialize_from_ground_truth(self, true_2edges, true_3edges, true_4edges, true_5edges, 
    true_6edges, true_7edges, remove_edges=None, init_strength=2.0):
        # Process ground truth edges (no grad needed for this part)
        true_2edges_0idx = set(tuple(sorted([x-1 for x in edge])) for edge in true_2edges)
        true_3edges_0idx = set(tuple(sorted([x-1 for x in edge])) for edge in true_3edges)
        true_4edges_0idx = set(tuple(sorted([x-1 for x in edge])) for edge in true_4edges)
        true_5edges_0idx = set(tuple(sorted([x-1 for x in edge])) for edge in true_5edges)
        true_6edges_0idx = set(tuple(sorted([x-1 for x in edge])) for edge in true_6edges)
        true_7edges_0idx = set(tuple(sorted([x-1 for x in edge])) for edge in true_7edges)
        
        if remove_edges:
            import random
            if isinstance(remove_edges, dict):
                if 2 in remove_edges and true_2edges_0idx:
                    true_2edges_0idx = set(true_2edges_0idx)
                    num_to_remove = min(remove_edges[2], len(true_2edges_0idx))
                    edges_to_remove = random.sample(list(true_2edges_0idx), num_to_remove)
                    true_2edges_0idx -= set(edges_to_remove)
                if 3 in remove_edges and true_3edges_0idx:
                    true_3edges_0idx = set(true_3edges_0idx)
                    num_to_remove = min(remove_edges[3], len(true_3edges_0idx))
                    edges_to_remove = random.sample(list(true_3edges_0idx), num_to_remove)
                    true_3edges_0idx -= set(edges_to_remove)
                if 4 in remove_edges and true_4edges_0idx:
                    true_4edges_0idx = set(true_4edges_0idx)
                    num_to_remove = min(remove_edges[4], len(true_4edges_0idx))
                    edges_to_remove = random.sample(list(true_4edges_0idx), num_to_remove)
                    true_4edges_0idx -= set(edges_to_remove)
                if 5 in remove_edges and true_5edges_0idx:
                    true_5edges_0idx = set(true_5edges_0idx)
                    num_to_remove = min(remove_edges[5], len(true_5edges_0idx))
                    edges_to_remove = random.sample(list(true_5edges_0idx), num_to_remove)
                    true_5edges_0idx -= set(edges_to_remove)
                if 6 in remove_edges and true_6edges_0idx:
                    true_6edges_0idx = set(true_6edges_0idx)
                    num_to_remove = min(remove_edges[6], len(true_6edges_0idx))
                    edges_to_remove = random.sample(list(true_6edges_0idx), num_to_remove)
                    true_6edges_0idx -= set(edges_to_remove)
                if 7 in remove_edges and true_7edges_0idx:
                    true_7edges_0idx = set(true_7edges_0idx)
                    num_to_remove = min(remove_edges[7], len(true_7edges_0idx))
                    edges_to_remove = random.sample(list(true_7edges_0idx), num_to_remove)
                    true_7edges_0idx -= set(edges_to_remove)
        
        # Initialize parameters WITHOUT no_grad() to preserve autograd graph
        with torch.no_grad():
            for idx, edge in enumerate(self.edge_indices):
                if tuple(edge) in true_2edges_0idx:
                    self.edge_weights.data[idx] = init_strength
                else:
                    self.edge_weights.data[idx] = -init_strength
            
            for idx, triangle in enumerate(self.triangle_indices):
                if tuple(triangle) in true_3edges_0idx:
                    self.triangle_weights.data[idx] = init_strength
                else:
                    self.triangle_weights.data[idx] = -init_strength
            
            for idx, quad in enumerate(self.quad_indices):
                if tuple(quad) in true_4edges_0idx:
                    self.quad_weights.data[idx] = init_strength
                else:
                    self.quad_weights.data[idx] = -init_strength
            
            for idx, quint in enumerate(self.quint_indices):
                if tuple(quint) in true_5edges_0idx:
                    self.quint_weights.data[idx] = init_strength
                else:
                    self.quint_weights.data[idx] = -init_strength
            
            for idx, sext in enumerate(self.sext_indices):
                if tuple(sext) in true_6edges_0idx:
                    self.sext_weights.data[idx] = init_strength
                else:
                    self.sext_weights.data[idx] = -init_strength
            
            for idx, sept in enumerate(self.sept_indices):
                if tuple(sept) in true_7edges_0idx:
                    self.sept_weights.data[idx] = init_strength
                else:
                    self.sept_weights.data[idx] = -init_strength

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
 
    def concrete_binary_gates(self, logits, temperature=1.0, hard=False):
        uniform = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20) * 0.1
        y_soft = torch.sigmoid((logits + gumbel) / temperature)
        
        if hard:
            y_hard = (y_soft > 0.5).float()
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft     
        return y
    
    def get_sparse_weights(self, use_concrete=True, hard=False):
        if use_concrete:
            edge_probs = self.concrete_binary_gates(self.edge_weights, self.temperature, hard)
            triangle_probs = self.concrete_binary_gates(self.triangle_weights, self.temperature, hard)
            quad_probs = self.concrete_binary_gates(self.quad_weights, self.temperature, hard)
            quint_probs = self.concrete_binary_gates(self.quint_weights, self.temperature, hard)
            sext_probs = self.concrete_binary_gates(self.sext_weights, self.temperature, hard)
            sept_probs = self.concrete_binary_gates(self.sept_weights, self.temperature, hard)
        else:
            edge_probs = torch.sigmoid(self.edge_weights)
            triangle_probs = torch.sigmoid(self.triangle_weights)
            quad_probs = torch.sigmoid(self.quad_weights)
            quint_probs = torch.sigmoid(self.quint_weights)
            sext_probs = torch.sigmoid(self.sext_weights)
            sept_probs = torch.sigmoid(self.sept_weights)
            if hard:
                edge_probs = (edge_probs > 0.5).float() - edge_probs.detach() + edge_probs
                triangle_probs = (triangle_probs > 0.5).float() - triangle_probs.detach() + triangle_probs
                quad_probs = (quad_probs > 0.5).float() - quad_probs.detach() + quad_probs
                quint_probs = (quint_probs > 0.5).float() - quint_probs.detach() + quint_probs
                sext_probs = (sext_probs > 0.5).float() - sext_probs.detach() + sext_probs
                sept_probs = (sept_probs > 0.5).float() - sept_probs.detach() + sept_probs

        return edge_probs, triangle_probs, quad_probs, quint_probs, sext_probs, sept_probs
    
    def sparsity_regularization(self):
        edge_probs = torch.sigmoid(self.edge_weights)
        triangle_probs = torch.sigmoid(self.triangle_weights)
        quad_probs = torch.sigmoid(self.quad_weights)
        quint_probs = torch.sigmoid(self.quint_weights)
        sext_probs = torch.sigmoid(self.sext_weights)
        sept_probs = torch.sigmoid(self.sept_weights)
        l1_edges = torch.sum(edge_probs)
        l1_triangles = torch.sum(triangle_probs)
        l1_quads = torch.sum(quad_probs)
        l1_quints = torch.sum(quint_probs)
        l1_sexts = torch.sum(sext_probs)
        l1_septs = torch.sum(sept_probs)

        l0_edges = torch.sum(edge_probs * (1 - edge_probs) * 4)
        l0_triangles = torch.sum(triangle_probs * (1 - triangle_probs) * 4)
        l0_quads = torch.sum(quad_probs * (1 - quad_probs) * 4)
        l0_quints = torch.sum(quint_probs * (1 - quint_probs) * 4)
        l0_sexts = torch.sum(sext_probs * (1 - sext_probs) * 4)
        l0_septs = torch.sum(sept_probs * (1 - sept_probs) * 4)

        sparsity_loss = (
            self.lambda_l1_edges * l1_edges + self.lambda_l1_triangles * l1_triangles +
            self.lambda_l1_quads * l1_quads + self.lambda_l1_quints * l1_quints +
            self.lambda_l1_sexts * l1_sexts + self.lambda_l1_septs * l1_septs +
            self.lambda_l0_edges * l0_edges + self.lambda_l0_triangles * l0_triangles +
            self.lambda_l0_quads * l0_quads + self.lambda_l0_quints * l0_quints +
            self.lambda_l0_sexts * l0_sexts + self.lambda_l0_septs * l0_septs
        )

        return sparsity_loss, {
            'l1_edges': l1_edges.item(), 'l1_triangles': l1_triangles.item(),
            'l1_quads': l1_quads.item(), 'l1_quints': l1_quints.item(),
            'l1_sexts': l1_sexts.item(), 'l1_septs': l1_septs.item(),
            'l0_edges': l0_edges.item(), 'l0_triangles': l0_triangles.item(),
            'l0_quads': l0_quads.item(), 'l0_quints': l0_quints.item(),
            'l0_sexts': l0_sexts.item(), 'l0_septs': l0_septs.item()
        }
     
    def physics_loss(self,t):
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

        B = x_pred.shape[0]
        device = x_pred.device

        coup_rete = torch.zeros((B, N), device=device, dtype=x_pred.dtype)
        coup_simplicial = torch.zeros((B, N), device=device, dtype=x_pred.dtype)
        coup_quads = torch.zeros((B, N), device=device, dtype=x_pred.dtype)
        coup_quints = torch.zeros((B, N), device=device, dtype=x_pred.dtype)
        coup_sexts = torch.zeros((B, N), device=device, dtype=x_pred.dtype)
        coup_septs = torch.zeros((B, N), device=device, dtype=x_pred.dtype)
        edge_probs, triangle_probs, quad_probs, quint_probs, sext_probs, sept_probs = \
            self.get_sparse_weights(use_concrete=False, hard=True)

        for idx, (i, j) in enumerate(self.edge_indices):
            w = edge_probs[idx]
            coup_rete[:, i] += w * (x_old[:, j] - x_old[:, i])
            coup_rete[:, j] += w * (x_old[:, i] - x_old[:, j])

        # triangle (3-body) coupling
        for idx, (i, j, k_idx) in enumerate(self.triangle_indices):
            w = triangle_probs[idx]
            coup_simplicial[:, i] += w * (x_old[:, j]**2 * x_old[:, k_idx] - x_old[:, i]**3 + x_old[:, j] * x_old[:, k_idx]**2 - x_old[:, i]**3)
            coup_simplicial[:, j] += w * (x_old[:, i]**2 * x_old[:, k_idx] - x_old[:, j]**3 + x_old[:, i] * x_old[:, k_idx]**2 - x_old[:, j]**3)
            coup_simplicial[:, k_idx] += w * (x_old[:, i]**2 * x_old[:, j] - x_old[:, k_idx]**3 + x_old[:, i] * x_old[:, j]**2 - x_old[:, k_idx]**3)

        # quad (4-body) coupling
        for idx, (i, j, k_idx, l) in enumerate(self.quad_indices):
            w = quad_probs[idx]
            coup_quads[:, i] += w * (x_old[:, j]**2 * x_old[:, k_idx] * x_old[:, l] - x_old[:, i]**3)
            coup_quads[:, j] += w * (x_old[:, i]**2 * x_old[:, k_idx] * x_old[:, l] - x_old[:, j]**3)
            coup_quads[:, k_idx] += w * (x_old[:, i]**2 * x_old[:, j] * x_old[:, l] - x_old[:, k_idx]**3)
            coup_quads[:, l] += w * (x_old[:, i]**2 * x_old[:, j] * x_old[:, k_idx] - x_old[:, l]**3)

        # quint (5-body) coupling
        for idx, (i, j, k_idx, l, m) in enumerate(self.quint_indices):
            w = quint_probs[idx]
            coup_quints[:, i] += w * (y_old[:, j]**2 * y_old[:, k_idx] * y_old[:, l] * y_old[:, m] - y_old[:, i]**3)
            coup_quints[:, j] += w * (y_old[:, i]**2 * y_old[:, k_idx] * y_old[:, l] * y_old[:, m] - y_old[:, j]**3)
            coup_quints[:, k_idx] += w * (y_old[:, i]**2 * y_old[:, j] * y_old[:, l] * y_old[:, m] - y_old[:, k_idx]**3)
            coup_quints[:, l] += w * (y_old[:, i]**2 * y_old[:, j] * y_old[:, k_idx] * y_old[:, m] - y_old[:, l]**3)
            coup_quints[:, m] += w * (y_old[:, i]**2 * y_old[:, j] * y_old[:, k_idx] * y_old[:, l] - y_old[:, m]**3)
        # sext (6-body) coupling
        for idx, (i, j, k_idx, l, m, n) in enumerate(self.sext_indices):
            w = sext_probs[idx]
            coup_sexts[:, i] += w * (y_old[:, j]**2 * y_old[:, k_idx] * y_old[:, l] * y_old[:, m] * y_old[:, n] - y_old[:, i]**3)
            coup_sexts[:, j] += w * (y_old[:, i]**2 * y_old[:, k_idx] * y_old[:, l] * y_old[:, m] * y_old[:, n] - y_old[:, j]**3)
            coup_sexts[:, k_idx] += w * (y_old[:, i]**2 * y_old[:, j] * y_old[:, l] * y_old[:, m] * y_old[:, n] - y_old[:, k_idx]**3)
            coup_sexts[:, l] += w * (y_old[:, i]**2 * y_old[:, j] * y_old[:, k_idx] * y_old[:, m] * y_old[:, n] - y_old[:, l]**3)
            coup_sexts[:, m] += w * (y_old[:, i]**2 * y_old[:, j] * y_old[:, k_idx] * y_old[:, l] * y_old[:, n] - y_old[:, m]**3)
            coup_sexts[:, n] += w * (y_old[:, i]**2 * y_old[:, j] * y_old[:, k_idx] * y_old[:, l] * y_old[:, m] - y_old[:, n]**3)

        # sept (7-body) coupling
        for idx, (i, j, k_idx, l, m, n, o) in enumerate(self.sept_indices):
            w = sept_probs[idx]
            coup_septs[:, i] += w * (z_old[:, j]**2 * z_old[:, k_idx] * z_old[:, l] * z_old[:, m] * z_old[:, n] * z_old[:, o] - z_old[:, i]**3)
            coup_septs[:, j] += w * (z_old[:, i]**2 * z_old[:, k_idx] * z_old[:, l] * z_old[:, m] * z_old[:, n] * z_old[:, o] - z_old[:, j]**3)
            coup_septs[:, k_idx] += w * (z_old[:, i]**2 * z_old[:, j] * z_old[:, l] * z_old[:, m] * z_old[:, n] * z_old[:, o] - z_old[:, k_idx]**3)
            coup_septs[:, l] += w * (z_old[:, i]**2 * z_old[:, j] * z_old[:, k_idx] * z_old[:, m] * z_old[:, n] * z_old[:, o] - z_old[:, l]**3)
            coup_septs[:, m] += w * (z_old[:, i]**2 * z_old[:, j] * z_old[:, k_idx] * z_old[:, l] * z_old[:, n] * z_old[:, o] - z_old[:, m]**3)
            coup_septs[:, n] += w * (z_old[:, i]**2 * z_old[:, j] * z_old[:, k_idx] * z_old[:, l] * z_old[:, m] * z_old[:, o] - z_old[:, n]**3)
            coup_septs[:, o] += w * (z_old[:, i]**2 * z_old[:, j] * z_old[:, k_idx] * z_old[:, l] * z_old[:, m] * z_old[:, n] - z_old[:, o]**3)

        dxdt_expected = -y_old - z_old + k * coup_rete + kD * coup_simplicial + kD * coup_quads
        dydt_expected = x_old + ar * y_old + kD * coup_quints + kD * coup_sexts
        dzdt_expected = br + z_old * (x_old - cr) + kD * coup_septs

        expected = torch.cat([dxdt_expected, dydt_expected, dzdt_expected], dim=1)

        loss = torch.mean((dx_dt_pred - expected) ** 2)
        return loss
