"""
Mamba Temporal Behavioral Model and Baseline for Zero Rabies-MMNet.
Pure PyTorch implementation of Mamba Selective State Space (S6) block.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaS6Block(nn.Module):
    """
    Pure PyTorch implementation of the Mamba S6 Selective State Space Model
    (Gu & Dao, 2023: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces").
    """
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection to inner dimension * 2 (branch u and branch z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Depthwise Causal Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
            padding=d_conv - 1
        )

        # Selective projection for SSM parameters: B (d_state), C (d_state), delta (d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Initialize dt projection bias to preserve scale
        dt_init_std = 2 ** 0.5 * (self.d_inner ** -0.5)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias to span [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Initialize A using HiPPO (log scale)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D skip parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # 1D causal convolution over L
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM parameters
        x_dbl = self.x_proj(x_conv)
        B_ssm, C_ssm, delta = torch.split(x_dbl, [self.d_state, self.d_state, self.d_inner], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) # (B, L, d_inner)

        A = -torch.exp(self.A_log) # (d_inner, d_state)

        # Selective scan recurrence (L=16 is fast in pure PyTorch loop)
        y = torch.zeros_like(x_conv)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(L):
            d_t = delta[:, t].unsqueeze(-1)  # (B, d_inner, 1)
            b_t = B_ssm[:, t].unsqueeze(1)   # (B, 1, d_state)
            c_t = C_ssm[:, t].unsqueeze(1)   # (B, 1, d_state)
            u_t = x_conv[:, t].unsqueeze(-1) # (B, d_inner, 1)

            dA = torch.exp(d_t * A)          # (B, d_inner, d_state)
            dB = d_t * b_t                   # (B, d_inner, d_state)

            h = dA * h + dB * u_t
            y[:, t] = (h * c_t).sum(-1) + self.D * x_conv[:, t]

        # Multiplicative gating with branch z
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaS6Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaBehaviorModel(nn.Module):
    """
    Temporal Mamba model for Canine Behavioral Screening (Stage 4).
    Input: (B, T=16, 72)
    Output: (B, 1) logits
    """
    def __init__(
        self,
        in_features=72,
        d_model=64,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        head_dropout=0.2
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Pose projection: 72 -> 64
        self.pose_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # 2. Stacked Mamba Layers
        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # 3. Small MLP Classification Head: 64 -> 32 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, reliability_weights=None):
        """
        x: (B, T=16, 72)
        reliability_weights: (B, T) or None
        """
        B, T, _ = x.shape
        h = self.pose_proj(x) # (B, T, d_model)

        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)

        # Masked temporal mean pooling
        if reliability_weights is not None:
            # reliability_weights: (B, T)
            w = reliability_weights.unsqueeze(-1) # (B, T, 1)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # Check for batch items with zero valid frames
            mask_has_weight = (w.sum(dim=1) > 1e-4).float() # (B, 1)
            pooled_weighted = (h * w).sum(dim=1, keepdim=True) / w_sum
            pooled_uniform = h.mean(dim=1, keepdim=True)
            pooled = mask_has_weight.unsqueeze(-1) * pooled_weighted + (1.0 - mask_has_weight.unsqueeze(-1)) * pooled_uniform
            pooled = pooled.squeeze(1) # (B, d_model)
        else:
            pooled = h.mean(dim=1)

        logits = self.classifier(pooled) # (B, 1)
        return logits


class PoseBaselineMLP(nn.Module):
    """
    Non-temporal baseline: Temporal mean pooling of keypoint features + small MLP.
    Serves to verify if temporal dynamics in Mamba add value beyond static feature pooling.
    """
    def __init__(self, in_features=72, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, reliability_weights=None):
        """
        x: (B, T, 72)
        """
        if reliability_weights is not None:
            w = reliability_weights.unsqueeze(-1)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            mask_has_weight = (w.sum(dim=1) > 1e-4).float()
            pooled_weighted = (x * w).sum(dim=1, keepdim=True) / w_sum
            pooled_uniform = x.mean(dim=1, keepdim=True)
            pooled = mask_has_weight.unsqueeze(-1) * pooled_weighted + (1.0 - mask_has_weight.unsqueeze(-1)) * pooled_uniform
            pooled = pooled.squeeze(1)
        else:
            pooled = x.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
