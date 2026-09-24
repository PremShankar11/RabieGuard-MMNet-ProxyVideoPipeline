"""
Unified Model Architectures for Video V2 Development.
Zero Rabies-MMNet Project.
Includes: Mamba S6 V2, Parameter-Matched GRU, Parameter-Matched LSTM, Non-Temporal MLP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.temporal.mamba_model import MambaS6Block, MambaLayer


class MambaBehaviorModelV2(nn.Module):
    """
    Enhanced Mamba Temporal Model with flexible reliability integration.
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
        head_dropout=0.2,
        reliability_mode="both" # 'pooling', 'input', 'both', 'gating'
    ):
        super().__init__()
        self.d_model = d_model
        self.reliability_mode = reliability_mode

        self.pose_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

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

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, reliability_weights=None):
        """
        x: (B, T, in_features)
        reliability_weights: (B, T)
        """
        if self.reliability_mode == "gating" and reliability_weights is not None:
            x = x * reliability_weights.unsqueeze(-1)

        h = self.pose_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)

        # Temporal pooling
        if self.reliability_mode in ("pooling", "both") and reliability_weights is not None:
            w = reliability_weights.unsqueeze(-1) # (B, T, 1)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            mask_has_weight = (w.sum(dim=1) > 1e-4).float().unsqueeze(-1)
            pooled_weighted = (h * w).sum(dim=1, keepdim=True) / w_sum
            pooled_uniform = h.mean(dim=1, keepdim=True)
            pooled = mask_has_weight * pooled_weighted + (1.0 - mask_has_weight) * pooled_uniform
            pooled = pooled.squeeze(1)
        else:
            pooled = h.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


class GRUBehaviorModelV2(nn.Module):
    """
    Parameter-matched GRU model (~130k params) for controlled baseline comparison.
    """
    def __init__(
        self,
        in_features=75,
        hidden_dim=96,
        num_layers=2,
        dropout=0.1,
        head_dropout=0.2,
        bidirectional=True
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.norm = nn.LayerNorm(self.out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.out_dim, 32),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, reliability_weights=None):
        h = self.proj(x)
        out, _ = self.gru(h)
        out = self.norm(out)

        if reliability_weights is not None:
            w = reliability_weights.unsqueeze(-1)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            mask_has_weight = (w.sum(dim=1) > 1e-4).float().unsqueeze(-1)
            pooled_weighted = (out * w).sum(dim=1, keepdim=True) / w_sum
            pooled_uniform = out.mean(dim=1, keepdim=True)
            pooled = mask_has_weight * pooled_weighted + (1.0 - mask_has_weight) * pooled_uniform
            pooled = pooled.squeeze(1)
        else:
            pooled = out.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


class LSTMBehaviorModelV2(nn.Module):
    """
    Parameter-matched LSTM model (~135k params) for controlled baseline comparison.
    """
    def __init__(
        self,
        in_features=75,
        hidden_dim=84,
        num_layers=2,
        dropout=0.1,
        head_dropout=0.2,
        bidirectional=True
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.norm = nn.LayerNorm(self.out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.out_dim, 32),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, reliability_weights=None):
        h = self.proj(x)
        out, _ = self.lstm(h)
        out = self.norm(out)

        if reliability_weights is not None:
            w = reliability_weights.unsqueeze(-1)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            mask_has_weight = (w.sum(dim=1) > 1e-4).float().unsqueeze(-1)
            pooled_weighted = (out * w).sum(dim=1, keepdim=True) / w_sum
            pooled_uniform = out.mean(dim=1, keepdim=True)
            pooled = mask_has_weight * pooled_weighted + (1.0 - mask_has_weight) * pooled_uniform
            pooled = pooled.squeeze(1)
        else:
            pooled = out.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


class PoseBaselineMLPV2(nn.Module):
    """
    Non-temporal baseline: Temporal mean pooling of keypoint features + small MLP.
    """
    def __init__(self, in_features=75, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, reliability_weights=None):
        if reliability_weights is not None:
            w = reliability_weights.unsqueeze(-1)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            mask_has_weight = (w.sum(dim=1) > 1e-4).float().unsqueeze(-1)
            pooled_weighted = (x * w).sum(dim=1, keepdim=True) / w_sum
            pooled_uniform = x.mean(dim=1, keepdim=True)
            pooled = mask_has_weight * pooled_weighted + (1.0 - mask_has_weight) * pooled_uniform
            pooled = pooled.squeeze(1)
        else:
            pooled = x.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
