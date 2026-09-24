"""
Frozen Mamba V2 Behavioral Classification Model Module.
Loads the strictly frozen checkpoints/mamba_behavior_v2/final.pt model for inference.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.temporal.mamba_model import MambaLayer


class ModelLoadError(Exception):
    """Raised when frozen model checkpoint cannot be loaded."""
    pass


class MambaBehaviorModelV2(nn.Module):
    """
    Frozen Mamba Temporal Model with reliability integration.
    Exact architecture: 2 layers, d_model=64, d_state=16, d_conv=4, expand=2,
    75 in_features (72 pose + 3 reliability features), reliability pooling.
    """
    def __init__(
        self,
        in_features: int = 75,
        d_model: int = 64,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        reliability_mode: str = "both"
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

    def forward(self, x: torch.Tensor, reliability_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
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


class FrozenMambaV2Classifier:
    """
    Inference interface for the frozen Video V2 Mamba S6 behavior model.
    Guarantees loading the authoritative frozen research checkpoint without modification.
    """
    FROZEN_CHECKPOINT_REL = "checkpoints/mamba_behavior_v2/final.pt"

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None
    ):
        if checkpoint_path is None:
            self.checkpoint_path = Path(self.FROZEN_CHECKPOINT_REL).resolve()
        else:
            self.checkpoint_path = Path(checkpoint_path).resolve()

        if not self.checkpoint_path.exists():
            raise ModelLoadError(
                f"Frozen V2 checkpoint not found at: {self.checkpoint_path}. "
                "Ensure checkpoints/mamba_behavior_v2/final.pt is present."
            )

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model, self.metadata = self._load_frozen_checkpoint()

    def _load_frozen_checkpoint(self) -> Tuple[MambaBehaviorModelV2, Dict[str, Any]]:
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            winner_cfg = ckpt.get("config", {})
            in_features = ckpt.get("in_features", 75)

            model = MambaBehaviorModelV2(
                in_features=in_features,
                d_model=winner_cfg.get("d_model", 64),
                num_layers=winner_cfg.get("num_layers", 2),
                d_state=winner_cfg.get("d_state", 16),
                d_conv=winner_cfg.get("d_conv", 4),
                expand=winner_cfg.get("expand", 2),
                dropout=0.1,
                head_dropout=0.2,
                reliability_mode=winner_cfg.get("reliability_mode", "both")
            ).to(self.device)

            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            meta = {
                "winner_name": ckpt.get("winner_name", "exp_d_temporal_tracking"),
                "config": winner_cfg,
                "in_features": in_features,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "device": str(self.device)
            }
            return model, meta
        except Exception as e:
            raise ModelLoadError(f"Failed to load frozen Mamba checkpoint: {e}") from e

    def predict_window(self, features: torch.Tensor, reliability: torch.Tensor) -> float:
        """
        Inference on a single temporal window.
        features: (1, 16, 75) or (16, 75)
        reliability: (1, 16) or (16,)
        Returns probability of AGITATED in [0.0, 1.0].
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if reliability.dim() == 1:
            reliability = reliability.unsqueeze(0)

        bx = features.to(self.device, dtype=torch.float32)
        br = reliability.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            logit = self.model(bx, br)
            prob = float(torch.sigmoid(logit).cpu().numpy().item())
        return prob

    def predict_batch(self, features: torch.Tensor, reliability: torch.Tensor) -> list[float]:
        """
        Batch inference on multiple temporal windows.
        features: (B, 16, 75)
        reliability: (B, 16)
        """
        bx = features.to(self.device, dtype=torch.float32)
        br = reliability.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(bx, br)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
            if isinstance(probs, float):
                probs = [probs]
        return [float(p) for p in probs]
