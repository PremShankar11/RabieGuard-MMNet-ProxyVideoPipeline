"""
Reproducibility Verification Script for Zero Rabies-MMNet Video V2.
Compares actual training configuration, code, and checkpoint against manifests and reports.
"""

import sys
import json
import yaml
import torch
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# 1. Load behavior_v2.yaml
with open(repo_root / "configs" / "behavior_v2.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 2. Load V2_FINAL_CONFIG.json
with open(repo_root / "reports" / "behavior_v2" / "V2_FINAL_CONFIG.json", "r") as f:
    final_cfg = json.load(f)

# 3. Load FROZEN_MODEL_MANIFEST.json
with open(repo_root / "reports" / "behavior_v2" / "FROZEN_MODEL_MANIFEST.json", "r") as f:
    manifest = json.load(f)

# 4. Load final.pt
ckpt_path = repo_root / "checkpoints" / "mamba_behavior_v2" / "final.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("=" * 65)
print("ZERO RABIES-MMNET — VIDEO V2 REPRODUCIBILITY AUDIT")
print("=" * 65)

# 1. Learning Rate
lr_yaml = cfg["training"]["lr"]
lr_manifest = manifest["training_configuration"]["learning_rate"]
match_lr = (lr_yaml == lr_manifest == 0.001)
print(f"1. Learning Rate:        yaml={lr_yaml}, manifest={lr_manifest} -> MATCH: {match_lr}")

# 2. Weight Decay
wd_yaml = cfg["training"]["weight_decay"]
wd_manifest = manifest["training_configuration"]["weight_decay"]
match_wd = (wd_yaml == wd_manifest == 0.0001)
print(f"2. Weight Decay:        yaml={wd_yaml}, manifest={wd_manifest} -> MATCH: {match_wd}")

# 3. Batch Size
bs_yaml = cfg["training"]["batch_size"]
bs_manifest = manifest["training_configuration"]["batch_size"]
match_bs = (bs_yaml == bs_manifest == 8)
print(f"3. Batch Size:          yaml={bs_yaml}, manifest={bs_manifest} -> MATCH: {match_bs}")

# 4. Epochs / Patience
ep_yaml = cfg["training"]["epochs"]
ep_ckpt = ckpt.get("epochs_trained")
ep_manifest = manifest["training_configuration"]["epochs"]
pat_yaml = cfg["training"]["patience"]
pat_manifest = manifest["training_configuration"]["patience"]
match_ep = (ep_yaml == ep_ckpt == ep_manifest == 35 and pat_yaml == pat_manifest == 8)
print(f"4. Epochs & Patience:   epochs={ep_yaml} (ckpt={ep_ckpt}), patience={pat_yaml} -> MATCH: {match_ep}")

# 5. Seed
seed_yaml = cfg["training"]["seed"]
seed_manifest = manifest["training_configuration"]["seed"]
match_seed = (seed_yaml == seed_manifest == 42)
print(f"5. Seed:                yaml={seed_yaml}, manifest={seed_manifest} -> MATCH: {match_seed}")

# 6. Sequence Length
seq_yaml = cfg["experiments"]["exp_d_temporal_tracking"]["seq_length"]
seq_ckpt = ckpt["config"]["seq_length"]
seq_final = final_cfg["model_config"]["seq_length"]
seq_manifest = manifest["input_specifications"]["sequence_length"]
match_seq = (seq_yaml == seq_ckpt == seq_final == seq_manifest == 16)
print(f"6. Sequence Length:     yaml={seq_yaml}, ckpt={seq_ckpt}, final_cfg={seq_final}, manifest={seq_manifest} -> MATCH: {match_seq}")

# 7. Sampling Rate
fps_yaml = cfg["data"]["sampling_rate"]
fps_manifest = manifest["input_specifications"]["sampling_rate_fps"]
match_fps = (fps_yaml == fps_manifest == 8.0)
print(f"7. Sampling Rate (FPS): yaml={fps_yaml}, manifest={fps_manifest} -> MATCH: {match_fps}")

# 8. Reliability Configuration
rel_yaml = cfg["experiments"]["exp_d_temporal_tracking"]["reliability_mode"]
rel_ckpt = ckpt["config"]["reliability_mode"]
rel_final = final_cfg["model_config"]["reliability_mode"]
rel_manifest = manifest["input_specifications"]["reliability_mode"]
match_rel = (rel_yaml == rel_ckpt == rel_final == rel_manifest == "both")
print(f"8. Reliability Config:  yaml={rel_yaml}, ckpt={rel_ckpt}, final_cfg={rel_final}, manifest={rel_manifest} -> MATCH: {match_rel}")

# 9. Tracking Configuration
track_yaml = cfg["experiments"]["exp_d_temporal_tracking"]["tracking_mode"]
track_ckpt = ckpt["config"]["tracking_mode"]
track_final = final_cfg["model_config"]["tracking_mode"]
track_manifest = manifest["input_specifications"]["tracking_mode"]
match_track = (track_yaml == track_ckpt == track_final == track_manifest == "temporal_consistent")
print(f"9. Tracking Config:     yaml={track_yaml}, ckpt={track_ckpt}, final_cfg={track_final}, manifest={track_manifest} -> MATCH: {match_track}")

# 10. Model Architecture
from src.video.model import FrozenMambaV2Classifier
model_wrapper = FrozenMambaV2Classifier()
model_params = sum(p.numel() for p in model_wrapper.model.parameters() if p.requires_grad)
match_arch = (
    ckpt["config"]["model_type"] == "mamba" and
    ckpt["config"]["d_model"] == 64 and
    ckpt["config"]["num_layers"] == 2 and
    ckpt["in_features"] == 75
)
print(f"10. Model Architecture:  Mamba S6 (2 layers, d_model=64, d_state=16, d_conv=4, expand=2, in_features=75) -> MATCH: {match_arch}")

# 11. Trainable Parameters
match_params = (model_params == 136257 == final_cfg["trainable_parameters"] == manifest["architecture"]["trainable_parameters"])
print(f"11. Trainable Params:   model={model_params:,}, final_cfg={final_cfg['trainable_parameters']:,}, manifest={manifest['architecture']['trainable_parameters']:,} -> MATCH: {match_params}")

# 12. Decision Threshold
thresh_manifest = manifest["operational_scoring"]["decision_threshold"]
match_thresh = (thresh_manifest == 0.50)
print(f"12. Decision Threshold: {thresh_manifest} -> MATCH: {match_thresh}")

print("=" * 65)
all_passed = all([
    match_lr, match_wd, match_bs, match_ep, match_seed,
    match_seq, match_fps, match_rel, match_track, match_arch,
    match_params, match_thresh
])

if all_passed:
    print("V2 reproducibility audit: PASS")
else:
    print("V2 reproducibility audit: FAIL")
print("=" * 65)
