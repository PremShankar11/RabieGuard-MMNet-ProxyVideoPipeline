import os
import sys
import yaml
import cv2
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("VERIFYING ULTRALYTICS DOG-POSE DATASET")
print("=" * 60)

config_path = Path("configs/dog_pose.yaml")
assert config_path.exists(), f"Config missing: {config_path}"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dataset_root = Path(cfg["path"])
print(f"Dataset root: {dataset_root}")
assert dataset_root.exists(), f"Dataset directory missing: {dataset_root}"

train_img_dir = dataset_root / cfg["train"]
val_img_dir = dataset_root / cfg["val"]
train_lbl_dir = dataset_root / "labels" / "train"
val_lbl_dir = dataset_root / "labels" / "val"

train_imgs = list(train_img_dir.glob("*.jpg"))
val_imgs = list(val_img_dir.glob("*.jpg"))
train_lbls = list(train_lbl_dir.glob("*.txt"))
val_lbls = list(val_lbl_dir.glob("*.txt"))

print(f"Train images: {len(train_imgs)} (Expected: 6773)")
print(f"Val images:   {len(val_imgs)} (Expected: 1703)")
print(f"Train labels: {len(train_lbls)} (Expected: 6773)")
print(f"Val labels:   {len(val_lbls)} (Expected: 1703)")

assert len(train_imgs) == 6773, f"Mismatch in train images: {len(train_imgs)}"
assert len(val_imgs) == 1703, f"Mismatch in val images: {len(val_imgs)}"
assert len(train_lbls) == 6773, f"Mismatch in train labels: {len(train_lbls)}"
assert len(val_lbls) == 1703, f"Mismatch in val labels: {len(val_lbls)}"

# Verify token format and token count (77 tokens per line)
print("\nChecking annotation format on sample 100 train & val labels...")
corrupt_count = 0
for lbl_file in (train_lbls[:100] + val_lbls[:100]):
    lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        tokens = line.split()
        if len(tokens) != 77:
            print(f"Corrupt label in {lbl_file.name}: {len(tokens)} tokens")
            corrupt_count += 1
assert corrupt_count == 0, f"Found {corrupt_count} corrupt annotations!"
print("Annotation format verified: exactly 77 tokens (1 class + 4 bbox + 24*3 keypoint values).")

# Verify sample image readability
sample_img = cv2.imread(str(train_imgs[0]))
assert sample_img is not None, f"Failed to read image {train_imgs[0]}"
h, w, c = sample_img.shape
print(f"\nSample image verified: {train_imgs[0].name}, shape: {w}x{h}x{c}")

print("\nAll Dog-Pose dataset checks PASSED successfully.")
