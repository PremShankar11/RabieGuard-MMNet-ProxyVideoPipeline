import os
import cv2
import yaml
from pathlib import Path

raw_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\dog_pose")

print("=" * 60)
print("INSPECTING DOG-POSE DIRECTORY STRUCTURE")
print("=" * 60)

for root, dirs, files in os.walk(raw_dir):
    rel_path = Path(root).relative_to(raw_dir)
    # only print top levels
    if len(rel_path.parts) <= 3:
        sample_files = files[:3]
        print(f"{rel_path}: {len(dirs)} dirs, {len(files)} files (samples: {sample_files})")

images_train_dir = raw_dir / "images" / "train"
images_val_dir = raw_dir / "images" / "val"
labels_train_dir = raw_dir / "labels" / "train"
labels_val_dir = raw_dir / "labels" / "val"

print("\nResolved paths:")
print("Images Train:", images_train_dir, f"(Exists: {images_train_dir.exists()})")
print("Images Val:", images_val_dir, f"(Exists: {images_val_dir.exists()})")
print("Labels Train:", labels_train_dir, f"(Exists: {labels_train_dir.exists()})")
print("Labels Val:", labels_val_dir, f"(Exists: {labels_val_dir.exists()})")

train_images = list(images_train_dir.glob("*.jpg")) + list(images_train_dir.glob("*.png"))
val_images = list(images_val_dir.glob("*.jpg")) + list(images_val_dir.glob("*.png"))
train_labels = list(labels_train_dir.glob("*.txt"))
val_labels = list(labels_val_dir.glob("*.txt"))

print("\nDataset Statistics:")
print(f"Train Images count: {len(train_images)}")
print(f"Val Images count: {len(val_images)}")
print(f"Train Labels count: {len(train_labels)}")
print(f"Val Labels count: {len(val_labels)}")

yaml_path = raw_dir / "dog-pose.yaml"
print(f"\nYAML file in raw_dir: {yaml_path} (Exists: {yaml_path.exists()})")
if yaml_path.exists():
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print("\nDataset YAML content:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

# Sample Image and Annotation Inspection
if train_images and train_labels:
    sample_img_path = train_images[0]
    sample_lbl_path = labels_train_dir / f"{sample_img_path.stem}.txt"
    if not sample_lbl_path.exists():
        sample_lbl_path = train_labels[0]
        sample_img_path = images_train_dir / f"{sample_lbl_path.stem}.jpg"

    print("\n" + "=" * 60)
    print("SAMPLE IMAGE & ANNOTATION INSPECTION")
    print("=" * 60)
    print(f"Image filename: {sample_img_path.name}")
    print(f"Image path: {sample_img_path}")
    if sample_img_path.exists():
        img = cv2.imread(str(sample_img_path))
        h, w, c = img.shape
        print(f"Image dimensions: width={w}, height={h}, channels={c}")

    print(f"Annotation filename: {sample_lbl_path.name}")
    print(f"Annotation path: {sample_lbl_path}")
    if sample_lbl_path.exists():
        with open(sample_lbl_path, 'r', encoding='utf-8') as f:
            raw_content = f.read().strip()
        print(f"\nRaw annotation contents:\n{raw_content}\n")
        tokens = raw_content.split()
        print(f"Total numeric values in line: {len(tokens)}")
        if len(tokens) >= 5:
            cls_id = tokens[0]
            bbox = [float(x) for x in tokens[1:5]]
            kpts = [float(x) for x in tokens[5:]]
            print(f"Class ID: {cls_id} ({cfg['names'][int(cls_id)] if 'names' in cfg else 'unknown'})")
            print(f"BBox (center_x, center_y, width, height): {bbox}")
            print(f"Keypoint values count: {len(kpts)}")
            num_kpts = len(kpts) // 3
            print(f"Keypoints count: {num_kpts}")
            kpt_names = cfg.get('kpt_names', {}).get(int(cls_id), [])
            print("\nKeypoint coordinates & visibility triplets [x, y, v]:")
            for i in range(num_kpts):
                name = kpt_names[i] if i < len(kpt_names) else f"kpt_{i+1}"
                x, y, v = kpts[i*3], kpts[i*3+1], kpts[i*3+2]
                print(f"  [{i+1:2d}] {name:18s} -> x={x:.4f}, y={y:.4f}, v={int(v)}")
