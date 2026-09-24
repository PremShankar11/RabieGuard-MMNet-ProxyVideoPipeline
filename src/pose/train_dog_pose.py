import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-Pose on Ultralytics Dog-Pose dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (default: 2)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="Pretrained model weights")
    parser.add_argument("--config", type=str, default="configs/dog_pose.yaml", help="Dataset config YAML")
    args = parser.parse_args()

    project_dir = Path("c:/Important_prem/FYP/Zero-Rabies-MMNet/runs/pose_training")
    run_name = "dog_pose_yolo11n"
    checkpoints_dir = Path("c:/Important_prem/FYP/Zero-Rabies-MMNet/checkpoints/dog_pose")
    reports_dir = Path("c:/Important_prem/FYP/Zero-Rabies-MMNet/reports/pose")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 3: YOLO POSE TRAINING ON DOG-POSE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Patience: {args.patience}")
    print(f"Workers: {args.workers}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    print(f"Device: {gpu_name} (VRAM: {vram_gb:.2f} GB)")

    # Load pretrained model
    model = YOLO(args.model)

    # Train
    print("\nStarting Ultralytics training...")
    try:
        results = model.train(
            data=args.config,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            workers=args.workers,
            device=0 if torch.cuda.is_available() else "cpu",
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            amp=True,
            plots=True,
            save=True,
            val=True
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nCUDA OOM with batch={args.batch}! Retrying with batch={args.batch // 2}...")
        torch.cuda.empty_cache()
        results = model.train(
            data=args.config,
            epochs=args.epochs,
            batch=args.batch // 2,
            imgsz=args.imgsz,
            patience=args.patience,
            workers=args.workers,
            device=0 if torch.cuda.is_available() else "cpu",
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            amp=True,
            plots=True,
            save=True,
            val=True
        )

    # Locate saved weights
    run_dir = project_dir / run_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    dest_best = checkpoints_dir / "best.pt"
    dest_last = checkpoints_dir / "last.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, dest_best)
        print(f"\nSuccessfully saved best checkpoint to: {dest_best} ({dest_best.stat().st_size / 1e6:.2f} MB)")
    if last_weights.exists():
        shutil.copy2(last_weights, dest_last)
        print(f"Successfully saved last checkpoint to: {dest_last} ({dest_last.stat().st_size / 1e6:.2f} MB)")

    # Generate training report
    report_path = reports_dir / "dog_pose_training_report.md"
    csv_results = run_dir / "results.csv"
    csv_snippet = ""
    if csv_results.exists():
        lines = csv_results.read_text(encoding="utf-8").strip().splitlines()
        csv_snippet = "\n".join(lines[:1] + lines[-5:])

    report_content = f"""# Dog-Pose Training Report

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 3 — YOLO Pose Fine-Tuning  
**Date:** 2026-09-23  
**Model Architecture:** YOLO11n-Pose (`yolo11n-pose.pt`, ~2.6M parameters)  
**Dataset:** Ultralytics Dog-Pose (6,773 train, 1,703 val images, 24 keypoints)  

---

## 1. Environment & Hardware
- **Python:** {sys.version.split()[0]}
- **PyTorch:** {torch.__version__}
- **CUDA Device:** {gpu_name} (Total VRAM: {vram_gb:.2f} GB)
- **AMP Enabled:** True
- **Workers:** {args.workers}

---

## 2. Hyperparameters & Configuration
- **Dataset Config:** `{args.config}`
- **Epochs:** {args.epochs}
- **Batch Size:** {args.batch}
- **Image Size:** {args.imgsz}
- **Patience:** {args.patience}
- **Pretrained Weights:** `{args.model}`
- **Best Checkpoint Path:** `{dest_best}`

---

## 3. Training Results & Metrics Summary
The model was fine-tuned to adapt its pose prediction head from 17 COCO human keypoints to 24 canine keypoints.

```csv
{csv_snippet}
```

Checkpoints generated:
- Best Checkpoint: `{dest_best}`
- Last Checkpoint: `{dest_last}`
"""
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Saved training report to: {report_path}")
    print("\nTraining procedure finished successfully!")

if __name__ == "__main__":
    main()
