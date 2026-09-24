import os
import cv2
import pandas as pd
from pathlib import Path

raw_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom")
annot_dir = raw_dir / "annotation"

print("=" * 60)
print("INSPECTING ANIMAL KINGDOM EXTRACTED DIRECTORY")
print("=" * 60)

# List top-level items
for item in raw_dir.iterdir():
    if item.is_dir():
        count = sum(1 for _ in item.rglob("*") if _.is_file())
        print(f"Dir: {item.name:25s} (Contains {count} files)")
    else:
        sz_mb = item.stat().st_size / (1024 * 1024)
        print(f"File: {item.name:25s} ({sz_mb:.2f} MB)")

# Find where videos are located
video_files = list(raw_dir.rglob("*.mp4")) + list(raw_dir.rglob("*.avi")) + list(raw_dir.rglob("*.mkv"))
print(f"\nTotal video files found: {len(video_files):,}")

if video_files:
    video_dir = video_files[0].parent
    print(f"Video directory: {video_dir}")
    print(f"Sample video filenames (first 5): {[v.name for v in video_files[:5]]}")
    
    # Inspect sample video with OpenCV
    sample_vid = video_files[0]
    print(f"\n--- Sample Video Inspection: {sample_vid.name} ---")
    cap = cv2.VideoCapture(str(sample_vid))
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        ret, frame = cap.read()
        cap.release()
        print(f"  Resolution: {width} x {height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frame count: {frame_count}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Frame decoding test: {'SUCCESS' if ret and frame is not None else 'FAILED'}")
    else:
        print("  Error: Could not open sample video.")

# Cross-reference with annotations
train_path = annot_dir / "train.csv"
val_path = annot_dir / "val.csv"
meta_path = raw_dir / "AR_metadata.xlsx"

if train_path.exists():
    df_train = pd.read_csv(train_path, sep=' ')
    train_clip_ids = set(df_train['original_vido_id'].unique())
    vid_stems = set(v.stem for v in video_files)
    common_ids = train_clip_ids.intersection(vid_stems)
    print(f"\nLinking Check:")
    print(f"  Unique clips in train.csv: {len(train_clip_ids):,}")
    print(f"  Unique video stems on disk: {len(vid_stems):,}")
    print(f"  Overlap (train clip IDs matching video file stems): {len(common_ids):,}")

# Check disk usage
def get_dir_size(path):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total

dog_pose_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\dog_pose")
dp_size = get_dir_size(dog_pose_dir) / (1024**3)
ak_size = get_dir_size(raw_dir) / (1024**3)

print("\n" + "=" * 60)
print("DISK USAGE SUMMARY")
print("=" * 60)
print(f"Dog-Pose:       {dp_size:.2f} GB")
print(f"Animal Kingdom: {ak_size:.2f} GB")
print(f"Total Raw Data: {dp_size + ak_size:.2f} GB")
print("=" * 60)
