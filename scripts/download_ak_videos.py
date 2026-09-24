import os
import tarfile
import gdown
from pathlib import Path

target_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom")
target_dir.mkdir(parents=True, exist_ok=True)
tar_path = target_dir / "video.tar.gz"

FILE_ID = "1X4rL5ey7M1_YM4GDa1DvvVdHoUfuHeJp"

def download_video_archive():
    if not tar_path.exists():
        print(f"Downloading Animal Kingdom video archive (video.tar.gz) from Google Drive (ID: {FILE_ID})...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, str(tar_path), quiet=False)
        print("Download completed.")
    else:
        print(f"{tar_path} already exists ({tar_path.stat().st_size / (1024*1024*1024):.2f} GB). Skipping download.")

def extract_video_archive():
    print(f"Extracting {tar_path} to {target_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print("Extraction completed.")

if __name__ == "__main__":
    download_video_archive()
    extract_video_archive()
