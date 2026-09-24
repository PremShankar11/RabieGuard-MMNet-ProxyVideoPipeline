import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\dog_pose")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DATA_DIR / "dog-pose.zip"
URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/dog-pose.zip"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print("Download completed.")

def extract_file(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed.")

if __name__ == "__main__":
    if not ZIP_PATH.exists():
        download_file(URL, ZIP_PATH)
    else:
        print(f"{ZIP_PATH} already exists. Skipping download.")

    extract_file(ZIP_PATH, DATA_DIR)
