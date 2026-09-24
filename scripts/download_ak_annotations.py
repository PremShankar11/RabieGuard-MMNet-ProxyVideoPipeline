import gdown
from pathlib import Path

target_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom")
annot_dir = target_dir / "annotation"
annot_dir.mkdir(parents=True, exist_ok=True)

files = [
    {
        "name": "df_action.xlsx",
        "id": "1Ic-QcniLDzHzXQjTd2jJx6mgz2ejPEID",
        "output": annot_dir / "df_action.xlsx"
    },
    {
        "name": "train.csv",
        "id": "1fCjP5bZx6Ku5NSXQivfroEeLL4A4v6Vl",
        "output": annot_dir / "train.csv"
    },
    {
        "name": "val.csv",
        "id": "1IpnY1sArVLxclBPxM7e6NZnVi5ZArSKn",
        "output": annot_dir / "val.csv"
    },
    {
        "name": "AR_metadata.xlsx",
        "id": "1qER0AtPRQlLoapn4-omlWE3uwfUi0KpX",
        "output": target_dir / "AR_metadata.xlsx"
    },
    {
        "name": "README_action_recognition.md",
        "id": "1p-mU2WzCcZXZ53hyqVklGF2vslVLqust",
        "output": target_dir / "README_action_recognition.md"
    }
]

for f in files:
    out = f["output"]
    if out.exists():
        print(f"{f['name']} already exists at {out}. Skipping.")
    else:
        print(f"Downloading {f['name']}...")
        url = f"https://drive.google.com/uc?id={f['id']}"
        gdown.download(url, str(out), quiet=False)
        print(f"Downloaded {f['name']} ({out.stat().st_size / (1024*1024):.2f} MB)")

print("All annotation and metadata files downloaded successfully.")
