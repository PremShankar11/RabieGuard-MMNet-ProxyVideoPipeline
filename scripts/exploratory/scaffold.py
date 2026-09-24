from pathlib import Path

base = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet")

dirs = [
    "data/raw/dog_pose",
    "data/raw/animal_kingdom",
    "data/processed",
    "src/pose",
    "src/features",
    "src/temporal",
    "src/evaluation",
    "configs",
    "checkpoints",
    "outputs",
    "notebooks",
    "scripts"
]

for d in dirs:
    p = base / d
    p.mkdir(parents=True, exist_ok=True)
    keep = p / ".gitkeep"
    if not keep.exists():
        keep.touch()

print("Directories initialized successfully.")
