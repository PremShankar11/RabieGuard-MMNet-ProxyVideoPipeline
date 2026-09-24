import pandas as pd
from pathlib import Path

raw_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom")

for p in raw_dir.rglob("*.xlsx"):
    xls = pd.ExcelFile(p)
    print(f"File: {p.relative_to(raw_dir)} -> Sheets: {xls.sheet_names}")
