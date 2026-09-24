import pandas as pd
from pathlib import Path

annot_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom\annotation")
df_action_path = annot_dir / "df_action.xlsx"

xls = pd.ExcelFile(df_action_path)
print("df_action.xlsx sheet names:", xls.sheet_names)
for s in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=s)
    print(f"Sheet '{s}': shape {df.shape}, cols {list(df.columns)}")
