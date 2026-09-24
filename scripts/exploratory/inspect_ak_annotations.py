import pandas as pd
from pathlib import Path

raw_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom")
annot_dir = raw_dir / "annotation"

print("=" * 60)
print("INSPECTING ANIMAL KINGDOM ANNOTATIONS")
print("=" * 60)

train_csv = annot_dir / "train.csv"
val_csv = annot_dir / "val.csv"
metadata_xlsx = raw_dir / "AR_metadata.xlsx"
df_action_xlsx = annot_dir / "df_action.xlsx"

print(f"train.csv exists: {train_csv.exists()} ({train_csv.stat().st_size / (1024*1024):.2f} MB)")
print(f"val.csv exists: {val_csv.exists()} ({val_csv.stat().st_size / (1024*1024):.2f} MB)")
print(f"AR_metadata.xlsx exists: {metadata_xlsx.exists()}")
print(f"df_action.xlsx exists: {df_action_xlsx.exists()}")

# Inspect train.csv and val.csv
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)

print(f"\nTrain rows: {len(df_train)}")
print(f"Val rows: {len(df_val)}")
print(f"Total annotations: {len(df_train) + len(df_val)}")
print(f"Columns: {list(df_train.columns)}")

print("\nTrain sample (first 3 rows):")
print(df_train.head(3).to_string())

# Inspect metadata
if metadata_xlsx.exists():
    xls = pd.ExcelFile(metadata_xlsx)
    print(f"\nAR_metadata.xlsx sheet names: {xls.sheet_names}")
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        print(f"\n--- Sheet: {sheet} (shape: {df_sheet.shape}) ---")
        print(f"Columns: {list(df_sheet.columns)}")
        print(df_sheet.head(2).to_string())

# Search for Dog in metadata sheets
dog_info = []
if metadata_xlsx.exists():
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        for col in df_sheet.columns:
            matches = df_sheet[df_sheet[col].astype(str).str.contains(r'\bdog\b|canis', case=False, na=False)]
            if len(matches) > 0:
                print(f"\nMatches for 'dog' in Sheet '{sheet}', Column '{col}': {len(matches)}")
                print(matches.head(10).to_string())
