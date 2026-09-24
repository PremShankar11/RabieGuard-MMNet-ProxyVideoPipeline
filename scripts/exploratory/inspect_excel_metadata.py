import pandas as pd
from pathlib import Path

raw_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom")
annot_dir = raw_dir / "annotation"

print("=" * 60)
print("INSPECTING df_action.xlsx & AR_metadata.xlsx")
print("=" * 60)

# Inspect df_action.xlsx
df_action_path = annot_dir / "df_action.xlsx"
if df_action_path.exists():
    df_act = pd.read_excel(df_action_path)
    print("\ndf_action.xlsx shape:", df_act.shape)
    print("df_action.xlsx columns:", list(df_act.columns))
    print("\nFirst 10 action labels in df_action.xlsx:")
    print(df_act.head(10).to_string())

# Inspect AR_metadata.xlsx
meta_path = raw_dir / "AR_metadata.xlsx"
if meta_path.exists():
    xls = pd.ExcelFile(meta_path)
    print("\nAR_metadata.xlsx sheets:", xls.sheet_names)
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        print(f"\n--- Sheet: {sheet} (shape: {df_sheet.shape}) ---")
        print("Columns:", list(df_sheet.columns))
        print(df_sheet.head(3).to_string())
        
        # Search for dog or canine
        for col in df_sheet.columns:
            matches = df_sheet[df_sheet[col].astype(str).str.contains(r'\bdog\b|canis|canine', case=False, na=False)]
            if len(matches) > 0:
                print(f"  -> Found {len(matches)} dog matches in col '{col}':")
                print(matches.head(5).to_string())
