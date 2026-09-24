import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

files = [
    Path("c:/Important_prem/FYP/extracted_info/21-1.txt"),
    Path("c:/Important_prem/FYP/extracted_info/21-2_1.txt"),
    Path("c:/Important_prem/FYP/extracted_info/BID_REPORT_41.txt"),
    Path("c:/Important_prem/FYP/extracted_info/FirstReviewLaTeX.txt"),
]

for f in files:
    print(f"=== {f.name} ===")
    lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, l in enumerate(lines):
        if "bioacoustic" in l.lower() or "dataset" in l.lower() or "play" in l.lower():
            start = max(0, idx - 2)
            end = min(len(lines), idx + 3)
            print(f"--- Line {idx+1} ---")
            for j in range(start, end):
                print(f"  {j+1}: {lines[j]}")
