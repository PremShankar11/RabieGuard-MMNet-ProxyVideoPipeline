import os
from pathlib import Path

root = Path("c:/Important_prem/FYP")

print("--- 1. Searching for filenames matching play, social, bioacoustic, dog-human ---")
for p in root.rglob("*"):
    if "tinytex" in str(p) or ".git" in str(p):
        continue
    name_low = p.name.lower()
    if any(k in name_low for k in ["play", "social", "bioacoustic", "dogspeak", "barkopedia", "human"]):
        print(f"File/Dir match: {p}")

print("\n--- 2. Searching text contents in extracted_info and project docs ---")
text_exts = {".txt", ".json", ".md", ".py", ".tex"}
for p in root.rglob("*"):
    if "tinytex" in str(p) or ".git" in str(p) or "node_modules" in str(p):
        continue
    if p.suffix.lower() in text_exts and p.is_file():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            for k in ["social play", "bioacoustic", "dog-human", "dogspeak", "barkopedia"]:
                if k in txt:
                    print(f"Content match for '{k}' in: {p}")
        except Exception as e:
            pass
