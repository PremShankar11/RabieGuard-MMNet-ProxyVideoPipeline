import urllib.request
import json
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

url = "https://raw.githubusercontent.com/rhernandez00/bioacoustic-dataset/main/database_analysis.ipynb"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        nb = json.loads(resp.read().decode("utf-8"))
        print(f"Total cells: {len(nb.get('cells', []))}")
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type')
            source = "".join(cell.get('source', []))
            if cell_type == 'markdown':
                print(f"\n--- [Markdown Cell {i+1}] ---")
                print(source[:500])
            elif cell_type == 'code':
                # check if it defines labels or reads metadata
                for kw in ['label', 'sound', 'category', 'class', 'session', 'duration', 'metadata', 'annotation', 'barks']:
                    if kw in source.lower():
                        print(f"\n--- [Code Cell {i+1}] (matches {kw}) ---")
                        lines = source.splitlines()
                        for l in lines[:20]:
                            print(" ", l)
                        break
except Exception as e:
    print("Error fetching notebook:", e)
