import urllib.request
import json
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("Fetching GitHub repository file listing...")
url = "https://api.github.com/repos/rhernandez00/bioacoustic-dataset/contents"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        items = json.loads(resp.read().decode("utf-8"))
        for item in items:
            name = item.get("name")
            t = item.get("type")
            size = item.get("size", 0)
            dl = item.get("download_url")
            print(f"  {name} ({t}, {size} bytes)")
except Exception as e:
    print("Error fetching GitHub contents:", e)
