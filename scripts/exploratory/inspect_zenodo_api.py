import urllib.request
import json
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

url = 'https://zenodo.org/api/records/18972388'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print('Title:', data.get('metadata', {}).get('title'))
        print('Publication Date:', data.get('metadata', {}).get('publication_date'))
        print('DOI:', data.get('doi'))
        print('\nFiles in Zenodo record:')
        for f in data.get('files', []):
            print(f"  {f.get('key')} - {f.get('size') / 1e6:.2f} MB ({f.get('size')} bytes)")
            print(f"    Download link: {f.get('links', {}).get('self')}")
except Exception as e:
    print('Error fetching Zenodo record:', e)
