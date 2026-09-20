"""Exact annotation closure count by source tree (run on westd)."""
import json
import pathlib
import re
import subprocess
from collections import Counter

r = subprocess.run(['find', '/root/autodl-tmp', '/root/autodl-fs', '-path', '*annotation_output*',
                    '-name', '*.json', '-path', '*status*'], capture_output=True, text=True)
files = [l for l in r.stdout.splitlines() if l.strip()]
print('status json files:', len(files))

by_prod = {}
for f in files:
    p = pathlib.Path(f)
    m = re.match(r'(S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_[0-9A-F]{6}_[0-9A-F]{6}_[0-9A-F]{4})', p.stem)
    if not m:
        continue
    try:
        st = json.loads(p.read_text(encoding='utf-8')).get('status')
    except Exception:
        continue
    by_prod.setdefault(m.group(1), set()).add(st)

completed = {p for p, s in by_prod.items() if 'completed' in s}
inv = set(l.split('\t')[0] for l in open('/root/ann841_inventory.tsv', encoding='utf-8') if l.strip())
print('unique products with status:', len(by_prod))
print('products completed:', len(completed))
print('841 inventory:', len(inv))
print('completed & 841:', len(completed & inv))
missing = sorted(inv - completed)
print('missing:', len(missing), [m[-9:] for m in missing][:8])

c = Counter()
for f in files:
    if '/annotation_output_s3/' in f:
        c['annotation_output_s3'] += 1
    elif '/annotation_output/' in f:
        c['annotation_output'] += 1
    elif 'autodl-fs' in f:
        c['autodl-fs'] += 1
    else:
        c['other'] += 1
print('status files by tree:', dict(c))

transfer = set()
for f in files:
    if 'local_transfer' in pathlib.Path(f).parts:
        pass
tr = subprocess.run(['grep', '-rl', 'local_transfer', '/root/autodl-tmp/safe841_batch/annotation_output/status/'],
                    capture_output=True, text=True)
print('local_transfer status files:', len([l for l in tr.stdout.splitlines() if l.strip()]))
