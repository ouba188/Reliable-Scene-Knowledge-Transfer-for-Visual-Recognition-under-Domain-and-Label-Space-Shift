"""Fetch authoritative ASF acquisition metadata for every product in the manifest.

ASF search needs no login for metadata (only downloads do). Batch granule_list, 40 per request.
Output: asf_meta.csv  (product, platform, flightDirection, pathNumber, frameNumber, startTime,
                       centerLat, centerLon, polarization, lookDirection)
"""
import csv, json, time, urllib.request
from pathlib import Path

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/asf_meta.csv')
API = 'https://api.daac.asf.alaska.edu/services/search/param'
FIELDS = ['granuleName', 'platform', 'flightDirection', 'pathNumber', 'frameNumber', 'startTime',
          'centerLat', 'centerLon', 'polarization', 'lookDirection']
B = 40
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))

prods = sorted({r['product'] for r in csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')) if r['product']})
print('unique products:', len(prods), flush=True)
have, rows = set(), []
if OUT.exists():
    rows = list(csv.DictReader(OUT.open(encoding='utf-8')))
    have = {r['product'] for r in rows if r.get('flightDirection')}
    print('already fetched:', len(have), flush=True)

f = OUT.open('a' if rows else 'w', encoding='utf-8', newline='')
w = csv.DictWriter(f, fieldnames=['product'] + FIELDS)
if not rows:
    w.writeheader()
todo = [p for p in prods if p not in have]
for i in range(0, len(todo), B):
    batch = todo[i:i + B]
    u = '%s?granule_list=%s&output=json' % (API, ','.join(batch))
    d = None
    for attempt in range(3):
        try:
            d = json.load(op.open(u, timeout=90)); break
        except Exception as e:
            print('   retry %d (%s)' % (attempt, type(e).__name__), flush=True); time.sleep(8)
    try:
        if d is None: raise RuntimeError('3 attempts failed')
        got = {r['granuleName']: r for r in (d[0] or [])}
        for p in batch:
            r = got.get(p)
            w.writerow({'product': p, **{k: (r.get(k, '') if r else '') for k in FIELDS}})
        f.flush()
        print('  %d/%d  got %d' % (min(i + B, len(todo)), len(todo), len(got)), flush=True)
    except Exception as e:
        print('  batch %d failed: %s %s' % (i, type(e).__name__, str(e)[:90]), flush=True)
    time.sleep(1.0)
f.close()
n = sum(1 for r in csv.DictReader(OUT.open(encoding='utf-8')) if r.get('flightDirection'))
print('DONE ->', OUT, ' rows with direction:', n, flush=True)
