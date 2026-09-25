"""nastar_fetch.py -- download every NASTaR ship_patches_uint8 collection into E:/nastar.

Ids come from nastar_ids2.json (built by enumeration, since search returns the whole repository). Each matched collection is a
scene and holds a handful of uint8 TIFF patches whose FILENAMES carry the ship type (e.g. ..._patch_3_Cargo_uint8.tif), so labels
need no separate file.

ponytail: stdlib + the local proxy, per-file size check so a re-run resumes, and a plain manifest CSV so the next step can read
the scene/type/path without re-listing CKAN.
"""
import csv
import json
import os
import urllib.request
from pathlib import Path

PROXY = 'http://127.0.0.1:7890'
for v in ('NO_PROXY', 'no_proxy'):
    os.environ.pop(v, None)
OUT = Path('E:/nastar')
OUT.mkdir(parents=True, exist_ok=True)
BASE = 'https://data.bris.ac.uk/data/api/3/action/'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY}))

d = json.load(open('nastar_ids2.json'))
ids = d['kinds'].get('ship_patches_uint8', [])
print('待取集合 %d 个（ship_patches_uint8）' % len(ids), flush=True)

rows = []
for k, pid in enumerate(ids, 1):
    try:
        with opener.open(BASE + 'package_show?id=%s' % pid, timeout=60) as r:
            pkg = json.loads(r.read().decode('utf-8'))['result']
    except Exception as e:
        print('  [%d] 元数据失败 %s' % (k, str(e)[:80]), flush=True)
        continue
    title = pkg.get('title') or ''
    scene = title.split('/')[-2] if title.count('/') >= 2 else title[-40:]
    for x in pkg.get('resources', []):
        nm = x.get('name') or ''
        url = x.get('url') or ''
        if not url.lower().endswith('.tif'):
            continue
        parts = nm.split('_')
        typ = parts[-1].replace('.tif', '') if len(parts) > 1 else 'unknown'
        dst = OUT / (scene + '__' + nm) if scene else OUT / nm
        want = int(x.get('size') or 0)
        if dst.exists() and (not want or dst.stat().st_size == want):
            rows.append((scene, typ, str(dst), dst.stat().st_size))
            continue
        try:
            with opener.open(url, timeout=180) as rr, dst.open('wb') as f:
                while True:
                    ch = rr.read(1 << 20)
                    if not ch:
                        break
                    f.write(ch)
            rows.append((scene, typ, str(dst), dst.stat().st_size))
        except Exception as e:
            print('  FAIL %s: %s' % (nm[:44], str(e)[:80]), flush=True)
    if k % 10 == 0:
        print('  [%d/%d] 已下 %d 片' % (k, len(ids), len(rows)), flush=True)

with (OUT / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['scene', 'ship_type', 'path', 'bytes'])
    w.writerows(rows)
import collections
print('完成 ✓ 片 %d ｜ 场景 %d ｜ 类型 %d' % (len(rows), len(set(r[0] for r in rows)), len(set(r[1] for r in rows))), flush=True)
print('类型分布:', dict(collections.Counter(r[1] for r in rows).most_common(12)), flush=True)
print('清单:', OUT / 'manifest.csv', flush=True)
