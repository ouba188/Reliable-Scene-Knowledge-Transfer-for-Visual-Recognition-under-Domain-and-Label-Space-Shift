"""nastar_ids2.py -- get the NASTaR sub-dataset ids WITH NAMES so they can be filtered.

The organisation listing returned 1000 packages (the whole university), and only ids were kept last time. This re-reads it keeping
name/title, greps for nastar, and also retries package_search now that the proxy is healthy.
"""
import json
import os
import urllib.request

PROXY = 'http://127.0.0.1:7890'
for v in ('NO_PROXY', 'no_proxy'):
    os.environ.pop(v, None)
BASE = 'https://data.bris.ac.uk/data/api/3/action/'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY}))


def gj(url, tries=3):
    for _ in range(tries):
        try:
            with opener.open(url, timeout=90) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:
            last = str(e)[:110]
    print('  ERR', url.split('?')[0].split('/')[-1], last)
    return None


hits = {}
s = gj(BASE + 'package_search?q=NASTaR&rows=1000')
if s and s.get('success'):
    for p in s['result'].get('results', []):
        hits[p['id']] = (p.get('name', ''), p.get('title', ''))
    print('package_search ✓ 命中 %d（总 %s）' % (len(hits), s['result'].get('count')))
else:
    print('package_search 仍不可用，退到组织列表')
    o = gj(BASE + 'organization_show?id=1cb7125e-9c7e-43ef-8619-c3517be622e8&include_datasets=true')
    pk = (o or {}).get('result', {}).get('packages', [])
    print('组织包数', len(pk))
    for p in pk:
        nm = (p.get('name') or '') + ' ' + (p.get('title') or '')
        if 'nastar' in nm.lower():
            hits[p['id']] = (p.get('name', ''), p.get('title', ''))

print('NASTaR 相关包: %d' % len(hits))
kinds = {}
for i, (nm, ti) in list(hits.items())[:6]:
    print('  %-38s %s' % (nm[:38], ti[-58:]))
for i, (nm, ti) in hits.items():
    for k in ('ship_patches_uint8', 'ship_patches', 'wake_patches_uint8_512x512', 'wake_patches_uint8', 'wake_patches_512x512', 'wake_patches'):
        if k in ti.lower() or k in nm.lower():
            kinds.setdefault(k, []).append(i)
            break
print('分类计数:', {k: len(v) for k, v in kinds.items()})
with open('nastar_ids2.json', 'w') as f:
    json.dump({'hits': hits, 'kinds': kinds}, f)
print('已写 nastar_ids2.json')
