"""e108: what exactly still has to be downloaded -- the missing scenes, per port, matched by filename.

Findings that motivate this: the object table already holds detections for all 24 ports (141,880 usable objects in
the AIS-matched tier), but crops can only be made where the SCENE is local. F: covers only 8 ports' worth of
products, and for those ports the products are a DIFFERENT acquisition batch from the objects (Sydney: 23 object
products, 14 F: products, intersection ZERO). The Quark disk's tif_local_done/ holds 430 scenes covering 215 of the
328 needed (port, product) pairs -- including all 23 of Sydney's -- so the port expansion is data-available, it just
needs a download. The Quark HTTP API is unreliable for this (400/403/23006-style errors, per the skill), and the
skill's recommendation is the desktop client.

This writes the exact shopping list: per port, the products whose scenes are missing locally and present on the
Quark disk, as the filenames the client will show.
"""
import csv
import gzip
import json
import os
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

KS = Path(r'E:/临时会话/knowledge_set_841')
OBJ = KS / 'objects/objects_classed.csv.gz'
SCENES = Path(r'F:/SAR_0922')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/needed_scenes.csv')
BASE = 'https://drive-pc.quark.cn/1/clouddrive'
Q = 'pr=ucpro&fr=pc&uc_param_str='
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/138.0.0.0 Safari/537.36')

# the Quark scene inventory (tif_local_done, 430 files)
ck = open(os.path.join(os.environ['TEMP'], 'quark_cookie.txt'), encoding='utf-8').read().strip()
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
    os.environ.pop(k, None)
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))


def ls(fid, page=1, size=500):
    u = '%s/file/sort?%s&%s' % (BASE, Q, urllib.parse.urlencode(
        {'pdir_fid': fid, '_page': page, '_size': size, '_sort': 'file_type:asc,updated_at:desc'}))
    r = urllib.request.Request(u, headers={'Cookie': ck, 'Referer': 'https://pan.quark.cn/', 'User-Agent': UA})
    return (json.loads(urllib.request.urlopen(r, timeout=60).read().decode()).get('data') or {}).get('list') or []


quark = []
for pg in (1, 2):
    L = ls('4395fd23170c4be7af3841a1381d7f49', page=pg)
    quark += [(x.get('file_name'), int(x.get('size') or 0), x.get('fid')) for x in L]
    if len(L) < 500:
        break
print('夸克场景文件 %d 个' % len(quark), flush=True)
byprod = {}
for nm, sz, fid in quark:
    m = re.match(r'^(S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d+_[0-9A-F]+_[0-9A-F]+)_(VV|VH)', nm or '')
    if m:
        byprod[(m.group(1), m.group(2))] = (nm, sz, fid)

# objects per (port, product, pol) in the AIS-matched tier
need = defaultdict(int)
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        if (row.get('match_status') or '').strip() != 'unique_spatial_candidate':
            continue
        pj = (row.get('port') or '').strip()
        prod = row['object_id'].split('|')[0]
        pol = (row.get('polarization') or '').strip()
        if pj and prod and pol in ('VV', 'VH'):
            need[(pj, prod, pol)] += 1
print('对象侧需要的 (港,产品,极化) 组: %d' % len(need), flush=True)

rows, tot = [], 0
per_port = defaultdict(lambda: [0, 0])
for (pj, prod, pol), n in sorted(need.items()):
    local = (SCENES / pj / pol / ('%s_%s_UTM_8bit.tif' % (prod, pol))).is_file()
    q = byprod.get((prod, pol))
    per_port[pj][0] += 1
    if local:
        per_port[pj][1] += 1
        continue
    rows.append({'port': pj, 'product': prod, 'pol': pol, 'objects': n,
                 'quark_file': q[0] if q else '', 'quark_mb': round((q[1] / 1048576.0), 1) if q else 0,
                 'quark_fid': q[2] if q else ''})
    if q:
        tot += q[1]
with OUT.open('w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
have = [r for r in rows if r['quark_file']]
print('')
print('缺场景 (港,产品,极化) 组: %d ；其中夸克有货: %d   合计 %.1f GB' % (
    len(rows), len(have), sum(r['quark_mb'] for r in have) / 1024))
print('')
print('%-18s %8s %8s %8s' % ('port', '需要组', '本地有', '夸克可补'))
for pj in sorted(per_port, key=lambda x: -(per_port[x][0] - per_port[x][1])):
    need_n, loc_n = per_port[pj]
    miss = [r for r in have if r['port'] == pj]
    print('%-18s %8d %8d %8d' % (pj, need_n, loc_n, len(miss)))
print('')
print('清单 ->', OUT)
