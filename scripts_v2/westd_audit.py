"""Read-only audit: is everything the 0917 KIRC experiment needs actually on westd?

Checks, against the local 841-product inventory:
  rasters   *_UTM_8bit.tif present per product
  labels    *_UTM_8bit.json under */labels_json/ present per product
  knowledge objects_final.csv.gz + d_p + traffic + dating + facilities
  env       python/torch/gdal/sklearn/numpy/pandas availability
  code      any 0917 implementation, and the design docs
  disk      headroom

  WESTD_PW=... python westd_audit.py
"""
import csv
import os
from collections import Counter
from pathlib import Path

import paramiko

INV = Path(r'E:/临时会话/safe841_redownload_plan_20260918/product_inventory_841.csv')
RMS = r'''
python3 - <<'PY'
import json, os
from collections import Counter

B = '/root/autodl-tmp'
prods_tif, prods_lab = set(), set()
tifs = labels = 0
for root, dirs, files in os.walk(B):
    if root.startswith(B + '/knowledge_841_20260920'):
        continue
    for f in files:
        if f.endswith('_UTM_8bit.tif'):
            tifs += 1
            prods_tif.add(f.split('_UTM_8bit.tif')[0].rsplit('_', 1)[0])
        elif f.endswith('_UTM_8bit.json') and '/labels_json/' in root.replace('\\', '/'):
            labels += 1
            prods_lab.add(f.split('_UTM_8bit.json')[0].rsplit('_', 1)[0])
print('TIFS %d' % tifs)
print('LABELS %d' % labels)
print('TIF_PRODUCTS %s' % json.dumps(sorted(prods_tif)))
print('LABEL_PRODUCTS %s' % json.dumps(sorted(prods_lab)))
print('KNOWLEDGE %s' % json.dumps(sorted(os.listdir(os.path.join(B, 'knowledge_841_20260920')))))
kb = os.path.join(B, 'knowledge_841_20260920')
for sub in ('objects', 'ports', 'traffic', 'facilities', 'mmsi', 'keys'):
    p = os.path.join(kb, sub)
    if os.path.isdir(p):
        print('KB_%s %d' % (sub, len(os.listdir(p))))
PY
df -h /root/autodl-tmp | tail -1
python3 -c "import sys;print('PYVER',sys.version.split()[0])"
python3 -c "
mods=['torch','torchvision','numpy','pandas','sklearn','scipy','rasterio','geopandas','shapely','yaml','tqdm']
ok=[];miss=[]
for m in mods:
    try:
        __import__(m); ok.append(m)
    except Exception: miss.append(m)
print('PKG_OK', ','.join(ok)); print('PKG_MISS', ','.join(miss))
" 2>&1 | tail -3
python3 -c "import torch;print('TORCH', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1 | tail -1
ls /root/*.py /root/*.md /root/autodl-tmp/*.md 2>/dev/null | head -12
'''


def main():
    pw = os.environ.get('WESTD_PW')
    if not pw:
        raise SystemExit('WESTD_PW 未设置')
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=16575, username='root', password=pw,
              timeout=40, banner_timeout=60, auth_timeout=60)
    _, out, _ = c.exec_command(RMS, timeout=600)
    text = out.read().decode('utf-8', 'replace')
    c.close()

    data = {}
    for line in text.splitlines():
        if line.startswith(('TIF_PRODUCTS ', 'LABEL_PRODUCTS ', 'KNOWLEDGE ', 'PKG_', 'TORCH', 'PYVER')) or line.startswith('KB_'):
            k, _, v = line.partition(' ')
            data[k] = v
        print(line)

    inv = [r['product_id'] for r in csv.DictReader(INV.open('r', encoding='utf-8-sig', newline=''))]
    inv = [p.strip() for p in inv if p.strip()]
    import json
    tif_p = set(json.loads(data.get('TIF_PRODUCTS', '[]')))
    lab_p = set(json.loads(data.get('LABEL_PRODUCTS', '[]')))
    inv_s = set(inv)
    print('\n=== 对账（841 清单）===')
    print('841 产品数: %d' % len(inv_s))
    print('有大图: %d (缺 %d)' % (len(inv_s & tif_p), len(inv_s - tif_p)))
    print('有标注: %d (缺 %d)' % (len(inv_s & lab_p), len(inv_s - lab_p)))
    print('大图与标注都有: %d' % len(inv_s & tif_p & lab_p))
    miss_lab = sorted(inv_s - lab_p)
    print('缺标注样例:', miss_lab[:10])
    Path(r'E:/Hermes/scripts/out/server_missing_labels.txt').write_text('\n'.join(miss_lab), encoding='utf-8')
    Path(r'E:/Hermes/scripts/out/server_missing_rasters.txt').write_text('\n'.join(sorted(inv_s - tif_p)), encoding='utf-8')


if __name__ == '__main__':
    main()
