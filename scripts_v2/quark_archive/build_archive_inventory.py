"""Build /root/ann841_inventory.tsv using find (robust): product, port, {pol: tif}, {pol: labels}."""
import csv
import json
import re
import subprocess

PID = re.compile(r'(S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_[0-9A-F]{6}_[0-9A-F]{6}_[0-9A-F]{4})_([VH]{2})_UTM_8bit\.tif$')
ANN = '/root/autodl-tmp/safe841_batch/annotation_output'

port_of = {}
with open('/root/product_inventory_841.csv', encoding='utf-8-sig') as fh:
    for r in csv.DictReader(fh):
        port_of[r['product_id']] = r['port']
print('targets:', len(port_of))

r = subprocess.run(['find', '/root/autodl-tmp', '-name', '*_UTM_8bit.tif'], capture_output=True, text=True)
tifs, fallback = {}, {}
for t in r.stdout.splitlines():
    m = PID.search(t)
    if not m:
        continue
    d = fallback if t.startswith(ANN) else tifs
    d.setdefault(m.group(1), {})[m.group(2)] = t
print('tif 主位置: %d 产品 | 仅副本(annotation_output): %d 产品' % (len(tifs), len(set(fallback) - set(tifs))))

r = subprocess.run(['find', '/root/autodl-tmp', '-path', '*annotation_output*/status/*.json', '-name', '*.json'], capture_output=True, text=True)
labels = {}
ANN_ROOTS = ['/root/autodl-tmp/safe841_batch/annotation_output',
             '/root/autodl-tmp/safe841_batch/annotation_output_s3',
             '/root/autodl-fs/safe841_annotation_output']
r = subprocess.run(['find'] + ANN_ROOTS + ['-path', '*labels_json*', '-name', '*.json'], capture_output=True, text=True)
labels = {}
for p in r.stdout.splitlines():
    m = re.search(r'/(S1[A-Z0-9_]+)/[^/]+/([VH]{2})/labels_json/', p)
    if m:
        labels.setdefault(m.group(1), {})[m.group(2)] = p
print('labels 就位: %d 产品' % len(labels))

merged, no_tif, no_lab = {}, [], []
for product in sorted(port_of):
    t = dict(fallback.get(product) or {})
    t.update(tifs.get(product) or {})
    l = labels.get(product) or {}
    if not t:
        no_tif.append(product)
    if not l:
        no_lab.append(product)
    merged[product] = (port_of[product], t, l)

with open('/root/ann841_inventory.tsv', 'w', encoding='utf-8') as fh:
    for product, (port, t, l) in merged.items():
        fh.write('%s\t%s\t%s\t%s\n' % (product, port, json.dumps(t), json.dumps(l)))
print('wrote 841 rows')
print('缺 tif: %d %s' % (len(no_tif), no_tif[:2]))
print('缺 labels: %d %s' % (len(no_lab), no_lab[:2]))
