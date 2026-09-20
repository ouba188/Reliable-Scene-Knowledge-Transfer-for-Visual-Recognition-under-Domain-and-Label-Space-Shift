"""Phase 0 fill A/B/D for the cross-port UDA plan: acquisition splits, per-port class support, geo write-back.

A  splits/port_acquisition_split.csv + lopo_folds.json + class_sets_per_fold.json
B  objects/support_by_port_class.csv   (labeled objects per port x class, AIS-derived)
D  geo/<port>.geojson                  (every scene box written back as lon/lat polygon)

Rules (documented in splits/README.md):
  * one product = one acquisition; adapt = oldest half by date, eval = newest half (temporal, per port)
  * LOPO: target port held out entirely; other ports are sources, each keeping its own adapt/eval
  * a class counts as supported in a port when it has >= MIN_OBJ labeled objects there
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from osgeo import gdal, osr

gdal.UseExceptions()

F = Path('F:/SAR_AIS_T')
KB = Path('E:/临时会话/knowledge_set_841')
SPLITS = KB / 'splits'
GEO = KB / 'geo'
MIN_OBJ = 20
DATE_RE = re.compile(r'_(\d{8})T(\d{6})_')

scenes = sorted(F.glob('*/utm_out/*/int8/*_labels_scene.json'))
print('scenes: %d' % len(scenes), flush=True)

# ---------------- B: per-port class support + reuse of object stats ----------------
rows = []
support = Counter()
port_objs = Counter()
scene_meta = []
for p in scenes:
    d = json.loads(p.read_text(encoding='utf-8'))
    parts = p.parts
    port = parts[parts.index('SAR_AIS_T') + 1]
    product, pol = d['product'], d['polarization']
    m = DATE_RE.search(product)
    date = '%s-%s-%s' % (m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:8]) if m else ''
    scene_meta.append(dict(port=port, product=product, pol=pol, date=date, path=str(p), objects=len(d['objects'])))
    for o in d['objects']:
        meta = o.get('meta') or {}
        cls = (meta.get('prelabel_class') or 'ship_untyped').strip() or 'ship_untyped'
        support[(port, cls)] += 1
        port_objs[port] += 1
        if meta.get('matched_mmsi'):
            support[(port, cls + '|mmsi_matched')] += 1

with (KB / 'objects' / 'support_by_port_class.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['port', 'class', 'objects', 'share_of_port', 'objects_with_matched_mmsi'])
    for (port, cls), n in sorted(support.items()):
        if cls.endswith('|mmsi_matched'):
            continue
        w.writerow([port, cls, n, round(n / max(1, port_objs[port]), 4), support.get((port, cls + '|mmsi_matched'), 0)])
print('B: support table written; ports=%d classes=%d' % (
    len(port_objs), len({c for _, c in support if not c.endswith('|mmsi_matched')})), flush=True)

# ---------------- A: acquisition splits, LOPO folds, per-fold class sets ----------------
port_classes = defaultdict(lambda: Counter())
for (port, cls), n in support.items():
    if not cls.endswith('|mmsi_matched'):
        port_classes[port][cls] = n

by_port = defaultdict(list)
for s in scene_meta:
    by_port[s['port']].append(s)

acq = {}
for port, ms in by_port.items():
    prods = sorted({(m['date'], m['product']) for m in ms})
    half = max(1, len(prods) // 2)
    adapt = {p for _, p in prods[:half]}
    eval_ = {p for _, p in prods[half:]}
    acq[port] = dict(adapt=sorted(adapt), eval=sorted(eval_), n_products=len(prods))
    print('  %-16s products=%2d  adapt=%2d eval=%2d' % (port, len(prods), len(adapt), len(eval_)), flush=True)

(SPLITS).mkdir(parents=True, exist_ok=True)
with (SPLITS / 'port_acquisition_split.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['port', 'product', 'pol', 'date', 'role', 'objects'])
    for s in scene_meta:
        role = 'adapt' if s['product'] in acq[s['port']]['adapt'] else 'eval'
        w.writerow([s['port'], s['product'], s['pol'], s['date'], role, s['objects']])

ports = sorted(acq)
folds = {}
for tgt in ports:
    srcs = [p for p in ports if p != tgt]
    src_pool = Counter()
    for p in srcs:
        for c, n in port_classes[p].items():
            if c != 'ship_untyped':
                src_pool[c] += n
    tgt_pool = {c: n for c, n in port_classes[tgt].items() if c != 'ship_untyped'}
    shared = sorted(c for c, n in src_pool.items() if n >= MIN_OBJ and tgt_pool.get(c, 0) >= MIN_OBJ)
    src_priv = sorted(c for c, n in src_pool.items() if n >= MIN_OBJ and tgt_pool.get(c, 0) < MIN_OBJ)
    tgt_priv = sorted(c for c, n in tgt_pool.items() if n >= MIN_OBJ and src_pool.get(c, 0) < MIN_OBJ)
    folds[tgt] = dict(sources=srcs, n_sources=len(srcs), target_products=acq[tgt]['n_products'],
                      shared=shared, source_private=src_priv, target_private=tgt_priv,
                      setting=('universal' if src_priv and tgt_priv else
                               'partial' if src_priv else 'open' if tgt_priv else 'closed'))

(SPLITS / 'lopo_folds.json').write_text(json.dumps(dict(rule='target port held out entirely; sources keep adapt/eval',
                                                        min_objects_per_class=MIN_OBJ, ports=acq, folds=folds),
                                                   ensure_ascii=False, indent=1), encoding='utf-8')
(SPLITS / 'class_sets_per_fold.json').write_text(json.dumps(
    {t: {k: v for k, v in f.items() if k in ('shared', 'source_private', 'target_private', 'setting')}
     for t, f in folds.items()}, ensure_ascii=False, indent=1), encoding='utf-8')
(SPLITS / 'README.md').write_text(
    '# 实验划分（Phase 0）\n\n'
    '- `port_acquisition_split.csv`：每个 (港口, 产品, 极化) 一行，`role=adapt|eval`；**一产品=一次采集**，按日期排序取前一半 adapt、后一半 eval（时间序，避免同采集泄漏）。\n'
    '- `lopo_folds.json`：外层留一港口；源港保持各自 adapt/eval（内层嵌套留一用于选超参）。\n'
    '- `class_sets_per_fold.json`：每折的 shared / source_private / target_private（判定阈值：该港该类带 AIS 类别标注的对象数 ≥ %d；`ship_untyped` 不计入类别空间）。\n'
    '- 类别来源：AIS 唯一匹配 + 细类解析（`mmsi_class_final.csv`），未匹配框一律 `ship_untyped`。\n'
    '- 口径提醒：小港（Busan/Tanger Med 1 个采集、Callao 少量目标）不适合单独做 LOPO 结论，建议作为附加目标港报告。\n' % MIN_OBJ,
    encoding='utf-8')
print('A: splits written', flush=True)

# ---------------- D: geo write-back per port ----------------
GEO.mkdir(parents=True, exist_ok=True)
wgs = osr.SpatialReference()
wgs.ImportFromEPSG(4326)
wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
per_port = defaultdict(list)
index_rows = []
for s in scene_meta:
    d = json.loads(Path(s['path']).read_text(encoding='utf-8'))
    raster = str(F / s['port'] / 'utm_out' / s['pol'] / 'int8' / d['image'])
    ds = gdal.Open(raster, gdal.GA_ReadOnly)
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    src = osr.SpatialReference(wkt=proj)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, wgs)
    feats, classes = 0, d.get('classes') or {}
    for o in d['objects']:
        pts = o['pts']
        lonlat = []
        for x, y in pts:
            lon, lat, _ = ct.TransformPoint(gt[0] + x * gt[1], gt[3] + y * gt[5])
            lonlat.append([round(lon, 6), round(lat, 6)])
        lonlat.append(lonlat[0])
        meta = o.get('meta') or {}
        per_port[s['port']].append(dict(type='Feature', geometry=dict(type='Polygon', coordinates=[lonlat]),
                                        properties=dict(product=s['product'], pol=s['pol'],
                                                        det=meta.get('detection_id'), cls=meta.get('prelabel_class'),
                                                        cls_id=o.get('cls_id'), conf=o.get('conf'),
                                                        mmsi=meta.get('matched_mmsi'),
                                                        coast=meta.get('coast_status'), shape=meta.get('shape_status'),
                                                        match=meta.get('match_status'), tile=meta.get('tile'))))
        feats += 1
    ds = None
    index_rows.append([s['port'], s['product'], s['pol'], feats, ','.join(sorted(classes))])
    print('  geo %-14s %s %s: %d boxes' % (s['port'], s['product'][-8:], s['pol'], feats), flush=True)

for port, feats in per_port.items():
    (GEO / ('%s.geojson' % port)).write_text(json.dumps(
        dict(type='FeatureCollection',
             crs=dict(type='name', properties=dict(name='urn:ogc:def:crs:OGC:1.3:CRS84')),
             features=feats), ensure_ascii=False), encoding='utf-8')
with (GEO / 'index.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['port', 'product', 'pol', 'objects', 'classes'])
    w.writerows(index_rows)
print('D: geo written; ports=%d boxes=%d' % (len(per_port), sum(len(v) for v in per_port.values())), flush=True)
