"""AIS-derived traffic-direction field per port, used where OSM has no fairway geometry.

Ships travel along the fairways, so the AIS track directions ARE the channel layout. For each port
the AIS positions of all its scenes are binned into a grid (CELL_M), and cells with enough points
get a principal direction from PCA of successive-position differences. Output per port:
  traffic/<port>.csv  cell_lon, cell_lat, cell_x, cell_y (UTM), points, direction_deg, anisotropy
plus traffic/<port>.provenance.json.

Directions are axial (0-180) like OSM fairway lines, so the object table can compare either with
the OBB long axis. Anisotropy = ratio of the PCA eigenvalues: near 1 means no dominant direction.
"""
import csv, json, math, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

KS = Path(r'E:/临时会话/knowledge_set_841')
BUNDLE = Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle')
PRODUCTS = BUNDLE / 'assets' / 'products'
KEYS = KS / 'keys' / 'products.csv'
OUT = KS / 'traffic'
PORTS = json.loads(Path(r'E:/Hermes/scripts/out/ports_bbox.json').read_text(encoding='utf-8'))
CELL_M = 1000.0
MIN_POINTS = 8
MIN_STEP_M = 20.0


def main():
    from pyproj import Transformer
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:]) or None
    by_port = defaultdict(list)
    with KEYS.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            by_port[r['port']].append(r['product_id'])

    for port, pids in sorted(by_port.items()):
        if only and port not in only:
            continue
        west, south, east, north = PORTS[port]['bbox']
        lat0 = (south + north) / 2
        proj = Transformer.from_crs(4326, f'+proj=utm +zone={int((west + east) / 2 // 6) + 31} +datum=WGS84', always_xy=True)
        cells = defaultdict(list)
        rows = 0
        for pid in pids:
            for txt in sorted((PRODUCTS / pid / 'ais').glob('*.txt')):
                prev = {}
                with txt.open('r', encoding='utf-8', errors='replace', newline='') as f:
                    for line in f:
                        cols = line.rstrip('\n').split(',')
                        if len(cols) < 18:
                            continue
                        mmsi = cols[1].strip()
                        if not (re.fullmatch(r'\d{9}', mmsi) and mmsi != '000000000'):
                            continue
                        try:
                            lon, lat = float(cols[6]), float(cols[7])
                            t = int(cols[0])
                        except ValueError:
                            continue
                        if not (west <= lon <= east and south <= lat <= north):
                            continue
                        rows += 1
                        x, y = proj.transform(lon, lat)
                        key = mmsi
                        if key in prev:
                            px, py, pt = prev[key]
                            if 0 < (t - pt) <= 600:
                                dx, dy = x - px, y - py
                                if math.hypot(dx, dy) >= MIN_STEP_M:
                                    cx = int(x // CELL_M)
                                    cy = int(y // CELL_M)
                                    cells[(cx, cy)].append((x, y, dx, dy))
                        prev[key] = (x, y, t)
        if not cells:
            print('%-18s no AIS movement cells' % port, flush=True)
            continue
        out_rows = []
        for (cx, cy), pts in cells.items():
            if len(pts) < MIN_POINTS:
                continue
            arr = np.array([(p[2], p[3]) for p in pts])
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
            cov = np.cov(arr.T)
            vals, vecs = np.linalg.eigh(cov)
            v = vecs[:, -1]
            direction = math.degrees(math.atan2(v[1], v[0])) % 180.0
            anisotropy = float(vals[-1] / (vals[0] + 1e-12))
            x, y = (cx + 0.5) * CELL_M, (cy + 0.5) * CELL_M
            lon, lat = Transformer.from_crs(f'+proj=utm +zone={int((west + east) / 2 // 6) + 31} +datum=WGS84',
                                            4326, always_xy=True).transform(x, y)
            out_rows.append(dict(cell_x=round(x, 1), cell_y=round(y, 1), cell_lon=round(lon, 5),
                                 cell_lat=round(lat, 5), points=len(pts),
                                 direction_deg=round(direction, 2), anisotropy=round(anisotropy, 2)))
        out_rows.sort(key=lambda r: -r['points'])
        with (OUT / (port.replace(' ', '_') + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        (OUT / (port.replace(' ', '_') + '.provenance.json')).write_text(json.dumps(dict(
            port=port, source='AIS positions from the 841-product bundle (extraction window per scene)',
            method='successive same-MMSI steps >= %d m, PCA per %.0f m cell, >= %d steps' % (MIN_STEP_M, CELL_M, MIN_POINTS),
            semantics='empirical traffic direction (fairway proxy); axial 0-180 degrees',
            ais_rows_in_bbox=rows, cells=len(out_rows),
            generated=time.strftime('%Y-%m-%dT%H:%M:%S%z')), ensure_ascii=False, indent=1), encoding='utf-8')
        print('%-18s rows=%-9d cells=%-5d top_points=%d' % (port, rows, len(out_rows), out_rows[0]['points']), flush=True)


if __name__ == '__main__':
    main()
