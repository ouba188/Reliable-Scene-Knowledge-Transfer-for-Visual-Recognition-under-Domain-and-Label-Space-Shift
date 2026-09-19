"""Audit and fill the WorldCover tiles actually needed by the 841 products.

Authoritative rule = the one the annotation bundle itself uses (prepare_assets.py):
the Sentinel-1 annotation geolocation grid boundary (iw-vv.xml) expanded by 1 km.
Availability is checked against the ESA bucket's own listing, so a tile that ESA does
not publish (ocean-only blocks) is reported, never silently re-requested.

Writes worldcover_per_product.csv (product_id, needed, have, missing) + worldcover_audit.json.
"""
import csv, json, math, sys
from pathlib import Path
from shapely.geometry import Polygon  # noqa: F401  (kept for parity with the bundle helper)

BUNDLE = Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle')
PRODUCTS = BUNDLE / 'assets' / 'products'
BUNDLE_WC = BUNDLE / 'assets' / 'worldcover'
MINE = Path(r'E:/临时会话/knowledge_set_841/worldcover')
OUT = Path(r'E:/临时会话/knowledge_set_841')
INDEX = MINE / 'bucket_index.txt'
sys.path.insert(0, str(BUNDLE / 'vendor'))
from finalize_safe841_ais_spatial_20260917 import grid_boundary  # noqa: E402


def tiles_for(boundary):
    """Same rule as annotation_bundle/prepare_assets.py: footprint bounds + 1 km margin."""
    west, south, east, north = boundary.bounds
    dy = 1000 / 110000
    dx = dy / max(.01, math.cos(math.radians(max(abs(south), abs(north)) + dy)))
    names = set()
    for lat in range(math.floor((south - dy) / 3) * 3, math.floor((north + dy) / 3) * 3 + 1, 3):
        for lon in range(math.floor((west - dx) / 3) * 3, math.floor((east + dx) / 3) * 3 + 1, 3):
            names.add('ESA_WorldCover_10m_2021_v200_%s%02d%s%03d_Map.tif' % (
                'N' if lat >= 0 else 'S', abs(lat), 'E' if lon >= 0 else 'W', abs(lon)))
    return names


def main():
    have = {p.name for p in BUNDLE_WC.glob('*.tif')} | {p.name for p in MINE.glob('*.tif')}
    bucket = set()
    if INDEX.is_file():
        bucket = {l.strip() for l in INDEX.read_text(encoding='utf-8').splitlines() if l.strip()}
    rows, needed_all, missing_all = [], set(), set()
    for pdir in sorted(p for p in PRODUCTS.iterdir() if p.is_dir()):
        xml = pdir / 'iw-vv.xml'
        if not xml.is_file():
            rows.append(dict(product_id=pdir.name, needed=0, have=0, missing='no_annotation_xml'))
            continue
        need = tiles_for(grid_boundary(xml))
        lack = need - have
        needed_all |= need
        missing_all |= lack
        rows.append(dict(product_id=pdir.name, needed=len(need), have=len(need - lack),
                         missing=';'.join(sorted(lack))))
    with (OUT / 'worldcover_per_product.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['product_id', 'needed', 'have', 'missing'])
        w.writeheader()
        w.writerows(rows)
    fetchable = sorted(m for m in missing_all if not bucket or m in bucket)
    unavailable = sorted(m for m in missing_all if bucket and m not in bucket)
    audit = dict(products=len(rows), needed_tiles=len(needed_all), local_tiles=len(needed_all & have),
                 missing_tiles=len(missing_all), missing=sorted(missing_all),
                 missing_fetchable=fetchable, unavailable_in_bucket=unavailable,
                 products_with_gaps=sum(1 for r in rows if r['missing'] and r['missing'] != 'no_annotation_xml'),
                 bucket_index_tiles=len(bucket))
    (OUT / 'worldcover_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if not isinstance(v, list)}, ensure_ascii=False))
    print('fetchable:', fetchable)
    print('unavailable:', unavailable)


if __name__ == '__main__':
    main()
