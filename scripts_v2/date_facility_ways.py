"""Date OSM facility features so the "current snapshot" facility layer gains per-feature time evidence.

The extracts carry no version/timestamp (both Geofabrik and BBBike strip metadata), but the OSM API
does expose it per element. Only the screening-relevant features are dated: quay/pier/breakwater/
tank/silo/anchorage/harbour/dock/terminal ways and navigation lines, from
  * the backfilled facilities/<port>.geojson extracts (osm_id column), and
  * the local per-port layers_all.gpkg facility layers (element='way', id).
Results land in facilities/osm_way_dates.csv (osm_id, version, timestamp, fetched_at, status).
Then facility_temporal_valid per scene = timestamp <= scene date - i.e. the feature existed when the
scene was imaged (its geometry may have changed since, which is why the flag is "existed", not "identical").

Checkpointed and resumable: re-running skips osm_ids already in the CSV. Polite fixed concurrency,
exponential backoff on 429/5xx, never retries 404 (deleted upstream).
"""
import csv, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KS = Path(r'E:/临时会话/knowledge_set_841')
FACILITIES = KS / 'facilities'
DOCMS = Path(r'E:/Docms/Port')
RUNWIDE = Path(r'E:/Install_packs/port_osm_run/port_osm_output/run_wide/ports')
OUT = FACILITIES / 'osm_way_dates.csv'
API = 'https://api.openstreetmap.org/api/0.6/%s/%d.json'
UA = {'User-Agent': 'hermes-research-facility-dating/1.0 (offline SAR-AIS study; contact: local)'}
WORKERS = 10
SCREEN_KINDS = ('quay', 'pier', 'breakwater', 'groyne', 'jetty', 'dock', 'terminal', 'storage_tank',
                'silo', 'crane', 'anchorage', 'mooring', 'harbour', 'basin', 'berth')
lock = threading.Lock()
done = set()


def load_done():
    """Keys already in the CSV, as 'element:id'. Legacy rows predate the element column (= ways)."""
    if not OUT.is_file():
        return
    with OUT.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        legacy = 'element' not in (reader.fieldnames or [])
        for r in reader:
            if legacy:
                done.add('way:' + r['osm_id'])
                done.add(r['osm_id'])
            else:
                done.add((r.get('element') or 'way') + ':' + r['osm_id'])


def migrate_legacy_file():
    """One-off: give an old 5-column CSV the element column so appends stay aligned."""
    if not OUT.is_file():
        return
    with OUT.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else []
    if 'element' in fields:
        return
    tmp = OUT.with_suffix('.migrating.csv')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['osm_id', 'element', 'version', 'timestamp', 'fetched_at', 'status'])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, element='way'))
    tmp.replace(OUT)
    print('已迁移旧 CSV 到 element 列（%d 行）' % len(rows), flush=True)


def norm_id(v):
    """OSM element ids come out of the gpkg as floats ('558793732.0'); normalise to a plain int string."""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return None


def screening_osm_ids():
    """(osm_id -> source tag) for the features that matter for object screening."""
    import geopandas as gpd
    ids = {}
    for gj in sorted(FACILITIES.glob('*.geojson')):
        if gj.name.endswith('.coastline.geojson'):
            continue
        try:
            frame = gpd.read_file(gj, columns=['osm_id', 'facility_kind'])
        except Exception:
            try:
                frame = gpd.read_file(gj)
            except Exception as exc:
                print('read fail %s: %s' % (gj.name, repr(exc)[:60]), flush=True)
                continue
        if 'osm_id' not in frame.columns:
            continue
        wanted = frame[frame['facility_kind'].astype(str).str.contains('|'.join(SCREEN_KINDS), case=False)]
        for oid in wanted['osm_id'].dropna().unique():
            key = norm_id(oid)
            if key:
                ids['way:' + key] = 'pbf_' + gj.stem
    for root in (DOCMS, RUNWIDE):
        for base in sorted(root.glob('*')):
            gpkgs = sorted(base.rglob('layers*.gpkg')) if base.is_dir() else []
            if not gpkgs:
                continue
            for layer in ('infra_quay_pier_breakwater', 'seamark_anchorages', 'seamark_berth',
                          'infra_oil_tanks', 'infra_silos_conveyors', 'harbour_cat_tanker',
                          'harbour_cat_container', 'harbour_cat_bulk', 'osm_all', 'water_context',
                          'infra_cranes', 'seamark_harbour', 'coastline'):
                try:
                    frame = gpd.read_file(gpkgs[0], layer=layer, columns=['element', 'id'])
                except Exception:
                    continue
                if not len(frame) or 'id' not in frame.columns:
                    continue
                elems = (frame['element'].astype(str).str.lower() if 'element' in frame.columns
                         else pd.Series('way', index=frame.index))
                for elem, oid in zip(elems, frame['id']):
                    if elem not in ('way', 'node', 'relation'):
                        continue
                    key = norm_id(oid)
                    if key:
                        ids.setdefault(elem + ':' + key, 'gpkg_' + base.name)
    return ids


def fetch(key):
    import urllib.request
    elem, _, oid = key.partition(':')
    url = API % (elem, int(oid))
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                el = json.load(resp)['elements'][0]
            return dict(osm_id=oid, element=elem, version=el.get('version', ''), timestamp=el.get('timestamp', ''),
                        fetched_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='ok')
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return dict(osm_id=oid, element=elem, version='', timestamp='',
                            fetched_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='deleted_or_missing')
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return dict(osm_id=oid, element=elem, version='', timestamp='',
                        fetched_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='http_%d' % exc.code)
        except Exception:
            time.sleep(2 ** attempt)
    return dict(osm_id=oid, element=elem, version='', timestamp='',
                fetched_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='error')


def main():
    only = sys.argv[1:]
    load_done()
    ids_file = None
    if '--ids-file' in only:
        i = only.index('--ids-file')
        ids_file = Path(only[i + 1])
        only = only[:i] + only[i + 2:]
    if ids_file:
        # priority queue: the features the object table actually hits (usually ~10x smaller than the
        # full screening set), so the temporal column is complete long before the wide sweep ends.
        # Lines are 'element:id' or a bare id (a way).
        ids = {}
        for line in ids_file.read_text(encoding='utf-8').split('\n'):
            line = line.strip()
            if not line:
                continue
            elem, sep, oid = line.partition(':')
            if not sep:
                elem, oid = 'way', line
            key = norm_id(oid)
            if key:
                ids[elem + ':' + key] = 'object_hits'
    else:
        ids = screening_osm_ids()
    if only:
        ids = {k: v for k, v in ids.items() if any(t in v for t in only)}
    todo = [oid for oid in ids if oid not in done]
    print('候选 %d 条，已完成 %d，本次 %d 条' % (len(ids), len(ids) - len(todo), len(todo)), flush=True)
    if not todo:
        return
    migrate_legacy_file()
    new_file = not OUT.is_file()
    with OUT.open('a', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['osm_id', 'element', 'version', 'timestamp', 'fetched_at', 'status'])
        if new_file:
            w.writeheader()
        n = 0
        with ThreadPoolExecutor(WORKERS) as pool:
            for row in pool.map(fetch, todo, chunksize=1):
                w.writerow(row)
                n += 1
                if n % 200 == 0:
                    f.flush()
                    print('  %d/%d' % (n, len(todo)), flush=True)
    print('完成 %d 条 -> %s' % (n, OUT), flush=True)


if __name__ == '__main__':
    main()
