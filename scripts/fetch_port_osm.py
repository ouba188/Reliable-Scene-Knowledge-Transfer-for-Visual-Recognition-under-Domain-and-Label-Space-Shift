"""Fetch per-port OSM facility geometry (Overpass raw JSON + dated provenance).

Feeds annotate_product.py --osm/--provenance and safe841_facility_review.load_features,
which expects raw Overpass JSON: elements[].type == 'way' with geometry + tags.

Snapshot policy: the *current* OSM snapshot is used, and provenance records that it is not
contemporaneous with the 2025 SAR scenes, so the pipeline marks facility overlaps 'pending'
(human review) rather than auto-excluding them. Attic (scene-date) snapshots were tried and
returned near-empty results on the public endpoints -- do not claim scene-date coverage.
"""
import json, time, sys, urllib.error, urllib.parse, urllib.request
from pathlib import Path

OUT = Path(r'E:/临时会话/knowledge_set_841/osm')
ENDPOINTS = ['https://overpass-api.de/api/interpreter',
             'https://overpass.kumi.systems/api/interpreter',
             'https://overpass.private.coffee/api/interpreter',
             'https://overpass.osm.jp/api/interpreter']
PORTS = json.loads(Path(r'E:/Hermes/scripts/out/ports_bbox.json').read_text(encoding='utf-8'))
TAGS = ('way["man_made"~"^(quay|pier|breakwater|groyne|jetty|dock|storage_tank|crane)$"]{B};'
        'way["harbour"]{B};way["waterway"="dock"]{B};way["natural"="coastline"]{B};'
        'way["amenity"="ferry_terminal"]{B};way["landuse"="harbour"]{B};way["industrial"="port"]{B};'
        'way["seamark:type"~"^(anchorage|berth|quay|harbour_basin|mooring|crane|terminal|dry_dock|pier|'
        'small_craft_facility|turning_basin|harbour)$"]{B};')


def build_query(bbox, date=None):
    s, w, n, e = bbox
    b = '(%f,%f,%f,%f)' % (s, w, n, e)
    head = '[out:json][timeout:300]' + ('[date:"%s"]' % date if date else '')
    return head + ';(' + TAGS.replace('{B}', b) + ');out geom;'


def fetch(query, tries=5):
    """Rotate endpoints, back off on rate limits. Returns (payload, endpoint) or raises."""
    errors = []
    for attempt in range(tries):
        ep = ENDPOINTS[attempt % len(ENDPOINTS)]
        data = urllib.parse.urlencode({'data': query}).encode()
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # direct: local proxy ports are dead
        try:
            req = urllib.request.Request(ep, data=data, headers={'User-Agent': 'hermes-knowledge-set/1.0'})
            payload = json.loads(op.open(req, timeout=420).read())
            if payload.get('remark'):
                raise RuntimeError('overpass remark: ' + str(payload['remark'])[:200])
            return payload, ep
        except urllib.error.HTTPError as exc:
            errors.append('%s: HTTP %s' % (ep.split('/')[2], exc.code))
            time.sleep(90 if exc.code == 429 else 10 * (attempt + 1))
        except Exception as exc:
            errors.append('%s: %s' % (ep.split('/')[2], repr(exc)[:120]))
            time.sleep(10 * (attempt + 1))
    raise RuntimeError('all endpoints failed: ' + ' | '.join(errors))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    failures = []
    for port, info in sorted(PORTS.items()):
        if only and port not in only:
            continue
        raw_path, prov_path = OUT / (port + '.json'), OUT / (port + '.provenance.json')
        if raw_path.is_file() and raw_path.stat().st_size > 200:
            print('skip %s (exists)' % port, flush=True)
            continue
        try:
            payload, ep = fetch(build_query(info['bbox']))
        except RuntimeError as exc:
            failures.append((port, str(exc)[:200]))
            print('FAIL %s %s' % (port, exc), flush=True)
            continue
        ways = [e for e in payload.get('elements', []) if e.get('type') == 'way' and e.get('geometry')]
        if not ways:
            # an empty result for a major port is far more likely a fetch problem than a real absence
            failures.append((port, 'zero ways returned; not writing an empty facility map'))
            print('EMPTY %s (no file written)' % port, flush=True)
            time.sleep(20)
            continue
        raw_path.write_text(json.dumps(payload), encoding='utf-8')
        prov_path.write_text(json.dumps(dict(
            source='OpenStreetMap via Overpass API', endpoint=ep, license='ODbL 1.0',
            snapshot_requested=time.strftime('%Y-%m-%d'),
            snapshot_mode='current',
            retrieved_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            relation_to_scene=('current map snapshot; NOT contemporaneous with the 2025 SAR scenes '
                               '-> facility overlaps are review-pending, never auto-excluded'),
            facility_map_temporal_match=False,
            bbox=info['bbox'], ports=[port], way_count=len(ways),
            scene_dates=info.get('scene_dates') or [],
            query=build_query(info['bbox'])),
            ensure_ascii=False, indent=1), encoding='utf-8')
        print('%s ok %d ways' % (port, len(ways)), flush=True)
        time.sleep(20)  # be polite with the public Overpass instance
    print('failures: %d' % len(failures), flush=True)
    for f in failures:
        print('  ', f, flush=True)


if __name__ == '__main__':
    main()
