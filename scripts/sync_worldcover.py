"""Sync the ESA WorldCover tiles needed by the 841 products.

Tile naming is %s%02d%s%03d (N03E102, S33E150) -- the 3-digit latitude form 404s.
Availability is checked against the bucket's own listing, never inferred, so a
missing tile is reported as unavailable rather than silently re-requested.
"""
import json, sys, threading, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path

# ponytail: stdlib XML on the first-party AWS S3 listing endpoint only; no untrusted XML is parsed here.

OUT = Path(r'E:/临时会话/knowledge_set_841/worldcover')
BUNDLE_WC = Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle/assets/worldcover')
INDEX = OUT / 'bucket_index.txt'
COVERAGE = Path(r'E:/临时会话/knowledge_set_841/knowledge_coverage_841.csv')
BUCKET = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/?list-type=2&max-keys=1000&prefix=v200/2021/map/'
BASE = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/'
WORKERS = 6
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def tile_name(lat, lon):
    return 'ESA_WorldCover_10m_2021_v200_%s%02d%s%03d_Map.tif' % (
        'N' if lat >= 0 else 'S', abs(lat), 'E' if lon >= 0 else 'W', abs(lon))


def bucket_index(force=False):
    if INDEX.is_file() and not force:
        return {l.strip() for l in INDEX.read_text(encoding='utf-8').splitlines() if l.strip()}
    names, token = [], ''
    while True:
        url = BUCKET + ('&continuation-token=' + urllib.parse.quote(token) if token else '')
        root = ET.fromstring(OP.open(urllib.request.Request(url, headers={'User-Agent': 'h/1.0'}), timeout=90).read())
        names += [k.text.split('/')[-1] for k in root.iter() if k.tag.endswith('Key')]
        token = next((e.text for e in root.iter() if e.tag.endswith('NextContinuationToken')), '')
        if not token:
            break
    INDEX.write_text('\n'.join(sorted(names)), encoding='utf-8')
    return set(names)


def needed_tiles():
    import csv
    need = set()
    with COVERAGE.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            try:
                s, w, n, e = (float(row['bbox_min_lat']), float(row['bbox_min_lon']),
                              float(row['bbox_max_lat']), float(row['bbox_max_lon']))
            except ValueError:
                continue
            for lat in range(int(s // 3) * 3, int(n // 3) * 3 + 1, 3):
                for lon in range(int(w // 3) * 3, int(e // 3) * 3 + 1, 3):
                    need.add(tile_name(lat, lon))
    return need


def download(name):
    dest = OUT / name
    for attempt in range(4):
        try:
            have = dest.stat().st_size if dest.exists() else 0
            headers = {'User-Agent': 'hermes-knowledge-set/1.0'}
            if have:
                headers['Range'] = 'bytes=%d-' % have
            resp = OP.open(urllib.request.Request(BASE + name, headers=headers), timeout=180)
            clen = int(resp.headers.get('Content-Length') or 0)
            resuming = bool(have) and resp.status == 206
            if not resuming:
                have = 0
            total = clen + have
            with dest.open('ab' if resuming else 'wb') as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if dest.stat().st_size < total:
                raise RuntimeError('short read')
            return True
        except Exception as exc:
            if attempt == 3:
                print('FAIL %s %s' % (name, repr(exc)[:100]), flush=True)
            time.sleep(4 * (attempt + 1))
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    force = '--refresh-index' in sys.argv
    have_bucket = bucket_index(force)
    need = needed_tiles()
    local = {p.name for p in OUT.glob('*.tif')} | {p.name for p in BUNDLE_WC.glob('*.tif')}
    missing = need - local
    absent = sorted(m for m in missing if m not in have_bucket)
    todo = sorted(m for m in missing if m in have_bucket)
    print('needed=%d local=%d missing=%d fetchable=%d unavailable=%d'
          % (len(need), len(need & local), len(missing), len(todo), len(absent)), flush=True)
    if absent:
        print('not in bucket (leave as gaps): %s' % absent, flush=True)
    q, lock, failed = list(todo), threading.Lock(), []

    def worker():
        while True:
            with lock:
                if not q:
                    return
                name = q.pop(0)
            if not download(name):
                failed.append(name)
            else:
                with lock:
                    print('ok %s (%.1f MB)' % (name, (OUT / name).stat().st_size / 1048576), flush=True)
    threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print('downloaded=%d failed=%d' % (len(todo) - len(failed), len(failed)), flush=True)
    (OUT / 'worldcover_sync.json').write_text(json.dumps(dict(
        bucket_tiles=len(have_bucket), needed=sorted(need), local=sorted(need & local),
        fetched=sorted(set(todo) - set(failed)), unavailable_in_bucket=absent, failed=failed,
        generated=time.strftime('%Y-%m-%dT%H:%M:%S%z')), ensure_ascii=False, indent=1), encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
