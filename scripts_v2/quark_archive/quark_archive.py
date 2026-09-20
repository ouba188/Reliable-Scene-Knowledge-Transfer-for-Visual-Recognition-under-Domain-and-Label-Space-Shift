"""Archive the 841 dataset to the Quark netdisk (tif + png + labels), resumable, parallel.

Layout: tif/<port>/<pol>/<name>  png/<port>/<pol>/<name>.png  labels/<port>/<pol>/<name>.json
State:  /root/quark_archive_state.json       PNG scratch: /root/png_tmp (deleted right after upload)
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, '/root')
import quark_upload as qu  # noqa: E402

STATE = Path('/root/quark_archive_state.json')
PNG_TMP = Path('/root/png_tmp')
LOCK = threading.Lock()
state = {}
dir_cache = {}


def save_state():
    with LOCK:
        tmp = STATE.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding='utf-8')
        tmp.replace(STATE)


def mkdir(name, pdir='0'):
    key = (pdir, name)
    if key in dir_cache:
        return dir_cache[key]
    st, d = qu.api('/file', {'pdir_fid': pdir, 'file_name': name, 'dir_path': '', 'dir_init_lock': False})
    fid = d['data']['fid'] if d.get('code') == 0 else None
    if fid is None:
        st2, listing = qu.api('/file/sort', None, method='GET',
                              params={'pdir_fid': pdir, 'force': 0, '_page': 1, '_size': 200, '_sort': 'file_type:asc'})
        for f in (listing.get('data') or {}).get('list', []):
            if f.get('file_name') == name and f.get('dir'):
                fid = f['fid']
    if fid is None:
        raise RuntimeError('mkdir %s failed: %s' % (name, json.dumps(d, ensure_ascii=False)[:200]))
    dir_cache[key] = fid
    return fid


def target_dir(root, port, pol):
    root_fid = dir_cache.get(('ROOT', root)) or mkdir(root, '0')
    dir_cache[('ROOT', root)] = root_fid
    port_fid = mkdir(port, root_fid)
    return mkdir(pol, port_fid)


def to_png(pair):
    tif, out = pair
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(tif)
        im.save(out, 'PNG', compress_level=3)
        return out, True, os.path.getsize(out)
    except Exception as e:
        return out, False, str(e)[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inventory', default='/root/ann841_inventory.tsv')
    ap.add_argument('--root', default='SAR_AIS_T_841_20260920')
    ap.add_argument('--converters', type=int, default=6)
    ap.add_argument('--uploaders', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--part-mb', type=int, default=32, help='OSS part size (MB); fewer parts = fewer auth round-trips')
    ap.add_argument('--skip-png', action='store_true', help='upload tif + labels only (PNG later)')
    a = ap.parse_args()

    global state
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding='utf-8'))
    PNG_TMP.mkdir(exist_ok=True)

    rows = []
    with open(a.inventory, encoding='utf-8') as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4:
                rows.append({'product': p[0], 'port': p[1],
                             'tifs': json.loads(p[2]), 'labels': json.loads(p[3])})
    if a.limit:
        rows = rows[:a.limit]

    fails, stats = [], {'files': 0, 'bytes': 0}
    t0 = time.time()

    def note(product, key, ok, size, err=''):
        with LOCK:
            if ok:
                state.setdefault(product, {})[key] = 'done'
                stats['files'] += 1
                stats['bytes'] += size
            else:
                fails.append((key, err))

    def do_upload(product, root, port, pol, path, key):
        size = os.path.getsize(path)
        try:
            pdir = target_dir(root, port, pol)
            qu.upload(path, pdir, part_size=a.part_mb << 20, log=lambda s: None)
            note(product, key, True, size)
            return True
        except Exception as e:
            note(product, key, False, size, repr(e)[:180])
            return False

    # ---------------- phase 1: tifs + labels ----------------
    queue = []
    for r in rows:
        st = state.setdefault(r['product'], {})
        for pol, tif in (r['tifs'] or {}).items():
            if pol in ('VV', 'VH') and st.get('tif/' + os.path.basename(tif)) != 'done':
                queue.append((r['product'], 'tif', r['port'], pol, tif))
        for pol, lab in (r['labels'] or {}).items():
            if st.get('lab/' + os.path.basename(lab)) != 'done':
                queue.append((r['product'], 'labels', r['port'], pol, lab))
    print('[P1] products=%d artifacts=%d uploaders=%d' % (len(rows), len(queue), a.uploaders), flush=True)

    def p1_work(item):
        product, root, port, pol, path = item
        if not os.path.isfile(path):
            note(product, '%s/%s' % (root, os.path.basename(path)), False, 0, 'missing local file')
            return
        do_upload(product, root, port, pol, path, '%s/%s' % (root, os.path.basename(path)))

    with ThreadPoolExecutor(max_workers=a.uploaders) as pool:
        chunk = 20
        for i in range(0, len(queue), chunk):
            list(pool.map(p1_work, queue[i:i + chunk]))
            el = time.time() - t0
            print('[P1] %d/%d files  %.2f GB  %.1f MB/s  fails=%d' % (
                stats['files'], len(queue), stats['bytes'] / 2**30, stats['bytes'] / 2**20 / max(el, .1), len(fails)), flush=True)
            save_state()

    # ---------------- phase 2: png (convert -> upload -> delete) ----------------
    if a.skip_png:
        save_state()
        print('ALL DONE (skip-png) files=%d bytes=%.2f GB fails=%d elapsed=%.1f h' % (
            stats['files'], stats['bytes'] / 2**30, len(fails), (time.time() - t0) / 3600), flush=True)
        for f in fails[:20]:
            print('   FAIL', f)
        return
    tasks = []
    for r in rows:
        st = state.setdefault(r['product'], {})
        for pol, tif in (r['tifs'] or {}).items():
            if pol not in ('VV', 'VH') or not os.path.isfile(tif):
                continue
            png = str(PNG_TMP / (os.path.splitext(os.path.basename(tif))[0] + '.png'))
            if st.get('png/' + os.path.basename(png)) != 'done':
                tasks.append((r['product'], r['port'], pol, tif, png))
    print('[P2] png tasks=%d converters=%d uploaders=%d' % (len(tasks), a.converters, a.uploaders), flush=True)
    t1 = time.time()
    png_bytes = [0]
    with ProcessPoolExecutor(max_workers=a.converters) as conv, ThreadPoolExecutor(max_workers=a.uploaders) as up:
        batch = 40
        for i in range(0, len(tasks), batch):
            group = tasks[i:i + batch]
            results = list(conv.map(to_png, [(t[3], t[4]) for t in group]))
            ups = []
            for (product, port, pol, tif, png), (out, ok, info) in zip(group, results):
                if not ok:
                    fails.append((os.path.basename(png), 'png:' + info))
                    continue
                ups.append((product, port, pol, out, png, info))
            def p2_work(t):
                product, port, pol, out, png, size = t
                try:
                    pdir = target_dir('png', port, pol)
                    qu.upload(out, pdir, part_size=a.part_mb << 20, log=lambda s: None)
                    note(product, 'png/' + os.path.basename(png), True, size)
                except Exception as e:
                    fails.append((os.path.basename(png), repr(e)[:180]))
                finally:
                    try:
                        os.remove(out)
                    except OSError:
                        pass
            list(up.map(p2_work, ups))
            png_bytes[0] = stats['bytes']
            el = time.time() - t1
            print('[P2] %d/%d png done  cumulative %.2f GB  %.1f MB/s  fails=%d' % (
                min(i + batch, len(tasks)), len(tasks), stats['bytes'] / 2**30, stats['bytes'] / 2**20 / max(time.time() - t0, .1), len(fails)), flush=True)
            save_state()

    save_state()
    print('ALL DONE files=%d bytes=%.2f GB fails=%d elapsed=%.1f h' % (
        stats['files'], stats['bytes'] / 2**30, len(fails), (time.time() - t0) / 3600), flush=True)
    for f in fails[:20]:
        print('   FAIL', f)


if __name__ == '__main__':
    main()
