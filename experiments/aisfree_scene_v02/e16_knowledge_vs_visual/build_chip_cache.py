"""Build a memory-mapped uint8 image cache for the 38,091 pairs (fine-tuning prerequisite).

Writes straight into .npy via memmap (no full-array rewrites). Resumable.
Output: chip_cache/images_uint8_{vv,vh}.npy  (38091, 1, 128, 128) uint8
Normalization is applied at training time, so raw 128x128 grayscale is stored.
"""
import csv, json, time
from pathlib import Path

import numpy as np
from PIL import Image
from numpy.lib.format import open_memmap

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/chip_cache')
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 128
CHUNK = 1000
SHAPE = None


def main():
    rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
    n = len(rows)
    shape = (n, 1, SIZE, SIZE)
    fvv, fvh = OUT / 'images_uint8_vv.npy', OUT / 'images_uint8_vh.npy'
    prog = OUT / 'progress.json'
    start = 0
    if prog.exists() and fvv.exists() and fvh.exists():
        d = json.loads(prog.read_text())
        if d.get('total') == n:
            start = int(d.get('done', 0))
    vv = open_memmap(fvv, mode='r+' if (start and fvv.exists()) else 'w+', dtype=np.uint8, shape=shape)
    vh = open_memmap(fvh, mode='r+' if (start and fvh.exists()) else 'w+', dtype=np.uint8, shape=shape)
    print('start', start, '/', n, flush=True)
    t0 = time.time(); miss = 0
    for i in range(start, n, CHUNK):
        for j, r in enumerate(rows[i:i + CHUNK]):
            try:
                a = Image.open(r['image_relpath_vv']).convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                b = Image.open(r['image_relpath_vh']).convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                vv[i + j, 0] = np.asarray(a, np.uint8)
                vh[i + j, 0] = np.asarray(b, np.uint8)
            except Exception:
                miss += 1
        k = min(i + CHUNK, n)
        vv.flush(); vh.flush()
        prog.write_text(json.dumps({'total': n, 'done': k, 'missing': miss}))
        el = time.time() - t0
        print('%d/%d  %.0fs  %.1f it/s  miss=%d' % (k, n, el, (k - start) / max(1e-9, el), miss), flush=True)
    vv.flush(); vh.flush()
    print('DONE', shape, 'missing', miss, flush=True)


if __name__ == '__main__':
    main()
