"""fetch_gshhg.py -- get a usable coastline archive, verifying CONTENT not exit status.

The first attempt "succeeded" with 6.4 KB of HTML: NOAA's latest/ URL is dead, and a shell loop that breaks on curl's exit code
accepts an error page. So: try several mirrors, require the zip magic (PK) and a sane size, and report the HTTP code each time.

GSHHG h (~1:1M, roughly 100 m) is chosen over Natural Earth 10m (~1 km) because the threshold being tested is 2 x LOA, i.e.
tens to a few hundred metres, where a 1 km coastline is too coarse.
"""
import os
import urllib.request
from pathlib import Path

OUT = Path(r'E:/gshhg')
OUT.mkdir(parents=True, exist_ok=True)
PROXY = 'http://127.0.0.1:7890'
for v in ('NO_PROXY', 'no_proxy'):
    os.environ.pop(v, None)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY}))

URLS = [
    'https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip',
    'https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/gshhg-shp-2.3.7.zip',
    'https://github.com/GenericMappingTools/gshhg-gmt/releases/download/2.3.7/gshhg-shp-2.3.7.zip',
    'https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip',
    'https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip',
]
for u in URLS:
    dst = OUT / (u.rsplit('/', 1)[-1])
    try:
        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
        with opener.open(req, timeout=120) as r:
            code = r.status
            total = int(r.headers.get('Content-Length') or 0)
            with dst.open('wb') as f:
                done = 0
                while True:
                    ch = r.read(1 << 20)
                    if not ch:
                        break
                    f.write(ch)
                    done += len(ch)
        head = dst.open('rb').read(4)
        ok = head[:2] == b'PK' and done > 1_000_000
        print('%-70s http=%s bytes=%d magic=%s ⇒ %s' % (u[-66:], code, done, head[:2], '✓ 可用' if ok else '✗ 丢弃（非 zip）'), flush=True)
        if ok:
            print('已获得可用档案: %s (%.1f MB)' % (dst, dst.stat().st_size / 1e6), flush=True)
            break
        dst.unlink(missing_ok=True)
    except Exception as e:
        print('%-70s ERR %s' % (u[-66:], str(e)[:70]), flush=True)
