"""quark_dl.py -- download small files from Quark (read-only usage), then report their headers.

The account has a download size limit (code 23018) for large files, so this is for the small ones -- the traffic
summaries here are 0.1-0.4 MB. Cookie from %TEMP%/quark_cookie.txt, proxies cleared, written as a file to keep the
search literals out of the shell's own command line.

Usage: python quark_dl.py <fid> <out_path> [<fid2> <out2> ...]
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = 'https://drive-pc.quark.cn/1/clouddrive'
Q = 'pr=ucpro&fr=pc&uc_param_str='
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/138.0.0.0 Safari/537.36')


def cookie():
    for p in (Path(os.environ.get('TEMP', '.')) / 'quark_cookie.txt',
              Path.home() / 'AppData/Local/Temp/quark_cookie.txt'):
        if p.is_file():
            return p.read_text(encoding='utf-8').strip()
    raise SystemExit('cookie not found')


def post(path, body):
    url = '%s/%s?%s' % (BASE, path, Q)
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={
        'Cookie': cookie(), 'Referer': 'https://pan.quark.cn/', 'User-Agent': UA,
        'Content-Type': 'application/json', 'Accept': 'application/json, text/plain, */*'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))


if __name__ == '__main__':
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
        os.environ.pop(k, None)
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    for i in range(0, len(args) - 1, 2):
        fid, out = args[i], Path(args[i + 1])
        d = post('file/download', {'fids': [fid]})
        url = ((d.get('data') or [{}])[0] or {}).get('download_url')
        if not url:
            print('取链失败 %s: %s' % (fid, json.dumps(d, ensure_ascii=False)[:160]))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Referer': 'https://pan.quark.cn/'})
        with urllib.request.urlopen(req, timeout=120) as r, out.open('wb') as fh:
            fh.write(r.read())
        print('下载 %s -> %s (%.2f MB)' % (fid, out, out.stat().st_size / 1048576))
        with out.open(encoding='utf-8', errors='ignore') as fh:
            head = [next(fh, '') for _ in range(3)]
        for h in head:
            print('   ', h.strip()[:300])
