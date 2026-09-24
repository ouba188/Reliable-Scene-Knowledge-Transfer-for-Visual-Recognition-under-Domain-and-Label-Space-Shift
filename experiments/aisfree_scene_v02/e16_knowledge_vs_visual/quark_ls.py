"""quark_ls.py -- list a Quark netdisk directory (read-only), find files by name.

Cookie from %TEMP%/quark_cookie.txt (never echoed). Proxies cleared: the local HTTP proxy is usually not listening
and urllib would fail with WinError 10061. Written as a FILE so the search literals never enter the shell's own
command line (the self-match trap that bit twice tonight).

Usage: python quark_ls.py [fid|0] [name_substring]
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


def get(path, params):
    url = '%s/%s?%s&%s' % (BASE, path, Q, urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={
        'Cookie': cookie(), 'Referer': 'https://pan.quark.cn/', 'User-Agent': UA,
        'Accept': 'application/json, text/plain, */*'})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode('utf-8'))


def ls(fid='0', page=1, size=500):
    d = get('file/sort', {'pdir_fid': fid, '_page': page, '_size': size,
                          '_sort': 'file_type:asc,updated_at:desc'})
    return (d.get('data') or {}).get('list') or []


def walk(fid='0', depth=0, maxdepth=3):
    out = []
    for it in ls(fid):
        out.append((depth, it.get('file_name'), it.get('dir'), int(it.get('size') or 0), it.get('fid')))
        if it.get('dir') and depth < maxdepth:
            out += walk(it.get('fid'), depth + 1, maxdepth)
    return out


if __name__ == '__main__':
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
        os.environ.pop(k, None)
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

    fid = sys.argv[1] if len(sys.argv) > 1 else '0'
    pat = (sys.argv[2] if len(sys.argv) > 2 else '').lower()
    if pat:
        rows = walk(fid, 0, 3)
        hit = [r for r in rows if pat in (r[1] or '').lower()]
        print('匹配 %d / 共 %d 项' % (len(hit), len(rows)))
        for d, nm, isd, sz, f_ in hit[:40]:
            print('  %s%-50s %s %8.1f MB  fid=%s' % ('  ' * d, nm, 'DIR ' if isd else 'FILE', sz / 1048576, f_))
    else:
        items = ls(fid)
        print('目录 %s 共 %d 项（前 40）' % (fid, len(items)))
        for it in items[:40]:
            print('  %-52s %s %8.1f MB  fid=%s' % (it.get('file_name'), 'DIR ' if it.get('dir') else 'FILE',
                                                   int(it.get('size') or 0) / 1048576, it.get('fid')))
