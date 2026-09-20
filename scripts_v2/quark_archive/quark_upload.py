"""Quark netdisk uploader (alist quark_uc flow): pre -> hash -> part PUT -> commit -> finish."""
import base64
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

T = os.path.join(os.environ.get('LOCALAPPDATA') or '/root', 'Temp')
_c = os.path.join(T, 'quark_cookie.txt')
COOKIE = open(_c if os.path.exists(_c) else '/root/quark_cookie.txt', encoding='utf-8').read().strip()
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/131.0.0.0 Safari/537.36')
API = 'https://drive-pc.quark.cn/1/clouddrive'
OSS_UA = 'aliyun-sdk-js/6.6.1 Chrome 98.0.4758.80 on Windows 10 64-bit'


def api(path, body=None, method='POST', params=None):
    q = {'pr': 'ucpro', 'fr': 'pc', 'uc_param_str': ''}
    q.update(params or {})
    url = API + path + '?' + urllib.parse.urlencode(q)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Cookie', COOKIE)
    req.add_header('User-Agent', UA)
    req.add_header('Referer', 'https://pan.quark.cn/')
    req.add_header('Origin', 'https://pan.quark.cn')
    req.add_header('Accept', 'application/json, text/plain, */*')
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode('utf-8', 'replace'))
    except urllib.error.HTTPError as e:
        raw = e.read().decode('utf-8', 'replace')
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {'raw': raw[:300]}


def hashes(path):
    m, s = hashlib.md5(), hashlib.sha1()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(8 << 20)
            if not b:
                break
            m.update(b)
            s.update(b)
    return m.hexdigest(), s.hexdigest()


def http_date():
    return time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())


def upload(path, pdir_fid, part_size=None, log=print):
    size = os.path.getsize(path)
    name = os.path.basename(path)
    mime = mimetypes.guess_type(name)[0] or 'application/octet-stream'
    md5s, sha1s = hashes(path)
    st, d = api('/file/upload/pre', {'file_name': name, 'format_type': 'file', 'size': size,
                                     'pdir_fid': pdir_fid, 'l_created_at': int(time.time() * 1000),
                                     'l_updated_at': int(time.time() * 1000), 'ccp_hash_update': True})
    if st != 200 or d.get('code') != 0:
        raise RuntimeError('pre failed: %s %s' % (st, json.dumps(d, ensure_ascii=False)[:300]))
    data = d['data']
    meta = d.get('metadata') or {}
    log('  pre ok upload_id=%s bucket=%s part_size=%s' % (data['upload_id'][:8], data['bucket'], meta.get('part_size')))
    st, h = api('/file/update/hash', {'md5': md5s, 'sha1': sha1s, 'task_id': data['task_id']})
    if (h.get('data') or {}).get('finish'):
        log('  秒传命中 (instant)')
        return {'finish': 'instant', 'name': name, 'size': size}
    part_size = part_size or meta.get('part_size') or (4 << 20)
    base = 'https://%s.%s/%s' % (data['bucket'], data['upload_url'][7:], data['obj_key'])
    etags, n, sent = [], 1, 0
    with open(path, 'rb') as fh:
        while True:
            chunk = fh.read(part_size)
            if not chunk:
                break
            tstr = http_date()
            auth_meta = ('PUT\n\n%s\n%s\nx-oss-date:%s\nx-oss-user-agent:%s\n/%s/%s?partNumber=%d&uploadId=%s'
                         % (mime, tstr, tstr, OSS_UA, data['bucket'], data['obj_key'], n, data['upload_id']))
            st, a = api('/file/upload/auth', {'auth_info': data['auth_info'], 'auth_meta': auth_meta, 'task_id': data['task_id']})
            key = (a.get('data') or {}).get('auth_key') or (a.get('data') or {}).get('authKey')
            if not key:
                raise RuntimeError('auth failed: %s %s' % (st, json.dumps(a, ensure_ascii=False)[:300]))
            url = base + '?' + urllib.parse.urlencode({'partNumber': n, 'uploadId': data['upload_id']})
            req = urllib.request.Request(url, data=chunk, method='PUT')
            req.add_header('Authorization', key)
            req.add_header('Content-Type', mime)
            req.add_header('Referer', 'https://pan.quark.cn/')
            req.add_header('x-oss-date', tstr)
            req.add_header('x-oss-user-agent', OSS_UA)
            with urllib.request.urlopen(req, timeout=600) as r:
                etag = r.headers.get('ETag')
            etags.append(etag)
            sent += len(chunk)
            log('  part %d: %d/%d bytes etag=%s' % (n, sent, size, (etag or '')[:16]))
            n += 1
    body = '<?xml version="1.0" encoding="UTF-8"?>\n<CompleteMultipartUpload>\n' + ''.join(
        '<Part>\n<PartNumber>%d</PartNumber>\n<ETag>%s</ETag>\n</Part>\n' % (i + 1, e) for i, e in enumerate(etags)
    ) + '</CompleteMultipartUpload>'
    content_md5 = base64.b64encode(hashlib.md5(body.encode()).digest()).decode()
    cb_b64 = base64.b64encode(json.dumps(data['callback']).encode()).decode()
    tstr = http_date()
    auth_meta = ('POST\n%s\napplication/xml\n%s\nx-oss-callback:%s\nx-oss-date:%s\nx-oss-user-agent:%s\n/%s/%s?uploadId=%s'
                 % (content_md5, tstr, cb_b64, tstr, OSS_UA, data['bucket'], data['obj_key'], data['upload_id']))
    st, a = api('/file/upload/auth', {'auth_info': data['auth_info'], 'auth_meta': auth_meta, 'task_id': data['task_id']})
    key = (a.get('data') or {}).get('auth_key') or (a.get('data') or {}).get('authKey')
    if not key:
        raise RuntimeError('commit auth failed: %s %s' % (st, json.dumps(a, ensure_ascii=False)[:300]))
    url = base + '?' + urllib.parse.urlencode({'uploadId': data['upload_id']})
    req = urllib.request.Request(url, data=body.encode(), method='POST')
    for k, v in (('Authorization', key), ('Content-MD5', content_md5), ('Content-Type', 'application/xml'),
                 ('Referer', 'https://pan.quark.cn/'), ('x-oss-callback', cb_b64),
                 ('x-oss-date', tstr), ('x-oss-user-agent', OSS_UA)):
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=300) as r:
        log('  commit: %s %s' % (r.status, r.read(200).decode('utf-8', 'replace')[:120]))
    st, f = api('/file/upload/finish', {'obj_key': data['obj_key'], 'task_id': data['task_id']})
    log('  finish: %s %s' % (st, json.dumps(f, ensure_ascii=False)[:200]))
    time.sleep(1)
    return {'finish': bool(st == 200 and f.get('code') == 0), 'name': name, 'size': size, 'fid': data.get('fid')}


if __name__ == '__main__':
    print(upload(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '0'))
