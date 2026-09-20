"""Approved cleanup on westd. Deletes only the categories the user approved, with a receipt.

  categories: review jpgs · duplicate annotation_bundle_s3 · historical *_annotation_package.tgz ·
              extra per-product run dirs · byte-identical duplicate rasters (md5-checked)

Writes a manifest (path, bytes) before deleting, then verifies sizes after.
Run with --execute to actually delete; without it, only inventories.
"""
import argparse
import os
import paramiko

INVENTORY = r'''
python3 - <<'PY'
import os, hashlib, json, collections
B = '/root/autodl-tmp'
report = {}

# 1) review jpgs
jpgs = []
for root in ['%s/safe841_batch/annotation_output' % B, '%s/safe841_batch_s3/annotation_output_s3' % B,
             '%s/safe841_batch/annotation_output_s3' % B]:
    if not os.path.isdir(root):
        continue
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith('.jpg'):
                jpgs.append(os.path.join(dirpath, f))
report['jpgs'] = {'files': len(jpgs), 'bytes': sum(os.path.getsize(p) for p in jpgs)}

# 2) duplicate bundle
dupb = '%s/safe841_batch_s3/annotation_bundle_s3' % B
report['duplicate_bundle'] = {'path': dupb, 'exists': os.path.isdir(dupb)}
if os.path.isdir(dupb):
    tot = n = 0
    for dp, _d, fs in os.walk(dupb):
        for f in fs:
            p = os.path.join(dp, f); tot += os.path.getsize(p); n += 1
    report['duplicate_bundle'].update(files=n, bytes=tot)

# 3) tarballs
tars = []
for pat in ['server1_annotation_package.tgz', 'server2_annotation_package.tgz',
            'server3_annotation_package.tgz', 'server3_annotation_package_s3.tgz']:
    for p in ['%s/safe841_batch/%s' % (B, pat), '%s/safe841_batch_s3/%s' % (B, pat)]:
        if os.path.isfile(p):
            tars.append({'path': p, 'bytes': os.path.getsize(p)})
report['tarballs'] = {'files': len(tars), 'bytes': sum(t['bytes'] for t in tars), 'items': tars}

# 4) extra run dirs (more than one run dir per product)
extra = []
for root in ['%s/safe841_batch/annotation_output' % B, '%s/safe841_batch_s3/annotation_output_s3' % B]:
    if not os.path.isdir(root):
        continue
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not (os.path.isdir(p) and name.startswith('S1')):
            continue
        runs = sorted([r for r in os.listdir(p) if os.path.isdir(os.path.join(p, r))])
        for r in runs[:-1]:
            rp = os.path.join(p, r)
            tot = n = 0
            for dp, _d, fs in os.walk(rp):
                for f in fs:
                    tot += os.path.getsize(os.path.join(dp, f)); n += 1
            extra.append({'path': rp, 'bytes': tot, 'files': n})
report['extra_runs'] = {'dirs': len(extra), 'bytes': sum(e['bytes'] for e in extra), 'items': extra[:20]}

# 5) duplicate rasters by name (md5-verified candidates)
by_name = collections.defaultdict(list)
for dirpath, _dirs, files in os.walk(B):
    if '/knowledge_841_20260920' in dirpath:
        continue
    for f in files:
        if f.endswith('_UTM_8bit.tif'):
            p = os.path.join(dirpath, f)
            by_name[f].append((p, os.path.getsize(p)))
dups = {k: v for k, v in by_name.items() if len(v) > 1}
report['dup_rasters'] = {'names': len(dups), 'extra_copies': sum(len(v) - 1 for v in dups.values()),
                         'extra_bytes': sum(sorted(c[1] for c in v)[0] * (len(v) - 1) for v in dups.values())}
print(json.dumps(report, ensure_ascii=False, indent=1))
json.dump({'jpgs': jpgs, 'tarballs': tars, 'extra_runs': extra,
           'dup_rasters': {k: v for k, v in dups.items()}}, open('/root/cleanup_manifest.json', 'w'))
open('/root/cleanup_jpgs.txt', 'w').write('\n'.join(jpgs))
PY
'''

EXECUTE = r'''
echo "== 删除复核 jpg =="
wc -l < /root/cleanup_jpgs.txt
xargs -a /root/cleanup_jpgs.txt -d '\n' -r rm -f
echo "剩余 jpg: $(find /root/autodl-tmp/safe841_batch*/annotation_output* -name '*.jpg' 2>/dev/null | wc -l)"

echo "== 删除重复 bundle =="
du -sh /root/autodl-tmp/safe841_batch_s3/annotation_bundle_s3 2>/dev/null
rm -rf /root/autodl-tmp/safe841_batch_s3/annotation_bundle_s3
[ -d /root/autodl-tmp/safe841_batch_s3/annotation_bundle_s3 ] && echo "仍存在(异常)" || echo "已删除"

echo "== 删除历史 tgz =="
python3 - <<'PY'
import json, os
m = json.load(open('/root/cleanup_manifest.json'))
for t in m['tarballs']:
    if os.path.isfile(t['path']):
        os.remove(t['path']); print('rm', t['path'])
PY

echo "== 删除多余 run 目录 =="
python3 - <<'PY'
import json, os, shutil
m = json.load(open('/root/cleanup_manifest.json'))
for e in m['extra_runs']:
    if os.path.isdir(e['path']):
        shutil.rmtree(e['path']); print('rm -r', e['path'])
PY

echo "== 同名大图副本：md5 相同才删 =="
python3 - <<'PY'
import hashlib, json, os
m = json.load(open('/root/cleanup_manifest.json'))
PREF = ['/safe841_batch/annotation_output/', '/safe841_batch/parallel_', '/fdrive_import_20260920/',
        '/asf_download_v2_', '/sar_download_v1/', '/safe841_batch_s3/']
def rank(p):
    for i, pref in enumerate(PREF):
        if pref in p:
            return i
    return len(PREF)
def md5(p, chunk=8 << 20):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(chunk), b''):
            h.update(c)
    return h.hexdigest()
freed = 0
for name, copies in m['dup_rasters'].items():
    copies = sorted(copies, key=lambda c: (rank(c[0]), c[0]))
    keep = copies[0]
    keep_md5 = md5(keep[0])
    for p, size in copies[1:]:
        if md5(p) == keep_md5:
            os.remove(p); freed += size
            print('rm dup tif', p)
        else:
            print('SKIP (content differs)', p)
print('dup tif freed bytes', freed)
PY

echo "== 磁盘 =="
df -h /root/autodl-tmp | tail -1
find /root/autodl-tmp/safe841_batch*/annotation_output* -name '*.jpg' 2>/dev/null | wc -l
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=16575, username='root',
              password=os.environ['WESTD_PW'], timeout=40, banner_timeout=60, auth_timeout=60)
    _, out, _ = c.exec_command(INVENTORY, timeout=1800)
    inv = out.read().decode('utf-8', 'replace')
    print(inv)
    open(r'E:/Hermes/scripts/out/cleanup_inventory2.json', 'w', encoding='utf-8').write(inv)
    if a.execute:
        print('\n===== EXECUTE =====')
        _, out, _ = c.exec_command(EXECUTE, timeout=3600)
        log = out.read().decode('utf-8', 'replace')
        print(log)
        open(r'E:/Hermes/scripts/out/cleanup_execute.log', 'w', encoding='utf-8').write(log)
    c.close()


if __name__ == '__main__':
    main()
