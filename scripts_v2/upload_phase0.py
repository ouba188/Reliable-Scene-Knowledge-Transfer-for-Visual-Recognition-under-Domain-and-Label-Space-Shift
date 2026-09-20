"""Upload the Phase 0 artifacts to westd /root/autodl-tmp/phase0/ (resumable, parallel, keeps others' dirs untouched).

Only pushes what the experiment needs on the server; the big objects_*.csv.gz dumps are already
being uploaded by the knowledge-set session, so they are skipped here.

ponytail: one sftp session per worker thread (the paramiko sftp channel is not thread-safe).
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paramiko

KB = Path('E:/临时会话/knowledge_set_841')
REMOTE_ROOT = '/root/autodl-tmp/phase0'
CREDS = json.loads(Path('E:/Hermes/config/westd.json').read_text(encoding='utf-8'))

ITEMS = [
    (KB / 'splits', 'splits'),
    (KB / 'geo', 'geo'),
    (KB / 'masks', 'masks'),
    (KB / 'ports', 'ports'),
    (KB / 'mmsi', 'mmsi'),
    (KB / 'facilities', 'facilities'),
    (KB / 'objects' / 'support_by_port_class.csv', 'objects/support_by_port_class.csv'),
]
local_state = threading.local()


def collect():
    files = []
    for local, rel in ITEMS:
        if local.is_dir():
            for p in sorted(local.rglob('*')):
                if p.is_file() and p.suffix in ('.csv', '.json', '.md', '.geojson', '.png'):
                    files.append((p, '%s/%s/%s' % (REMOTE_ROOT, rel, p.relative_to(local).as_posix())))
        elif local.is_file():
            files.append((local, '%s/%s' % (REMOTE_ROOT, rel)))
    return files


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(CREDS['host'], port=CREDS['port'], username=CREDS['user'],
                password=CREDS['password'], timeout=25, banner_timeout=25, auth_timeout=25)
    return ssh, ssh.open_sftp()


def ensuredirs(sftp, path, made):
    cur = ''
    for part in path.strip('/').split('/'):
        cur += '/' + part
        if cur in made:
            continue
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except IOError:
                pass          # another worker won the race, or it already exists
        made.add(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    files = collect()
    total = sum(p.stat().st_size for p, _ in files)
    print('files=%d total=%.1f MB -> %s' % (len(files), total / 2**20, REMOTE_ROOT), flush=True)
    if not files:
        return

    batches = [files[i::a.workers] for i in range(a.workers)]
    stats = dict(sent=0, skipped=0, failed=0)
    lock = threading.Lock()
    t0 = time.time()

    def worker(batch):
        ssh, sftp = connect()
        made = set()
        try:
            for local, remote in batch:
                ensuredirs(sftp, str(Path(remote).parent).replace('\\', '/'), made)
                try:
                    st = sftp.stat(remote)
                    if st.st_size == local.stat().st_size:
                        with lock:
                            stats['skipped'] += 1
                        continue
                except IOError:
                    pass
                try:
                    sftp.put(str(local), remote)
                    with lock:
                        stats['sent'] += local.stat().st_size
                except Exception as exc:
                    with lock:
                        stats['failed'] += 1
                    print('FAIL %s: %r' % (remote, exc), flush=True)
        finally:
            sftp.close()
            ssh.close()

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(worker, batches))
    ssh, sftp = connect()
    q = "du -sh %s/* 2>/dev/null; find %s -type f | wc -l" % (REMOTE_ROOT, REMOTE_ROOT)
    _, stdout, _ = ssh.exec_command(q)
    listing = stdout.read().decode('utf-8', 'replace')
    sftp.close()
    ssh.close()
    print('upload done in %.1f min: sent %.1f MB, skipped %d, failed %d' % (
        (time.time() - t0) / 60, stats['sent'] / 2**20, stats['skipped'], stats['failed']), flush=True)
    print('remote:\n%s' % listing, flush=True)


if __name__ == '__main__':
    main()
