"""Upload the 841 knowledge set's experiment-relevant artifacts to the AutoDL box (westd).

Resumable: files already present with the same size are skipped; SFTP put for the rest.
The SSH password is read from WESTD_PW (never written into this file), so the script can be
deleted afterwards without leaving a credential behind.

  WESTD_PW=... python upload_to_westd.py
"""
import os, sys, time
from pathlib import Path

import paramiko

HOST, PORT, USER = 'connect.westd.seetacloud.com', 16575, 'root'
REMOTE = '/root/autodl-tmp/knowledge_841_20260920'
KS = Path(r'E:/临时会话/knowledge_set_841')

FILES = [
    ('objects/objects_final.csv.gz', 'objects/objects_final.csv.gz'),
    ('objects/objects_by_port_summary.csv', 'objects/objects_by_port_summary.csv'),
    ('objects/objects_summary.json', 'objects/objects_summary.json'),
    ('ports/port_knowledge.csv', 'ports/port_knowledge.csv'),
    ('ports/port_knowledge.json', 'ports/port_knowledge.json'),
    ('facilities/osm_way_dates.csv', 'facilities/osm_way_dates.csv'),
    ('mmsi/mmsi_class_final.csv', 'mmsi/mmsi_class_final.csv'),
    ('mmsi/FILL_REPORT.md', 'mmsi/FILL_REPORT.md'),
    ('keys/products.csv', 'keys/products.csv'),
    ('README_STATUS.md', 'README_STATUS.md'),
]
TRAFFIC = sorted((KS / 'traffic').glob('*.csv')) + sorted((KS / 'traffic').glob('*.provenance.json'))


def main():
    pw = os.environ.get('WESTD_PW')
    if not pw:
        sys.exit('WESTD_PW 未设置')
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=pw, timeout=30, banner_timeout=60)
    sftp = cli.open_sftp()
    cli.exec_command('mkdir -p %s/objects %s/ports %s/facilities %s/mmsi %s/keys %s/traffic' % (
        REMOTE, REMOTE, REMOTE, REMOTE, REMOTE, REMOTE))[1].channel.recv_exit_status()

    items = [(KS / rel, '%s/%s' % (REMOTE, rrel.replace('\\', '/'))) for rel, rrel in FILES]
    items += [(p, '%s/traffic/%s' % (REMOTE, p.name)) for p in TRAFFIC]

    sent = skipped = 0
    for local, remote in items:
        if not local.is_file():
            print('缺文件，跳过: %s' % local, flush=True)
            continue
        size = local.stat().st_size
        try:
            if sftp.stat(remote).st_size == size:
                print('已存在且大小一致，跳过: %s' % remote, flush=True)
                skipped += 1
                continue
        except IOError:
            pass
        t0 = time.time()
        sftp.put(str(local), remote)
        dt = time.time() - t0
        print('上传 %-52s %8.1f MB  %5.1f s  %5.2f MB/s' % (
            remote.replace(REMOTE + '/', ''), size / 1048576, dt, size / 1048576 / max(dt, 0.01)), flush=True)
        sent += 1

    print('--- 校验 ---', flush=True)
    ok = bad = 0
    for local, remote in items:
        if not local.is_file():
            continue
        try:
            rsize = sftp.stat(remote).st_size
        except IOError:
            print('缺失 %s' % remote)
            bad += 1
            continue
        if rsize == local.stat().st_size:
            ok += 1
        else:
            print('大小不符 %s 本地 %d 远端 %d' % (remote, local.stat().st_size, rsize))
            bad += 1
    print('已上传 %d，跳过 %d；校验一致 %d，异常 %d' % (sent, skipped, ok, bad), flush=True)
    _, out, _ = cli.exec_command('du -sh %s && ls %s' % (REMOTE, REMOTE))
    print(out.read().decode('utf-8', 'replace'), flush=True)
    sftp.close()
    cli.close()


if __name__ == '__main__':
    main()
