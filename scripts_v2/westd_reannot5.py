"""Re-annotate the 5 products whose outputs died with the shared disk.

Writes queue items + a driver script into my own server dir (knowledge_841_20260920/recover5/),
then launches it under tmux. Outputs go to the standard annotation tree (append-only new
<product>/ dirs) because that is where the experiment reads them.
"""
import io
import json
import os
from pathlib import Path

import paramiko

PRODUCTS = [
    ('S1A_IW_GRDH_1SDV_20250315T191539_20250315T191604_058319_0735A5_19A8', 'Newcastle',
     '/root/autodl-tmp/fdrive_import_20260920/v5/geocode/Newcastle/utm_out'),
    ('S1A_IW_GRDH_1SDV_20250316T153905_20250316T153934_058331_073622_85D9', 'Mombasa',
     '/root/autodl-tmp/fdrive_import_20260920/v5/geocode/Mombasa/utm_out'),
    ('S1A_IW_GRDH_1SDV_20250317T035236_20250317T035301_058339_07366E_1A19', 'Port Said',
     '/root/autodl-tmp/fdrive_import_20260920/v5/geocode/Port Said/utm_out'),
    ('S1A_IW_GRDH_1SDV_20250317T095619_20250317T095644_058343_073693_570A', 'Qingdao',
     '/root/autodl-tmp/fdrive_import_20260920/v5/geocode/Qingdao/utm_out'),
    ('S1A_IW_GRDH_1SDV_20250318T224811_20250318T224836_058365_073771_B1C2', 'Singapore',
     '/root/autodl-tmp/fdrive_import_20260920/v5/geocode/Singapore/utm_out'),
]
RECOVER = '/root/autodl-tmp/knowledge_841_20260920/recover5'
OUT_ROOT = '/root/autodl-tmp/safe841_batch/annotation_output'
ASSETS = '/root/autodl-tmp/safe841_batch/annotation_bundle_s3/assets'
TOOL = '/root/autodl-tmp/safe841_batch/annotation_bundle/run_queue_item.py'
PY = '/root/miniconda3/bin/python'

DRIVER = '''#!/bin/bash
cd {rec} || exit 1
for item in queue/*.json; do
  echo "=== $(date +%H:%M:%S) start $item"
  {py} {tool} --queue-item "$item" --output-root {out} --assets {assets} \\
      --detector-python {py} --device 0 --allow-missing-facility-map --retry-failed
  echo "=== $(date +%H:%M:%S) rc=$? $item"
done
echo "=== ALL DONE $(date +%H:%M:%S)"
'''.format(rec=RECOVER, py=PY, tool=TOOL, out=OUT_ROOT, assets=ASSETS)


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=16575, username='root',
              password=os.environ['WESTD_PW'], timeout=40, banner_timeout=60, auth_timeout=60)
    sftp = c.open_sftp()

    def mkdir(p):
        try:
            sftp.mkdir(p)
        except IOError:
            pass

    mkdir(RECOVER)
    mkdir(RECOVER + '/queue')
    mkdir(RECOVER + '/logs')

    for product, port, base in PRODUCTS:
        item = {
            'product': product,
            'rasters': ['%s/VV/int8/%s_VV_UTM_8bit.tif' % (base, product),
                        '%s/VH/int8/%s_VH_UTM_8bit.tif' % (base, product)],
            'port': port,
            'annotation': 'd0b3b3_recover5',
        }
        with sftp.open('%s/queue/%s.json' % (RECOVER, product), 'w') as fh:
            fh.write(json.dumps(item, ensure_ascii=False, indent=1))
        print('queue item:', product[-9:], port)
    with sftp.open(RECOVER + '/run5.sh', 'w') as fh:
        fh.write(DRIVER)

    cmd = ('chmod +x {r}/run5.sh; tmux kill-session -t recover5 2>/dev/null; '
           'tmux new-session -d -s recover5 "bash {r}/run5.sh 2>&1 | tee {r}/logs/run5.log"; '
           'sleep 25; tmux ls; echo ---; tail -5 {r}/logs/run5.log').format(r=RECOVER)
    _, out, _ = c.exec_command(cmd, timeout=300)
    print(out.read().decode('utf-8', 'replace'))
    sftp.close()
    c.close()


if __name__ == '__main__':
    main()
