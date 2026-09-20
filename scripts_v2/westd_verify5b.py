"""Clean verification of the 5 recovered products (python on the server, no shell-quoting games)."""
import os
import paramiko

REMOTE_PY = r'''
import os, glob
targets = {
 '19A8': 'S1A_IW_GRDH_1SDV_20250315T191539_20250315T191604_058319_0735A5_19A8',
 '85D9': 'S1A_IW_GRDH_1SDV_20250316T153905_20250316T153934_058331_073622_85D9',
 '1A19': 'S1A_IW_GRDH_1SDV_20250317T035236_20250317T035301_058339_07366E_1A19',
 '570A': 'S1A_IW_GRDH_1SDV_20250317T095619_20250317T095644_058343_073693_570A',
 'B1C2': 'S1A_IW_GRDH_1SDV_20250318T224811_20250318T224836_058365_073771_B1C2',
}
for k, p in targets.items():
    hits = glob.glob('/root/autodl-tmp/safe841_batch/annotation_output/%s/*/labels_json/*_UTM_8bit.json' % p)
    hits += glob.glob('/root/autodl-tmp/safe841_batch/annotation_output/%s/*/*/labels_json/*_UTM_8bit.json' % p)
    print('%s  labels_json=%d  %s' % (k, len(hits), os.path.basename(hits[0])[:40] if hits else ''))
print('--- 全局 ---')
allp = set()
for f in glob.glob('/root/autodl-tmp/safe841_batch/annotation_output/**/labels_json/*_UTM_8bit.json', recursive=True):
    allp.add(os.path.basename(f).replace('_UTM_8bit.json', '')[:-3])
print('annotation_output 产品数:', len(allp))
'''


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=16575, username='root',
              password=os.environ['WESTD_PW'], timeout=40, banner_timeout=60, auth_timeout=60)
    sftp = c.open_sftp()
    with sftp.open('/root/verify5.py', 'w') as fh:
        fh.write(REMOTE_PY)
    _, out, _ = c.exec_command('/root/miniconda3/bin/python /root/verify5.py', timeout=600)
    print(out.read().decode('utf-8', 'replace'))
    sftp.close()
    c.close()


if __name__ == '__main__':
    main()
