"""Confirm the 5 products' rasters exist and the shared mount state."""
import os
import paramiko

CMD = r'''
echo "== 5 个产品的 tif =="
for k in 19A8 85D9 1A19 570A B1C2; do
  n=$(find /root/autodl-tmp -name "*_$k*_UTM_8bit.tif" 2>/dev/null | wc -l)
  echo "  $k -> $n 张大图"
done
echo "== 共享盘 =="
mount | grep -c autodl-fs
ls -la /root/autodl-fs/ 2>/dev/null | head -3
df -h /root/autodl-fs/data 2>/dev/null | tail -1
echo "== 标注台 status（最新计数） =="
ls /root/autodl-tmp/safe841_batch/annotation_output/status 2>/dev/null | wc -l
ls /root/autodl-tmp/safe841_batch_s3/annotation_output_s3/status 2>/dev/null | wc -l
echo "== 归档进度 =="
tail -1 /root/autodl-tmp/redl_work/quark_archive.log 2>/dev/null
echo "== 磁盘 =="
df -h /root/autodl-tmp | tail -1
'''


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=16575, username='root',
              password=os.environ['WESTD_PW'], timeout=40, banner_timeout=60, auth_timeout=60)
    _, out, _ = c.exec_command(CMD, timeout=600)
    print(out.read().decode('utf-8', 'replace'))
    c.close()


if __name__ == '__main__':
    main()
