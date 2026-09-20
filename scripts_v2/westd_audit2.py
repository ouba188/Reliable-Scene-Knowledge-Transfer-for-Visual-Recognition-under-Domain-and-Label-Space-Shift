"""Second pass of the westd readiness audit: labels via find, conda env packages."""
import os
from pathlib import Path

import paramiko

RMS = r'''
echo "== 标注文件 =="
find /root/autodl-tmp -path '*labels_json*' -name '*_UTM_8bit.json' 2>/dev/null | sed 's#.*/##' | sed 's/_UTM_8bit.json//' | sed 's/_[VH][VH]$//' | sort -u > /tmp/lab_products.txt
wc -l < /tmp/lab_products.txt
echo "== 标注目录（每产品一次运行） =="
find /root/autodl-tmp -type d -name 'labels_json' 2>/dev/null | wc -l
echo "== 大图产品数 =="
find /root/autodl-tmp -name '*_UTM_8bit.tif' 2>/dev/null | sed 's#.*/##' | sed 's/_UTM_8bit.tif//' | sed 's/_[VH][VH]$//' | sort -u | wc -l
echo "== miniconda python 依赖 =="
for PY in /root/miniconda3/bin/python; do
  echo "-- $PY"
  $PY -c "import sys;print('PYVER',sys.version.split()[0])"
  $PY - <<'PY'
mods=['torch','torchvision','numpy','pandas','sklearn','scipy','rasterio','geopandas','shapely','tqdm','yaml','ultralytics','cv2']
ok=[];miss=[]
for m in mods:
    try:
        __import__(m); ok.append(m)
    except Exception: miss.append(m)
print('PKG_OK', ','.join(ok))
print('PKG_MISS', ','.join(miss))
import torch
print('TORCH', torch.__version__, 'cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
PY
done
echo "== 0917 实现代码 =="
ls /root/*.md /root/autodl-tmp/*.md 2>/dev/null | head
find /root -maxdepth 3 -name '*.py' -newermt '2026-09-15' 2>/dev/null | grep -viE 'quark|asf|autopanel|annotation_bundle' | head -10
echo "== conda 环境 =="
ls /root/miniconda3/envs 2>/dev/null
echo "== 磁盘 =="
df -h /root/autodl-tmp | tail -1
'''


def main():
    pw = os.environ.get('WESTD_PW')
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=16575, username='root', password=pw,
              timeout=40, banner_timeout=60, auth_timeout=60)
    _, out, _ = c.exec_command(RMS, timeout=900)
    text = out.read().decode('utf-8', 'replace')
    Path(r'E:/Hermes/scripts/out/westd_audit2.txt').write_text(text, encoding='utf-8')
    print(text)
    # 下载标注产品清单以本地对账
    sftp = c.open_sftp()
    try:
        sftp.get('/tmp/lab_products.txt', r'E:/Hermes/scripts/out/server_label_products.txt')
        print('已取回标注产品清单')
    except Exception as exc:
        print('取回失败:', repr(exc)[:80])
    sftp.close()
    c.close()


if __name__ == '__main__':
    main()
