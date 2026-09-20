"""Install the two missing experiment deps into the server's miniconda env and verify."""
import os
import paramiko

CMD = r'''
/root/miniconda3/bin/python -m pip install -q scikit-learn geopandas 2>&1 | tail -3
/root/miniconda3/bin/python - <<'PY'
mods = ['sklearn', 'geopandas', 'torch', 'rasterio', 'shapely']
import importlib
for m in mods:
    try:
        mod = importlib.import_module(m)
        print('OK  %-12s %s' % (m, getattr(mod, '__version__', '?')))
    except Exception as exc:
        print('MISS %-12s %s' % (m, repr(exc)[:60]))
import torch
print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
PY
'''


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=16575, username='root',
              password=os.environ['WESTD_PW'], timeout=40, banner_timeout=60, auth_timeout=60)
    _, out, _ = c.exec_command(CMD, timeout=900)
    print(out.read().decode('utf-8', 'replace'))
    c.close()


if __name__ == '__main__':
    main()
