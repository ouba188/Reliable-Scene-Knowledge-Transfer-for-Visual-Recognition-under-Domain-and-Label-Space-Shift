"""Fresh read of the westd state that matters for the three-session sync.

  WESTD_PW=... python westd_status.py
"""
import os
import paramiko

CMD = r'''
echo "== 时间/磁盘 =="; date '+%F %H:%M'; df -h /root/autodl-tmp | tail -1
echo; echo "== 标注台 completed 计数 =="
python3 - <<'PY'
import json, glob
for pat, label in [('/root/autodl-tmp/safe841_batch/annotation_output/status/*.json', 'westd-main'),
                   ('/root/autodl-tmp/safe841_batch_s3/annotation_output_s3/status/*.json', 'westd-s3')]:
    ok = fail = other = 0
    for f in glob.glob(pat):
        try:
            s = json.load(open(f)).get('status')
        except Exception:
            s = '?'
        if s == 'completed':
            ok += 1
        elif s in ('failed', 'error'):
            fail += 1
        else:
            other += 1
    print('%s: completed=%d failed=%d other=%d' % (label, ok, fail, other))
PY
echo; echo "== labels_json 产品数（main / s3） =="
find /root/autodl-tmp/safe841_batch/annotation_output -path '*labels_json*' -name '*_UTM_8bit.json' 2>/dev/null | sed 's#.*/##' | sort -u | wc -l
find /root/autodl-tmp/safe841_batch_s3/annotation_output_s3 -path '*labels_json*' -name '*_UTM_8bit.json' 2>/dev/null | sed 's#.*/##' | sort -u | wc -l
echo; echo "== 大图总数 =="; find /root/autodl-tmp -name '*_UTM_8bit.tif' 2>/dev/null | wc -l
echo; echo "== 知识集目录 =="; du -sh /root/autodl-tmp/knowledge_841_20260920 2>/dev/null
echo; echo "== 活动任务 =="; pgrep -af 'python|quark|asf|rsync' | grep -vE 'jupyter|tensorboard|autopanel' | head -6
'''


def main():
    pw = os.environ.get('WESTD_PW')
    if not pw:
        raise SystemExit('WESTD_PW 未设置')
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=16575, username='root', password=pw,
              timeout=40, banner_timeout=60, auth_timeout=60)
    _, out, _ = c.exec_command(CMD, timeout=300)
    print(out.read().decode('utf-8', 'replace'))
    c.close()


if __name__ == '__main__':
    main()
