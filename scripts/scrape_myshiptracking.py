"""Resolve fine-grained ship classes for unresolved MMSI by reading MyShipTracking (read-only).

The curated label -> fine_class layer in mapping.csv is 100% pure for its top labels, so a
scraped vessel label can be mapped to a fine class with the same semantics the corpus already uses.

Resumable: every lookup appends to mmsi_myshiptracking.jsonl and is skipped on rerun.
Ordered by AIS-row coverage, so an interrupted run still resolves the highest-impact MMSI first.
"""
import argparse, csv, json, re, threading, time, urllib.request
from collections import Counter
from pathlib import Path

WORK = Path(r'E:/临时会话/knowledge_set_841/mmsi')
JOURNAL = WORK / 'mmsi_myshiptracking.jsonl'
FILLED = WORK / 'mmsi_fine_class_filled.csv'
TITLE = re.compile(r'<title>(.*?)</title>', re.S)
LABEL_FROM_TITLE = re.compile(r'Current Position\s*\(([^,)]*)')
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36'
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def label_rules():
    """source label -> fine class, learned from the corpus's own curated mapping rows."""
    rules = {}
    with (Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle/assets/mapping.csv')
          ).open('r', encoding='utf-8-sig', newline='') as f:
        seen = {}
        for r in csv.DictReader(f):
            src, fin = (r['selected_mapping_type'] or '').strip(), (r['final_class'] or '').strip()
            if src and fin:
                seen.setdefault(src, Counter())[fin] += 1
    for src, counter in seen.items():
        total = sum(counter.values())
        fin, hits = counter.most_common(1)[0]
        if hits / total >= 0.9:
            rules[src.lower()] = fin
    return rules


def todo_list():
    obs = {}
    with (WORK / 'mmsi_observed.csv').open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            obs[r['mmsi']] = int(r['ais_rows'])
    with FILLED.open('r', encoding='utf-8-sig', newline='') as f:
        pending = [r['mmsi'] for r in csv.DictReader(f) if not r['final_class']]
    pending.sort(key=lambda m: -obs.get(m, 0))
    return pending, obs


def lookup(mmsi, timeout=30):
    url = 'https://www.myshiptracking.com/?mmsi=' + mmsi
    try:
        html = OP.open(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=timeout).read().decode('utf-8', 'replace')
        m = TITLE.search(html)
        title = re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
        lm = LABEL_FROM_TITLE.search(title)
        label = (lm.group(1) if lm else '').strip()
        name = title.split(' Current Position')[0].strip() if title else ''
        return dict(status='ok', name=name, label=label)
    except Exception as exc:
        return dict(status='error', error=repr(exc)[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=30.0, help='wall-clock budget for this run')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--delay', type=float, default=0.35, help='seconds between a worker\'s requests')
    args = ap.parse_args()

    done = set()
    if JOURNAL.is_file():
        with JOURNAL.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    done.add(json.loads(line)['mmsi'])
                except Exception:
                    pass
    pending, obs = todo_list()
    queue = [m for m in pending if m not in done]
    print('unresolved=%d already looked up=%d this run=%d' % (len(pending), len(done), len(queue)), flush=True)

    deadline = time.time() + args.minutes * 60
    lock = threading.Lock()
    stats = Counter()

    def worker():
        while True:
            with lock:
                if not queue or time.time() > deadline:
                    return
                mmsi = queue.pop(0)
            res = lookup(mmsi)
            with lock:
                stats[res['status']] += 1
                stats['rows_' + res['status']] += obs.get(mmsi, 0)
                with JOURNAL.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(mmsi=mmsi, ais_rows=obs.get(mmsi, 0), **res), ensure_ascii=False) + '\n')
                if stats['ok'] and stats['ok'] % 200 == 0:
                    print('%d lookups, ok=%d, labels=%d, %.0fs left' % (
                        stats['ok'] + stats['error'], stats['ok'], stats['labelled'],
                        deadline - time.time()), flush=True)
            if res['status'] == 'ok' and res.get('label'):
                with lock:
                    stats['labelled'] += 1
            time.sleep(args.delay)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print('done: %s | rows covered ok=%d' % (dict(stats), stats['rows_ok']), flush=True)


if __name__ == '__main__':
    main()
