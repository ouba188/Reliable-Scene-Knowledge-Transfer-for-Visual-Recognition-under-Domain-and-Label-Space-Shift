"""Fill MMSI -> fine-grained ship class for the 841-product knowledge set.

Stages (run all, or pass one or more as argv):
  scan    stream the bundle AIS text once -> per-MMSI observed AIS evidence
  rules   derive type-code -> fine-class rules from the already-resolved mapping rows
  sarais  batch-query the local SARAIS backend for MMSI with no usable AIS type code
  fill    classify unresolved MMSI with observed/SARAIS type codes and the derived rules

Provenance is explicit: every filled class records which evidence and which rule produced it,
plus the rule's purity in the resolved corpus. Low-purity codes are never silently assigned.
"""
import csv, json, re, sys, time, urllib.parse, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

PLAN = Path(r'E:/临时会话/safe841_redownload_plan_20260918')
BUNDLE = PLAN / 'annotation_bundle' / 'assets'
PRODUCTS = BUNDLE / 'products'
MAPPING = BUNDLE / 'mapping.csv'
OUT = Path(r'E:/临时会话/knowledge_set_841/mmsi')
OBSERVED = OUT / 'mmsi_observed.csv'
RULES = OUT / 'type_code_rules.json'
SARAIS = OUT / 'mmsi_sarais_lookup.csv'
FILLED = OUT / 'mmsi_fine_class_filled.csv'
SUMMARY = OUT / 'mmsi_fill_summary.json'
SARAIS_API = 'http://nsoasais.gogotrade.info/api/ship/baseInfo'
PURITY_MIN = 0.95          # a code->class rule must be this pure to auto-assign
SUPPORT_MIN = 5            # ...and backed by at least this many resolved MMSI (blocks one-off junk codes)
BATCH = 400                # SARAIS accepts 500 MMSI, 1000 trips a 414 URI-too-long
SCRAPE = OUT / 'mmsi_myshiptracking.jsonl'
CLEAN = lambda v: re.sub(r'\s+', ' ', (v or '').strip())


def iter_ais():
    for pdir in sorted(p for p in PRODUCTS.iterdir() if p.is_dir()):
        for txt in sorted((pdir / 'ais').glob('*.txt')):
            with txt.open('r', encoding='utf-8', errors='replace', newline='') as f:
                for line in f:
                    cols = line.rstrip('\n').split(',')
                    if len(cols) >= 18:
                        yield pdir.name, cols


def stage_scan():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = Counter()
    codes = defaultdict(Counter)
    names = defaultdict(Counter)
    dims = defaultdict(lambda: [Counter(), Counter()])
    with OBSERVED.open('w', encoding='utf-8-sig', newline='') as out:
        w = csv.writer(out)
        w.writerow(['mmsi', 'ais_rows', 'type_codes', 'names', 'length_mode', 'beam_mode'])
        for n, (product, cols) in enumerate(iter_ais(), 1):
            mmsi = CLEAN(cols[1])
            if not (re.fullmatch(r'\d{9}', mmsi) and mmsi != '000000000'):
                continue
            rows[mmsi] += 1
            codes[mmsi][CLEAN(cols[11])] += 1
            if CLEAN(cols[10]):
                names[mmsi][CLEAN(cols[10])] += 1
            if CLEAN(cols[14]):
                dims[mmsi][0][CLEAN(cols[14])] += 1
            if CLEAN(cols[15]):
                dims[mmsi][1][CLEAN(cols[15])] += 1
            if n % 5000000 == 0:
                print('scan %d rows, %d mmsi, %.0fs' % (n, len(rows), time.time() - t0), flush=True)
        for mmsi, count in sorted(rows.items()):
            w.writerow([mmsi, count, json.dumps(codes[mmsi], ensure_ascii=False),
                        json.dumps(dict(names[mmsi].most_common(3)), ensure_ascii=False),
                        dims[mmsi][0].most_common(1)[0][0] if dims[mmsi][0] else '',
                        dims[mmsi][1].most_common(1)[0][0] if dims[mmsi][1] else ''])
    print('scan done: %d mmsi, %d rows, %.0fs' % (len(rows), sum(rows.values()), time.time() - t0), flush=True)


def load_mapping_rows():
    with MAPPING.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def stage_rules():
    """Learn code -> fine class from MMSI the curated layer already resolved."""
    rules = defaultdict(Counter)
    for row in load_mapping_rows():
        final = CLEAN(row.get('final_class'))
        status = CLEAN(row.get('mapping_status'))
        if not final or status != 'fine_class_mapped':
            continue
        for code in CLEAN(row.get('ais_types')).split(';'):
            if code:
                rules[code][final] += 1
    table = {}
    for code, counter in rules.items():
        total = sum(counter.values())
        final, hits = counter.most_common(1)[0]
        table[code] = dict(final_class=final, purity=round(hits / total, 4),
                           support=total, distribution=dict(counter.most_common(6)))
    OUT.mkdir(parents=True, exist_ok=True)
    RULES.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding='utf-8')
    print('rules: %d codes, %d pure at >=%.2f' % (len(table), sum(1 for v in table.values() if v['purity'] >= PURITY_MIN), PURITY_MIN))


def load_observed():
    obs = {}
    with OBSERVED.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            obs[row['mmsi']] = row
    return obs


def stage_sarais():
    """Query the backend only for MMSI whose own AIS rows carry no usable type code."""
    obs = load_observed()
    rules = json.loads(RULES.read_text(encoding='utf-8'))
    known = {r['mmsi'] for r in load_mapping_rows()}
    todo = []
    for mmsi, row in obs.items():
        codes = [c for c in json.loads(row['type_codes'] or '{}') if c and c != '0']
        if mmsi not in known or not any(c in rules for c in codes):
            todo.append((int(row['ais_rows']), mmsi))
    todo.sort(reverse=True)
    todo = [m for _, m in todo]
    print('sarais lookup candidates: %d' % len(todo), flush=True)

    done = set()
    if SARAIS.exists():
        with SARAIS.open('r', encoding='utf-8-sig', newline='') as f:
            done = {r['mmsi'] for r in csv.DictReader(f)}
    todo = [m for m in todo if m not in done]
    print('remaining after resume: %d' % len(todo), flush=True)

    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    new_file = not SARAIS.exists()
    with SARAIS.open('a', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['mmsi', 'found', 'name', 'imo', 'callsign', 'ais_type_code', 'length', 'width', 'dwt', 'flag'])
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            url = SARAIS_API + '?' + urllib.parse.urlencode({'mmsis': ','.join(chunk)})
            try:
                payload = json.loads(op.open(urllib.request.Request(url, headers={'User-Agent': 'hermes/1.0'}), timeout=180).read())
            except Exception as exc:
                print('batch %d failed: %s' % (i // BATCH, repr(exc)[:120]), flush=True)
                time.sleep(5)
                continue
            hits = {}
            for item in payload.get('data') or []:
                hits[str(item.get('mmsi'))] = item
            for mmsi in chunk:
                item = hits.get(mmsi, {})
                w.writerow([mmsi, bool(item), CLEAN(item.get('name')), CLEAN(item.get('imo')), CLEAN(item.get('callsign')),
                            CLEAN(str(item.get('type', ''))), item.get('length', ''), item.get('width', ''), item.get('dwt', ''), CLEAN(item.get('flag'))])
            f.flush()
            if (i // BATCH) % 10 == 0:
                print('sarais %d/%d batches, %.0fs' % (i // BATCH, (len(todo) + BATCH - 1) // BATCH, time.time()), flush=True)
    print('sarais stage done', flush=True)


def top_name(row):
    names = json.loads(row['names'] or '{}')
    return next(iter(names), '') if names else ''


def load_scrape_labels():
    """MMSI -> scraped vessel label from MyShipTracking (read-only lookups)."""
    if not SCRAPE.is_file():
        return {}
    out = {}
    with SCRAPE.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get('label'):
                out[rec['mmsi']] = rec['label']
    return out


def label_rules():
    """Scraped label -> fine class, learned from the corpus's own curated mapping rows."""
    seen = defaultdict(Counter)
    for row in load_mapping_rows():
        src, fin = CLEAN(row.get('selected_mapping_type')), CLEAN(row.get('final_class'))
        if src and fin:
            seen[src.lower()][fin] += 1
    rules = {}
    for src, counter in seen.items():
        total = sum(counter.values())
        fin, hits = counter.most_common(1)[0]
        if hits / total >= 0.9:
            rules[src] = dict(final_class=fin, purity=round(hits / total, 4), support=total)
    return rules


def stage_fill():
    rules = json.loads(RULES.read_text(encoding='utf-8'))
    labels = label_rules()
    scraped = load_scrape_labels()
    obs = load_observed()
    mapping = {r['mmsi']: r for r in load_mapping_rows()}
    sarais = {}
    if SARAIS.exists():
        with SARAIS.open('r', encoding='utf-8-sig', newline='') as f:
            sarais = {r['mmsi']: r for r in csv.DictReader(f)}

    filled = []
    stats = Counter()
    for mmsi, row in obs.items():
        cur = mapping.get(mmsi, {})
        if CLEAN(cur.get('final_class')) and CLEAN(cur.get('mapping_status')) == 'fine_class_mapped':
            stats['already_resolved'] += 1
            continue
        codes = [c for c in json.loads(row['type_codes'] or '{}') if c and c != '0']
        chosen = None
        for code in sorted(codes, key=lambda c: -rules.get(c, {}).get('support', 0)):
            rule = rules.get(code)
            if rule and rule['purity'] >= PURITY_MIN and rule['support'] >= SUPPORT_MIN:
                chosen = ('bundle_ais_type_code', code, rule)
                break
        if chosen is None:
            sc = CLEAN(sarais.get(mmsi, {}).get('ais_type_code'))
            rule = rules.get(sc)
            if sc and rule and rule['purity'] >= PURITY_MIN and rule['support'] >= SUPPORT_MIN:
                chosen = ('sarais_backend_type_code', sc, rule)
        if chosen is None:
            lab = CLEAN(scraped.get(mmsi))
            lrule = labels.get(lab.lower()) if lab else None
            if lrule:
                chosen = ('myshiptracking_label', lab, lrule)
        if chosen is None:
            stats['needs_manual'] += 1
            filled.append(dict(mmsi=mmsi, ais_rows=row['ais_rows'], final_class='', source='needs_manual',
                               type_code=';'.join(codes), rule_purity='', rule_support='',
                               sarais_type_code=CLEAN(sarais.get(mmsi, {}).get('ais_type_code')),
                               name=top_name(row),
                               prev_status=CLEAN(cur.get('mapping_status')) or 'absent_from_mapping'))
            continue
        source, code, rule = chosen
        stats['filled_' + source] += 1
        confidence = ('high' if rule['support'] >= 100 else 'medium' if rule['support'] >= 10 else 'low')
        filled.append(dict(mmsi=mmsi, ais_rows=row['ais_rows'], final_class=rule['final_class'], source=source,
                           type_code=code, rule_purity=rule['purity'], rule_support=rule['support'],
                           confidence=confidence,
                           sarais_type_code=CLEAN(sarais.get(mmsi, {}).get('ais_type_code')),
                           name=top_name(row),
                           prev_status=CLEAN(cur.get('mapping_status')) or 'absent_from_mapping'))
    fields = ['mmsi', 'ais_rows', 'final_class', 'source', 'type_code', 'rule_purity', 'rule_support',
              'confidence', 'sarais_type_code', 'name', 'prev_status']
    with FILLED.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(filled)
    rows_covered = sum(int(r['ais_rows']) for r in filled if r['final_class'])
    unresolved_rows = sum(int(r['ais_rows']) for r in filled if not r['final_class'])
    summary = dict(stats=dict(stats), new_classes=len({r['final_class'] for r in filled if r['final_class']}),
                   ais_rows_newly_resolved=rows_covered, ais_rows_still_unresolved=unresolved_rows,
                   purity_min=PURITY_MIN, generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


STAGES = dict(scan=stage_scan, rules=stage_rules, sarais=stage_sarais, fill=stage_fill)
if __name__ == '__main__':
    for name in (sys.argv[1:] or ['scan', 'rules', 'sarais', 'fill']):
        print('=== stage %s ===' % name, flush=True)
        STAGES[name]()
