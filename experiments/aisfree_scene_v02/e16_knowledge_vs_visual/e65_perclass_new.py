"""e65: per-port x per-class breakdown for the e63 arms (the new configurations).

Reconstructs each arm's predictions from the stored gate scores (argmax / the theta rule), then prints
(1) the per-class aggregate over ports, n-weighted, for every arm, and (2) the full per-port x per-class
delta grid for the recommended arm. Writes e65_perclass_new.csv with every cell so nothing is hidden in
a summary. Read-only on our own pickles.
"""
import csv
import pickle
from collections import defaultdict

import numpy as np

C = 8
CANDS = ['ridge_V', 'ridge_VK', 'gbm_zk']
NAMES = {0: 'bulk_carrier', 1: 'fishing_vessel', 2: 'general_cargo', 3: 'product_chem_tanker',
         4: 'container_ship', 5: 'crude_oil_tanker', 6: 'tug_towing', 7: 'offshore_supply'}


def load(tag):
    d = pickle.load(open('e63_preds_%s.pkl' % tag, 'rb'))      # our own artifact, written by e63
    out = []
    for rec in d['dump']:
        y, P = rec['y'], rec['pred']
        a = {'ridge_V': P['ridge_V'], 'ridge_VK': P['ridge_VK'], 'gbm_zk': P['gbm_zk']}
        s_old, s_new, th = rec['sc_old'], rec['sc_new'], rec['theta']
        a['gate_old'] = np.array([P[CANDS[j]][i] for i, j in enumerate(s_old.argmax(1))])
        a['gate_new'] = np.array([P[CANDS[j]][i] for i, j in enumerate(s_new.argmax(1))])
        pick = np.where(s_new[:, 2] - s_new[:, 1] > th, 2, 1)
        a['gate_new_theta'] = np.array([P[CANDS[j]][i] for i, j in enumerate(pick)])
        ok = np.zeros(len(y), bool)
        for nm in CANDS:
            ok |= (P[nm] == y)
        a['oracle3'] = np.where(ok, y, -1)
        out.append({'port': rec['port'], 'y': y, 'arms': a})
    return out


def rec(y, p, c):
    m = (y == c)
    return float((p[m] == c).mean()) if m.any() else float('nan')


for tag, label in [('n8', '8 源港'), ('n3nk', '3 源港(删原始知识)')]:
    d = load(tag)
    arm_names = ['ridge_V', 'ridge_VK', 'gbm_zk', 'gate_old', 'gate_new', 'gate_new_theta', 'oracle3']
    print('=' * 96)
    print('【%s】逐类召回（按 n 跨 24 港加权）' % label)
    print('%-22s %6s' % ('class', 'n') + ''.join('%9s' % a for a in arm_names))
    num = defaultdict(lambda: defaultdict(float)); den = defaultdict(float)
    for r in d:
        for c in range(C):
            m = (r['y'] == c)
            if not m.any():
                continue
            den[c] += int(m.sum())
            for a in arm_names:
                num[a][c] += rec(r['y'], r['arms'][a], c) * m.sum()
    for c in range(C):
        if den[c] == 0:
            continue
        print('%-22s %6d' % (NAMES[c], int(den[c])) + ''.join('%9.3f' % (num[a][c] / den[c]) for a in arm_names))
    print()
    port_rows = []
    for r in d:
        for c in range(C):
            m = (r['y'] == c)
            if not m.any():
                continue
            port_rows.append([r['port'], c, NAMES[c], int(m.sum())] +
                             [rec(r['y'], r['arms'][a], c) for a in arm_names])
    with open('e65_perclass_%s.csv' % tag, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['port', 'class', 'class_name', 'n'] + arm_names); w.writerows(port_rows)
    print('全网格(每港每类每臂) -> e65_perclass_%s.csv  共 %d 个单元' % (tag, len(port_rows)))

# the recommended arm's per-port x per-class delta grid, in pp
d = load('n8')
print()
print('=' * 96)
print('【8 源港】Δ(gate_old − ridge_V)，单位 pp；· = 该港无此类')
print('| port | ' + ' | '.join(['bulk', 'fish', 'gcargo', 'pchem', 'cont', 'crude', 'tug', 'offsh']) + ' | portΔ |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')


def ba(y, p):
    rs = [rec(y, p, c) for c in range(C) if (y == c).any()]
    return float(np.nanmean(rs))


for r in sorted(d, key=lambda z: ba(z['y'], z['arms']['gate_old']) - ba(z['y'], z['arms']['ridge_V'])):
    y = r['y']; cells = []
    for c in range(C):
        m = (y == c)
        if not m.any():
            cells.append('·')
        else:
            cells.append('%+.0f' % ((rec(y, r['arms']['gate_old'], c) - rec(y, r['arms']['ridge_V'], c)) * 100))
    pd_ = (ba(y, r['arms']['gate_old']) - ba(y, r['arms']['ridge_V'])) * 100
    print('| %s | %s | %+.0f |' % (r['port'], ' | '.join(cells), pd_))
