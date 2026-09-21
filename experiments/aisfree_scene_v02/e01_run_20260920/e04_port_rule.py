"""E04: port-level leave-out evaluation of a Cross/Mean-only selection rule (BA-primary, regret-based).

Protocol (per reviewer):
  * episode = (fold, held-out port P): reference pool = the fold's source_fit-labelled instances
    EXCLUDING P's own instances ("no donor from the held-out port"); adaptation pool U_P = P's other
    products (unlabelled statistics only); evaluation = P's labelled instances.
  * the statistic, its direction and its threshold are chosen ONLY on the training side (all episodes of
    the other ports, across folds); the held-out port contributes no episode to training.
  * primary metric is BA; Acc and per-class support are reported alongside. Regret is measured against
    the per-episode two-action oracle (Cross vs Mean-only), which no rule of this family can exceed.
  * Mean-only acts as a single per-class score bias: entropy/confidence changes are recorded as
    auxiliary signals only, never as evidence that "observation conditions were fixed".

usage: python e04_port_rule.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
SRCDEV = ROOT / 'e01_runs' / 'aux_source_dev'
KS = ROOT / 'knowledge_841_20260920'
OUT = ROOT / 'e04_port_rule'
FOLDS = ['Rotterdam', 'Shanghai', 'PortKlang', 'Fujairah', 'JebelAli', 'PortSaid']
META = {'Rotterdam': 'Rotterdam', 'Shanghai': 'Shanghai', 'PortKlang': 'Port Klang',
        'Fujairah': 'Fujairah', 'JebelAli': 'Jebel Ali', 'PortSaid': 'Port Said'}
PCA_DIMS = 64
MIN_PORT, MIN_REF = 150, 200


def ba(y, pred, C):
    rs = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def ent(p):
    p = np.clip(p, 1e-9, 1)
    return float(-(p * np.log(p)).sum(1).mean())


def energy_distance(X, Y, cap=800, seed=20260920):
    rng = np.random.default_rng(seed)
    def sub(Z):
        return Z if len(Z) <= cap else Z[rng.choice(len(Z), cap, replace=False)]
    X, Y = sub(X), sub(Y)
    d_xy = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2).mean()
    d_xx = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2).mean()
    d_yy = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2).mean()
    return float(2 * d_xy - d_xx - d_yy)


def build_episodes():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    man = json.loads((KS / 'e01_first_batch' / 'split_manifest.json').read_text())['folds']
    rows = []
    for tag in FOLDS:
        d = np.load(SRCDEV / (tag + '_source.npz'), allow_pickle=False)
        z_raw, y, port = d['z'].astype(np.float64), d['y'], d['port'].astype(str)
        vocab = [str(x) for x in d['class_names']]
        C = len(vocab)
        fit_ports = set(man[META[tag]].get('source_fit_ports', []))
        ports = [p for p in sorted(set(port.tolist())) if (port == p).sum() >= MIN_PORT
                 and len(set(y[port == p].tolist())) >= 2]
        for P in ports:
            m_ref = np.array([(pt != P) and (pt in fit_ports) for pt in port.tolist()])
            if m_ref.sum() < MIN_REF or len(set(y[m_ref].tolist())) < 2:
                continue
            Xr_raw, yr = z_raw[m_ref], y[m_ref]
            Xa_raw, ya = z_raw[port == P], y[port == P]
            pca = PCA(n_components=min(PCA_DIMS, Xr_raw.shape[1], max(2, Xr_raw.shape[0] - 1)),
                      random_state=20260920).fit(Xr_raw)
            Xr, Xa = pca.transform(Xr_raw), pca.transform(Xa_raw)
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xr, yr)
            cross = clf.predict(Xa)
            Xa_corr = Xa - Xa.mean(0) + Xr.mean(0)
            meanp = clf.predict(Xa_corr)
            pr, pa, pac = clf.predict_proba(Xa), clf.predict_proba(Xa_corr), None
            b_cross, b_mean = ba(ya, cross, C), ba(ya, meanp, C)
            a_cross, a_mean = float((cross == ya).mean()), float((meanp == ya).mean())
            sup = {vocab[c]: int((ya == c).sum()) for c in range(C) if (ya == c).any()}
            rows.append({
                'fold': tag, 'held_out_port': P, 'held_out_port_role':
                    'fit' if P in fit_ports else ('meta' if P in set(man[META[tag]].get('source_meta_query_ports', [])) else 'other'),
                'n_ref': int(m_ref.sum()), 'n_adapt': int(len(ya)),
                'stat_shift_norm': float(np.linalg.norm(Xa.mean(0) - Xr.mean(0)) / max(1e-9, Xr.std(0).mean())),
                'stat_energy_dist': energy_distance(Xr, Xa),
                'stat_entropy_before': ent(pr), 'stat_entropy_after': ent(pa),
                'stat_delta_entropy': ent(pa) - ent(pr),
                'stat_conf_before': float(pr.max(1).mean()), 'stat_conf_after': float(pa.max(1).mean()),
                'stat_delta_conf': float(pa.max(1).mean() - pr.max(1).mean()),
                'stat_bias_shift_max_abs': float(np.abs(Xa.mean(0) - Xr.mean(0)).max()),
                'cross_acc': a_cross, 'mean_acc': a_mean, 'cross_ba': b_cross, 'mean_ba': b_mean,
                'delta_acc': a_mean - a_cross, 'delta_ba': b_mean - b_cross,
                'oracle_acc': max(a_cross, a_mean), 'oracle_ba': max(b_cross, b_mean),
                'class_support': json.dumps(sup, ensure_ascii=False),
            })
    return rows


def select_rule(train, stat):
    """Training-side only: pick direction+threshold maximising mean BA gain (ties -> smaller |gain|)."""
    v = np.array([r[stat] for r in train])
    dba = np.array([r['delta_ba'] for r in train])
    best = None
    for direction in (1, -1):
        for t in np.unique(np.round(v, 6)):
            sel = (direction * v) >= (direction * t)
            if sel.sum() == 0 or sel.sum() == len(sel):
                continue
            gain = float(dba[sel].mean() * sel.mean())     # BA gain over always-Cross
            if best is None or gain > best[0] + 1e-12:
                best = (gain, direction, float(t))
    return best or (0.0, 1, float(np.median(v)))


def main() -> None:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    rows = build_episodes()
    OUT.mkdir(parents=True, exist_ok=True)
    stats = [k for k in rows[0] if k.startswith('stat_')]
    ports = sorted({r['held_out_port'] for r in rows})

    report = {'protocol': 'leave-one-port-out (whole held-out port, no donor from that port)',
              'primary_metric': 'BA', 'n_episodes': len(rows), 'ports': ports, 'rules': {}}
    table = []
    for stat in stats:
        picks, acts = [], []
        for P in ports:
            tr = [r for r in rows if r['held_out_port'] != P]
            te = [r for r in rows if r['held_out_port'] == P]
            if not tr or not te:
                continue
            gain, direction, t = select_rule(tr, stat)
            for r in te:
                use = direction * r[stat] >= direction * t
                picks.append({'fold': r['fold'], 'held_out_port': P, 'statistic': stat,
                              'direction': direction, 'threshold_selected_on_train': t,
                              'train_mean_ba_gain': gain, 'signal_value': r[stat], 'action': 'mean' if use else 'cross',
                              'ba': r['mean_ba'] if use else r['cross_ba'],
                              'acc': r['mean_acc'] if use else r['cross_acc'],
                              'cross_ba': r['cross_ba'], 'mean_ba': r['mean_ba'],
                              'oracle_ba': r['oracle_ba'], 'delta_ba': r['delta_ba'],
                              'regret_vs_oracle_ba': r['oracle_ba'] - (r['mean_ba'] if use else r['cross_ba']),
                              'class_support': r['class_support']})
        if not picks:
            continue
        bah = np.array([p['ba'] for p in picks]); ora = np.array([p['oracle_ba'] for p in picks])
        cba = np.array([p['cross_ba'] for p in picks]); mba = np.array([p['mean_ba'] for p in picks])
        table.extend(picks)
        report['rules'][stat] = {
            'rule_ba': float(bah.mean()), 'fixed_cross_ba': float(cba.mean()),
            'fixed_mean_ba': float(mba.mean()), 'oracle_ba': float(ora.mean()),
            'gain_over_cross': float(bah.mean() - cba.mean()),
            'gain_over_mean': float(bah.mean() - mba.mean()),
            'mean_regret_vs_oracle': float((ora - bah).mean()),
            'picked_mean_only': int(sum(1 for p in picks if p['action'] == 'mean')),
            'n_episodes': len(picks)}
    with (OUT / 'pair_signal_table.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    with (OUT / 'episode_table.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (OUT / 'source_leave_port_rule_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print('episodes %d，留出港 %d 个' % (len(rows), len(ports)))
    print('%-22s %8s %12s %11s %9s %9s %10s %8s' % ('统计量', '规则BA', '固定Cross', '固定Mean',
                                                    'oracle', 'vs Cross', 'vs Mean', '后悔'))
    for s, v in sorted(report['rules'].items(), key=lambda kv: -kv[1]['gain_over_cross']):
        print('%-22s %8.3f %12.3f %11.3f %9.3f %+9.3f %+10.3f %8.3f' % (
            s, v['rule_ba'], v['fixed_cross_ba'], v['fixed_mean_ba'], v['oracle_ba'],
            v['gain_over_cross'], v['gain_over_mean'], v['mean_regret_vs_oracle']))


if __name__ == '__main__':
    main()
