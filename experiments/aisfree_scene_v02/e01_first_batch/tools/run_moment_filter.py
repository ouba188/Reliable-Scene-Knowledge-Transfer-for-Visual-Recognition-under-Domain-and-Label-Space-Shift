#!/usr/bin/env python3
"""Run aggregation/filter on model outputs. Does not train the models."""
import argparse,json
from pathlib import Path
import numpy as np
from e01_core import aggregate,filter_bank,predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--calibration',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text());data=np.load(a.input,allow_pickle=False)
    forbidden={'y','labels','target_labels','ais_final_class','matched_mmsi','support_i','support_missing'}
    if forbidden&set(data.files):raise ValueError(f'Forbidden input keys {forbidden&set(data.files)}')
    if a.out.exists():raise FileExistsError(a.out)
    valid=cfg['moment_validity'];cal=cfg['calibration']
    if a.smoke: c={'status':'ready','tau':cal['tau_fixed_smoke'],'scope':'SMOKE_ONLY'}
    elif a.calibration:c=json.loads(a.calibration.read_text())
    else:raise ValueError('Real run requires --calibration; fixed smoke threshold is not a result threshold')
    m=aggregate(data['phi'],data['available'],data['groups'],data['spatial_blocks'],data['probabilities'],data['relation_means'],
        B=cfg['B'],min_cell=valid['min_available_objects_per_cell'],min_blocks=valid['min_spatial_blocks_per_cell'],
        enabled_dimensions=data['enabled_dimensions'] if 'enabled_dimensions' in data.files else None)
    if c['status']!='ready':r={'retained':list(range(cfg['H'])),'feasible':[],'conflict':False,'status':c['status']}
    else:r=filter_bank(m.observed,m.predicted,m.active,float(c['tau']))
    pred=predictions(data['probabilities'],r['retained']);a.out.mkdir(parents=True)
    np.savez_compressed(a.out/'moments.npz',observed=m.observed,predicted=m.predicted,active=m.active,counts=m.counts,blocks=m.blocks,reasons=m.reasons)
    np.savez_compressed(a.out/'predictions.npz',**pred)
    (a.out/'retention.json').write_text(json.dumps({**r,'smoke':a.smoke,'calibration':c},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
