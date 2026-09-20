#!/usr/bin/env python3
"""Train the fixed H=8 bank from a frozen encoder cache.
Requires features.npz z[N,512], m[N,4], sample_id[N], port[N], product_id[N].
Requires source-fit-only labels CSV sample_id,label (NO target/meta/cal rows).
Does not train the encoder, infer target labels, choose B0 or tune thresholds.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models import CandidateHead

def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--target',required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int)
 a=p.parse_args();cfg=json.loads(a.config.read_text());m=json.loads(a.manifest.read_text());fold=m['folds'][a.target]
 if a.out.exists():raise FileExistsError(a.out)
 cache=np.load(a.features,allow_pickle=False)
 if {'y','labels','ais_final_class','matched_mmsi','support_i'}&set(cache.files):raise ValueError('Feature cache must contain no labels/AIS fields')
 ids=cache['sample_id'].astype(str);ports=cache['port'].astype(str);products=cache['product_id'].astype(str)
 z=cache['z'];mm=cache['m'];N=len(ids)
 if z.shape!=(N,512) or mm.shape!=(N,4) or len(set(ids))!=N:raise ValueError('Invalid feature cache shape/IDs')
 if not np.isfinite(z).all() or not np.isfinite(mm).all():raise ValueError('Features must be finite, explicit preprocessing required')
 lookup={s:i for i,s in enumerate(ids)};labels={}
 with a.labels.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   key=row['sample_id'];c=row['label']
   if key not in lookup:raise ValueError(f'Label ID absent from feature cache: {key}')
   i=lookup[key]
   if ports[i] not in fold['source_fit_ports'] or products[i] not in fold['source_fit_products']:
    raise ValueError('Labels file contains a target/meta/calibration product; provide source-fit-only file')
   if key in labels:raise ValueError('Duplicate label ID')
   if c.lower() in ('unknown','ship_untyped','untyped','non_ship','') or c.endswith('_coarse'):continue
   labels[key]=c
 cnt=Counter(labels.values());ps=defaultdict(set)
 for key,c in labels.items():ps[c].add(ports[lookup[key]])
 vocab=sorted(c for c,n in cnt.items() if n>=100 and len(ps[c])>=3)
 if len(vocab)<2:raise ValueError('Source-fit class support insufficient; do not choose classes with target labels')
 ci={c:i for i,c in enumerate(vocab)};ii=np.array([lookup[k] for k,c in labels.items() if c in ci]);yy=np.array([ci[labels[ids[i]]] for i in ii])
 xx=np.concatenate([z,mm],1).astype('float32');mean=xx[ii].mean(0);std=xx[ii].std(0).clip(.001)
 x=torch.from_numpy(((xx[ii]-mean)/std).astype('float32'));y=torch.from_numpy(yy).long();pp=ports[ii]
 device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; pass --device cpu for smoke only')
 torch.set_num_threads(8);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 a.out.mkdir(parents=True);np.savez(a.out/'feature_scaler.npz',mean=mean,std=std)
 (a.out/'class_vocab.json').write_text(json.dumps(vocab,indent=2))
 batch=cfg['candidate_bank']['batch_size'];epochs=a.epochs or cfg['candidate_bank']['epochs'];head_id=0;records=[]
 for mode in cfg['candidate_bank']['sampling_rules']:
  count=Counter(pp if mode=='uniform_port' else list(zip(pp,yy)))
  keys=pp if mode=='uniform_port' else list(zip(pp,yy))
  nclass_by_port={v:len(set(yy[pp==v])) for v in set(pp)}
  weights=np.array([1/count[k] if mode=='uniform_port' else 1/(nclass_by_port[k[0]]*count[k]) for k in keys],dtype=float);weights/=weights.sum()
  for arch in cfg['candidate_bank']['architectures']:
   for off in cfg['seeds']['head_offsets']:
    seed=cfg['seeds']['first_run']+off;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    net=CandidateHead(516,len(vocab),arch).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['candidate_bank']['lr'],weight_decay=cfg['candidate_bank']['weight_decay'])
    rng=np.random.default_rng(seed);losses=[]
    for ep in range(epochs):
     net.train();perm=rng.choice(len(ii),size=len(ii),replace=True,p=weights);total=0.
     for start in range(0,len(perm),batch):
      ix=perm[start:start+batch];xb=x[ix].to(device);yb=y[ix].to(device)
      opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(net(xb),yb)
      loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();total+=float(loss.detach())*len(ix)
     losses.append(total/len(ii))
    path=a.out/f'h{head_id}.pt';torch.save({'state_dict':net.cpu().state_dict(),'architecture':arch,'in_dim':516,'classes':len(vocab)},path)
    records.append({'head_id':f'h{head_id}','architecture':arch,'sampling':mode,'seed':seed,'epochs':epochs,'last_loss':losses[-1],
                    'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'train_loss_history':losses});head_id+=1
 (a.out/'bank_manifest.json').write_text(json.dumps({'H':head_id,'target':a.target,'source_fit_ports':fold['source_fit_ports'],'source_fit_labelled_objects':len(ii),'heads':records,
     'features_sha256':hashlib.sha256(a.features.read_bytes()).hexdigest(),'epochs_override':a.epochs,'target_labels_read':False},ensure_ascii=False,indent=2))
 print(json.dumps({'H':head_id,'source_examples':len(ii),'classes':len(vocab),'output':str(a.out)},ensure_ascii=False))
if __name__=='__main__':main()
