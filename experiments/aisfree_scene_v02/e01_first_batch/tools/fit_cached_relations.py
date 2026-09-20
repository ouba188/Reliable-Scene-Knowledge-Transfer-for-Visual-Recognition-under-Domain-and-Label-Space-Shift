#!/usr/bin/env python3
"""Train the J=3 conditional MOMENT models on source-fit labels only.
No target pseudo-labels, no calibration/meta gradients. Fixed source group projection.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,csv,json,random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models import RelationMomentModel

def main():
 p=argparse.ArgumentParser()
 for arg in ('features','relations','group_model','heads','labels','manifest','config','out'):
  p.add_argument('--'+arg.replace('_','-'),type=Path,required=True)
 p.add_argument('--target',required=True);p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int)
 a=p.parse_args();cfg=json.loads(a.config.read_text());fold=json.loads(a.manifest.read_text())['folds'][a.target]
 if a.out.exists():raise FileExistsError(a.out)
 f=np.load(a.features,allow_pickle=False);r=np.load(a.relations,allow_pickle=False);g=np.load(a.group_model,allow_pickle=False)
 ids=f['sample_id'].astype(str);ports=f['port'].astype(str);products=f['product_id'].astype(str);N=len(ids)
 if len(set(ids))!=N:raise ValueError('Duplicate feature ids')
 rids=r['sample_id'].astype(str);lookup={s:i for i,s in enumerate(ids)};rlookup={s:i for i,s in enumerate(rids)}
 if len(rlookup)!=len(rids):raise ValueError('Duplicate relation ids')
 vocab=json.loads((a.heads/'class_vocab.json').read_text());ci={c:i for i,c in enumerate(vocab)};rows=[];ys=[]
 with a.labels.open(encoding='utf-8-sig',newline='') as stream:
  for row in csv.DictReader(stream):
   key=row['sample_id'];c=row['label']
   if key not in lookup:raise ValueError('Label missing feature id')
   i=lookup[key]
   if ports[i] not in fold['source_fit_ports'] or products[i] not in fold['source_fit_products']:raise ValueError('Non-source-fit label supplied')
   if c in ci and key in rlookup:rows.append(i);ys.append(ci[c])
 if not rows or len(set(rows))!=len(rows):raise ValueError('Empty or duplicate source rows')
 ii=np.asarray(rows);yy=np.asarray(ys);ri=np.asarray([rlookup[ids[i]] for i in ii])
 phi=r['phi'][ri].astype('float32');mask=r['available'][ri].astype(bool);C=len(vocab);D=cfg['D']
 enabled=np.ones(D,bool);details={}
 for d in range(D):
  good=mask[:,d]&np.isfinite(phi[:,d]);
  if good.mean()<.10 or (good.any() and phi[good,d].var()<1e-4):enabled[d]=False
  checks=[]
  for c in range(C):
   sel=good&(yy==c);n=int(sel.sum());nport=len(set(ports[ii][sel]));checks.append([vocab[c],n,nport])
   if n<30 or nport<2:enabled[d]=False
  details[str(d)]=checks
 # Same disabled dimensions across J; cannot make the mechanism bank exploit NaN patterns.
 mask &= enabled[None,:]
 if not enabled.any():raise ValueError('No estimable D from source-fit; report and use B1, do not use target to activate dimensions')
 z16=((f['z'][ii]-g['scaler_mean'])/g['scaler_scale']-g['pca_mean'])@g['pca_components'].T
 sc=np.load(a.heads/'feature_scaler.npz');mm=(f['m'][ii]-sc['mean'][-4:])/sc['std'][-4:]
 inp=np.concatenate([z16,mm],1).astype('float32')
 if not np.isfinite(inp).all():raise ValueError('Nonfinite inputs')
 X=torch.from_numpy(inp);Y=torch.from_numpy(yy).long();T=torch.from_numpy(np.nan_to_num(phi,nan=0));V=torch.from_numpy(mask)
 dev=torch.device(a.device)
 if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.set_num_threads(8);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 count=Counter(zip(ports[ii],yy));nc={p:len(set(yy[ports[ii]==p])) for p in set(ports[ii])}
 base=np.asarray([1/(nc[p]*count[(p,c)]) for p,c in zip(ports[ii],yy)])
 a.out.mkdir(parents=True);epochs=a.epochs or cfg['relation_models']['epochs'];batch=cfg['relation_models']['batch_size'];records=[]
 for j,(name,pweights) in enumerate(fold['mechanism_port_weight'].items()):
  seed=cfg['seeds']['relation'][j];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
  if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
  model=RelationMomentModel(20,C,D).to(dev)
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
  w=base*np.array([pweights[p] for p in ports[ii]]);w/=w.sum();rng=np.random.default_rng(seed);history=[]
  for ep in range(epochs):
   sel=rng.choice(len(ii),size=len(ii),replace=True,p=w);loss_sum=0.;used=0
   for start in range(0,len(sel),batch):
    ix=sel[start:start+batch];vm=V[ix].to(dev)
    if not vm.any():continue
    opt.zero_grad(set_to_none=True)
    loss=nn.functional.binary_cross_entropy_with_logits(model.logits(X[ix].to(dev),Y[ix].to(dev)),T[ix].to(dev),reduction='none')
    loss=loss[vm].mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();loss_sum+=float(loss.detach());used+=1
   history.append(loss_sum/max(used,1))
  torch.save({'state_dict':model.cpu().state_dict(),'in_dim':20,'classes':C,'D':D},a.out/f'j{j}.pt')
  records.append({'mechanism':name,'seed':seed,'epochs':epochs,'train_loss_history':history,'port_weights':pweights})
 (a.out/'relation_manifest.json').write_text(json.dumps({'J':3,'D':D,'enabled_dimensions':enabled.tolist(),'support':details,'models':records,'target_labels_read':False,'source_fit_ports':fold['source_fit_ports'],'epochs_override':a.epochs},ensure_ascii=False,indent=2))
 print(json.dumps({'enabled_dimensions':enabled.tolist(),'output':str(a.out)}))
if __name__=='__main__':main()
