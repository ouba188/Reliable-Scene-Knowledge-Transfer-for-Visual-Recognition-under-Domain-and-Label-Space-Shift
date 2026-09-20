#!/usr/bin/env python3
"""Prepare D=4 from a sanitized E00 P0 table. No AIS or label columns are read.
Coverage CSV contract: product_id,coast_layer_observed,anchorage_layer_observed,
fairway_layer_observed. These booleans must reflect layer evidence, not hit counts.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def boolean(x):
 s=x.astype(str).str.lower().str.strip()
 if not s.isin(['true','false','1','0','1.0','0.0']).all():raise ValueError('Coverage booleans must be explicit 0/1/true/false')
 return s.isin(['true','1','1.0']).to_numpy()

def main():
 p=argparse.ArgumentParser();p.add_argument('--objects',type=Path,required=True);p.add_argument('--coverage',type=Path,required=True)
 p.add_argument('--mask-column',required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--density-scope',choices=['p0_offshore_dedup_leave_one_out'],required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 need=['object_id','product_id','port','world_x','world_y','valid_fraction','d_coast_m','d_anchorage_m','d_fairway_m','local_ships_1km',a.mask_column]
 df=pd.read_csv(a.objects,usecols=need,dtype={'object_id':str,'product_id':str,'port':str})
 # Mask assignment comes from the audited E00 join; do not reclassify by AIS/support.
 df=df.loc[pd.to_numeric(df[a.mask_column],errors='coerce')==3].copy()
 expected=408981
 if len(df)!=expected:raise ValueError(f'Expected E00 P0 {expected} rows before image gates, got {len(df)}; do not silently change task population')
 if df.object_id.duplicated().any():raise ValueError('Duplicate representative object_id')
 cov=pd.read_csv(a.coverage,dtype={'product_id':str})
 if cov.product_id.duplicated().any():raise ValueError('Coverage must have one row/product')
 df=df.merge(cov,on='product_id',how='left',validate='many_to_one')
 names=['coast_layer_observed','anchorage_layer_observed','fairway_layer_observed']
 vals=[];masks=[]
 for col,flag in zip(['d_coast_m','d_anchorage_m','d_fairway_m'],names):
  x=pd.to_numeric(df[col],errors='coerce').to_numpy(float)
  v=boolean(df[flag])&np.isfinite(x)&(x>=0)&(x<=2000)
  y=np.clip((2000-x)/(1500 if col=='d_coast_m' else 2000),0,1)
  vals.append(np.where(v,y,np.nan));masks.append(v)
 n=pd.to_numeric(df.local_ships_1km,errors='coerce').to_numpy(float)
 ok=np.isfinite(n)&(n>=0)
 vals.append(np.where(ok,np.log1p(np.minimum(n,64))/np.log(65),np.nan));masks.append(ok)
 x=pd.to_numeric(df.world_x,errors='raise').to_numpy(float);y=pd.to_numeric(df.world_y,errors='raise').to_numpy(float)
 block=np.array([f'{pr}|{int(xx//1000)}|{int(yy//1000)}' for pr,xx,yy in zip(df.product_id,x,y)])
 a.out.mkdir(parents=True)
 np.savez_compressed(a.out/'relations.npz',sample_id=df.object_id.to_numpy(str),product_id=df.product_id.to_numpy(str),port=df.port.to_numpy(str),
     phi=np.stack(vals,1).astype('float32'),available=np.stack(masks,1),spatial_blocks=block,valid_fraction=pd.to_numeric(df.valid_fraction,errors='coerce').to_numpy(float))
 (a.out/'report.json').write_text(json.dumps({'P0_objects':len(df),'D':4,'valid_counts':np.stack(masks,1).sum(0).tolist(),'density_scope':a.density_scope,'labels_read':False,'new_sample_filter':False},indent=2))
if __name__=='__main__':main()
