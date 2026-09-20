#!/usr/bin/env python3
"""Fit B=4 on source-fit only. Apply to all feature rows, never fit on target."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--target',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 data=np.load(a.features,allow_pickle=False);z=data['z'];port=data['port'].astype(str);ids=data['sample_id'].astype(str)
 fold=json.loads(a.manifest.read_text())['folds'][a.target]
 if z.ndim!=2 or z.shape[1]!=512 or len(z)!=len(port) or len(z)!=len(ids):raise ValueError('Need z[N,512], port[N], sample_id[N]')
 if not np.isfinite(z).all():raise ValueError('z must be finite')
 idx=[]
 for pp in fold['source_fit_ports']:
  ii=np.flatnonzero(port==pp).tolist()
  ii.sort(key=lambda i:hashlib.sha256(ids[i].encode()).hexdigest());idx.extend(ii[:2000])
 if len(idx)<64:raise ValueError('Insufficient source-fit examples')
 s=StandardScaler().fit(z[idx]);pca=PCA(n_components=16,whiten=False,random_state=20260920).fit(s.transform(z[idx]))
 km=KMeans(n_clusters=4,n_init=20,random_state=20260920).fit(pca.transform(s.transform(z[idx])))
 zz=pca.transform(s.transform(z));g=km.predict(zz)
 a.out.mkdir(parents=True)
 np.savez_compressed(a.out/'groups.npz',sample_id=ids,group_id=g,z16=zz.astype('float32'))
 np.savez_compressed(a.out/'group_model.npz',scaler_mean=s.mean_,scaler_scale=s.scale_,pca_mean=pca.mean_,pca_components=pca.components_,centers=km.cluster_centers_)
 (a.out/'provenance.json').write_text(json.dumps({'target':a.target,'fit_ports':fold['source_fit_ports'],'fit_objects':len(idx),'B':4,'input_feature_dim':512,'target_fit':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
