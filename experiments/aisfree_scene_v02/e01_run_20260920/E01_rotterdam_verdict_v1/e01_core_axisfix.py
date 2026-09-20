"""E01 numerical interface. NO target labels accepted.
Inputs:
  phi[N,D], available[N,D], groups[N], blocks[N]
  probabilities[H,N,C], relation_means[J,N,C,D]
A single J must fit the complete decision scope.
Model ranges are NOT calibrated class posteriors or novelty certificates.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Moments:
    observed: np.ndarray
    predicted: np.ndarray
    active: np.ndarray
    counts: np.ndarray
    blocks: np.ndarray
    reasons: np.ndarray

def _probs(p:np.ndarray):
    if p.ndim!=3 or np.any(~np.isfinite(p)) or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-5):
        raise ValueError('probabilities must be finite [H,N,C], nonnegative, sum to 1')

def aggregate(phi,available,groups,spatial_blocks,probabilities,relation_means,
              B=4,min_cell=64,min_blocks=4,enabled_dimensions=None)->Moments:
    phi=np.asarray(phi,dtype=np.float64); mask=np.asarray(available,dtype=bool)
    g=np.asarray(groups);blocks=np.asarray(spatial_blocks)
    p=np.asarray(probabilities,dtype=np.float64);q=np.asarray(relation_means,dtype=np.float64)
    _probs(p)
    if phi.ndim!=2 or mask.shape!=phi.shape:raise ValueError('phi/mask must be [N,D]')
    N,D=phi.shape;H,Np,C=p.shape
    if Np!=N or g.shape!=(N,) or blocks.shape!=(N,):raise ValueError('N mismatch')
    if g.dtype.kind not in 'iu' or np.any(g<0) or np.any(g>=B):raise ValueError('Invalid frozen group ids')
    if q.ndim!=4 or q.shape[1:]!=(N,C,D) or q.shape[0]<1:raise ValueError('relation_means must be [J,N,C,D]')
    if np.any(~np.isfinite(phi[mask])) or np.any((phi[mask]<0)|(phi[mask]>1)):raise ValueError('Active phi must be finite in [0,1]')
    en=np.ones(D,dtype=bool) if enabled_dimensions is None else np.asarray(enabled_dimensions,dtype=bool)
    if en.shape!=(D,):raise ValueError('enabled_dimensions shape')
    J=q.shape[0];obs=np.full((B,D),np.nan);pred=np.full((H,J,B,D),np.nan)
    active=np.zeros((B,D),bool);cnt=np.zeros((B,D),int);nb=np.zeros((B,D),int)
    reasons=np.full((B,D),'',dtype='<U40')
    for b in range(B):
        for d in range(D):
            use=(g==b)&mask[:,d];n=int(use.sum());cnt[b,d]=n;nb[b,d]=len(np.unique(blocks[use]))
            if not en[d]:reasons[b,d]='disabled_by_source_support';continue
            if n<min_cell:reasons[b,d]='too_few_available_objects';continue
            if nb[b,d]<min_blocks:reasons[b,d]='too_few_spatial_blocks';continue
            qq=q[:,use][:,:,:,d]  # keep [J,n,C]; mixed advanced indexing reorders axes
            if np.any(~np.isfinite(qq)) or np.any((qq<0)|(qq>1)):
                reasons[b,d]='relation_model_unavailable';continue
            obs[b,d]=phi[use,d].mean()
            # Identical eligible rows in observed AND predicted moments.
            pred[:,:,b,d]=np.einsum('hnc,jnc->hj',p[:,use,:],qq)/n
            active[b,d]=True;reasons[b,d]='active'
    return Moments(obs,pred,active,cnt,nb,reasons)

def filter_bank(observed,predicted,active,tau):
    obs=np.asarray(observed,dtype=float);pr=np.asarray(predicted,dtype=float);act=np.asarray(active,dtype=bool)
    if obs.ndim!=2 or act.shape!=obs.shape or pr.ndim!=4 or pr.shape[2:]!=obs.shape:raise ValueError('moment shape mismatch')
    H,J=pr.shape[:2]
    if H<1 or J<1 or not np.isfinite(tau) or tau<0:raise ValueError('invalid H/J/tau')
    original=list(range(H))
    if not act.any():return {'retained':original,'feasible':original,'conflict':False,'status':'no_active_knowledge','score':[None]*H,'best_mechanism':[None]*H}
    if np.any(~np.isfinite(obs[act])) or np.any(~np.isfinite(pr[...,act])):raise ValueError('active moments nonfinite')
    residual=np.abs(pr[...,act]-obs[act])
    worst=residual.max(-1)                 # [H,J], all groups jointly
    scores=worst.min(1);which=worst.argmin(1)
    feasible=np.flatnonzero(scores<=tau+1e-12).tolist()
    return {'retained':feasible if feasible else original,'feasible':feasible,'conflict':not bool(feasible),
            'status':'compatible' if feasible else 'mechanism_conflict','score':scores.tolist(),'best_mechanism':which.tolist()}

def calibrate(observed_by_port,predicted_anchor_by_port,active_by_port,margin=.02,floor=.05,min_ports=4,uninformative=.50):
    """One anchor hypothesis fixed on source-meta-query BEFORE calibration.
    Per calibration port: min_J max_BD residual. No target data or Y arguments.
    A conservative empirical maximum is NOT distribution-free new-port coverage.
    """
    obs=np.asarray(observed_by_port,float);pr=np.asarray(predicted_anchor_by_port,float);ma=np.asarray(active_by_port,bool)
    if obs.ndim!=3 or ma.shape!=obs.shape or pr.ndim!=4 or pr.shape[0]!=obs.shape[0] or pr.shape[2:]!=obs.shape[1:]:
        raise ValueError('Need obs[A,B,D], anchor_predictions[A,J,B,D], mask[A,B,D]')
    scores=[]
    for a in range(obs.shape[0]):
        if not ma[a].any():scores.append(None);continue
        r=np.abs(pr[a][...,ma[a]]-obs[a][ma[a]])
        if np.any(~np.isfinite(r)):raise ValueError('Nonfinite calibration residual')
        scores.append(float(r.max(-1).min()))
    values=[x for x in scores if x is not None]
    out={'mode':'empirical_source_port_max','scores':scores,'n_valid_ports':len(values),'coverage_guarantee':False}
    if len(values)<min_ports:return {**out,'status':'calibration_unavailable','tau':None}
    tau=max(floor,max(values)+margin)
    return {**out,'status':'uninformative' if tau>=uninformative else 'ready','tau':tau}

def predictions(probabilities,retained):
    p=np.asarray(probabilities,float);_probs(p)
    if not retained or len(set(retained))!=len(retained) or min(retained)<0 or max(retained)>=p.shape[0]:raise ValueError('invalid retained indices')
    kept=p[retained];args=kept.argmax(-1)
    return {'B1':p.mean(0),'Full':kept.mean(0),'lower':kept.min(0),'upper':kept.max(0),'unanimous_argmax':np.all(args==args[:1],axis=0)}
