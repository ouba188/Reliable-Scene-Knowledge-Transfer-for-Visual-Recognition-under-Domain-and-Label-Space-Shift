"""Reference correction for label-tier candidate enumeration, not an AIS identity verifier.
Pass the COMPLETE set from tree.query_ball_point(xy, r=500, eps=0), not query(k=8).
Input coordinates must use a metric projected CRS consistent with the SAR object position.
No trajectory interpolation or scene-time model is added; preserve the existing time convention.
Return independent T1/T2 evidence, including T2-only match records. Never export raw IDs to public repos.
"""
import numpy as np


def match_tiers_from_complete_radius(distances, dt_seconds, mmsis):
    d=np.asarray(distances,dtype=float)
    t=np.asarray(dt_seconds,dtype=float)
    m=np.asarray(mmsis,dtype=str)
    if d.ndim!=1 or t.shape!=d.shape or m.shape!=d.shape:
        raise ValueError('1D equal-length complete-radius candidate arrays required')
    if np.any(~np.isfinite(d)) or np.any(~np.isfinite(t)) or np.any(d<0) or np.any(m==''):
        raise ValueError('candidate records must have finite valid coordinates/time/IDs')
    ans={}
    for tier,rd,rt in [('T1',150.0,180.0),('T2',500.0,300.0)]:
        eligible=np.flatnonzero((d<=rd)&(np.abs(t)<=rt))
        unique=np.unique(m[eligible])
        row={'n_candidate_records':int(len(eligible)), 'n_candidate_mmsi':int(len(unique)),
             'matched_mmsi':None,'distance_m':None,'abs_dt_s':None,
             'status':'no_candidate' if len(unique)==0 else ('unique_candidate' if len(unique)==1 else 'ambiguous')}
        if len(unique)==1:
            # Existing geometric-nearest representative, tie broken by absolute dt then index.
            best=min(eligible.tolist(),key=lambda i:(d[i],abs(t[i]),i))
            row.update(matched_mmsi=str(unique[0]),distance_m=float(d[best]),abs_dt_s=float(abs(t[best])))
        ans[tier]=row
    return ans
