#!/usr/bin/env python3
"""Verify the shipped 329-product list against a LOCAL GitHub checkout.
No network request or target label access. Exit nonzero on any mismatch.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, subprocess
from pathlib import Path

def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def verify(manifest_path: Path, csv_bytes: bytes) -> dict:
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    rr=list(csv.DictReader(io.StringIO(csv_bytes.decode('utf-8-sig'))))
    if len(rr)!=658: raise ValueError(f'Expected 658 source CSV rows, got {len(rr)}')
    got={}; pols={}
    for r in rr:
        key=(r['port'],r['product'])
        if key in got and got[key]!=r['role']: raise ValueError(f'Polarization split conflict {key}')
        got[key]=r['role'];pols.setdefault(key,[]).append(r['pol'])
    expected={(p,prod):role for p,sp in m['ports'].items() for role in ('adapt','eval') for prod in sp[role]}
    if got!=expected:
        absent=list(set(expected)-set(got)); extra=list(set(got)-set(expected))
        changed=[k for k in set(got)&set(expected) if got[k]!=expected[k]]
        raise ValueError(f'Manifest mismatch; missing={absent[:5]} extra={extra[:5]} changed={changed[:5]}')
    if len(got)!=329 or any(sorted(v)!=['VH','VV'] for v in pols.values()):
        raise ValueError('Product/polarization count or duplicate polarization failed')
    for p,sp in m['ports'].items():
        a=[x.split('_')[4] for x in sp['adapt']];e=[x.split('_')[4] for x in sp['eval']]
        if a and e and max(a)>min(e): raise ValueError(f'Temporal order failed: {p}')
    for target,f in m['folds'].items():
        ss=[set(f[k]) for k in ('source_fit_ports','source_meta_query_ports','source_calibration_ports')]
        if any(target in s for s in ss):raise ValueError('Target entered source fit/meta/cal')
        if any(ss[i]&ss[j] for i in range(3) for j in range(i)):raise ValueError('Source role overlap')
        if len(set.union(*ss))!=23:raise ValueError('Incomplete source role coverage')
    return {'passed':True,'products':329,'polarization_rows':658,'adapt':sum(v=='adapt' for v in got.values()),
            'eval':sum(v=='eval' for v in got.values()),'csv_git_blob_sha':git_blob_sha(csv_bytes),
            'blob_matches_pinned':git_blob_sha(csv_bytes)==m['source_files']['csv_git_blob_sha'],
            'note':'Exact product/role equality checked; different newline bytes may change blob hash.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'split_manifest.json')
    p.add_argument('--repo-root',type=Path);p.add_argument('--split-csv',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();m=json.loads(a.manifest.read_text())
    if a.split_csv:data=a.split_csv.read_bytes()
    elif a.repo_root:
        r=subprocess.run(['git','-C',str(a.repo_root),'show',f"{m['source_commit']}:{m['source_files']['csv_path']}"],capture_output=True)
        if r.returncode:raise SystemExit('Pinned Git blob unavailable. Supply --split-csv or fetch the commit locally. '+r.stderr.decode(errors='replace')[:300])
        data=r.stdout
    else: p.error('Supply --repo-root or --split-csv')
    result=verify(a.manifest,data);text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.out:
        if a.out.exists():raise FileExistsError(a.out)
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text+'\n')
    print(text)
if __name__=='__main__':main()
