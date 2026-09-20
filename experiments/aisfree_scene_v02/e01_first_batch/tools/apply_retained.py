#!/usr/bin/env python3
"""G protocol: apply an ADAPT-derived retained bank to EVAL probabilities; no eval K."""
import argparse,json
from pathlib import Path
import numpy as np
from e01_core import predictions
p=argparse.ArgumentParser();p.add_argument('--probabilities',type=Path,required=True);p.add_argument('--retention',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
r=json.loads(a.retention.read_text());pr=np.load(a.probabilities,allow_pickle=False)
o=predictions(pr,r['retained']);a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,**o)
