#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from e01_core import calibrate
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();d=np.load(a.input,allow_pickle=False)
if {'target_labels','y','ais_final_class'}&set(d.files):raise ValueError('Calibration API does not take labels')
if a.out.exists():raise FileExistsError(a.out)
r=calibrate(d['observed'],d['predicted_anchor'],d['active'])
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
