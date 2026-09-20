import csv,io,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from e01_core import aggregate,filter_bank,calibrate,predictions
from verify_manifest import verify

class ManifestTests(unittest.TestCase):
 def setUp(self):self.m=json.loads((ROOT/'split_manifest.json').read_text())
 def test_329_unique(self):
  xs=[p for sp in self.m['ports'].values() for role in ('adapt','eval') for p in sp[role]]
  self.assertEqual(len(xs),329);self.assertEqual(len(set(xs)),329)
 def test_counts(self):
  self.assertEqual(sum(len(x['adapt']) for x in self.m['ports'].values()),161)
  self.assertEqual(sum(len(x['eval']) for x in self.m['ports'].values()),168)
 def test_single_capture(self):
  self.assertEqual([p for p,v in self.m['ports'].items() if not v['eval']],['Busan','Tanger Med'])
 def test_folds_disjoint(self):
  for p,f in self.m['folds'].items():
   ss=[set(f[k]) for k in ('source_fit_ports','source_meta_query_ports','source_calibration_ports')]
   self.assertEqual([len(s) for s in ss],[16,3,4]);self.assertNotIn(p,set.union(*ss))
   self.assertFalse(ss[0]&ss[1] or ss[0]&ss[2] or ss[1]&ss[2])
 def test_temporal_order(self):
  for p,f in self.m['ports'].items():
   if f['eval']:self.assertLessEqual(max(x.split('_')[4] for x in f['adapt']),min(x.split('_')[4] for x in f['eval']))
 def test_local_csv_verifier(self):
  # Synthetic CSV reconstructed from manifest tests parser; NOT remote-byte verification.
  s=io.StringIO();w=csv.writer(s);w.writerow(['port','product','pol','date','role','objects'])
  for p,v in self.m['ports'].items():
   for r in ('adapt','eval'):
    for x in v[r]:
     for pol in ('VV','VH'):w.writerow([p,x,pol,x.split('_')[4][:8],r,0])
  out=verify(ROOT/'split_manifest.json',s.getvalue().encode());self.assertTrue(out['passed'])
 def test_config_dimensions(self):
  c=json.loads((ROOT/'config.json').read_text());self.assertEqual((c['H'],c['J'],c['B'],c['D']),(8,3,4,4))
  self.assertEqual(len(json.loads((ROOT/'candidate_bank.json').read_text())),8)

class MomentTests(unittest.TestCase):
 def test_identity_selection(self):
  o=np.array([[.1],[.9]]);q=np.array([[[[.1],[.9]]],[[[.9],[.1]]]])
  self.assertEqual(filter_bank(o,q,np.ones_like(o,bool),.1)['retained'],[0])
 def test_missing(self):
  r=filter_bank(np.full((2,1),np.nan),np.full((2,1,2,1),np.nan),np.zeros((2,1),bool),.1)
  self.assertEqual(r['retained'],[0,1]);self.assertFalse(r['conflict'])
 def test_conflict_not_unknown(self):
  r=filter_bank(np.full((2,1),.5),np.array([[[[.1],[.9]]],[[[.9],[.1]]]]),np.ones((2,1),bool),.1)
  self.assertTrue(r['conflict']);self.assertEqual(r['retained'],[0,1]);self.assertNotIn('unknown_probability',r)
 def test_one_mechanism_global(self):
  q=np.array([[[[.1],[.9]],[[.9],[.1]]]])
  self.assertTrue(filter_bank(np.array([[.1],[.1]]),q,np.ones((2,1),bool),.05)['conflict'])
 def test_equivalent_mechanisms_preserve_ambiguity(self):
  q=np.array([[[[.1],[.9]]],[[[.9],[.1]]]])
  q=np.concatenate([q,1-q],1)
  self.assertEqual(filter_bank(np.array([[.1],[.9]]),q,np.ones((2,1),bool),.05)['retained'],[0,1])
 def test_same_mask(self):
  phi=np.array([[.2],[np.nan]]);p=np.array([[[1.,0.],[0.,1.]]]);q=np.array([[[[.2],[.8]],[[.9],[.1]]]])
  m=aggregate(phi,np.array([[True],[False]]),np.zeros(2,int),np.array(['a','b']),p,q,B=1,min_cell=1,min_blocks=1)
  self.assertAlmostEqual(m.observed[0,0],.2);self.assertAlmostEqual(m.predicted[0,0,0,0],.2)
 def test_small_cell_inactive(self):
  m=aggregate(np.ones((2,1)),np.ones((2,1),bool),np.zeros(2,int),np.array(['a','b']),np.ones((1,2,1)),np.ones((1,2,1,1)),B=1)
  self.assertFalse(m.active.any())
 def test_predictions_fallback(self):
  p=np.array([[[.1,.9]],[[.9,.1]]]);o=predictions(p,[0,1]);np.testing.assert_allclose(o['B1'],o['Full']);self.assertFalse(o['unanimous_argmax'][0])
 def test_calibration_four_ports(self):
  r=calibrate(np.zeros((4,1,1)),np.ones((4,3,1,1))*.1,np.ones((4,1,1),bool))
  self.assertEqual(r['status'],'ready');self.assertAlmostEqual(r['tau'],.12);self.assertFalse(r['coverage_guarantee'])
 def test_calibration_insufficient(self):
  r=calibrate(np.zeros((3,1,1)),np.zeros((3,3,1,1)),np.ones((3,1,1),bool));self.assertIsNone(r['tau'])
 def test_uninformative_calibration(self):
  r=calibrate(np.zeros((4,1,1)),np.ones((4,3,1,1))*.8,np.ones((4,1,1),bool));self.assertEqual(r['status'],'uninformative')
 def test_invalid_probabilities(self):
  with self.assertRaises(ValueError): predictions(np.zeros((2,3,2)),[0])
 def test_geometry_not_iid_claim(self):
  c=json.loads((ROOT/'config.json').read_text());self.assertIn('acquisition',c['report']['bootstrap_unit'])

class ModelTests(unittest.TestCase):
 def test_shapes(self):
  import torch
  from models import SARResNet18GN,CandidateHead,RelationMomentModel
  torch.set_num_threads(2)
  with torch.no_grad():
   e=SARResNet18GN(3).eval();x=torch.zeros(1,2,128,128)
   self.assertEqual(tuple(e.encode(x).shape),(1,512));self.assertEqual(tuple(e(x).shape),(1,3))
   self.assertEqual(tuple(CandidateHead(516,3,'mlp128')(torch.zeros(2,516)).shape),(2,3))
   q=RelationMomentModel(20,3);o=q(torch.zeros(2,20),torch.tensor([0,1]));self.assertEqual(tuple(o.shape),(2,4))

if __name__=='__main__':unittest.main(verbosity=2)
