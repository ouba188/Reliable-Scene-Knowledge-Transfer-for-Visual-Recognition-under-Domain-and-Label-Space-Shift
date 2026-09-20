import unittest
import tempfile
from pathlib import Path
import numpy as np
from e01_execution_checks import *

PRODUCT = 'S1A_IW_GRDH_1SDV_20250112T055841_20250112T055906_057407_07110D_854D'

class TestChecks(unittest.TestCase):
    def test_fixed_u64_reproduces_collision(self):
        self.assertEqual(len(PRODUCT), 67)
        ids = strings_exact([PRODUCT+'|VV|0,0|1', PRODUCT+'|VH|0,1|1'], unique=True)
        self.assertEqual(len(set(ids.tolist())), 2)
        self.assertEqual(len(set(ids.astype('<U64').tolist())), 1)
    def test_dynamic_unicode_roundtrip(self):
        vals = [PRODUCT+'|VV|0,0|1', PRODUCT+'|VH|0,100000|10']
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/'x.npz'
            np.savez(f, sample_id=strings_exact(vals))
            with np.load(f, allow_pickle=False) as r:
                self.assertEqual(r['sample_id'].tolist(), vals)
    def test_strict_duplicate_join(self):
        with self.assertRaises(ValueError): strict_index(['a','a'])
    def test_ba_not_accuracy(self):
        m=classification_metrics([0]*9+[1],[0]*10)
        self.assertAlmostEqual(m['accuracy'],.9)
        self.assertAlmostEqual(m['balanced_accuracy'],.5)
    def test_equal_port_balanced_accuracy(self):
        self.assertAlmostEqual(equal_port_ba([0]*9+[1]+[0,1], [0]*10+[0,1], ['A']*10+['B']*2),.75)
    def test_sampling_mass(self):
        ports=['A']*10+['B']*2; labels=[0]*9+[1]+[0]*2
        w=port_class_probabilities(ports,labels)
        self.assertAlmostEqual(w[:9].sum(),.25)
        self.assertAlmostEqual(w[9],.25)
        self.assertAlmostEqual(w[10:].sum(),.5)
    def test_epoch_draw(self):
        p=['A','A','B']; y=[0,1,0]
        a=balanced_epoch_draw(p,y,42,0,100)
        np.testing.assert_array_equal(a,balanced_epoch_draw(p,y,42,0,100))
        self.assertFalse(np.array_equal(a,balanced_epoch_draw(p,y,42,1,100)))
    def test_augmentation_sync_and_epoch(self):
        base=np.arange(128*128).reshape(128,128)
        x=np.stack([base,base+100000])
        a=augment_pair_reflect(x,42,0,1)
        np.testing.assert_array_equal(a[1]-a[0],np.full((128,128),100000))
        np.testing.assert_array_equal(a,augment_pair_reflect(x,42,0,1))
        self.assertTrue(any(not np.array_equal(a,augment_pair_reflect(x,42,e,1)) for e in range(1,5)))
    def test_calibration_mapping(self):
        m={'ports':{'A':{'adapt':['a'],'eval':['aa']},'B':{'adapt':['b'],'eval':[]}},
           'folds':{'T':{'source_calibration_ports':['A','B'],'source_calibration_adapt_products':['a','b']}}}
        self.assertEqual(calibration_products(m,'T'),{'A':['a'],'B':['b']})
        object_meta={'a|VV|0,0|1':{'product_id':'a','port':'A'}}
        self.assertIsNone(object_meta.get('a'))
    def test_raster_window_order(self):
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        from rasterio.windows import Window
        data=np.zeros((360,400),dtype='uint8'); data[90,250]=231
        with MemoryFile() as mem:
            with mem.open(driver='GTiff',width=400,height=360,count=1,dtype='uint8',
                          crs='EPSG:32631',transform=from_origin(1000,6000,10,10)) as ds:
                ds.write(data,1)
                x,y=ds.xy(90,250)
                correct=ds.read(1,window=centered_window(ds,x,y))
                col,row=ds.index(x,y)
                wrong=ds.read(1,window=Window(col-64,row-64,128,128))
                self.assertEqual(correct[64,64],231)
                self.assertEqual(wrong[64,64],0)
    def test_mask_intersection_counterexample(self):
        vv=np.ones((128,128),bool); vh=np.zeros_like(vv)
        self.assertEqual(float((vv|vh).mean()),1)
        self.assertEqual(float((vv&vh).mean()),0)
    def test_no_empty_metric(self):
        with self.assertRaises(ValueError): classification_metrics([],[])

if __name__=='__main__': unittest.main(verbosity=2)
