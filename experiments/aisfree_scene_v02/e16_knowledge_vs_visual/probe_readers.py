import glob

f = sorted(glob.glob('E:/NASTaR/**/ship_patches_uint8/*.tif', recursive=True))[0]
print('样例', f[-50:], flush=True)

for mod in ('tifffile', 'imageio.v3', 'cv2', 'skimage.io'):
    try:
        if mod == 'tifffile':
            import tifffile
            a = tifffile.imread(f)
        elif mod == 'imageio.v3':
            import imageio.v3 as iio
            a = iio.imread(f)
        elif mod == 'cv2':
            import cv2
            a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        else:
            from skimage import io as skio
            a = skio.imread(f)
        import numpy as np
        print('%-12s ✓ dtype %s shape %s range %d..%d' % (mod, a.dtype, a.shape, int(np.min(a)), int(np.max(a))), flush=True)
    except Exception as e:
        print('%-12s ✗ %s: %s' % (mod, type(e).__name__, str(e)[:90]), flush=True)
