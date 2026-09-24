"""e110: smoke test -- can the YOLO-OBB detector actually run on an F: scene and produce detections?

This is the first concrete step of the 'detect on the F: scenes' route, chosen because the F: batch is DISJOINT from
the batch the Quark download brings (Sydney: F: products vs object products intersect at zero), so the two routes
expand the pool in different directions. The full route is long (detection, then AIS matching, then labels, then
screening -- how the current 141,880 objects were made), so before committing to it, prove the detector itself works:
load the weights, tile one 1024-px window over a harbour area, and count the detections.

Uses the same model and parameters as safe841_obb_infer.py: YOLO task='obb', imgsz 1024, conf 0.08, iou 0.30.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SCENE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r'F:/SAR_0922/Sydney Botany/VV/S1A_IW_GRDH_1SDV_20250102T191606_20250102T191631_057269_070B97_08C8_VV_UTM_8bit.tif')
WEIGHTS = r'E:/Docms/SARlabelimg/550k.pt'
WIN = 1024
CONF = 0.08

print('场景:', SCENE.name, '存在:', SCENE.is_file(), flush=True)
if not SCENE.is_file():
    raise SystemExit('场景不存在')

from ultralytics import YOLO
print('加载权重 ...', flush=True)
model = YOLO(WEIGHTS, task='obb')
print('类别:', model.names, flush=True)

im = Image.open(SCENE)
print('整景尺寸 %s' % (im.size,), flush=True)
W, H = im.size
# a few windows spread over the scene (not the whole scene: this is a smoke test)
wins = [(W // 2 - WIN // 2, H // 2 - WIN // 2), (W // 4 - WIN // 2, H // 4 - WIN // 2),
        (3 * W // 4 - WIN // 2, 3 * H // 4 - WIN // 2), (W // 2 - WIN // 2, H // 4 - WIN // 2)]
total = 0
for wi, (x, y) in enumerate(wins):
    x = max(0, min(x, W - WIN)); y = max(0, min(y, H - WIN))
    crop = np.asarray(im.crop((x, y, x + WIN, y + WIN)))
    r = model.predict(np.repeat(crop[..., None], 3, axis=2), imgsz=WIN, conf=CONF, iou=0.30, verbose=False)[0]
    n = 0 if r.obb is None else len(r.obb)
    total += n
    confs = [] if r.obb is None else r.obb.conf.cpu().numpy().tolist()
    print('  窗口 %d @(%d,%d): 检出 %d 个  conf 中位 %.3f' % (wi, x, y, n, float(np.median(confs)) if confs else 0.0), flush=True)
print('')
print('★ 4 个窗口合计检出 %d 个目标 ⇒ 检测器可用 %s' % (total, '✓✓' if total > 0 else '✗'))
print('（全港整景需 ~600 个重叠窗口，约 1 分钟/景；F: 上这些港每个 14-28 景 ⇒ 每港 ~30 分钟检测）')
