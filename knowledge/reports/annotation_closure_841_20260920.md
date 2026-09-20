# 841 标注收口报告（372993 · 2026-09-20）

> 数据全部来自服务器 westd 实读（`/root/autodl-tmp`），非沿用旧汇总。

## 1. 结论

**841/841 产品标注 completed，missing = 0。**

```
status 文件（逐产品去重）          841
  ├─ annotation_output/           738   （含 218 个 = 本机 482 图汇总后传输的 labels）
  └─ annotation_output_s3/        103   （服务器3 那棵平行树）
completed 产品 ∩ 841 清单          841
missing                            0
```

⚠ 口径提醒：**只数 `annotation_output/` 一个目录会读到 738 的偏低值**；必须把
`annotation_output_s3/` 与 `autodl-fs`（5 个产品的产物在共享盘）一起算。

## 2. 三条来源（同一 detector 权重 550k.pt，字段一致，可互转）

| 来源 | 产品数 | 说明 |
|---|---:|---|
| 服务器既有 | 623 | 09-18/19 在 westd 跑完（442 状态文件读数 = 440 completed + 2 卡在 running）|
| 本机传输 | 218 | 本机 `E:/safe841_local_20260918` 658 景检测 → 汇总场景 JSON → 打包 25.2 MB 上传落位 |
| 服务器新跑 | 181 | 大图只在服务器（无本机副本）→ 服务器端 6 worker 标注，产物只留 labels（复核 jpg + 大图副本跑完即删）|
| 补跑 | 2 | 09-19 磁盘满时中断的 2 个（B317_BE74 / C2A2_0DD7），重跑后 completed |

## 3. 每产品标注内容（与原 442 完全同构）

```
<output>/<product>/<run>/<VV|VH>/labels_json/<product>_<pol>_UTM_8bit.json
    {image, classes[18], objects[{pts, cls_id, conf, meta}]}
<output>/<product>/<run>/<VV|VH>/{detection_audit.json, ais_audit.json, facility_audit.json, raw_detections.json, ...}
```

本机传输件保留了 `product/polarization/human_reviewed/final_dataset_eligible/source`
等来源字段（服务器消费者只读 `image/classes/objects`，多余键无害）。

## 4. 分港口产品数（合计 841）

| 港口 | n | 港口 | n | 港口 | n |
|---|---:|---|---:|---|---:|
| New York | 56 | Fujairah | 36 | Qingdao | 25 |
| Jebel Ali | 52 | Richards Bay | 36 | Melbourne | 25 |
| Los Angeles | 51 | Sydney Botany | 37 | Antwerp-Bruges | 23 |
| Hamburg | 47 | Callao | 36 | Tanger Med | 22 |
| Port Said | 45 | Mombasa | 34 | Busan | 16 |
| Santos | 43 | Newcastle | 32 | Rotterdam | 11 |
| Port Klang | 43 | Port Hedland | 32 | | |
| Houston | 41 | Ningbo-Zhoushan | 31 | | |
| Singapore | 38 | Shanghai | 29 | | |

## 5. 大图覆盖

服务器 `*_UTM_8bit.tif` 覆盖 **841/841 产品**（缺 0）。
其中 348 个产品的 tif 原件位于早前标注时复制进的 `annotation_output/` 内（字节一致）；其余在主下载树。
84 个产品是本次补下来的（网盘直拉 4 + ASF 重下 + geocode 85 → 实到 84，完成度以 89 缺口的最终核验为准）。

## 6. 相关脚本（本仓 scripts_v2/quark_archive/ 与服务器 /root/）

- `quark_upload.py` — 夸克 web 上传协议实现（PDS 分片；cookie 从 `quark_cookie.txt` 读，0600）
- `quark_archive.py` — 841 归档流水线（P1 tif+labels / P2 png；state 断点续传；md5 秒传）
- `build_archive_inventory.py` — 生成 841 产品 → (tif, labels) 清单
- `count_ann_exact.py` — 本报告第 1 节的逐产品去重统计脚本
