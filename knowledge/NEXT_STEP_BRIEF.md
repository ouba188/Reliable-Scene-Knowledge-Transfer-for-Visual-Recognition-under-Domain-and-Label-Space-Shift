# 跨港知识集状态简报（2026-09-20 收口版）

**用途**：给下一步实验方案讨论用。本文件只描述**已经产出并核对过的**资产；所有数字来自对最终表的整表扫描（2,897,382 行，33 秒全表 pass），不是抽样估计。

---

## 1. 一句话状态

841 产品（本机可检测的 329 产品 / 658 影像）已产出**对象级 $k_i$ 知识表**（289.7 万对象 × 64 列），
港口设施的**逐要素时间证据**、**AIS 经验航道方向场**、**派生岸线**、**MMSI 类别层**、**港口级 $d_p$ 向量**全部就位。
四项此前的结构缺口：三项已解，一项（细类别）用外部类别层替代并如实标注。

---

## 2. 资产清单（仓库内路径）

| 路径 | 规模 | 内容 |
|---|---|---|
| `knowledge/objects_by_port_summary.csv` | 24 行 | 分港口：对象数 / 设施命中 / 带要素 id / 已定年 / 成像时已存在 / 航道角来源 / `support_i` 分布 |
| `knowledge/objects_summary_final.json` | — | 全表汇总数字（本文件第 4 节的机器可读版本） |
| `knowledge/ports/port_knowledge.csv` | 24 × 67 | 港口级 $d_p$：(meta) 场景日期跨度/轨道/成像时刻 + (facility) 各类型要素计数 + (function) 油品/集装箱/散货/渡轮/船厂/铁路代理指标 + (scene) AIS 船类占比、平均船长宽、AIS 空间范围 |
| `knowledge/traffic/*.csv` | 24 港 | AIS 经验航道方向场：1 km 网格、主轴方向（0–180°）、各向异性、点数 + provenance |
| `knowledge/dating/osm_way_dates.csv` | 8.6 万行 | OSM API 逐要素定年：`osm_id, element(way/node/relation), version, timestamp` |
| `knowledge/mmsi/mmsi_class_final.csv` | 126,606 MMSI | 类别层：`final_class, class_level(fine/coarse/non_ship/untyped/unknown), confidence, source, imo, flag, name` |
| `knowledge/keys/products.csv` | 841 | 产品主键 + 港口名归一（31→24 别名） |
| `knowledge/README_STATUS.md` | — | 逐层构建记录 + 复现命令 |
| `scripts_v2/` | 13 个脚本 | 覆盖表、MMSI 补齐、方向场、派生岸线、定年、对象表（3 分片并行）、回填与统计 |
| `knowledge/*.geojson`（**未入库**） | 97 MB | 24 港设施几何 + 14 港派生岸线（体积原因只放在服务器 `/root/autodl-tmp/knowledge_841_20260920/facilities/`） |

对象表本体（195 MB `objects_final.csv.gz`）同样只在服务器与本机，不入库。

---

## 3. 对象表列字典（64 列，按用途分组）

**标识/来源**：`object_id, product_id, port, polarization, start_utc, detection_id, roi_origin, conf`

**几何**：`centroid_x/y`（像素）、`world_x/y`（场景 UTM 米）、`area_px2, obb_long_px, obb_short_px, obb_long_deg`（长轴 0–180°）

**AIS 匹配与类别**：`screen_status, match_status, candidate_mmsi, candidate_distance_m, matched_mmsi, prelabel_class,
ais_final_class, ais_class_level, ais_class_confidence, ais_class_source, fine_class, fine_class_source`

**有效区域/近岸**：`pixel_status, valid_fraction, zero_fraction, coast_status, coast_exclusion_m`

**设施（$k_i$ 实例关系）**：`facility_kind, facility_distance_m, facility_osm_id, facility_osm_element,
facility_edit_ts, facility_temporal_valid`

**几何关系特征**：`d_coast_m, d_quay_m, distance_source, d_coast_pipeline_m, d_anchorage_m, d_fairway_m,
on_fairway, channel_angle_deg, channel_angle_source`

**场景群体特征**：`local_ships_500m, local_ships_1km, nn_distance_m, heading_consistency_deg`

**语义地表**：`worldcover_class, semantic_area(water/land/built/nodata)`

**标注状态/证据计数**：`human_reviewed, final_dataset_eligible, support_i(0–6), support_missing,
ais_unique_match, fine_class_resolved, valid_area_ok, coast_known, facility_context, facility_map_available`

约定：`support_i` 是 6 个信号的计数；`support_missing` 列出缺哪几项；`facility_temporal_valid=1` 表示该设施要素最后编辑时间 ≤ 场景日期（即成像时已存在），`0` 表示编辑晚于场景（送人工复核），空表示未定年。**`human_reviewed=0` 全表未人工复核**。

---

## 4. 覆盖与质检（整表核对）

| 指标 | 数值 |
|---|---|
| 对象 | **2,897,382**（329 产品 = 658 影像） |
| 设施命中 | 918,464（31.7%） |
| 带设施要素 id | 858,023（占命中 93.4%） |
| 已逐要素定年 | 856,000（占带 id 的 99.8%） |
| 时间证据判定「成像时已存在」 | **618,282（72.2%）** |
| 时间证据判定「编辑晚于场景」 | 237,718（27.8%，送人工复核） |
| `on_fairway` | 18,283 |
| 航道夹角来源 | `ais_traffic` **772,047** / `osm_fairway` 15,519 |
| 岸线距离来源 | `pipeline_worldcover` 2,128,210 / `worldcover_derived` 552,267 / `osm_coastline` 216,905 |
| 语义地表 | land 1,606,244 / water 691,340 / built 375,328 / nodata 222,373 |
| 船类级别 | fine 25,538 / coarse 45,667 / non_ship 3,386 / untyped 1,458 |
| `support_i` 分布 | 3 → 1,829,896；4 → 974,480；5 → 91,145；**6 → 1,861** |

---

## 5. 四个历史缺口的最终状态

1. **设施图层是当前快照** → 已解：Geofabrik/BBBike 的 PBF 都丢 `version/timestamp`，改走 OSM API 逐要素取最后编辑时间；
   `facility_edit_ts` / `facility_temporal_valid` 因此可判「成像时是否已存在」。**几何仍是 2026 快照**（只有时间是逐要素的）。
2. **OSM 航道线稀疏** → 已解：用 AIS 轨迹方向场替代（同 MMSI 相邻位置差 → 1 km 网格 PCA 主轴 + 各向异性）。
   影响：`channel_angle_deg` 在多数港口来自 `ais_traffic`，不是 OSM 语义航道。
3. **12 港无 OSM 岸线** → 已解：WorldCover 水体等值线派生岸线（14 港），`distance_source` 逐行标注来源与精度等级。
4. **细类别** → 由外部类别层替代（fine 22,615 MMSI，其中严格细类 21,265）；`unknown` 约 32% 的 MMSI（多为渔船小艇）**按设计不硬猜**。

**已知残余**（不影响主线，供讨论时打折）：backfill geojson 那一路加载时漏了 `osm_id`，涉及 7 个港口 **4,259 个命中（占全部命中 0.46%）** 无时间证据（代码已修，重跑 30 分钟可补）；222,373 个对象的 WorldCover 采样落在无数据处；2,023 条要素 API 未返回时间戳。

---

## 6. 给下一步实验的输入建议

- **默认切片**：`support_i >= 5`（91,145 + 1,861 个对象）或 `support_i >= 4`（含 4 → 97.6 万）；训练用对象建议再叠 `final_dataset_eligible=1`。
- **剔除伪目标**：`ais_class_level='non_ship'`（3,386）应作为 AIS 台站剔除；`semantic_area='built'` 且 `facility_kind in (storage_tank, silo, crane)` 的是典型设施误检候选，可作负样本或先可视化。
- **时间敏感实验**：用 `facility_temporal_valid=1` 的对象构造「设施已存在」的子集，`=0` 的子集留作对照。
- **`k_i` 扩展**需要一个端口级几何重算（服务器已有 97 MB 原始几何：可算到堆场/铁路/储罐/船厂的距离、对象↔设施二部图）。
- **禁止项**（设计约束）：目标港单船 AIS 类型不得进入 $k_i$；AIS 只用于源侧类别支撑与场景统计 $\xi_t$。

---

## 7. 待讨论（给下一步方案）

1. $\Delta$（知识增量）在对象表上的**可操作定义**：用 `support_i` 差分、还是用 $R_{ig}$ 三因子里的 $r^{V|S}$ 残差？表里现有列够不够算？
2. 消融 K 的构造：`K` 正确 / 错误 / 打乱三档，是否分别对应 `support_i` 高/低、`facility_temporal_valid` 真/假、`channel_angle_source` 真/随机？
3. 场景级 $\xi_t$：`local_ships_1km + heading_consistency + semantic_area` 是否足以代表「船群状态」？还需要什么场景统计（如泊位占用率）？
4. ρ（闭包残差）的落点：用 `support_missing` 的类别分布来定义「目标港私有」是否成立？

---

## 8. 复现命令

```bash
python scripts_v2/build_object_table_local.py --products $(cat products_shard0.txt)  # 3 分片并行
python scripts_v2/merge_object_tables.py out/objects_final.csv.gz objects_part*.csv.gz
python scripts_v2/date_facility_ways.py --ids-file need_dates.txt      # 只定年被命中的要素
python scripts_v2/join_facility_dates.py objects.csv.gz                # 定年回填
python scripts_v2/join_ais_class.py objects.csv.gz                     # 类别层回填
python scripts_v2/build_port_knowledge.py                              # d_p
python scripts_v2/build_ais_traffic_field.py                           # 方向场
python scripts_v2/derive_coastline_from_worldcover.py <ports...>       # 派生岸线
python scripts_v2/final_stats.py objects_final.csv.gz                   # 整表核对
```
