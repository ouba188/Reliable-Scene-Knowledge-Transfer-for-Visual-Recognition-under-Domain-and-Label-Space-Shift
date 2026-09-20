# E01 权限复核：只作执行收口，不要求重做 E00

依据：仓库 `experiments/aisfree_scene_v02/feature_permissions.yaml`（commit 435fa4b...）和本次用户报告的 E00 去重/掩膜完成状态。仓库被读取的旧权限文件仍含 before_dedup 文案；以新对象表的数据哈希绑定重算结果，不用旧行数覆盖新报告。

## 同意保留
- `support_i`、`support_missing` 不进入目标特征，也不决定目标候选保留。
- AIS 字段只在源监督/隔离评价表中出现，目标端包括聚合 AIS 类比例/方向场都不读（P0）。
- 设施编辑时间仅证据，不等于建成时间或真假知识标签。
- 2 km 截断和图层缺失独立记录。
- 去重后 SAR 群体属性可使用，不使用 AIS 代表记录选择簇代表。

## E01 进一步缩小输入权限
`p0_usable=true` 不等于“送进神经网络”。

| 字段 | E01 作用 |
|---|---|
| `object_id, product_id, port, detection_id, roi_origin` | join/split/log，只作字符串标识 |
| `world_x/world_y, centroid_x/centroid_y` | 找到双极化 chip、构造空间 block，不直接进候选/关系模型 |
| `facility_osm_id/element` | 证据关联，不嵌入模型 |
| `start_utc` | 时间划分，不编码绝对日期作为港口代理 |
| `obb_long_px, obb_short_px` | 换算到米，生成 m 的3个几何量 |
| `valid_fraction` | 唯一质量 m 量及 chip有效像素检查 |
| `d_coast_m,d_anchorage_m,d_fairway_m` | D0–D2，需要“图层实际被观察/查询”的独立掩膜 |
| `local_ships_1km` | D3，必须是离岸、去重、自身排除的候选计数 |
| `dedup_cluster_size/pols/tiles/kind` | 去重追溯，不作类别特征 |
| `channel_angle_deg` | 本批不启用，避免稀疏 OSM 航道与 AIS 回退混用 |
| `heading_consistency_deg, nn_distance_m` | 留作后续单独消融，本批不进入 B/m，避免把 D3 近似复制给条件 |
| `facility_temporal_valid` | 证据，不作 correctK/wrongK 依据 |
| `final_dataset_eligible,human_reviewed,screen_status` | 本批不作目标筛选键 |

**B 分组只看冻结 z；m 不包含 D，不包含邻船数，不包含设施、港口 ID。**

可用性需在 `coverage` sidecar 映射到三个明确布尔字段：
`coast_layer_observed, anchorage_layer_observed, fairway_layer_observed`。
这是已有 E00 证据的接口映射，不是要求重抓 OSM。
有有效查询范围且结果 >2000m：该截断变换的值=0，有效；图层无资料/未查询：值=NaN，mask=false。

海域掩膜=3仅是固定研究域，不自动保证候选是真船。标签未知不是未知船型。组统计全部采用同一个标签盲 P0 候选流程；真值只在隔离评价器与源监督生成中使用。

## 必须执行的权限断言
1. 将评价专用表的标签/AIS值任意置换，部署特征、B分组、H筛选输出必须字节一致。
2. 删除全部评价专用列，所有预测命令照常运行。
3. 任何 source-fit encoder/head 的训练 ID 都不得属于该折目标、meta-query、calibration 港口。
4. 源域细类监督文件也要与 E00 的唯一源匹配与标签来源相联系；`candidate_mmsi` 非唯一猜测不能不加区分升级为真值。
