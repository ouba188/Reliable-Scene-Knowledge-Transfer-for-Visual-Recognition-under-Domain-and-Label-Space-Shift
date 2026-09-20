# 841 产品知识集 — 补齐状态（2026-09-19）

结论一句话：**能靠本机 + 公开数据补的层已经补到位；两层被外部因素卡住（OSM 网络访问、影像下载），另有 9 块 WorldCover 瓦片 ESA 官方不发布。**

原始资产目录（只读使用，未改动）：`E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle/`
本次产出目录：`E:/临时会话/knowledge_set_841/`

## 一、产物

| 文件 | 内容 |
|---|---|
| `knowledge_coverage_841.csv` | 841 行 × 每产品知识字段（元数据/AIS/细类别/WorldCover/OSM/影像/阻断项） |
| `knowledge_summary.json` | 各层完成度统计 |
| `knowledge_gaps.json` | 逐层缺口清单（未解决 MMSI 及其 AIS 行数、缺瓦片、无影像产品） |
| `worldcover_per_product.csv` | 每产品所需/已有/缺失瓦片（口径＝标注足迹 + 1 km） |
| `worldcover_audit.json` | 瓦片总量核对（含 ESA 未发布清单） |
| `mmsi/mmsi_observed.csv` | 202,266 个 MMSI 的 AIS 观测量（行数、类型码、名称、尺寸） |
| `mmsi/mmsi_fine_class_filled.csv` | 本次新判定 17,700 个 MMSI 的细类别 + 证据 + 置信度 |
| `mmsi/type_code_rules.json` | 从语料自身学到的 AIS 类型码 → 细类别规则（含纯度/支撑数） |
| `mmsi/mmsi_sarais_lookup.csv` | SARAIS 后台库查询记录 47,473 条 |
| `mmsi/mmsi_myshiptracking.jsonl` | MyShipTracking 只读查询流水（可续跑） |

## 二、六层状态

| 层 | 状态 | 数字 | 证据 |
|---|---|---|---|
| A 产品级 S1 元数据 | ✅ 齐 | 841/841 有 iw-vv.xml + iw-vh.xml | `asset_summary.json` |
| B AIS 记录 | ✅ 齐 | 841/841 有 AIS txt；31,176,385 行；202,266 个 MMSI；字段错行 0 | `knowledge_summary.json` |
| C 细粒度船型 | 🟡 部分 | 已判定 768,724 产品×MMSI 对（本次新判 128,423）；粗类 320,539；未判定 285,012（64,465 个 MMSI） | `mmsi/*` |
| D WorldCover 地表 | 🟡 92/101 | 需要 101 块，本地 92 块；9 块 ESA 官方未发布（纯海洋块）；252 个产品受影响 | `worldcover_audit.json` |
| E OSM 港口设施 | ❌ 0/841 | 24 个港口全缺 | 见下方阻断说明 |
| F 影像 + 检测 | 🟡 311/841 | 本机完整 311、仅远端 165、两边都缺 365；检测审核待影像 | `product_inventory_841.csv` |

`fully_ready = 0`：E 层全缺、F 层 365 个缺影像，这两个是当前唯一的硬阻断。

## 三、本次实际补齐（含证据）

1. **口径纠错（重要）**
   - 港口名有 31 种写法（`Jebel_Ali`/`NewYork`/`Port_ Klang`…），已归一到 24 个规范港口。
   - WorldCover 瓦片名此前按 3 位纬度生成（`N027E120`）导致全部 404；ESA 实际为 `N27E120`（纬度 2 位、经度 3 位）。已改为按桶内真实清单核对：**需要的 101 块里，只有 9 块不存在于 ESA 桶（纯海洋块），其余全部可获取**。之前“缺 47 块/113 产品”是命名错误造成的假缺口。
   - 瓦片需求口径改为与标注包一致：**标注 footprint（iw-vv.xml）+ 1 km**（`prepare_assets.py` 的原规则），不再用 manifest bbox。
2. **WorldCover 补 3 块**：`N27E027`、`N27E030`、`N30E027`，并已硬链接进 `annotation_bundle/assets/worldcover/`，`CoastFilter` 默认根目录即可命中。剩余 9 块为 ESA 未发布的海洋块 → `CoastFilter` 对这些区域返回 `None`（不可用），**不会**被当成“无岸线”静默放行。
3. **细类别补 17,700 个 MMSI**，覆盖 1,749,774 条 AIS 行：
   - 17,645 个来自产品自带 AIS 类型码 + 从语料自身 116k 已标注样本学到的规则（要求纯度 ≥0.95 且支撑 ≥5，避免一次性垃圾码）；
   - 41 个来自 SARAIS 后台库（该库 39,812 条命中里 39,468 条 `type=0`，所以这条路基本到顶）；
   - 14 个来自 MyShipTracking 标签（本轮只跑了 45 分钟；实测标签命中率约 68%，可续跑）。
4. **规则表**：`type_code_rules.json` 覆盖 168 个类型码，其中 53 个纯度 ≥0.95（例如 30→fishing_vessel 0.957、36→sailing_vessel 0.971、52→tug_towing 0.912）。

## 四、阻断项（含确切原因与复现方式）

### E 层 OSM 港口设施：本机网络走不通
实测（2026-09-19 21:00–22:00）：

- `overpass-api.de` → `SSLEOFError: UNEXPECTED_EOF_WHILE_READING`
- `overpass.kumi.systems` / `overpass.private.coffee` → `RemoteDisconnected`
- `overpass.osm.jp` → 证书主机名不匹配
- `overpass.osm.ch` → HTTP 200 但 `total=0`（该实例已知会返回假空结果，技能文档已警告）
- 挂本机代理 `127.0.0.1:7890` 后仍全部失败；`api.ohsome.org/v1/metadata` 可通但 `/v1/elements/geometry` 返回 **403 Forbidden**
- 可通的替代源：`download.geofabrik.de`、`osm-pds` S3、`api.ohsome.org`（仅 metadata）

脚本已写好、参数就绪（24 个港口 bbox + 场景日期，输出 Overpass 原始 JSON + 带来源/时点的 provenance，直接满足 `annotate_product.py --osm/--provenance` 契约）：

```bash
python E:/Hermes/scripts/fetch_port_osm.py            # 全部 24 港口，失败自动换端点并退避
python E:/Hermes/scripts/fetch_port_osm.py "Jebel Ali" "Shanghai"   # 指定港口
```

注意：当前只能取**当前快照**，与 2025 场景不同期 → provenance 里 `facility_map_temporal_match=False`，管线会把设施重叠判为 `pending`（人工复核）而不是自动排除。要拿到同期几何需走 attic/ohsome，本机网络未通。

### F 层影像：365 个产品两边都缺
正在由 Codex 侧的多机流水线下载（server1 183 / server2 182 + 本机 F 盘 302 队列，见 `acceleration_status_20260919.md`）。本次未重复启动下载。

## 五、复现与续跑

```bash
# 1) 瓦片：审计 + 补齐（幂等，按桶内真实清单）
python E:/Hermes/scripts/audit_worldcover_tiles.py
python E:/Hermes/scripts/sync_worldcover.py

# 2) MMSI 细类别：scan → rules → sarais → fill（各阶段可单独跑）
python E:/Hermes/scripts/fill_mmsi_fine_class.py scan
python E:/Hermes/scripts/fill_mmsi_fine_class.py rules fill

# 3) 外部逐船标签（只读、可续跑、按 AIS 行数优先级）
python E:/Hermes/scripts/scrape_myshiptracking.py --minutes 480 --workers 4 --delay 0.35
python E:/Hermes/scripts/fill_mmsi_fine_class.py fill      # 把新抓到的标签并回细类别

# 4) 重建覆盖表（约 150 秒，只读原始资产）
python E:/Hermes/scripts/build_knowledge_841.py
```

## 六、验证

- 覆盖表行数断言 841；与 `product_inventory_841.csv` 的 841 个产品一一对应。
- WorldCover 需求 101 块与标注包 `asset_summary.json` 的 101 一致；本地 92 + ESA 未发布 9 = 101，无悬空缺口。
- 缺口清单中 `worldcover_tiles_missing`（9）与 `worldcover_audit.json` 的 `unavailable_in_bucket` 完全一致。
- 新判定细类别的规则全部 ≥0.95 纯度且 ≥5 支撑；低置信（支撑 5–9）单独标 `confidence=low`。
- 类型码规则由语料自身已标注行推导，未引入外部船型判断。
- SARAIS 查询仅只读 `baseInfo`；MyShipTracking 仅读取公开页面标题，未登录、未提交任何表单。

## 七、MyShipTracking 抓取实测（用于判断要不要长跑）

已跑样本 830 条（成功率 96%，约 1.0 条/秒/4 线程），644 条有标签，标签构成：

| 标签 | 数量 | 映射结果 |
|---|---:|---|
| Cargo | 254 | `cargo_coarse`（粗类） |
| Tanker | 91 | `tanker_coarse`（粗类） |
| Other Type / Not available / Law Enforcement / Reserved | 159 | 不可用 |
| Passenger | 43 | `passenger_ship`（细类） |
| Tug | 19 | `tug_towing`（细类） |
| High speed craft / Tanker B / Pilot Vessel / Diving ops | 28 | 部分可用（细类） |

结论：**标签里有细类价值的只占约 10%，粗类约 55%，无效约 35%。** 所以对剩下 64,465 个未判定
MMSI 做长时抓取，性价比很低——它们需要 IMO/船舶注册级数据源，而不是继续抓网页。已跑的标签仍会
并入（粗类也有知识价值），抓取进程可随时中断、可续跑。

## 八、2026-09-20 第二轮补齐（本文件新增）

四个此前的「结构缺口」中三个已解，一个为硬约束：

| 缺口 | 处理 | 产物 |
|---|---|---|
| 设施层是当前快照 | 逐要素定年：Geofabrik/BBBike 的 PBF 都丢掉了 `osm_version/timestamp`，改走 OSM API（`api.openstreetmap.org/api/0.6/way/<id>.json`，直连可达 ~1.35 s/条） | `facilities/osm_way_dates.csv`（osm_id→最后编辑时间），对象表 `facility_edit_ts / facility_temporal_valid` |
| OSM 航道线稀疏 | AIS 轨迹方向场：同 MMSI 相邻位置差（≥20 m）按 1 km 网格做 PCA 主方向 | `traffic/<port>.csv`（方向 + 各向异性），对象表 `channel_angle_deg` 的 `ais_traffic` 来源 |
| 12 港无 OSM 岸线 | WorldCover 水体(80) 抽稀到 ~80 m 后等值线化 | `facilities/<port>.coastline.geojson`（`coastline_derived` + provenance，占 14 港） |
| 64,465 个 MMSI 无细类别 | 硬约束：网页抓取细类产出仅 ~10%，需 IMO/注册级源 | 表中标未判定，不猜 |

同时完成：

- **$d_p$ 港口知识向量** `ports/port_knowledge.csv`（24 港 × 67 列）：meta（场景日期跨度/轨道/成像时刻）、
  facility（各类型要素计数）、function（油品/集装箱/散货/渡轮/船厂/铁路代理指标）、
  scene（AIS 船类占比、平均船长宽、AIS 空间范围）。
- **对象级 $k_i$ 表** `objects/objects.csv.gz`：逐对象给岸线/泊位/锚地/航道距离、`on_fairway`、
  航道夹角（含来源）、500 m/1 km 密度、最近邻距离、航向一致性、WorldCover 语义海域、
  设施类型/距离/`osm_id`/编辑时间，以及 `support_i`（6 信号）与 `support_missing`。
  `d_coast_m` 的来源用 `distance_source` 列区分：`osm_coastline` / `worldcover_derived` / `pipeline_worldcover`。
- 关键修正：对象表同时合并**两套本机港口图层**（`port_osm_output/run_wide` 的通用层 +
  `E:/Docms/Port` 的细分类层），否则设施类型会退化成单一 `osm_all`。

重跑命令：

```bash
python E:/Hermes/scripts/build_port_knowledge.py              # d_p（约 5 分钟）
python E:/Hermes/scripts/build_ais_traffic_field.py           # 方向场（约 20 分钟）
python E:/Hermes/scripts/derive_coastline_from_worldcover.py <无岸线的港口...>
python E:/Hermes/scripts/date_facility_ways.py                # 定年（可续跑，~400 条/分钟）
python E:/Hermes/scripts/build_object_table_local.py --products $(cat scripts/out/objtable_products.txt)
python E:/Hermes/scripts/join_facility_dates.py objects/objects.csv.gz   # 定年结果回填
```

## 九、建议的下一步（按性价比排序）

1. 剩余 64,465 个未判定 MMSI：**不要再靠网页抓取**（见第七节，细类产出仅约 10%）。要么接受粗类标注，要么换 IMO/船舶注册级数据源。
2. 等 Codex 侧 365 个产品影像下载完，跑检测审核（`550k.pt` + 近岸 500 m + 黑边/NoData + 设施排除），F 层才真正闭环。
3. 设施几何仍是 2026 快照（只有时间戳是逐要素的）；若要 2025 同期几何，需要 OSM 历史星球或 ohsome 历史接口（`/elements/geometry` 目前 403）。
4. 目标港 AIS 类型不得进入 $k_i$（设计约束）；本轮 AIS 只用于源侧类别支撑与场景统计 $\xi_t$。
