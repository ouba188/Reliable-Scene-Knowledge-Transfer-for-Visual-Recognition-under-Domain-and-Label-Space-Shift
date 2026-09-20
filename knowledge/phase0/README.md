# Phase 0 · 数据管线产物（19c027）

这里放的是 0917 计划 **Phase 0** 的产物与索引。**大文件不入库**（PyPI 风格：仓库只放文本/脚本/索引），几何与掩膜在本地与服务器各一份。

## 文件

| 文件 | 内容 |
|---|---|
| `splits/port_acquisition_split.csv` | 658 行 = 329 产品 × 2 极化；`role=adapt\|eval`（一产品 = 一次采集，按日期排序前后各半） |
| `splits/lopo_folds.json` | 24 折留一港口；源港保留各自 adapt/eval（内层嵌套留一用于选超参） |
| `splits/class_sets_per_fold.json` | 每折 `shared / source_private / target_private / setting`（阈值：该港该类带 AIS 类别对象 ≥ 20，`ship_untyped` 不计） |
| `splits/README.md` | 划分规则与口径提醒（小港不适合单独下 LOPO 结论） |
| `support_by_port_class.csv` | 每港 × 类别的对象数、占该港比例、其中带唯一 MMSI 匹配的对象数 |
| `masks_index.csv` | 每景掩膜网格（宽高、原点、像元 80 m）与 陆/近岸/离岸/无效 占比、WorldCover 可用标志 |
| `geo_index.csv` | 每景回写框数与类别列表 |
| `scene_label_index.csv` | 大景标注导出索引（658 景的目标数、端口、耗时） |
| `objects_by_port_summary.csv` | 对象表按港汇总（来自知识集） |
| `STATUS.md` | 自动生成的现状快照（由 `scripts_v2/sync_project_repo.py` 刷新） |

## 大文件位置

- 本地：`E:/临时会话/knowledge_set_841/geo/`（24 港 GeoJSON，217 MB）、`masks/`（658 景 PNG，56 MB）
- 服务器：`westd:/root/autodl-tmp/phase0/`（755 文件 / 418 MB；含 splits、geo、masks、mmsi、ports、facilities、objects）——回读核对过文件大小一致

## 掩膜取值

`0` 无效（nodata/黑边）· `1` 陆地 · `2` 近岸（距陆地 ≤ 500 m）· `3` 离岸（> 500 m，**任务域**）· `4` 该景 WorldCover 瓦片不可用（9 块海洋瓦片 ESA 未发布）。
网格与影像同投影（UTM），像元 80 m（1/8 缩放），`masks_index.csv` 里有原点与像元尺寸可反算像素坐标。

## 目前口径下的关键结论

1. **24 折 LOPO 全部是 partial-set**（源港有目标港没有的类），`target_private` 天然为 0 → I3 的"目标私有类"必须用 episodic 模拟制造，不能指望自然 AIS 分布（与计划 Phase 4 一致）。
2. 离岸有效域平均占景 **45.4%**，近岸带 1.4% —— 近岸带正好用作实验 1C（offshore 任务边界是否合理）的对照。
3. 带 AIS 类别的对象 **79,059** 个，头部类别（product_chemical_tanker 21,635 / bulk_carrier 9,205 / container_ship 8,594 / tug_towing 8,025）跨 22–24 港，长尾（dredger 13 港、passenger_ship 19 港）需按折裁剪类别集合。
