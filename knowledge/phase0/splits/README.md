# 实验划分（Phase 0）

- `port_acquisition_split.csv`：每个 (港口, 产品, 极化) 一行，`role=adapt|eval`；**一产品=一次采集**，按日期排序取前一半 adapt、后一半 eval（时间序，避免同采集泄漏）。
- `lopo_folds.json`：外层留一港口；源港保持各自 adapt/eval（内层嵌套留一用于选超参）。
- `class_sets_per_fold.json`：每折的 shared / source_private / target_private（判定阈值：该港该类带 AIS 类别标注的对象数 ≥ 20；`ship_untyped` 不计入类别空间）。
- 类别来源：AIS 唯一匹配 + 细类解析（`mmsi_class_final.csv`），未匹配框一律 `ship_untyped`。
- 口径提醒：小港（Busan/Tanger Med 1 个采集、Callao 少量目标）不适合单独做 LOPO 结论，建议作为附加目标港报告。
