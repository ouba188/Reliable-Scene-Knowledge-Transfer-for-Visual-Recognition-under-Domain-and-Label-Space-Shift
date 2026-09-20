# E01 rev1 — Shanghai（2026-09-21 02:41 完成）

冻结实现 sha256：chips=b4c5964178f2 features=ff926d1685d3 glue=e72fa557f276 core=cd207624d7f1

## 支撑面

| 项 | 值 |
|---|---|
| 源 fit 有标签（词表内） | **612** |
| 词表 | **2 类**：fishing_vessel / tug_towing |
| features.npz | 408,981 行 sha256 `a1d142433b8aa8f387b55d9d80f604d67511a943ca7b28a58bdba9b104c4ba4e`（636 s） |
| 目标 adapt / eval | 32,980 / 26,653 全可用（0 个因 I/O 排除） |
| eval 可评价 | **430**（词表内标签）；词表外标签 7,581；无标签 18,642 |

## 机制

- enabled_dimensions = [**D0 ✓, D1 ✓, D2 ✗, D3 ✓**]（本折锚地维达标，航道维不达标）
- B0 = h4（跨港 BA 均值 0.6320；各头 0.582–0.632）
- 校准 4 港：Mombasa 0.2379 / Santos 0.2953 / Fujairah 0.1390 / Port Said 0.1753 → **tau = 0.3153**（ready）
- 筛选：8 头残差 0.1032–0.1054 全 ≤ tau → retained=8、conflict=false、best_mechanism 全 j0

## 任务指标（同分母 n=430）

| 方案 | Acc | BA |
|---|---:|---:|
| B0 (h4) | 0.7093 | 0.6841 |
| B1 | 0.7070 | 0.6825 |
| Full (retained=8) | 0.7070 | 0.6825 |

- rescue/harm vs B1 = 0 / 0（Full≡B1）
- 逐类（Full）：fishing_vessel n=314 r=0.736；tug_towing n=116 r=0.629

## 与 Rotterdam 的差异（记录，不作方法结论）

- 词表从 5 类缩到 2 类、可评价从 46 升到 430 —— 这两项是**折别数据差异**，不是实现变化。
- Shanghai 的 D1 达标（Rotterdam 不达标），说明维度启用是**逐折**由源端支撑决定的。
