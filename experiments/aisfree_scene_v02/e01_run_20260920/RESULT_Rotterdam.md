# E01 Rotterdam 第一折结果（2026-09-21 01:05）

链路：encoder(30 ep, loss 1.64→1.28) → fit_groups(B=4) → H=8 heads → J=3 relation models →
glue → calibrate → run_moment_filter → apply_retained → metrics。全部按包内工具执行。

## 1. 支撑面（决定可解释性）

| 量 | 值 |
|---|---|
| 源 fit 产品 / 港 | 188 / 16 |
| 源 fit 有标签实例（词表内） | **2,918** |
| 词表 | 5 类（container_ship / dredger / fishing_vessel / product_chemical_tanker / tug_towing） |
| 目标 adapt 实例 | 13,949（5 产品） |
| 目标 eval 实例 | 13,200（6 产品） |
| **eval 中带词表内标签的实例** | **46** ⚠️ |
| B0（source-meta 选锚） | h3，equal-port BA 0.267（其余 0.0–0.2） |

## 2. 关系维（J=3 源支撑检查）

`enabled_dimensions = [D0 ✓, D1 ✗, D2 ✗, D3 ✓]`：锚地/航道两维因源 fit 每类 ≥30 实例且 ≥2 港的门槛不达标而**按规则关闭并记录**；本折可检验的知识面 = 岸距 + SAR 群体密度。

## 3. 校准与筛选

- 校准港 4 个：Sydney Botany 0.1295 / Qingdao 0.4012 / Antwerp-Bruges 0.1346 / Hamburg 0.1578
  → **tau = 0.4212**（status=ready，非 uninformative）
- 筛选：8 头全部 feasible（per-head 残差 0.1147–0.1218 ≤ tau），`status=compatible`、无 conflict
  → **retained = 全部 8 头 ⇒ Full ≡ B1**（本折没有可观测的候选剪枝）

## 4. 评估（隔离评价器，仅 46 个词表内标签实例）

| 方案 | Acc | n |
|---|---:|---:|
| B0 | 0.196 | 46 |
| B1 | 0.261 | 46 |
| Full | 0.261 | 46 |
| rescue_vs_B1 / harm_vs_B1 | 0 / 0 | — |

**结论（谨慎）**：管线端到端可用；本折的 Acc 差异在 n=46 下无统计意义，Full−B1=0 是
「tau 高于所有候选残差」的直接结果，而非知识无增益的证据。

## 5. 待请示（已发给网页 GPT）

1. eval 词表内标签只有 46/13,200：是否改为「全 24 港 P0 对象一起评估」或「按港分层报告标签覆盖率」，
   或把粗类作为辅助指标单独报告？
2. tau=0.42（被 Qingdao 0.401 顶起）导致零剪枝：校准规程是否应改用分位数/多锚点，还是维持当前
   「保守经验校准 + 记录 uninformative」的设计？
3. 是否按同配置续跑 Shanghai → Port Klang → Fujairah。
