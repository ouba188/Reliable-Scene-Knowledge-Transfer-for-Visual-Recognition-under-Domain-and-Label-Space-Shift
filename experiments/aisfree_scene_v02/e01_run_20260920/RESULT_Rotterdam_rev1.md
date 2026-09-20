# E01 rev1 — Rotterdam（2026-09-21 02:24 完成）

冻结实现 sha256：chips=b4c5964178f2 features=ff926d1685d3 glue=e72fa557f276 core=cd207624d7f1
（rev0 保留为 pilot，不与 rev1 作方法对照；本折改善属数据修复效果。）

## 执行接口（P0）

| 项 | 值 |
|---|---|
| chips 分片 | 329/329，`valid_joint` 全部 1.000、越界 0（几何口径；旧非零口径的 75–90% 是误判） |
| features.npz | 408,981 行；sha256 `0bd21c9edea0af965acb7915358225cff465f9bfa03eb8ff946e569a6e4fe6f2` |
| 训练池 | 2,918 条（词表内、源 fit、`valid_joint`≥0.95）全部进 RAM（91.2 MiB） |
| 采样 | P(i)=1/(P·C_p·N_(p,c))，11 港有放回；逐 epoch 抽样计数落盘 `epoch_draws.json` |
| 归一化 | source-fit 有效 chip：VV 0.1753/0.1199，VH 0.1933/0.1045 |
| 词表 | 5 类（container_ship / dredger / fishing_vessel / product_chemical_tanker / tug_towing），`class_level=fine` 且排除 `*_coarse`/占位类 |
| 训练 | 30 epoch 共 212 s（7 s/epoch）、末轮存档、AMP bf16 |
| 图像门 | train/export 均 ≥0.95；目标 eval 13,200 全部可用、**0 个因 I/O 排除**；adapt 13,949 全部可用 |

## 机制（P1）

- enabled_dimensions = [**D0 ✓, D1 ✗, D2 ✗, D3 ✓**]（锚地/航道源端每类 ≥30 & ≥2 港不达标，按规则停用）
- B0 锚点 = h0（逐港 BA→跨港平均，规则：max mean per-port BA，平局取小 id）
- 校准 4 港分数 0.1952 / 0.2241 / 0.2115 / 0.1616 → **tau = 0.2441**（status=ready）
- 筛选：8 头残差 0.1849–0.1874 全部 ≤ tau → retained=8、feasible=8、conflict=false、best_mechanism 全为 j1
- ⇒ 本折里知识筛选仍未产生剪枝（与 rev0 同性质，但原因不同：rev0 是 tau 被 0.401 顶起，rev1 是所有候选都远低于 tau）

## 任务指标（P2，同分母）

| 方案 | Acc | BA | n |
|---|---:|---:|---:|
| B0 (h0) | 0.4783 | 0.4960 | 46 |
| B1（8 头平均） | 0.4348 | 0.4786 | 46 |
| Full（retained=8） | 0.4348 | 0.4786 | 46 |

- rescue_vs_B1 = 0，harm_vs_B1 = 0，delta_acc_vs_B1 = 0.0（Full≡B1）
- 逐类（Full）：container_ship n=1 r=1.00；dredger n=5 r=0.00；fishing_vessel n=7 r=0.57；
  product_chemical_tanker n=10 r=0.30；tug_towing n=23 r=0.52
- 固定分母覆盖：eval 全部 13,200 实例中，词表内标签 46、词表外标签 268、无标签 12,886 —— **可评价面仍然只有 46**，
  该口径问题在本折未被图像门改变（因此 B0/B1 的差异不可作结论）

## 结论（谨慎）

rev1 的实现修正让「端到端记录」在正确窗口与正确采样下成立；本折仍未提供知识正迁移证据，
主要限制仍是**可评价标签支撑过小**（46）与**关系维只启用 D0/D3**。继续按冻结配置跑 Shanghai →
Port Klang → Fujairah，不据本折分数改动研究规则。
