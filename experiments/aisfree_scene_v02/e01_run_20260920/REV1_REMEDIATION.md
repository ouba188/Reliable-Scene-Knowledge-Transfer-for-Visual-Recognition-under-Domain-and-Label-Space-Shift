# E01 执行复核整改记录（rev1，2026-09-21 01:45）

依据：网页 GPT 的《E01执行回执裁定与首轮回传要求》（`E01_execution_review_v1.zip`，本地
`0917/e01_execution_review_v1/EXECUTION_DECISION.md`）+《Rotterdam首折判决/索引回归测试/阈值诊断包》
（`E01_rotterdam_verdict_v1.zip`）。

## 判定：rev0 全部作废（保留为 pilot，不删除）

**实测证据（不是静态推断）**：
```
样例 det_00012 world=(208789,2724810) -> ds.index 返回 row=19355 col=4286（差 15069）
修正窗口(row,col)   中心16×16 均值 162.5 最大 255   ← 目标在中心、饱和
交换窗口(col,row)   中心16×16 均值  85.7 最大 136   ← 无目标
已落盘 rev0 芯片 vs 修正窗口 0.5% 相同 | vs 交换窗口 100.0% 相同
```
→ rev0 `crop_pair` 的 `col,row = ds.index(x,y)` 把 rasterio 的 (row,col) 写反，芯片位置转置。
rev0 的 chips/features/heads/relations/metrics 全部只作探索记录，主表不使用。

## rev1 整改清单（对应复核意见 1–8）

| # | 复核意见 | rev1 处理 |
|---|---|---|
| 1 | crop 行列写反 | `row,col = ds.index(...)`；重抽到 `e01_chips_rev1/`（新目录、原子写 tmp→rename、每分片 `.done` + 全局 `_COMPLETE`） |
| 2 | `<U64` 截断 | 全部改 `<U128`（产品名 68 字符、对象 id 95 字符）；落盘后断言唯一 |
| 3 | glue 用 meta.get(product_id) | 改为 split_manifest 的产品→港映射（`prod2port`），并逐港记录 cal 产品数/实例数/活跃单元 |
| 4 | equal_port_ba 实为 Acc | 改为**逐港宏平均召回（BA）→跨港平均**；B0 规则写明「max mean per-port BA，平局取小 id」；metrics 同时输出 Acc/BA/逐类 N/recall |
| 5 | 有效比例口径 | rev1 输出 `valid_vv/valid_vh/valid_joint/nonzero_fraction/clipped`；`>=0.95` 门用于训练与导出；m4 第 4 列改用 **chip 级 valid_joint**；低有效对象保留 P0 登记并给 `status=io_coverage_missing` |
| 6 | np.roll 环绕 + 变换每 epoch 重复 | 改 **reflect-pad 平移**；增广参数在**主进程逐 epoch 抽样**并以 (row_index, aug_seed) 交给 worker，落盘 `epoch_draws.json` |
| 7 | 分块顺序≠uniform_port_class | 训练池（2,918 chip ≈ 91 MiB）整体进 RAM，按 **P(i)=1/(P·C_p·N_(p,c)) 有放回抽样**；逐 epoch 记录各港/各类实际抽样次数 |
| 8 | shell 缺 pipefail / 半写文件 | `set -Eeuo pipefail` + 完整日志 + `features.done`（含 sha256/行数）+ 训练/导出原子写；每折独立 run 目录 `*_rev1` |

## 续跑口径（按裁定）

- 词表：`class_level=fine` 且排除 `*_coarse`/`ship_untyped`/`untyped`/`non_ship`/`unknown`，只用 source-fit 且
  图像有效（valid_joint≥0.95）的标签，≥100 且 ≥3 港；冻结后不因 target 分数变更。
- 四折顺序 Rotterdam → Shanghai → Port Klang → Fujairah，**同一版冻结实现**（脚本 sha256 记录在每折日志首行）；
  「同配置」= 相同规则，非同权重、非同 tau。
- 允许自动阻断的只有：接口错误、权限错误、训练/读写失败、校准文件结构错误。其余（全缺失/机制冲突/校准不足）
  按原算法 fallback 并保留结果；不因 target 涨分/零增益/负增益改配置。
- 已停掉 rev0 的 Shanghai 折（当时在跑），改用 rev1 脚本统一重跑。

## 待补（本轮已做/未做）

- 已做：rev1 chips 重抽（进行中，3 分片）、rev1 features/glue/chain 脚本落盘并入仓。
- 未做：`summarize_screening.py` 本地实跑（需 rev1 filter 产物）、关系误差分解（P3 归因）——待 Rotterdam rev1 出结果后执行。
