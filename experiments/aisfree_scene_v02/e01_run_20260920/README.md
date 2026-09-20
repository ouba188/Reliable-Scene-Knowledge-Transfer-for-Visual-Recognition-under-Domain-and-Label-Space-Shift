# E01 运行目录索引（2026-09-21 早）

**先读**：[`MORNING_SUMMARY_20260921.md`](MORNING_SUMMARY_20260921.md) —— 一夜的循环、六个折的结果、三个硬结论、下一步建议。

## 文档

| 文件 | 内容 |
|---|---|
| `MORNING_SUMMARY_20260921.md` | 早间总汇总（先读这个） |
| `STATUS.md` | 数据层交付时的状态（01:15 时的快照） |
| `REV1_REMEDIATION.md` | GPT 执行复核 8 条意见 → 逐条整改对照 |
| `RESULT_ALL_FOLDS_rev1.md` | 冻结四折总汇总 |
| `RESULT_Rotterdam_rev1.md` / `RESULT_Shanghai_rev1.md` / `RESULT_Fujairah_rev1.md` / `RESULT_PortKlang` 相关 | 逐折结果 |
| `RESULT_JebelAli_rev1_provisional.md` / `RESULT_PortSaid_rev1_provisional.md` | 两个探索折（层覆盖问题） |
| `e01_execution_review_v1/`、`E01_rotterdam_verdict_v1/` | GPT 交付的复核包原文（含回归测试） |

## 数据

| 目录 | 内容 |
|---|---|
| `e01_first_batch/` | GPT 交付的执行包（config / split_manifest / relation_spec / tools / tests） |
| `Rotterdam_rev1/`、`Shanghai_rev1/`、`Fujairah_rev1/`、`PortKlang_rev1/`、`JebelAli_rev1/`、`PortSaid_rev1/` | 各折最终产物 JSON（metrics / calibration / retention / 逐头 BA / features.done 含 sha256） |
| `aux_coarse_eval_v1/` | 粗类辅助评价（单列，不改主表） |
| `aux_p1/` | H 分歧 / 机制矩差 / 逐 (B,D) 单元分解 |
| `aux_p3/` | Static/Group/Full 消融 + B3/B4/B5 归因 |
| `layer_coverage.csv` | 每产品的层观测标记（prepare_relations 的输入） |

## 脚本（服务器端执行的同一版本）

`e01_chips_rev1.py`（重抽芯片）、`e01_features_rev1.py`（训练+特征）、`e01_glue_rev1.py`（接口胶水+指标）、
`e01_coarse_eval.py`、`e01_p1_analysis.py`、`e01_p3_attribution.py`、`e01_chain_rev1.sh`（单折全链）。

## 一句话结论

机制跑通了、数据修对了、四折+两探索折再无「知识筛选零剪枝」之外的新证据；
瓶颈已经从「管线正确性」转移到「目标端可评价标签量 + 层覆盖」。
