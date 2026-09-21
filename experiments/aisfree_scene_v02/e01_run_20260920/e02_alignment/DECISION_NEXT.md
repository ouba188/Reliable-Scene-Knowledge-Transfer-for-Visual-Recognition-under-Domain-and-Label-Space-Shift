# DECISION_NEXT：本轮只选一个干预进入下一版

## 一、按裁定表的落位（依据 E02_ALIGNMENT_v1.md）

| 裁定表的行 | 是否命中 | 依据 |
|---|---|---|
| **Mean-only 已经改善** | **命中（3/5 可判定配对）** | Rotterdam→Qingdao +7.5 Acc/+6.7 BA；Port Said→Shanghai +2.0 Acc；Shanghai→Qingdao BA +1.8 |
| **正确 CORAL 能够改善** | **命中（2/5）** | Rotterdam→Qingdao +7.1/+7.9；Shanghai→Qingdao +1.6/+0.8 → 二阶适配为**真实基线** |
| 两者均未改善但干净差距明显 | 不命中 | 至少一个方法在多对改善 |
| 差距主要由类别覆盖/少数采集造成 | **部分命中** | `in_ref` 达 .71–.99 Acc，标签驱动空间极大；且 `mean_only` 的 BA 收益常大于 Acc |
| 旧的大差距在规范划分后明显缩小 | **命中** | 规范协议下 cross 为 .24–.57，且**对齐效果随配对可正可负**（Jebel Ali→Shanghai T2S −15.2 Acc）→ 旧 0.39–0.58 不再作算法依据 |

**关键新事实**：简单对齐的效果**配对相关**且**可以严重为负**。这既是可用的真实基线，也直接指出下一步该做什么。

## 二、下一版的唯一干预（不再同时改多个环节）

> **配对感知的观测条件校正（pair-aware observation-condition correction）。**
> 只使用**部署时允许的无标签信息**（伪目标港 U_q 的特征统计），在 `cross` 与 `mean_only` 之间**逐对做选择**，
> 选择规则本身必须**在源端留出港上验证**（源港留一：用其它源港对训练，判断在留出源港上该校正是否为正）。
> 不引入新的表示、不引入知识、不改词表、不改筛选。

- **要打败的基线**：`cross`、`mean_only`、`coral_t2s`（同一表的同分母同词表）；同时报告**逐对 oracle 上界**（仅作参考，非可部署）。
- **验证协议**：源伪目标 5 个可判定配对（n=865–1,588）+ 源港留一；指标 Acc **与 BA 及逐类支撑**并报。
- **可被证伪的形式**：若源港留一上的选择规则在 5 个伪目标上不能稳定超过 `cross` 与 `mean_only` 的较好者，
  则判该干预失败，回到「表示层」问题（此时才考虑编码器级留一）。

## 三、明确不做（防止多环节同改）

- 不把知识（港口/关系证据）与表示适配、路由、词表同时改变；知识的作用留到**纯视觉适配基线可信之后**单独比较。
- 不在本轮新增真实目标港实验；不重抽 chip；不重训 encoder/H/J；不改原 E01 主表。
- 不复用旧的 0.39–0.58 作为动机或算法依据。

## 四、复现入口

```bash
# 第一步回归（旧缓存回放）
python e02_coral_regression.py --fold Rotterdam
# 第二步基线表（源伪目标协议）
python e02_pseudo_target.py --fold Rotterdam --run /root/autodl-tmp/e01_runs/Rotterdam_rev1
```
参考实现与 9 项单测：`E02_alignment_protocol_v1/coral_reference.py` + `test_coral_reference.py`（9/9 通过）。
