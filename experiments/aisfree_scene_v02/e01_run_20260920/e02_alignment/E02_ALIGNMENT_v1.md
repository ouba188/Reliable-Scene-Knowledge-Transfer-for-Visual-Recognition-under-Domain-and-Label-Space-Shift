# E02 对齐数值回归与源伪目标适配基线 v1（2026-09-21）

**边界**：CPU only；不重抽 chip、不重训 encoder/H/J、不动原 E01 主表。本文件替代此前 `aux_domain_a` 的
`coral` 列（该列实现错误，见 `aux_domain_a/README.md` 顶部勘误）。

---

## 一、第一步：CORAL 数值回归（旧缓存 + 旧分母回放）

修正内容（评审参考实现 `E02_alignment_protocol_v1/coral_reference.py`，9/9 单测通过）：

| | 变换 |
|---|---|
| 旧（错误，保留作勘误） | `W = Cs^{-1/2} Ct^{-1/2}`（两个逆平方根、无重着色） |
| **正确 T2S**（映射目标到源，源分类器冻结） | `X' = (X-μt) Ct^{-1/2} Cs^{1/2} + μs` |
| 正确 S2T（映射源后重训分类器） | `X' = (X-μs) Cs^{-1/2} Ct^{1/2} + μt` |

数值证据（六折 × 24 个源港，ridge=1e-3，PCA-64）：

- **恒等性**：把同一池代入变换，最大绝对误差 ~1e-14（旧实现不满足）
- **协方差残差**：正确 T2S ~1e-15；**旧实现 3 – 125**（相差 15 个数量级 → 公式错误被数值坐实）
- 准确率：`cross` 六折平均 .269–.531，**旧错误变换 .073–.475**，**正确 T2S .124–.400**，S2T .127–.408
  → 在**旧探索协议**（全池 PCA + 随机 CV）下，修正后的 CORAL **仍未超过 cross**（个别港小胜，如
  Rotterdam→Qingdao .292 vs .233、Fujairah→Singapore .291 vs .234）。

**口径**：此回放标记为**旧探索协议**，不得因公式修好而升级为严格泛化结果；它只说明「该二阶方法在此协议下未改善分类」，
**不能证明数据中不存在二阶偏移**。

产出：`coral_regression_<fold>.csv`（逐港 cross / wrong / T2S / S2T / 残差 / 恒等误差）、
`coral_regression_<fold>_meta.json`（映射方向、ridge、PCA 拟合范围、端口规模）。

---

## 二、第二步：源伪目标协议与三列基线表

**协议**（每次运行唯一确定并落盘 `source_pseudo_target_splits_<fold>.json`）：

| 角色 | 内容 |
|---|---|
| `D_fit` | 该折 `source_fit_products` 的**带标签**实例（PCA-64 与逻辑回归**都只在 D_fit 上拟合**） |
| `U_q^adapt` | 伪目标港 q 的**较早采集日期**产品，**只用视觉特征、不看标签** |
| `E_q^eval` | q 的**独立后期日期产品**，标签**只交给评价器** |
| `in_ref` | 仅作参照：用 `U_q` 的标签训练独立分类器，在同一 `E_q` 上评价（标签路径与所有无监督方法隔离） |

统一设置：PCA ≤64 维仅在 D_fit 拟合；`LogisticRegression(C=1, max_iter=3000)`；相同词表；
相同评价分母；`ridge=1e-3`。方法：`cross` / `mean_only`（用 U_q 的均值把 E_q 平移回源坐标）/
`coral_t2s`（正确白化+重着色，源分类器冻结）/ `coral_s2t`（源侧变换后重训）。

### 结果：5 个可判定伪目标（n=865–1,588）

| 折 | 伪目标港 q | 角色 | n_eval | Cross | Mean-only | CORAL-T2S | in_ref（参考） |
|---|---|---|---:|---|---|---|---|
| Rotterdam | Qingdao | cal* | 1,588 | .240/.318 | **.315/.385** | **.311/.397** | .924/.354 |
| Shanghai | Qingdao | cal* | 1,451 | .391/.560 | .360/.578 | **.407/.568** | .990/.500 |
| Fujairah | Shanghai | meta | 1,094 | **.419/.344** | .397/.285 | .398/.281 | .736/.607 |
| Port Said | Shanghai | meta | 1,094 | .404/.339 | **.424/.315** | .396/.288 | .709/.585 |
| Jebel Ali | Shanghai | meta | 865 | **.572/.494** | .535/.447 | .420/.371 | .840/.736 |

（`*` 角色披露见下。其余 11 行为 n=32–60 的小样本，只登记不作结论。）

### 读数

1. **均值平移（Mean-only）与正确的二阶对齐都能改善**，但**只在部分配对上是正收益**：
   Rotterdam→Qingdao +7.5 Acc/+6.7 BA、+7.1/+7.9；Shanghai→Qingdao T2S +1.6/+0.8；
   而 **Jebel Ali→Shanghai 的 T2S 掉 15.2 Acc 点**、Fujairah→Shanghai 掉 2.1，Port Said→Shanghai 也略负。
   → 简单对齐**不是普遍有效**，是**配对相关**的，且可以显著为负。这正是一条可用的真实基线。
2. `in_ref`（用 U_q 标签监督）在所有大样本伪目标上都远高于无监督方法（.71–.99 Acc），说明
   **标签驱动的空间确实存在**；但它不是可部署方法，也不是理论上限。
3. 三列基线由此确定：**Cross / Mean-only / CORAL-T2S**（CORAL-S2T 作为独立附加列，因需重训分类器，
   不与其混列）。

### 角色与披露

- `D_fit` 与 `U_q^adapt` 永不重叠；`E_q^eval` 与前面两者按**产品 + 采集日期**隔离（`source_pseudo_target_splits_*.json` 记录）。
- **披露**：`Qingdao` 在两个折中属于 `source_calibration_ports`（cal）而非 meta。评审要求「原 source-cal 不再并入
  新模型训练或选择」——本表**未**把 cal 并入 D_fit 或任何选择过程，仅作为**伪目标港**（其标签只进评价器）。
  为可判定的样本量考虑，这一使用在此显式登记；meta 港中除 Shanghai 外仅 3–5 条标签，按规则登记为**不可评估**
  （`source_pseudo_target_splits_*.json` 内含原因）。
- **未做**：按伪目标真实标签挑选「已知类/有 AIS/标签完整」的对象再适配；也未随机拆 chip 凑跨采集。

---

## 三、交付件（评审要求的五个）

| 文件 | 内容 |
|---|---|
| `coral_regression_*.csv` / `_meta.json` | CORAL 数值回归（cross / 错误 / 正确 T2S / S2T + 残差与恒等误差 + 元数据） |
| `role_lineage.csv` | 8,743 行：每折 × 产品 × 角色（source_fit / meta / cal / shadow_eval / target_adapt / target_eval）+ 采集日期 |
| `source_pseudo_target_splits_*.json` | 每个伪目标港的角色、可评估性、D_fit/U_q/E_q 的产品与日期清单、不可评估原因 |
| `cross_mean_coral_corrected.csv` | 三列基线表（16 行，含小样本行；`n_eval` 列区分是否可判定） |
| `DECISION_NEXT.md` | 下一步单一干预的选择与理由 |
