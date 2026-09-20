# E01 Rotterdam首折：结果判决与聚合索引回归检查

这是对 commit `58654191eed5678959470437046f3cd640ef3e6e` 中结果的只读分析与最小修复参考。
没有运行服务器SAR训练；没有读取原始目标预测/真值；没有改变正式tau规则；没有上传GitHub。

## 1. 已确认的数学判断

8头分数范围0.114724–0.121776，tau=0.421208，全部保留。因此按等权规则Full=B1是结构性结果，不依赖46条评价标签。
这说明本配置在本折没有产生实际知识增益，但不能推出港口知识一般无用。
即使诊断性去掉Qingdao，tau=0.177810仍保留8头。取4个cal分数最小值+0.02为0.149476，也仍保留8头。
这些重算只解释结果，不允许用来选一个会剪枝或会涨分的新tau。

## 2. 评价决定

保留46条原细类结果并明确计数/覆盖。新增逐类支撑、原始标签→可评价标签的过滤瀑布。
24港聚合只能来自每港自身外折目标eval的out-of-fold结果，不能用Rotterdam模型评估它见过的源港后合并。
粗类指标可独立增加，但必须冻结fine→coarse本体；不把粗标签升级成细类；不替换原细类主表。
校准锚点记录只有14条meta标签，应报告支撑；不能声称0.267已经是可靠的跨港BA。
仓库glue代码仍用港内Accuracy后跨港平均，却命名equal_port_ba，需要和服务器实际版本对账。

## 3. 运行决定

保留当前run。后三折不根据Rotterdam分数改超参；但实际运行代码、chip窗口、ID完整性、BA锚点和cal产品映射未核对前，不批准为正式批次。
已经运行的任务在阶段边界保存并暂停后续，勿删产物。
若服务器已经修复，仅补交对应代码与文件SHA；若发现实质错误，按依赖最小重算，再用同一修订版本完成四折。

## 4. 索引修复

我原包的 aggregate 使用 q[:,use,:,d]，混合高级索引可能返回[n,J,C]，不是[J,n,C]。
执行端拆成 q[:,use][:,:,:,d] 是正确修复。e01_core_axisfix.py仅替换原包这一行，不改变算法或校准。
它不是覆盖服务器文件的自动补丁；实际代码可能还有别的修改，须手动合并并记录两端SHA。

运行回归：

```bash
python test_aggregate_regression.py --core /实际包/tools/e01_core.py --out aggregate_regression.json
```

7项测试包括n不等于J及n等于J但轴静默交换、不同缺失mask、停用维度、无可用知识。
对修复版7项通过；对原交付代码测试出现错误，证明原21项检查漏掉了这个生产形状。

只读重算分数：

```bash
python summarize_screening.py --retention /运行目录/adapt_filter/retention.json --out screening_geometry.json
```

工具不选择阈值、不修改预测、不读目标标签。

## 5. 下一个最小证据包

- runtime_provenance.json：实际encoder/core/chips/features/glue脚本SHA，chip与features哈希，修复时间及受影响产物。
- eval_support_waterfall.json：固定P0、图像有效、ID join、任意可靠标签、strict fine、in-vocab的逐级计数；meta单列。
- source_meta_class_counts.json：每港每类N与每候选真实BA。
- moment_diagnostics.npz：cal/target-adapt的observed、predicted[H,J,B,D]、active、count、空间块数；不含目标标签。
- candidate_disagreement.json：H预测分歧、类条件关系输出range；用于判断q是否对c有区分力。
- per_class_metrics.json：真值N、预测数、正确数、recall、rescue/harm；未标注不是unknown真值。

文件名为建议新增记录，不表示它们已经在实验端生成。
