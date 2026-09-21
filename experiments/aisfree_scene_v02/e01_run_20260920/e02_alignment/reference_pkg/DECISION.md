# E02：优先适配方向，但先修正CORAL基线与源港留一协议

审查提交：76331f1c4a6c16efcecfa997b9cebb59c95d8e17。
状态：执行建议与数值参考组件；未运行服务器数据，未推送仓库。

## 决定
选择(a)，暂缓新的(b)训练。不重抽chip，不改变E01模型、tau或主表。
现有a/b产物保留为探索记录。先完成下面两步CPU工作，再决定一个适配原型。

## 发现1：当前CORAL公式不是白化加重着色
原e01_domain_a.py执行：
  W = invsqrt(Cs) @ invsqrt(Ct)
  Xt_new = (Xt - mut) @ W + mus

按样本为行、目标映射到源的定义，应当是：
  A = invsqrt(Ct_reg) @ sqrt(Cs_reg)
  Xt_new = (Xt - mut) @ A + mus

无正则且协方差满秩时，A.T @ Ct @ A = Cs。
同一源/目标统计下应为恒等变换；原公式对非单位协方差会破坏恒等性。
coral_reference.py分别提供target-to-source与source-to-target两个明确方向。
后者必须在映射后的源特征上重新拟合分类器，不能沿用未映射源训练的分类器。

加ridge时验证的是REGULARIZED协方差身份；不要声称原始经验协方差完全一致。
固定ridge=1e-6是对原脚本的最小修复，不是声称该值统计最优。

## 发现2：STK不属于255个等权子集的函数族
STK是各头概率拼接后的多分类逻辑回归；可跨类重组并使用任意符号权重。
OBJ_SUBSET不能作为STK上限。Fujairah粗类STK=.201058 > OBJ_SUBSET=.153439并不矛盾。
不再使用“所有可用策略都低于OBJ”或“因此只能选择已有正确头”的表述。

## 步骤A：最小数学校正回放（CPU，不改模型）
- 使用原缓存及原分母，只修CORAL公式，另存结果，隔离数值错误的影响。
- 旧cross/in列继续明确标注exploratory/global-PCA/head-level，不提升为干净LOPO。
- 每个比较记录映射方向、ridge、covariance residual和所用产品ID摘要。
- 先运行test_coral_reference.py，全部通过后再接数据。

## 步骤B：建立真正用于方法选择的源伪目标开发协议
- 基础编码器及归一化必须未以监督方式用过伪目标港。
- 最小复用方式：每个旧外折中以original source-fit训练；original source-meta-query港作为伪目标。
  不把source-cal混入适配训练或模型选择。
- 对伪目标港按产品/日期冻结adapt与eval，相关船舶/同次采集按组隔离；不随机拆chip。
- 模型只读取伪目标adapt的合法无标签实例；不可按隐藏真值、是否有AIS或是否in-vocabulary筛适配输入。
- scaler/PCA只fit source-fit，所有方法使用同一冻结投影；保持PCA64作为明确的表示限制。
- cross: source-fit监督训练的LogisticRegression(C=1,max_iter=3000) -> 同一伪目标eval。
- mean-only: 由source与伪目标adapt均值得到平移，冻结后作用于eval。
- CORAL-T2S: 仅由source参考与伪目标adapt估计变换，冻结后作用于eval；source分类器不变。
- 可另报标准CORAL-S2T: 变换source并重拟合线性分类器，eval不变换；必须单独列名。
- 如保留in_ref，使用伪目标adapt的标签训练单独的监督参照，并只在同一eval评估。
  该标签路径与所有无监督方法物理分离；in_ref是带目标监督的参考，不是理论上限。
- 稀有类/产品不足给出NA及原因，不随机改分组、不降低词表门槛凑统计。
- 现有原编码器见过的内留出港只能作frozen-representation probe，不能叫完整encoder-held-out泛化。

## 主要比较目标
主要指标：同一eval上的BA_adapt-BA_cross，同时报告Acc、逐类召回、分母和覆盖。
gap = in_ref - cross是辅助描述，不能靠降低in_ref取得“gap缩小”。
适配方法本身可用原z加低秩修正等作为后续候选，但必须先将本轮基线修正完成。
本包没有实现或宣称提出新适配算法。

## 本轮结束时交付
- coral_numerical_regression.json（或测试文本）
- role_lineage.csv（每个伪目标港是否被encoder、scaler/PCA、head、STK见过及角色）
- source_pseudo_target_splits.json
- cross_mean_coral_corrected.csv（Acc/BA/逐类支撑、原子计数）
- DECISION_NEXT.md（按修正基线只选一项后续干预）

## 文献核验
Sun, Feng, Saenko, Correlation Alignment for Unsupervised Domain Adaptation, arXiv:1612.01939,
Section 2.2: whitening then recolouring, not multiplying two inverse square roots.
scikit-learn Common pitfalls: split before fitting preprocessing; PCA fitting counts as fitting.
scikit-learn StackingClassifier: prefit base-model predictions used on the same training rows
can induce overfitting; standard stacking uses held-out/cross-validated base predictions.

## 测试边界
9项本地合成单元测试通过。未加载真实源缓存/目标样本，未重算CORAL准确率，未运行原型训练。
