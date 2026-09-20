# E01 第一批视觉候选库及关系模型训练规范

这是首轮预设工程配置，不是最优超参，也不计为新的 backbone 创新。

## 1. 数据与编码器
一个去重对象只对应一条实例记录；VV/VH 是两通道而不是两个实例。
固定地理窗口：以该簇代表中心截取 native 128×128，不沿 OBB 把每条船拉到一样大，不放大到224。不自动扩大长船窗口；若船体越界，记录 `object_extent_clipped` 并分层报告，不能默默删除。

两极化须同 CRS/affine/grid，0.25像素容差；否则先显式重投影到声明网格。10m只是本批预期网格，需要验证；不同spacing>10%先报inventory，不做暗中resize。padding标记无效；有效比例<.95的对象仍留P0登记，预测状态记录为I/O/coverage缺失。

第一批复用现有uint8派生栅格，数值/255后只用source-fit计算每通道mean/std。不能对每个港口或chip独立minmax。float、uint16、signed-int8输入必须先明确解析方案；不猜辐射单位。uint8方案不声称保留全部辐射信息。

`models.py` 提供无torchvision、无外部权重下载的 SARResNet18GN：2通道、3×3 stride1 stem、无maxpool、[2,2,2,2]残差块、GroupNorm32、GAP512。每外折源训练30 epochs，AdamW 3e-4/WD1e-4，microbatch64、累积2（effective128），3epoch warmup+cosine。末轮checkpoint固定。全encoder冻结后抽取无增广z。

**已有encoder只有在可证明未见该外折目标及独立meta/cal港口时才能复用。**不确定其训练来源就从头训练，不能把旧24港全量模型当作未见目标基线。

增广只在源encoder阶段：180°旋转p=.5、±4px平移（VV/VH同步），不加90°旋转、强度jitter、mixup/cutmix、随机尺度、模拟speckle。它是有限数据增强假设，不是SAR物理等变定理。提取z及训练关系模型时无增广。元数据不参与几何增广下的角度目标监督。

## 2. m 的四列
1. log1p(OBB长边米)
2. log1p(OBB短边米)
3. log(长/短)
4. valid_fraction

m不包含port、坐标、密度、航道/锚地距离和类别。第一批不把稀缺入射角的补值作为新变量；后续 metadata-rich 消融单独进行。候选输入=[z512,m4]，source-fit标准化。

## 3. H=8
2种采样 × 2种头 × 2种种子。
- uniform_port：先等概率港口，再港内样本。
- uniform_port_class：先等概率港口，再港内等概率已有类别，再该单元样本。
- 头：Linear 或 MLP(516→128→C, SiLU, Dropout .1)。
- 头种子：run_seed、run_seed+1009。

每头40 epochs、AdamW lr=.001/WD=.001，batch256，不early-stop、不按目标K或Y修改候选。仅source-meta-query标签用于选择B0参照（equal-port BA，平局固定head_id）；B1始终8头等权平均。温度首轮全部1，不再额外引入目标校准。

候选差异若非常低，记录后按预定义规则暂停优化，不人为换标签制造歧义。知识过滤只能选择已有解释。

## 4. J=3
共用source-fit样本及类别空间：
- J0：port-class平衡。
- J1：固定hash半数源港权重×3，其余×1。
- J2：另一半源港×3，其余×1。

这是有限的源机制扰动库，不宣称三个真实世界穷尽所有目标关系。相同J必须适用于决策范围的全部B/D。
关系模型输入=[PCA16(z),m4,onehot(c)]，MLP64→D4，sigmoid，masked fractional BCE，35epochs、AdamW .001/WD.01、batch512。
只在真实source-fit有标签配对上学习；不以target pseudo-Y训练关系模型。D每类至少30有效实例、2源港，达不到的维度关闭并记原因，不把缺失值填0。

## 5. 训练单位
source-fit/meta-query/calibration在PORT层完全互斥；详细完整产品名单在split_manifest.json。source-fit可用其港口原adapt+eval两部分（它们在此折不是目标）。meta选择完成后不并回训练；cal独立。

训练器与数据读取器需要由实验端接到本机chip shard接口。此包有可实例化架构、cached head训练器、分组、矩聚合、校准和筛选；不宣称已在服务器执行encoder或关系模型训练。
