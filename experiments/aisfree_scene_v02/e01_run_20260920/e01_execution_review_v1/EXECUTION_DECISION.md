# E01执行回执裁定与首轮回传要求

## 审查范围与结论

本次静态读取的仓库提交：`7b4a46bc452c3978ec64dec7adf3c82140442787`。
文件范围：`experiments/aisfree_scene_v02/e01_run_20260920/{e01_chips.py,e01_features.py,e01_glue.py,e01_chain_rotterdam.sh}`；并对照已交付的E01.1 config与设计文档。

执行端报告的E00已完成工作不重做。服务器实际脚本与缓存未在此环境读取，不能据静态代码断言现有全部chips错误。应先绑定实际脚本SHA/哈希和产物生成版本。

**决定：严格fine词表原则通过；增广与source-fit归一化原则通过、实现需修正/对账；否决产品顺序打乱等同于uniform_port_class。后续三折的自动正式训练暂缓，先做执行接口检查。不要因目标是否涨分决定是否续跑。不要删除已有运行。**

## 已发现的执行代码问题

1. `e01_chips.py::crop_pair` 使用 `col,row=ds.index(x,y)`，但Rasterio实际返回(row,col)。`Window`参数是(col_off,row_off,width,height)。若实际执行相同，应修为`row,col=...`。用已落盘样本与独立窗口像素逐值核对，选非对角row!=col目标；不是只核验标签世界坐标。现有可视化回执与该代码存在待解释差异，可能本地已有修复未入库。
2. `e01_features.py`将product_id与sample_id转成`<U64`；glue导出filter输入也再次转`<U64`。产品名示例为67字符，对象ID更长，导致后缀截断、不同对象碰撞、无法匹配源标签与split。用动态宽度Unicode字符串；每次保存/读取断言完全相等、sample_id唯一。已截断数组不能靠`.astype(str)`恢复。
3. `e01_glue.py`用sample_id键的`meta`执行`meta.get(product_id)`。calibration循环中先尝试推断port但未用该变量，随后`r=meta.get(p)`仍然为空，可能不产生任何校准港。应使用split_manifest的产品到港口映射，且四个cal港适配样本量和有效矩要分别记录。数据确实不足时按原规则fallback，不能人为造校准港。
4. glue名为`equal_port_ba`的循环实际计算逐港Accuracy，不是逐港class recall宏平均。B0锚点选择应先每港BA再跨港平均，最终metrics同时输出Acc、BA、逐类分母/召回。修正B0选择后需要重新做校准/filter，不能只改报告标题。
5. chips有效比例目前是两极化非零像素的OR；VV全有效而VH全空也可能得到vf=1。应绑定明确的栅格validity规则、读取数据mask/界外mask，输出VV、VH和共同有效率。不得默认真实数值0一定无效。`e01_features.py`中未见按chip有效率>=.95筛选训练/导出有效预测；m4还取旧对象表valid_fraction。应保持P0对象登记完整，但把不可读/双通道不完整对象的预测状态独立标注；m4和筛选采用同一芯片级有效性版本。
6. augmentation使用np.roll，是环绕而非配置的reflect平移；随机种子只有seed+样本索引，同一对象各epoch重复同一个变换。修复需传epoch/draw，不能假定persistent workers能看到主进程epoch更新。正式重跑使用reflect规则。
7. encoder分块采样遍历每个样本一次，未实现config声明的uniform_port_class。不能将其称为等价纯I/O修改。现run可保留为pilot_product_block_unweighted，正式E01.1恢复目标分布。
8. shell仅set -u并把训练命令管道到tail，失败可能被tail成功码吞掉；通过存在features.npz判断就绪可能读到半写文件/旧文件。使用set -Eeuo pipefail、完整日志、输入哈希/完成标记；生产文件先写tmp再原子rename，完成标记最后写。新run目录，禁止静默复用不匹配旧输出。

这是对已读代码的静态判断。若服务器实际运行的是修复版，应提交运行版及产物来源证据，不能据旧代码要求无谓重算。

## 三项口径的细化

### 1. 词表
同意class_level=fine同时排除*_coarse、ship_untyped以及本体中的untyped/non_ship/unknown占位类别。原始表不覆盖，记录规范化规则和冲突计数。只用source-fit、去重且图像有效的标签支撑统计>=100且>=3港。最终5类名单/顺序冻结后不根据target分数变更。meta/evaluator同一规范化映射；coarse不能提升为细类。

目标词表外的已知标签、粗标签、无标签、非船候选分开登记。目标候选池不按真实类别筛选。本轮不因为source词表只剩5类就称目标池必然闭集。

### 2. 增广/归一化
设计同意。归一化统计用source-fit且图像有效的数据，记录参与统计的sample_id/采样seed/通道顺序/均值标准差。现值不凭大小否决。若chips或有效性版本改变，需要重新计算。不得跨折复用Rotterdam统计或encoder，因为其它折角色不同。

### 3. 采样
原目标概率：P(i)=1/[P * C_p * N_(p,c)]。产品局部性属于读取层，不应改变优化层抽样。
推荐只收集当前fold有监督的uint8训练chip进RAM/单一npy memmap，再按平衡概率有放回抽取。2918条×2×128×128 uint8共约91.19 MiB；不需要为平衡训练转换全40.9万条。

如果采用prefetch，可按产品读取批内数据，但必须还原预先采样顺序再交优化器。记录每epoch各港/各类实际抽样次数；随机抽样期望平衡，不要求每次恰好相等。

## 如何最小化重算

- 实际chips是正确版本，仓库脚本落后：提交运行代码和trace即可，不重做E00。
- 实际chips存在行列错误：新目录重抽受影响chip；重新统计、训练encoder、导出features和下游。原表、split、真实关系几何可保留，除非依赖更改数据。
- 只有feature字符串截断：若可证明完整导出行序与原chip索引一致，可仅重建ID和重新导出关联产物；否则重新导出features。不要盲拼回截断后缀。
- 只有glue错误：修复B0、calibration映射、指标，重跑prep→calibrate→filter→apply→metrics。H/J权重若来源正确可复用。
- 只有sampling/augmentation偏差：旧encoder记探索性变体；正式版本重新训练。禁止改旧配置让它看似原方案。

## 首轮回传优先级

### P0：执行接口
actual_code_sha；chips/features/labels/relations/group/heads的ID完整性与join覆盖；manifest5角色对象数；crop落盘窗口对照；image eligibility；4个cal港有效实例/矩；每epoch抽样支撑。

### P1：机制是否实际作用
每H的meta BA与pairwise disagreement；每J的预测矩差；B×D有效数量/空间块/均值/方差/零值比例；source维度停用理由；B0锚点；4cal误差；tau；target-adapt各(h,j)最大残差；feasible与returned集合；conflict/缺失/uncalibrated原因。

关系valid_counts只是非缺失数量，不能说明值有变化或类别关系有区分力。D0全部有效仍可能多数为0。不得为制造筛选作用改tau、D、H。

### P2：真实任务增量
同一target-eval可读且有可比细标签子集：B0/B1/Full的Acc、BA、逐类N/recall、rescue/harm、标签翻转总数；并报告完整P0/IO有效/可评价/词表外/无标签的数量。

同样本、同权重时：Delta Acc=(Rescue-Harm)/N。Delta BA=mean_c[(Rescue_c-Harm_c)/N_c]。类别N=0记NA，不在公式中暗填0。模型比较使用完全相同分母。

### P3：归因/不确定性
Static(D0–D2) / Group(D3) / Full；B3无条件关系；B4单解；B5逐组机制；按采集成对效应（Rotterdam eval只有6采集，勿夸大CI）；所有失败结果保留。

## 四折续跑规则

先绑定并通过实际代码/产物接口，修复/记录偏差，然后以冻结的一版顺序Rotterdam→Shanghai→Port Klang→Fujairah。每折独立fit编码器/归一化/B/H/J/meta选择/源校准/词表；同配置指相同规则，不是同权重和同tau。

允许自动阻断的条件是接口错误、错误权限、训练/读写失败、校准文件结构错误。全缺失/关系冲突/合法校准不足按原算法fallback并保留结果，不通过改target超参挽救。目标涨分、零增益、负增益不改变四折配置和顺序。

本轮不承诺后台监控；没有修改GitHub或服务器。附属检查工具需在本地执行。
