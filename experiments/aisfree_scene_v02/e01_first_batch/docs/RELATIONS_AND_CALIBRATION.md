# D、B 和阈值的执行定义

## B=4：不是四个知识字段
B是视觉分组数。冻结z512→source-fit StandardScaler→PCA16（不whiten）→KMeans4，n_init20。
fit池仅source-fit P0图像有效对象，可包含未标注对象，每港以sample_id的SHA256固定抽最多2000条，防止大港支配。meta/cal/target只transform，不能重新拟合或按目标类别排列簇。
不使用 `port`、经纬度、AIS、任何D值、density或support参与聚类。小簇不重分，不按目标Y合并，只停用对应检验单元。

## D=4：四个有界关系矩
D0 外部静态：`clip((2000-d_coast_m)/1500,0,1)`，在离岸任务中500m对应1、>=2km对应0。
D1 外部静态：`clip(1-d_anchorage_m/2000,0,1)`。
D2 外部静态：`clip(1-d_fairway_m/2000,0,1)`。
D3 SAR群体：`log1p(min(local_ships_1km,64))/log(65)`。

D3来自图像群体，不得把仅D3带来的收益表述成外部港口知识的收益。必须同时报告 static(D0–D2)、group(D3)、full(D0–D3)。NN及500m密度本轮不用来重复堆同一信号。
距离层有实际覆盖时，2000为合法截断值；没有层时该矩缺失。密度必须注明去重、offshore、leave-one-out以及边界处理。只剩裁切后的局部名单时需保留至少1km halo用于统计，不能按训练/标注过滤对象后重算邻船。
角度、OSM设施类别、d_quay在本批关闭：无需为了凑D继续使用稀疏、范围外或AIS回退属性。

## 活跃单元
预设 min_count=64、min_spatial_blocks=4（1km网格，以product_id区分）。它们是工程门槛，不是独立样本数证明。
source-fit有效比例<.10、有效值variance<1e-4或类别条件支持不足，停用该D。
同一B,d的观测矩和每个h,j预测矩使用完全相同的available对象。推理不删除知识缺失对象：只让相应检验单元无效。

## source校准
tau是共享标量，所有phi在[0,1]。源meta选好B0后冻结。
对每个独立calibration port的adapt产品，按与目标G相同方式合并关系矩，计算：
    score_port = min_j max_active_(b,d) |t_obs - t_pred[B0,j]|
    tau = max(0.05, max_valid_calibration_port(score_port) + 0.02)
至少4个有效校准港；不足则正式结果fallback到B1并记录calibration_unavailable。tau>=.50时记uninformative，不强行压小阈值来制造正收益。

这是可执行的保守经验校准，不是95%跨港覆盖保证；4个港口不能据此宣称高置信distribution-free safety。校准输入也用标签盲候选总体，使false proposals/未标注已知状态至少进入源端误差测量，但不证明目标私有污染被控制。
`tau=.15`只用于smoke运行，不得悄悄用于主结果。不能凭目标Acc或保留率挑tau。

## G/T要分开
首轮主协议G：用target_adapt产品聚合检验，固定保留H集合和同一J解释，再只对target_eval做预测。不得用eval关系再次筛候选。
T：每个产品自己作标签盲关系检验，单独标记transductive。Busan/Tanger Med只有1产品，只做T；其余小港产品少也不包装成稳定统计结论。

## 开放类别边界
本批不造target-private真值。源类别只由source-fit标签生成。全部P0对象都预测；只有隔离评价器依据真实标签计算可评指标。缺标签、未匹配、机制冲突都不等于unknown类。
目标池若有新类/虚警，可能破坏已知类关系混合假设；本批将其作为可检验的污染与fallback来源，不宣称识别保证。
