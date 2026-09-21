# E01：选择(ii)，先修正辅助真值与粗层后处理，再做冻结候选库的有限上界诊断

## 本轮决定

- 不扩跑未完成的外折，不训练新模型，不放宽源词表门槛，不重选B0/tau。
- 已完成六折的原始结果和修订记录归档；主表不覆盖。
- 使用修正的评估专用标签，在同一组冻结eval概率上做四个视图：fine_T1、fine_T1plusT2、coarse_T1、coarse_T1plusT2。
- 这是探索性、使用evaluation truth的能力诊断。Oracle结果不是可部署算法，不得把最优子集写回retention.json。

## 源码复核范围

仓库：ouba188/Reliable-Scene-Knowledge-Transfer-for-Visual-Recognition-under-Domain-and-Label-Space-Shift
基准提交：fabd72d08b2e2e7cef7b4e481f2c69d1cec94cc6
实际读取：e01_label_boost.py、e01_boost_eval.py、aux_label_boost/boost_eval_JebelAli.json。
没有访问服务器AIS或真实预测数组，没有重新计算六折真实指标，没有推送仓库。

## 必修1：半径内单一MMSI，不能只看8条最近AIS记录

当前label_boost使用tree.query(...,k=8,distance_upper_bound=500)，它只返回至多8条点记录。
一个MMSI的8条近记录会挡住第9条、仍在半径内的其他MMSI，从而错误认定唯一。

改为完整半径枚举：

```python
idx = np.asarray(tree.query_ball_point([x, y], r=500.0, eps=0.0), dtype=np.int64)
distance = np.linalg.norm(tree.data[idx] - np.array([x, y]), axis=1)
records = match_tiers_from_complete_radius(distance,
                                         kep_e[idx] - scene_epoch,
                                         kep_m[idx])
```

分别对T1(150m,180s)、T2(500m,300s)做时空过滤后计算唯一MMSI。
match_tiers_reference.py提供候选记录计算，不读取船型表，不认证目标身份，也不新加时间补偿模型。
每个tier独立保存MMSI/距离/时差/候选MMSI数；T2-only也必须有证据。
T1优先、T2补齐的组合应命名T1plusT2，不能误称纯T2；T2-only单列质量统计。
本修复生成boost_v2，不覆盖旧辅助层。已冻结模型及eval概率无需重训。

逐(product_id,MMSI)检查是否有多个去重对象同时被分配为同一船；冲突不能靠“单候选”自动认证。
一致率是与旧标签的一致性，不是独立标注准确率；两者复用了AIS/MMSI类层，重叠样本不能证明新增远距离匹配正确。
审核新增T1/T2-only子集，排除不可靠身份；不以港口知识或本模型预测决定真值。

## 必修2：粗层必须先求概率和、再argmax

```python
p_coarse, coarse_names = parent_sum(p_fine, fine_names, explicit_mapping)
```

不能先fine argmax再映射。未知字符串不默认“other”。
输出词表必须全部有明确合法映射；未解析/非船标签不强制映射成已知船型。
父类不受输出支持的对象单列coverage，不塞入受支持子集；主细类表保持不变。

## 新的诊断输入接口

每个fold/view一个评价专用NPZ，按完整sample_id关联，所有方法使用相同对象集：

- `p`：float[H,N,C]，固定eval每头概率；粗层时先对每头的细类概率按父类求和。
- `y`：int[N]，当前视图中可信且输出空间支持的真值类别索引；不支持/无标签在外部coverage记录，不能硬转成0。
- `sample_id`：完整唯一Unicode[N]；不存MMSI。
- `class_names`：Unicode[C]，与p的列顺序一致，不得按目标频率重排。
- `score`（建议）：float[H]，已冻结target-adapt分数，不用evaluation truth重新拟合。
- `product_id`（建议）：Unicode[N]，统计采集支撑。
- `b1_pred`（建议）：int[N]，同视图官方B1预测，用来核对一致性；不一致直接停止诊断。

主报告保留有效评价对象构成与每类支撑。不合并重复船舶当独立样本。
N=0只记录不可评价，不运行上界；样本量少仍可做描述性诊断，不提供无条件显著性结论。

运行：

```bash
python -m unittest -v test_diagnostics.py
python oracle_audit.py --input /path/to/evaluator/fold_coarse_t1.npz \
  --view coarse_t1 --out /path/to/evaluator/aux_candidate_ceiling_v1/fold/coarse_t1
```

严格在隔离评价器运行。输出目录不能已存在，避免覆盖证据。
报告会包含目标标签后见选择出的子集，文件前缀ORACLE_ONLY；不要让训练/部署自动读取这些输出。

## 三种不同上界

1. `single_head_ORACLE`：事后在整个评价集选一个最佳头，只是固定单头诊断。
2. `fixed_subset_ORACLE`：H=8时枚举255个非空、等权、整折固定子集；这是固定预测族下允许改变保留策略的经验能力上界。
3. `samplewise_subset_ORACLE`：每个对象允许独立选择最有利子集，是更宽松、当前场景级策略不能直接实现的上界。

另提供`score_prefix_ORACLE`：保持冻结s_h排序，只允许s_h<=tau的嵌套前缀，空集按旧规则回退全H。
不输出“推荐tau”，不允许用目标真值部署最优阈值。
额外的uninformative等策略限制可使实际可实现集合更小，所以此项仍是上界而非已获保证。

重要：单头argmax没有真类，不等于子集均值永远不能预测真类。
示例：[.55,.05,.40]和[.05,.55,.40]都不选第3类，但均值[.30,.30,.40]选第3类。
因此工具同时报告单头union和全子集可恢复性，不能将两者混称“候选库里没有答案”。

## 下一阶段决策

- 固定子集上界很小：仅改全局保留策略收益受限，先研究视觉/类别覆盖，不批量调tau。
- 固定子集上界大、按分数前缀上界小：关系分数排序没对准分类效用，单调降tau不能解决。
- 两者都大、当前Full零增益：再用源伪目标开发新校准/选择规则；目标oracle不可部署。
- 对象级上界大、固定子集小：场景级统一选择存在权衡；考虑局部机制属于下一版方法，不在本批改。
- 所有结论都相对于当前辅助真值；补标错误可改变这些读数。

## 统计文字修正

Jebel Ali coarse_t2=1056，但该JSON中的类计数是cargo813/fishing6/tug237，并非最少类>=50。
此前“1000+每类50保证Acc/BA 95%半宽约3pp”已撤回，不恢复。
B0/B1差3.6pp不能仅由N判成噪声，应基于同样对象的配对变化及采集/同船相关性分析。
本工具不提供显著性检验，Oracle极值也不是无偏的测试性能估计。

## 测试范围

20项合成单元测试：ID唯一性、概率校验、parent sum、255子集、净收益恒等式、固定/逐对象上界、
同分数不可拆开、按分数排序的受限上界、单头union遗漏ensemble正确类、k8假唯一反例、T2-only证据。
另执行一次合成CPU命令行冒烟测试。未读取任何真实SAR预测或AIS。
