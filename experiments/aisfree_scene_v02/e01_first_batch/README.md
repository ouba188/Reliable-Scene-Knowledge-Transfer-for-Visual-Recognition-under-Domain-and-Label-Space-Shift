# E01 第一批执行包：P0 AIS-free / v0.2 条件关系矩

## 本包是什么
针对用户已完成的 E00，提供固定参数、329产品完整名单、外折source-fit/meta/cal分工、字段权限覆盖、B/D/J定义、可执行的分组/矩聚合/校准/筛选工具，以及PyTorch网络定义和H=8缓存特征训练器。

**不是**已经跑完的SAR实验，不含模型权重、未取得的服务器路径，也不是完整大景读取/encoder训练/评价的一键调度系统。现有实验端负责将E00对象/影像接到所规定的数据接口。没有远端push或服务器训练。

## 立刻使用的文件
- `config.json`：正式首轮预设H8/J3/B4/D4、seed、epoch、source-only阈值规则。
- `split_manifest.json`：329真实产品、161 adapt/168 eval，24港完整名单和24外折。四个首批目标另外在manifests/。
- `feature_permissions.review.json`、`docs/FEATURE_PERMISSIONS_REVIEW.md`：对E00权限文件的严格覆盖建议。
- `relation_spec.json`、`docs/RELATIONS_AND_CALIBRATION.md`：明确每个B/D含义与采样mask。
- `candidate_bank.json`、`docs/CANDIDATE_BANK.md`：8头规格、源encoder与增广。
- `configs/paths.local.example.json`：仅需由实验端填写真实路径。空路径不能被当成数据已存在。

## 版本和证据
本次成功读取的仓库main为435fa4b38e77d494c45ff6d91a6f0ed08f41c6ef。
数据状态采用用户最新E00同步：去重1787454行、P0 offshore408981行；本环境没有重新扫描对象表。
329划分来自该commit的lopo_folds.json ports段，不重新随机分割。全量CSV复制到执行容器的桥接不可用，因此本包根据已读取的完整ports段构造名单，并附本地逐产品/role核对命令；不会假称已完成原CSV字节哈希核对。
**训练前verify_manifest必须通过**。其产品/role比对失败即停止，不可静默重建别的329集合。

## 首批任务
先只跑seed20260920，顺序Rotterdam → Shanghai → Port Klang → Fujairah；选择按已有多采集和不同知识覆盖开发，不由新目标性能排名产生。协议G为主，T单列；Busan/Tanger Med只能T。
4折通过I/O、校准与机制诊断后，再以同一配置跑其余种子及其他合格港口。所有配置是首轮建议，不是已证明最优。

## 命令：这些组件现在就可以执行
在包根目录：

```bash
# 0. 保留已有torch/CUDA，不重装。附加依赖仅numpy/pandas/scikit-learn；依赖不存在才安装。
python -m unittest discover -s tests -v

# 1. 对你的本地仓库做逐产品核验，不走网络，不读取目标标签。
python tools/verify_manifest.py --repo-root /YOUR/LOCAL/REPO --out work/split_verified.json
# 或 --split-csv /YOUR/REPO/knowledge/phase0/splits/port_acquisition_split.csv

# 2. 使用E00部署视图和覆盖sidecar构建D，不读取AIS标签列。
python tools/prepare_relations.py \
  --objects /DATA/objects_dedup_with_mask.csv.gz \
  --coverage /DATA/layer_coverage.csv \
  --mask-column YOUR_ACTUAL_E00_MASK_COLUMN \
  --density-scope p0_offshore_dedup_leave_one_out --out work/relations

# 3. 实验端按docs/CANDIDATE_BANK.md训练共同encoder并导出无增广z512。
# tools/models.py是可直接实例化的网络；本包不提供你的大景/标签loader。

# 4. features.npz中只有z,m,sample_id,product_id,port，不含标签。
python tools/fit_groups.py --features /DATA/features.npz \
  --manifest split_manifest.json --target Rotterdam --out work/rotterdam/groups

# 5. 可执行H=8缓存头训练器。标签CSV仅含该折source-fit样本！
CUBLAS_WORKSPACE_CONFIG=:4096:8 python tools/fit_cached_heads.py \
  --features /DATA/features.npz --labels /DATA/rotterdam_source_fit_labels.csv \
  --manifest split_manifest.json --config config.json --target Rotterdam \
  --out work/rotterdam/heads --device cuda

# 6. 可执行关系矩模型训练器（源fit标签不进入目标输入）。
python tools/fit_cached_relations.py --features /DATA/features.npz \
  --relations work/relations/relations.npz --group-model work/rotterdam/groups/group_model.npz \
  --heads work/rotterdam/heads --labels /DATA/rotterdam_source_fit_labels.csv \
  --manifest split_manifest.json --config config.json --target Rotterdam \
  --out work/rotterdam/relations_model --device cuda

# 实验端用 q_j 和候选头导出 source-calibration 关系矩（形状见下）。
python tools/calibrate_tolerance.py --input /DATA/rotterdam_calibration_moments.npz \
  --out work/rotterdam/calibration.json

# 7. G协议只把目标adapt的模型输出和关系交给筛选器。
python tools/run_moment_filter.py --input /DATA/rotterdam_adapt_filter_inputs.npz \
  --config config.json --calibration work/rotterdam/calibration.json \
  --out work/rotterdam/adapt_filter

# 8. G协议固定上述解释集合，只预测eval，不在eval重新筛选。
python tools/apply_retained.py --probabilities /DATA/rotterdam_eval_probabilities.npy \
  --retention work/rotterdam/adapt_filter/retention.json \
  --out work/rotterdam/eval_predictions.npz
```

`--smoke`可使用固定tau=.15检查数值流程，但其输出必须标为smoke；不能作为主结果。代码不读取目标标签，不内置未知类概率。

## 数组接口
### features.npz
`z[N,512]`, `m[N,4]`, `sample_id[N]`, `product_id[N]`, `port[N]`，字符串禁止object pickle。
标签另文件：`sample_id,label`；预先按源fit独立导出，信赖的fine标签才入训练。候选词表由source-fit内>=100实例且>=3港的类产生；不看目标有哪些类。

### relation/phase0绑定
`relations.npz`为phi[N,4], available[N,4], sample_id/product/port, spatial_blocks[N]。
必须以sample_id一一对齐所有预测数组，不按现存排序猜。sourcefit/cal/target也必须使用相同可观测接口。

### filter_inputs.npz
`phi[N,4]`, `available[N,4]`, `groups[N]`, `spatial_blocks[N]`,
`probabilities[8,N,C]`, `relation_means[3,N,C,4]`；可带`enabled_dimensions[4]`。
每个q_j都要对每个候选类别c给出预测矩，不是只用argmax类。

### calibration_moments.npz
`observed[A,4,4]`, `predicted_anchor[A,3,4,4]`, `active[A,4,4]`。
A是独立source-calibration港口，不是船舶数；使用各港adapt产品和已在source-meta上固定的B0。
阈值不足4有效港则不制造阈值，正式run回退B1。保存各港ID、代码/银行/关系模型哈希的sidecar，确保校准文件不是目标输出改名。

## 中间产物与结果
至少提交：配置、代码SHA、329名单验证结果、新E00数据哈希、逐港类支撑、B大小、每维有效率、q_j源端误差、calibration.json、relation_checks、hypothesis_retention、逐港逐类Acc/BA与rescue/harm、I/O剔除登记。
主要效应Full−B1；D0–D2静态知识与D3图像群体分开消融。保存B0/B1/B3/B4/B5及错误/缺失知识；不要用target Y选阈值、港口或checkpoint。

## 不能默默跳过的启动条件
- 329列表与本地CSV逐项匹配；现场数据映射到列表，不扩大到841。
- source encoder、heads、q_j、PCA/KMeans均无目标/独立meta/cal梯度。
- 实际双极化像素映射和radiometry已登记；本包未声称PNG/raw TIFF数值自动相同。
- 三个静态图层availability来自E00证据，不由距离是否=2000猜。
- 10m去重沿用E00，不把密集真实双船合并可疑案例当算法正常训练数据。
- 全部P0候选（含未标注/虚警/未见类型）可能污染关系检验；校准只是经验规则，不保证已知纯度。
- 报告oracle/高可信标签子集与完整P0候选的区别；缺标签不当unknown真值，错误检测不当新船型。
