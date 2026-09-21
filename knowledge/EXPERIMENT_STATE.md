# 实验状态 · 单一事实来源（三会话共享）

> 本文件是**唯一权威状态**：`20260919_143043_19c027`、`20260920_101220_372993`、`20260919_204838_d0b3b3` 三个会话都在为同一个实验服务。
> 约定：**谁完成一轮工作，就更新本文件里自己的那一行 + 时间戳，并推仓库**（仓库镜像 `knowledge/EXPERIMENT_STATE.md`）。
> 数字必须来自本次实读，不得沿用旧汇总；不确定的写「未知」而不是猜。

最后更新：**2026-09-20 20:55**（by 20260919_204838_d0b3b3）
> ✅ 20:45 复核（d0b3b3）：标注 **841/841 实打实**（此前 5 个缺口已重标修复，见 §5）；E00 审计件已产出（`next_experiment_v03/e00/`，含 feature_permissions.yaml / id 碰撞 207 万行 / 距离截断 / 近重复 174 万对）。
> ✅ 20:52 清理（用户批准）：复核 jpg 106.5 GB + 重复 bundle 4.3 GB + 历史 tgz 1.9 GB + 多余 run 目录 + md5 相同的重复大图 → **回收 ≈115 GB，磁盘 106G→221G 空闲**。

---

## 1. 实验是什么

跨港口 Sentinel-1 SAR 船只分类的**场景结构化知识迁移**（KIRC）：841 个产品 / 24 港，
$\Delta = \log r^{VK} - \log r^{V}$，闭合分解 $1 = \sum_g \beta_g R_{ig} + \rho_i$；
消融 B0 → Full；全部全量实验在服务器 westd（RTX 4080 SUPER 32G）上跑。
设计文档目录：`E:/Docms/无监督域适应/translated/0917/`（主文档 `实验设计.md`、`0918_实验设计.md`、`0918_实验模块设计2.md`）。

## 2. 三会话分工（各自只碰自己的目录）

| 会话 | 负责 | 关键产物（实读） |
|---|---|---|
| **19c027** | 本机检测+标注（`E:/safe841_local_20260918`，658 影像）、MMSI 类别层、**Phase 0 数据管线**、服务器标注/实例运维 | `mmsi_class_final.csv` 126,606 MMSI（fine 22,615 / coarse 45,409 / non_ship 18,043 / untyped 146 / unknown 40,393）；本机 labels_scene.json 汇总；**`knowledge/phase0/`**（splits / masks_index / geo_index / support_by_port_class / README / STATUS） |
| **372993**（本会话） | 栅格恢复与 841 覆盖对账、服务器并机（westd 为主）、**夸克网盘归档**、标注执行计划 | 服务器大图 **1,921 张**；`标注执行计划_20260920.md`；**841 标注收口（841/841，19:45 实读）**；**夸克上传通道打通 + 归档进行中**（`quark_upload.py`/`quark_archive.py`；协议沉淀为 skill `quark-netdisk-upload`） |
| **d0b3b3** | **知识集**：对象表 $k_i$、设施几何+逐要素定年、$d_p$、AIS 经验航道方向场、派生岸线；GitHub 实时同步；**服务器输入就绪度审计**（20:40） | `objects_final.csv.gz`（2,897,382 对象 × 64 列）、`port_knowledge.csv`（24×67）、`traffic/*.csv`（24 港）、`osm_way_dates.csv`、`facilities/*.geojson`（97 MB）；审结：大图 841/841 ✓、标注 836/841（5 个待重标）、环境依赖齐全、实验代码未上服务器 |

## 3. 共享资产分布（截至 19:45 实读）

| 位置 | 内容 |
|---|---|
| **服务器 westd** `/root/autodl-tmp/`（1.1T，剩 110G） | `knowledge_841_20260920/` 303 MB（知识集）· `safe841_batch/` 标注台（**completed 841**，含 442 原有 + 218 本机传输 + 181 服务器新跑 + 2 补跑）· 大图 1,921 张 · 活动任务：`quark_archive.py`（12 uploaders / 4 converters，跳过 PNG 阶段） |
| **本机 E:/临时会话/knowledge_set_841/** | 知识集全量（对象表 195 MB、设施几何、方向场、定年、类别层、`NEXT_STEP_BRIEF.md`） |
| **本机 E:/safe841_local_20260918/** | 本机检测输出（658 影像 / 109,118 块），19c027 负责 |
| **服务器 westd** `/root/autodl-tmp/phase0/`（755 文件 / 418 MB，20:20 实读回读核对） | Phase 0 全量：splits · geo（24 港 GeoJSON，217 MB）· masks（658 景 PNG，56 MB）· mmsi · ports · facilities · support_by_port_class.csv |
| **本机 E:/临时会话/knowledge_set_841/{splits,geo,masks}** | Phase 0 本地产物（geo 217 MB / masks 56 MB / splits 136 KB），与服务器同源 |
| **GitHub** `ouba188/Reliable-Scene-Knowledge-Transfer-…` | 文本/脚本/汇总（commit `6c721bf`）：`knowledge/`、`scripts_v2/`；**大文件不入库**（对象表/几何只在服务器+本机） |
| **夸克网盘** | SAR_8bit 1.58T + 0915增量 271.7G 归档；**841 数据集归档目录 `SAR_AIS_T_841_20260920/{tif,png,labels}`**（服务器侧上传中） |
| **服务器脚本** | `/root/quark_upload.py`（夸克 web 上传协议实现，cookie 在 `/root/quark_cookie.txt` 0600）· `/root/quark_archive.py`（841 归档流水线，state 断点续传）· `/root/run_quark_archive.sh`（tmux 内守护） |

## 4. 接口与硬约定

- 服务器目录根：`/root/autodl-tmp`；d0b3b3 只写 `knowledge_841_20260920/`，**不碰** `safe841_batch*`（19c027/372993 的域）。
- 对象表列语义与切片建议见 `NEXT_STEP_BRIEF.md`（同样在仓库 `knowledge/` 下）。
- **禁止**：目标港单船 AIS 类型进入 $k_i$；AIS 只用于源侧类别支撑与场景统计 $\xi_t$。
- **删除/迁移前**：先查 0917 设计文档 + 列出清单等用户点头（历史上有过「清理多了」的教训）。
- autodl-fs 共享盘按量计费（免费 20G），数据只放数据盘。
- 凭据不写进任何文件（脚本从环境变量取；夸克 cookie 单独 0600 存放）。
- **长任务必须 tmux 托管**：SSH 会话断开会杀子进程组（nohup/setsid 拦不住），表现为日志停在启动行、进程静默消失；`pkill -f` 的模式别匹配到自己的 shell。

## 5. 当前里程碑

| 环节 | 状态 |
|---|---|
| 841 产品大图 | 服务器 1,921 张；覆盖对账由 372993 维护（841/841 有图 ✓） |
| **标注** | ✅ **841/841 实打实**（20:42，d0b3b3 逐产品复核 labs_json 存在性）：status 文件 841（`annotation_output` 738 + `annotation_output_s3` 103）。**20:36–20:42 已修复**此前发现的 5 个缺口（`…19A8` Newcastle、`…85D9` Mombasa、`…1A19` Port Said、`…570A` Qingdao、`…B1C2` Singapore）——它们的产物原在共享盘（已清空），本次用 `annotation_bundle_s3` 的资产在服务器重跑，5/5 全部 rc=0 且 VV+VH 的 labels_json 各 2 个；队列与日志在 `knowledge_841_20260920/recover5/` |
| **服务器实验环境** | ✅（20:20 实读 + 补装）miniconda py3.12 · torch 2.12.1+cu130（CUDA ✓ RTX 4080 SUPER）· ultralytics/rasterio/shapely/pandas/scipy · **sklearn 1.9.1 + geopandas 1.1.4**（20:15 由 d0b3b3 补装） |
| MMSI 类别层 | ✅ 126,606 MMSI（19c027） |
| 知识集（$k_i$/设施/定年/$d_p$/方向场） | ✅ 已收口并上传服务器 + 入库 GitHub |
| **夸克归档** | 🔄 进行中（19:51 实读：**660/3067 文件 / 98.4 GB / 17.9 MB/s / 0 失败**）；已定：**先传原格式 tif + 标注 JSON**，PNG 后补；ETA ≈ 明早 7–8 点（账号带宽上限 ~25MB/s 已实测）。工具链与协议见  |
| 0917 机制实验（B0→Full） | **未开始**——输入层已就绪（Phase 0 产物 + 知识集 + 841 标注均已完成） |
| **Phase 0 数据管线**（19c027） | ✅ **20:30 收口**：大景框标注 658 景 / 448,080 框 · 地理回写 24 港 GeoJSON · offshore mask 658 景（离岸 45.4% / 近岸 1.4% / 陆 27.1%）· 采集级 split 24 折 LOPO（全为 partial-set）· 每港每类支撑 79,059 个带类对象；**仅 native-res COG 瓦片未做** |
| **磁盘 / 清理**（d0b3b3） | ✅ 20:52 执行（用户批准）：复核 jpg **3,845,486 个 / 106.5 GB**（删后剩 0）· 重复 `safe841_batch_s3/annotation_bundle_s3` 4.27 GB · 4 个历史标注包 tgz 1.92 GB · 4 个多余 run 目录 · 同名大图副本按 **md5 相同才删**（保留序：annotation_output > parallel_* > fdrive_import > asf_download_v2_* > sar_download_v1 > safe841_batch_s3；内容不同不删）。**`/root/autodl-tmp` 现 830G 用 / 221G 空**；删除清单 `/root/cleanup_manifest.json`、`cleanup_jpgs.txt`，日志 `E:/Hermes/scripts/out/cleanup_execute.log` |

## 6. 下一步（待用户与 ChatGPT 讨论后定）

0. **Phase 0 已闭合 → Phase 1 可直接开跑**：划分、掩膜、地理回写、类别支撑都在 `knowledge/phase0/`（服务器 `phase0/` 同源）；source-only class-aware / class-agnostic 基线直接用 `splits/port_acquisition_split.csv` 的 adapt/eval。注意 **target_private 天然为 0**，I3 需 episodic 模拟造目标私有类。
1. 用对象表跑 0917 设计的四组小规模实验（visual-only / metadata / 旧 scene graph / 条件关系矩筛选）。
2. 确定 $\Delta$ 的可操作定义与 K 的三档消融映射（见 `NEXT_STEP_BRIEF.md` 第 7 节）。
3. 若实验需要扩展 $k_i$（到堆场/铁路/储罐/船厂距离、对象↔设施二部图），服务器已有 97 MB 原始几何可直接算。
4. **PNG 后补**（用户已定）：归档跑完后跑 `quark_archive.py` 的 P2 阶段（`--skip-png` 去掉即可，state 自动跳过已完成的）；若只要预览可改为 1/4 缩略图（全尺寸 PNG 实测仅省 15% 体积、且丢地理参考）。

## 7. 本文件更新规则

- 每次收口一轮：更新「最后更新」行 + 自己那一行 + 第 5 节对应里程碑。
- 同步动作：本机 `E:/Docms/无监督域适应/translated/0917/EXPERIMENT_STATE.md` ← 权威；副本推仓库 `knowledge/EXPERIMENT_STATE.md` + `E:/临时会话/knowledge_set_841/`。
- 三个会话的系统提示里都记了本路径（memory），会话启动时先读本文件。

## E01 第一轮（rev1）执行结果 — 2026-09-21 03:45（by d0b3b3）

- **E01 第一批执行包已按网页 GPT 交付执行完毕**：四折 Rotterdam / Shanghai / Port Klang / Fujairah
  全部跑通（冻结实现 sha256：chips=b4c5964178f2 features=ff926d1685d3 glue=e72fa557f276 core=cd207624d7f1）。
- 服务器产物：`/root/autodl-tmp/e01_chips_rev1/`（329 分片，408,981 实例，valid_joint 全 1.000）、
  `/root/autodl-tmp/e01_runs/<折>_rev1/`（features/groups/heads/relations/校准/筛选/指标）。
- 主要结论：四折 retained 全 8/8 ⇒ Full≡B1；可评价标签支撑 7–430 是主瓶颈（非图像问题：0 个对象因 I/O 被排除）。
- 仓库：`experiments/aisfree_scene_v02/e01_run_20260920/`（总汇总见 RESULT_ALL_FOLDS_rev1.md）；
  rev0 因 crop 行列写反（ds.index 返回 row,col）保留为 pilot，不作方法对照。
- 待办：P1 关系误差分解 / P3 归因（Static/Group/Full）；标签支撑方案待网页 GPT 裁定。

## E01 轮次收口（2026-09-21 11:20，commit 7b64f77）

**本轮范围**：补标枚举修正 + 粗类概率聚合修正 + 候选库可改进上界诊断（只读），不重训、不扩港、不放宽词表。

### 两项实现修正
- 补标 v2：用 `query_ball_point(r=500)` 取完整候选集（替换 `query(k=8)`），T1 150m/180s、T2 500m/300s 分档独立证据；114 产品 T1 19,361 / T2-only 4,964；(product,MMSI) 复用 11,022 对（待核）。旧层保留未覆盖。
- 粗类评估改为**先按父类概率求和再 argmax**。

### 上界诊断（ORACLE_ONLY, deployable=false；六折 × 四视图）
- 最佳固定子集 ≈ 最佳单头（差中位 **+0.000**，20/24 视图最优子集=单头）→ 整折统一子集族**已饱和**
- B1（8 头等权）**弱于**最佳单头（差中位 **+0.036**，六折全非负）
- 逐对象上限 − B1 中位 **+0.071**（最大 Port Said coarse_t1 +0.184）⇒ 空间在**决策粒度**
- B1 错误合计 **3,208**，仅 **402（12.5%）** 可被某单头答对；**87.5% 任何单头皆错** ⇒ 输出空间内无正确答案

### 结论修正（措辞，记录在案）
1. Jebel Ali **未达**「每类≥50」（fishing 4–6）；该门槛不重新启用
2. B0 vs B1 差（+3.6 点）**不作**「噪声边缘」结论，且非知识增量（B0=跨港 BA 选出的单头，B1=8 头等权）
3. 零增量已由全/保留子集概率相等（`p_Full = p_B1`）确定，与新标签量无关；新标签价值在检查其它候选与错误类型

### 产出
- `experiments/aisfree_scene_v02/e01_run_20260920/aux_candidate_ceiling_v1/CANDIDATE_CEILING_v1.md`（+24 报告 + 255 子集明细 + 参考包）
- `.../paper_materials/PAPER_MATERIALS_v1.md`、`.../paper_materials/ARCHIVE_INDEX.md`

### 下一步（待裁定）
(a) 提升候选头覆盖与可分性（源端类别覆盖/视觉表示诊断）；(b) 决策粒度改为逐对象/条件选择（源端原型）。
**已停止的方向**：继续调 tau、继续换保留子集、扩港找涨分。

### 磁盘
服务器 930G 中 858G 已用 / **73G 可用（93%）**，夸克归档仍在跑，需盯余量；可删候选见 `ARCHIVE_INDEX.md` 第四节（等确认）。

## E01 (a)+(b) 执行结果（2026-09-21 12:40，源端训练/目标端评估，均 CPU）

**共用台子**：源端开发集（每折 2,971–5,822 条源端真标签，冻结头前向；`e01_source_dev.py`），
源端已复现目标端的同一结构：固定子集≈最佳单头、B1 弱于最佳单头、单头并集上限更高。

### (b) 决策粒度策略（`aux_policy_b/README.md`）
- B0 ≈ B1 ≈ FIX（差 ≤.03，噪声内）→ 固定子集族无空间
- 源端训练的逐对象堆叠 STK：**仅 Shanghai 稳定正收益**（coarse +.035 acc/+.013 BA），其余持平或正负相抵
- 逐对象 oracle 在所有视图高于任何可用策略（+.047~+.154）→ 空间存在，但**朴素源端选择器拿不到**
- 可停止：调 tau / 换子集 / 换聚合

### (a) 域相关可分性（`aux_domain_a/README.md`）
- 源港留一线性探针：**平均域间隙 +.388 ~ +.575**（cross .23–.57 vs in .70–.98）→ 表示强港口特异
- **CORAL 二阶对齐无帮助**（6 折 5 折更差，−.11~−.28）→ 不是二阶偏移
- 域间隙比决策粒度缺口大一个量级 → **优先级给表示/适配层**

### 本轮结论（面向论文）
知识筛选零增量（p_Full=p_B1）之后，真正限制来自：①固定子集族已饱和 ②朴素逐对象策略拿不到 oracle 空间
③表示跨港损失 40–60 点且非二阶可修。**可停止的方向**：tau、子集、聚合、扩港、放宽词表。

### 磁盘
服务器 848G/930G 已用（83G 可用）；rev0 作废项已删（清单 `/root/rev0_delete_manifest.txt`）。

## E02 对齐修正与源伪目标基线（2026-09-21 14:50，评审裁定后执行）

**边界**：CPU only；不重抽 chip、不重训 encoder/H/J、不动 E01 主表。

### 勘误（已写入对应 README 顶部）
1. **旧 CORAL 实现错误**：写成 `Cs^{-1/2} Ct^{-1/2}`（两个逆平方根、无重着色，连恒等性都不满足）
   → `aux_domain_a` 的 coral 列作废；正确 T2S 为 `(X-μt) Ct^{-1/2} Cs^{1/2} + μs`
   （恒等误差 ~1e-14，协方差残差 ~1e-15，而旧实现残差 3–125）
2. **旧「域间隙 +0.39~+0.58」混入未隔离因素**（全池 PCA、内层留出港可能在该 encoder 的 source-fit、随机 CV、只报 Acc）
   → 降级为「港内监督探针优于跨港探针，提示跨港泛化值得优先研究」，不再作算法依据
3. **(b) 中「OBJ 高于任何可用策略」不成立**（Fujairah coarse STK .201 > OBJ .153）：STK 是 `softmax(W[p1..p8]+b)` 的新分类器，
   而 OBJ 只在 255 个等权子集内取最优，**不是 STK 的上界**；STK 亦无港口知识输入、未做交叉拟合

### 新基线（源伪目标协议；PCA-64 与分类器只在 source_fit 拟合；U_q 无标签；E_q 标签只进评价器）
5 个可判定伪目标（n=865–1588），Acc/BA：

| 折 | 伪目标 q | 角色 | n | Cross | Mean-only | CORAL-T2S |
|---|---|---|---:|---|---|---|
| Rotterdam | Qingdao | cal* | 1588 | .240/.318 | **.315/.385** | **.311/.397** |
| Shanghai | Qingdao | cal* | 1451 | .391/.560 | .360/.578 | **.407/.568** |
| Fujairah | Shanghai | meta | 1094 | **.419/.344** | .397/.285 | .398/.281 |
| Port Said | Shanghai | meta | 1094 | .404/.339 | **.424/.315** | .396/.288 |
| Jebel Ali | Shanghai | meta | 865 | **.572/.494** | .535/.447 | .420/.371 |

结论：均值平移与正确二阶对齐**都能改善，但配对相关且可严重为负**（Jebel Ali→Shanghai 的 T2S 掉 15.2 Acc 点）
→ 三列基线成立（Cross / Mean-only / CORAL-T2S）；`in_ref`（用 U_q 标签监督，仅参考）达 .71–.99 Acc，
说明标签驱动的空间极大。`*` Qingdao 属 cal 港，**仅作伪目标、未并入 D_fit 或任何选择过程**（显式披露）。

### 下一版唯一干预（见 `DECISION_NEXT.md`）
**配对感知的观测条件校正**：只用无标签统计，在 cross 与 mean-only 间逐对选择，选择规则在**源港留一**上验证；
要打败 Cross/Mean-only/CORAL-T2S，并与逐对 oracle 对照。**不**同时改表示/知识/词表/筛选。

### 交付件
`e02_alignment/`：`E02_ALIGNMENT_v1.md`、`DECISION_NEXT.md`、`coral_regression_*.csv` 与 `_meta.json`、
`role_lineage.csv`、`source_pseudo_target_splits_*.json`、`cross_mean_coral_corrected.csv`、两个脚本 + 参考包（含 9 项单测）。

## E02 唯一干预结果：配对感知校正 —— 预登记证伪，判失败（2026-09-21 16:10）

**干预**：只用无标签统计（`s = ||μ(D_fit) − μ(U_q)|| / scale`，在 source-fit 上拟合的 PCA-64 坐标内），
在 `cross` 与 `mean_only` 间逐对选择；阈值由**源港留一**搜索；目标标签只在评价器内读取。

### 源端留一（规则拟合处）
| 折 | 阈值 | src Cross | src Mean | src RULE |
|---|---:|---|---|---|
| Rotterdam | 4.431 | .618 | .457 | **.618** |
| Shanghai | 0.961 | .845 | .844 | **.852** |
| Port Klang | 2.085 | .571 | .529 | **.572** |
| Fujairah | 4.311 | .661 | .549 | .652 |
| Jebel Ali | 7.723 | .794 | .620 | .667 |
| Port Said | 5.276 | .623 | .518 | **.623** |

### 迁移到 5 个可判定伪目标（Acc）
| 折 | q | n | s | 选择 | Cross | Mean | T2S | RULE | oracle |
|---|---|---:|---:|---|---|---|---|---|---|
| Rotterdam | Qingdao | 1588 | 5.27 | mean | .240 | .315 | .311 | **.315（+7.5）** | .315 |
| Shanghai | Qingdao | 1451 | 2.94 | mean | .391 | .360 | .407 | .360（−3.1） | .407 |
| Fujairah | Shanghai | 1094 | 3.89 | cross | **.419** | .397 | .398 | **.419（0.0）** | .419 |
| Port Said | Shanghai | 1094 | 4.65 | cross | .404 | .424 | .396 | .404（−2.0） | .424 |
| Jebel Ali | Shanghai | 865 | 7.88 | mean | **.572** | .535 | .420 | .535（−3.7） | .572 |

**RULE − cross 平均 −0.26 点；RULE − mean_only −0.10 点；2 胜 3 负** ⇒ 未超过 cross/mean_only 的较好者
⇒ 按预登记标准**判该干预失败**，不进入下一版。

**附带观察**：信号大小与校正收益无单调关系（s=7.88/2.94 均负、s=3.89 时 cross 已最优，唯一正收益在 s=5.27）
⇒ 一阶均值偏移量**不是**可迁移的判据；候选信号应改为**按类/按结构**的量。

### 下一版范围（按失败分支）
回到**表示层**，此时才考虑**编码器级留一**（为某 fit 港另建排除该港的 encoder）；候选信号改为按类或结构化的偏移量。
本轮不动 encoder、词表、筛选、E01 主表。

### 交付件
`e02_alignment/PAIR_RULE_v1.md` + `pair_rule_<fold>.json`（阈值、源端留出明细、逐伪目标信号/选择/四列指标/逐对 oracle）+ `e02_pair_rule.py`。

### E03 配对感知校正：诊断完成，**不进入下一版**（2026-09-21 15:25，commit 0da47a4）

- 协议：六折源端口两两有序配对 = 52 对；统计量只用无标签特征，标签仅事后验证；阈值 leave-one-source-fold-out
- 诊断：mean-only 为正仅 **8/52（15%）**，delta_acc 均值 **−0.094**（范围 **−0.621 ~ +0.235**）；
  熵/置信度类统计量对符号的定向 AUC **0.92**，位移范数/能量距离仅 0.65–0.66
- 源折留一：最好规则 **0.319 vs 总是 cross 0.315（+0.004，噪声）**；**配对 oracle 自身仅 0.328（+0.013）**
  ⇒ 符号可预测但幅度可忽略 → **预测符号换不来准确率**
- 5 个可判定伪目标的同口径配对 oracle：+7.5 / +2.0 / +1.6 / 0 / 0，**平均 +2.2 点**
- **裁定**：不把配对感知校正开发成模块；三条基线（Cross / Mean-only / CORAL-T2S）保留为参照，
  后续任何适配模块必须超过它们各自的较好者**且超过 +2.2 点**才有意义
- 依评审裁定表的剩余分支：下一版做**受源监督约束的小型表示适配模块**（同一 5 个伪目标 + 源港留一评估；
  不同时改知识/路由/词表）；已请 GPT 确认放弃决定与模块输入边界
- 文件：`e02_alignment/E03_PAIR_RULE.md`、`pair_diagnosis.csv`（52 对）、`rule_loo_eval.json`、两个脚本

### E05 源监督线性表示适配模块：未达标（2026-09-21 17:35，commit 99dd4b1）

- 模块：`L = CE(W z_fit, y_fit) + λ‖Cov(W z_fit)−Cov(z_adapt)‖²_F + μ‖mean(W z_fit)−mean(z_adapt)‖²`，W 初始化为 I（λ=μ=0 正好退化回 Cross）；分类器按冻结设置重训；**超参只在源端留出港 20 个 episode 上选**
- 源端选择 λ=10/μ=0：源端平均 BA **0.563**（λ=μ=0 为 0.546，**+1.7**）
- 5 个伪目标：模块 vs Cross BA = **+0.0003**（无增益）；vs 每对最好基线 = **−0.016**；对角版 −0.018
  （Rotterdam→Qingdao −.063、Port Said→Shanghai −.010 明显更差）
- **判读：源端增益不迁移，第三次独立复现**（E03 信号幅度可忽略 / E04 规则坍缩为恒选 Cross、二选一 oracle 仅 +0.002 BA / E05 模块增益 0）
  ⇒ **在冻结表示上做特征空间（一阶/二阶/源监督线性）适配，收益不跨港迁移**
- 下一步待评审裁定两条分支：**(A) 编码器级**（重建排除留出港的 encoder，GPU，把港级留出做实到表示层）；
  **(B) 源监督与类别覆盖**（Cross BA 仅 .28–.56 vs in_ref BA .35–.74 / Acc .71–.99）。已发 GPT 请求指定与协议边界
- 文件：`e02_alignment/E05_LINEAR_MODULE.md`、`linear_adapt_results.csv`、`linear_adapt_summary.json`、`e05_linear_adapt.py`
