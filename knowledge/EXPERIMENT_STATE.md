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
