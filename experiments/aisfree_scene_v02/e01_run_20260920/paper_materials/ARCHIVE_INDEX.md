# 归档索引 v1（E01 轮次完成态，2026-09-21）

口径：**文字/表格/脚本/小体积 JSON 进 GitHub；原始大件（芯片、栅格、对象表、AIS）留服务器与 E 盘，不入库。**

## 一、GitHub（私有仓 `ouba188/Reliable-Scene-Knowledge-Transfer-...`）

当前 HEAD：`751235a`（文本/脚本/报告已同步；`.git` 后缀不要出现在分享链接里）

| 路径 | 内容 |
|---|---|
| `knowledge/EXPERIMENT_STATE.md` | 三会话单一事实源镜像 |
| `knowledge/NEXT_STEP_BRIEF.md` | 下一步交接摘要 |
| `experiments/aisfree_scene_v02/e01_run_20260920/STATUS.md` | 本轮状态 |
| …`/RESULT_ALL_FOLDS_rev1.md` | rev1 六折主结果 + 修复清单 |
| …`/RESULT_{Rotterdam,Shanghai,Fujairah,PortKlang,JebelAli,PortSaid}_rev1*.md` | 逐折结果 |
| …`/REV1_REMEDIATION.md` | 修复与回归说明 |
| …`/aux_coarse_eval_v1/README.md` | 粗类辅助评估（单列） |
| …`/aux_label_boost/README.md` | 辅助标签层（boost v1 记录） |
| …`/aux_candidate_ceiling_v1/CANDIDATE_CEILING_v1.md` | **上界诊断主文档**（ORACLE_ONLY） |
| …`/aux_candidate_ceiling_v1/reports/` | 24 份 `ORACLE_ONLY_report.json` + 255 子集明细 CSV（48 文件 / 1.0 MB） |
| …`/aux_candidate_ceiling_v1/reference_pkg/` | 参考工具 `oracle_audit.py`、`match_tiers_reference.py`、README |
| …`/paper_materials/PAPER_MATERIALS_v1.md` | **论文材料包**（方法/结果/限制/复现/图表位） |
| …`/e01_*.py`、`e01_*.sh` | 全链脚本（chips/features/glue/label_boost_v2/prep_ceiling 等） |
| `knowledge/`(镜像) | 知识集小件：`port_knowledge.csv`、`layer_coverage.csv`、Phase-0 覆盖率 CSV |
| `e01_first_batch/` | 协议包（config/split_manifest/manifests/tools ×11/docs/SHA256SUMS） |

## 二、服务器（`root@connect.westd.seetacloud.com:16575`，`/root/autodl-tmp`）

| 路径 | 体积 | 内容 | 备注 |
|---|---:|---|---|
| `e01_chips_rev1/` | 9.3 GB | 329 分片 / 408,981 芯片（修正版） | **唯一有效芯片集** |
| `e01_chips/` | — | rev0 芯片（行列转置 bug 版） | **已删除 2026-09-21**（清单 `/root/rev0_delete_manifest.txt`，385 文件） |
| `e01_runs/Rotterdam`、`e01_runs/Shanghai`（rev0 pilot） | — | rev0 pilot 运行目录 | **已删除 2026-09-21**（结论以 `RESULT_Rotterdam.md` 文字留档） |
| `e01_runs/aux_candidate_ceiling_v1/` | 1.7 MB | 上界诊断输入 NPZ + 报告 + rescue 计数 | 已入库（报告） |
| `e01_runs/{aux_p1,aux_p3,aux_coarse_eval_v1}` | <100 KB | 前几轮辅助分析 | 已入库 |
| `e01_label_boost_v2/` | 2.6 MB | 114 产品辅助标签（分档证据） | 留服务器 |
| `knowledge_841_20260920/` | 556 MB | e01_inputs + objects + mmsi + ports + facilities + 协议包 | 知识集主目录 |
| `phase0/` | 420 MB | 可达性/距离场 | 留服务器 |
| `safe841_batch_s3/` | 79 GB | 原始产品 + AIS 语料（`annotation_bundle_s3`） | **AIS 语料源**，勿动 |
| 其他 | — | `asf_download_v2_*`、`fdrive_import_*`、`sar_download_v1`、`redl_*` | 原始下载/重下工作区 |

**磁盘**：930 GB 中 **858 GB 已用 / 73 GB 可用（93%）**；夸克归档任务仍在运行 → 空间余量需盯。

## 三、本机（Windows）

| 路径 | 内容 |
|---|---|
| `E:/Docms/无监督域适应/translated/0917/` | 协议包、网页端 GPT 交付包（4 个 zip + 解压）、`EXPERIMENT_STATE.md`、`0918_实验设计.md` |
| `E:/临时会话/knowledge_set_841/` | 去重对象表 v2（125.5 MB）、`e01_relations/relations.npz`、知识集小件 |
| `E:/临时会话/repo_rsk/` | GitHub 克隆（工作树） |
| `E:/临时会话/ceiling_v1/` | 上界诊断本地快照（24 报告 + 汇总） |
| `E:/Hermes/scripts/` | 全部执行脚本 + `chatgpt_web.py`/`chatgpt_download2.py`（网页端 GPT 驱动） |
| `E:/safe841_local_20260918/` | 另一会话的本地标注批次，**勿动** |

## 四、可删候选（rev0 两项已删除，其余等确认）

| 项 | 体积 | 理由 | 风险 |
|---|---:|---|---|
| `asf_download_v2_tmp/` | 待测 | 下载临时区 | 需先核对无未合并产物 |
| `redl_work/`、`redl_zips/` | 待测 | 重下工作区 | 需先核对 |

## 五、凭据与安全

- SSH 密码、GitHub token、代理 key **只经环境变量传递**，不入库、不写入任何文件
- 推送前做密钥扫描；分享仓库链接不带 `.git`
