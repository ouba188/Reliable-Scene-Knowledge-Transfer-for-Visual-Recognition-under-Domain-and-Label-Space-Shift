# HANDOFF — 在线状态（供飞书端会话接手）

**更新时刻**：2026-09-25 深夜；本项目当前进度以 `EXPERIMENT_STATE.md`（E12–E19b）为唯一真源。

## 1. 三条文档（论文主体已成稿）
- `paper/main.tex` —— **IEEEtran 期刊版（编译通过，9 页，10 表 + 3 图）**；`paper/refs.bib` 26 条真实文献（逐条对 Crossref/NeurIPS/PMLR/JMLR/MLSys/arXiv 页核过，无编造）；`paper/main.pdf`；图由 `paper/figs/make_figs.py` 从记录数字生成。编译：`tectonic -X compile main.tex`。
- `PAPER_SKELETON.md` —— 骨架 + 主张→产物对照（含 E19 更正）
- `PAPER_DRAFT.md` —— Abstract + §1 Introduction + §2 Method
- `PAPER_METHOD_DETAIL.md` —— Method 详细版（数据/协议/臂与模块/诊断/实现/复现清单）
- `PAPER_RESULTS.md` —— §3–§9 Results（7 张表，全部挂实测）
- `PAPER_DISCUSSION.md` —— Discussion + Conclusion + 待引文献脉络（**未编造引用**）

## 2. 结论一句话
> 跨港迁移**保留"序"、破坏"值"** ⇒ 绝对阈值/保证类闸门结构性死路 ✗；可部署形态是**预算式秩分配** ✓（合规版 20% 预算 → 106.8%，p=0.0194）；AIS-free 的代价已**定价** ✓（+4.23pp vs +0.37pp）；跨港掉分**全部是类先验**（−12.79pp，条件反而 +6.54pp）。

## 3. 正在跑 / 待跑（"深"的三条路，用户已同意"都做"）
| 项 | 内容 | 状态 |
|---|---|---|
| **C** | 预测相位图 `e139_phase_diagram.py`：无标签量预测增益符号（判据 ≥70% 且 p<0.05） | 运行中 |
| **A** | 理论 + 可检命题 `e140_theory_check.py`：P1 单调变换下 top-k 不变 ✓ / P2 标签偏移下阈值脆弱 ✓ / M 实测 R_gate vs AUC / B 破坏条件分布 | 待启动（脚本已写好 ✓） |
| **B** | 稳定先验校正复开 BBSE（判据：收复 ≥50% 的 −12.79pp 分量，24 港 p<0.05） | 未开始 |

## 4. 环境与纪律（接手必读）
- 运行：`cd "E:/临时会话/visual_reliable_baseline"`，`KMP_DUPLICATE_LIB_OK=TRUE /d/Program_files/anaconda3/python.exe -X utf8 -u <script> > <script>_out.txt`
- 内存：已腾出 ~4 GB（可用 3–4 GB ✓）；**重活一次只跑一个** ✓
- **协议红线**：知识维**只能用 0:71**（71:82 是 AIS 派生 ⇒ 违反 P0 AIS-free ✗）；目标港 AIS 绝不进输入
- **预注册纪律**：判据先写、结果照报、构造不稳的数**撤回不打折**
- 判据未过也要入档 + 推仓（`git` 直连时加 `-c http.proxy= -c https.proxy=` 并 `env -u *_PROXY`）
