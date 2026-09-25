# HANDOFF — 当前状态（供任何新会话接手）

**更新**：2026-09-25 深夜；唯一真源 = `EXPERIMENT_STATE.md`（E12–**E21c**）；镜像仓库已推至 `489d501`。

## 1. 一句话结论
> **跨港迁移保留「序」、破坏「值」** ⇒ 绝对阈值/保证类闸门结构性死路 ✗；可部署形态 = **序 + 预算** ✓。
> 由此合成**一个两级、可部署、每层带控制臂的方法** ✓✓。

## 2. 两级方法（今晚最终成品 ✓）
| 层 | 方法 | 证据 |
|---|---|---|
| **块级** | 源港**逐对可分性** AUC < τ ⇒ 合并类对 ⇒ 连通分量给 **k（k 是输出）** | 同尺寸随机划分配对 **p=0.0047** ✓✓ |
| **细类级** | **margin 排序 + 预算 k**（替代不可迁移的覆盖规则） | 旧池 10% 预算 **+13.05pp** ✓✓；**新池 +18.24pp，= 全量的 194%** ✓✓；p<0.0001 |
| **已知/未知** | **Mahalanobis 新颖度**（不是 margin ✗） | 0.555 vs margin 0.437，n=121，**p<0.0001** ✓✓（e117 数在新池独立复现 ✓） |
| 联合门控 | z(margin)+z(−novelty)：点估计支配两单分数 ✓⚠ | 相对 margin **未达显著**（p=0.0641）✗ |

## 3. 必须遵守的边界（都已实测，写进论文 §8 ✓）
1. **门控+专家**的价值只能表述为「**回收专家增益的预算效率**」✓；**不能写「选出的那批更准」** ✗（实测劣于随机 ✓ e175）；
2. **细类半边**在 oracle 覆盖规则下有效、可部署规则下失效 ✗ ⇒ 用预算替代规则后成立 ✓（e173/e174）；
3. **"一个分数两用"已否** ✗（margin 逐实例 AUC<0.5 ✓）⇒ margin 管细粒度、novelty 管已知/未知 ✓；
4. 知识维**只能用 0:71** ✗（71:82 是 AIS 派生 ⇒ 违反 P0 AIS-free ✓）；
5. 阈值/校准/保形/加权/代理风险 **全部实测失败** ✗；cap：合规专家 ≤ +2.3pp ✓。

## 4. 环境与运维（今晚动过的）
- 内存：已腾出 ~4 GB ✓（可用 3–4 GB ✓）；**重活一次只跑一个** ✓。
- **飞书投递坑已修** ✓：`.env` 的 `NO_PROXY` 追加 `open.feishu.cn` ✓（之前 FlClash 一停，gateway 侧投递静默失败 ✓）；gateway 已重启验证 ✓。
- 静默守护：cron `2c08cff909cf`（每 20 分钟 ✓，只在「关键产物被改写 / 进程 CPU 停滞 / 新输出文件」时才出声 ✓）。
- 仓库同步：cron `5b88a12a07bc` 每 15 分钟自动推 ✓（本地 ahead 会自己上去 ✓）。
- 另一会话（飞书线）：仍在跑 OSM **锚地恢复**（`fix_missing_anchorages.py` → 子进程扫 PBF ✓），其 dossier = `AIS_FREE_LIMITS_DOSSIER.md`；其 LaTeX 工程 = `paper/`（main.tex + refs.bib + figs ✓）。

## 5. 待办（等用户点头）
① 把 **E20a–E21c** 写进论文（方法章节最强素材 ✓，并补两条诚实边界 ✓）；
② 合并两份状态文档（`EXPERIMENT_STATE.md` ↔ 对方的 `AIS_FREE_LIMITS_DOSSIER.md` ✓ —— 后者属对方会话 ✓，动前先问 ✓）。

## 6. 关键路径速查
- 脚本/输出：`E:/临时会话/visual_reliable_baseline/`（e1xx_*.py 与 *_out.txt ✓）
- 状态：`E:/Docms/无监督域适应/translated/0917/EXPERIMENT_STATE.md` ✓
- 论文：同目录 `PAPER_SKELETON / DRAFT / METHOD_DETAIL / RESULTS / DISCUSSION / MANUSCRIPT.md` ✓
- 运行：`KMP_DUPLICATE_LIB_OK=TRUE /d/Program_files/anaconda3/python.exe -X utf8 -u <script>` ✓
- git 推：`env -u *_PROXY git -c http.proxy= -c https.proxy= push` ✓（提交信息用 `-F` 落盘 ✓，别用内联引号 ✗）
