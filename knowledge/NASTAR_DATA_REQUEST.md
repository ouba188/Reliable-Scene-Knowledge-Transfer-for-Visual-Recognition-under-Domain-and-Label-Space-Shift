# 如何申请 NASTaR 完整数据集（按成功率排序，附可直接复制的文案）

**事实（已核实 ✓）**：公开发布物是**样本集**（`ship_patches_uint8` 共 68 场景 / **126 片** / 每场景中位 2 片 ✓），源自仓库 readme 原话
"*This repository includes the NASTaR dataset **samples***" ⇒ 完整集（23 类 / 500 wake 片）**须向作者索取** ✓。

---

## 路线 ①（首选）在作者仓库开 issue

**地址**：`https://github.com/Benyaminhosseiny/NASTaR/issues/new/choose`（仓库：`Benyaminhosseiny/NASTaR` ✓ 作者 = b. Hosseiny ✓）

**判定理由**：公开 ✓、无需邮箱凭据 ✓、owner 就是作者 ✓、可被引用留痕 ✓；★ 发之前先看 Issues 列表是否已有同类请求（我没能查 —— GitHub API 限流 ✗）。

**标题**：`Request: full ship_patches split of NASTaR (released archive appears to be the samples subset)`

**正文（可直接复制）**：
```markdown
Hello, and thank you for releasing NASTaR.

We work on cross-domain SAR ship-type classification under an AIS-free protocol (leave-one-region-out,
no target labels used for any selection). NASTaR is a strong fit for us because of its 23 classes and its
inshore/offshore separation.

Downloading from data.bris (collections under `2tfa6x37oerz2lyiw6hp47058`) we obtain the `ship_patches_uint8`
set: 68 scene collections, 126 patches in total, i.e. a median of 2 patches per scene. That looks like the
"dataset samples" the readme refers to, rather than the full set.

Could you let us know how to obtain the complete `ship_patches` collection (and, if it is part of the same
release, the 500 wake patches)? We are happy to follow any access procedure, will cite the dataset and the
LGRS paper, and will use it strictly for non-commercial research as the licence requires. We are also glad
to share our benchmark results (leave-one-scene-out BA with a matched-size random control) once they exist.

Thank you,
<你的名字> — <单位 / 实验室>
```

---

## 路线 ②（并行，正式）邮件给通讯作者

- **取邮箱**：打开 arXiv 页面 `https://arxiv.org/abs/2512.18503` → 作者区点 **view email**（需浏览器过验证）；或 IEEE LGRS 论文首页的 corresponding author 行。
- **不要**猜邮箱地址 ✗（`firstname.lastname@bristol.ac.uk` 属于猜测 ✓，发错就浪费一轮 ✓）。
- **标题**：`Request for the full NASTaR ship_patches set (non-commercial research)`
- **正文**：同 issue 正文，**再加一句收件人相关**：`I am writing to the corresponding author of "NASTaR: NovaSAR Automated Ship Target Recognition Dataset" (IEEE GRSL / arXiv:2512.18503).`
- **签名务必写真实姓名与单位** ✓（学术数据索取几乎都看这个 ✓）。

---

## 路线 ③（数据仓库官方渠道）

`https://data.bris.ac.uk/data/dataset/2tfa6x37oerz2lyiw6hp47058` 页面底部通常有 **"Request access / Contact"** 入口 ✓
（Bristol 的研究数据仓库对受限/未完全公开的数据集提供 such a form ✓）⇒ 走它可在作者不回 issue 时形成第二路径 ✓。

---

## 发出去之前你要做的三件事

1. 填 `<你的名字>` / `<单位>`（**别用化名** ✗ —— 数据索取看身份 ✓）；
2. 在仓库 Issues 列表里扫一眼是否已有人问过同样的（若有，跟帖即可 ✓，别重复开 ✓）；
3. 告诉我一声，我把"已发出 / 待回复 / 已获数据"记进 `EXPERIMENT_STATE.md` ✓，并在拿到数据后**直接跑第四套验证**（按场景 LOPO ✓ 23 类 ✓ 同协议 ✓）。

**备选（不必等）**：若只想要"第四套验证"，可先用 **OpenSARShip2 的 shared5 / shared6 清单**（同一 5 折 ✓）把粒度轴补完 ✓（零成本 ✓）—— 它回答的是"更多类别下机制是否仍成立" ✓。
