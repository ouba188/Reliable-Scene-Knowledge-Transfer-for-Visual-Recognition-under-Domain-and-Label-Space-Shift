# 粗类辅助评价（aux_coarse_eval_v1）

**本分析不参与训练、不参与模型选择、不改变主表任何口径。** 主表（`metrics.json`，细层）原样保留；
这里是单列的辅助视图，用于同时观察「细层表现 / 粗层表现 / 父类覆盖 / 知识实际增量」四件事。

## 口径

- 映射版本：`coarse_parent_v1`（fine → coarse parent），定义写在 `e01_coarse_eval.py::COARSE`：
  cargo（container_ship/general_cargo/bulk_carrier/ro-ro…）、tanker（product_chemical/crude/lpg_lng…）、
  passenger、fishing、tug（tug_towing/pilot_port_tender）、dredger、offshore、pleasure、other、untyped、non_ship。
- 可评价子集 `coarse_supported`：eval 实例的真实细标签映射到「该折词表的粗父类集合」之内者。
  模型预测先取细类 argmax，再投影到粗父类比较。
- 固定分母：全部 eval 实例（含无标签、父类外、未标注），覆盖数一并报告。

## 结果（rev1 四折）

| 折 | eval 实例 | 有标签 | coarse_supported | 父类外 | 无标签 | B0 acc/ba | B1 acc/ba | Full acc/ba |
|---|---:|---:|---:|---:|---:|---|---|---|
| Rotterdam | 13,200 | 314 | 284 | 30 | 12,886 | .264/.320 | .254/.302 | .254/.302 |
| Shanghai | 26,653 | 8,011 | 443 | 7,568 | 18,642 | .716/.699 | .713/.697 | .713/.697 |
| Port Klang | 3,074 | 99 | 99 | 0 | 2,975 | .121/.080 | .141/.083 | .141/.083 |
| Fujairah | 3,460 | 145 | 141 | 4 | 3,315 | .064/.232 | .071/.237 | .071/.237 |

对照细层主表可评价数：46 / 430 / 9 / 7 —— 粗类视图把可评价面扩大了 4–16 倍
（Rotterdam 46→284、Fujairah 7→141），但仍受「无标签占绝对多数」限制（如 Shanghai 有标签 8,011 中
7,568 个的父类不在该折词表内）。

## 溯源（每次运行记录）

见各折 `coarse_metrics_*.json`：`run`（运行目录）、`predictions_hash`、`vocabulary_hash`、
`features_sha256`、`mapping_version`、`coverage`、`metrics`、以及
`note: single-column auxiliary analysis; never used for training or model selection`。

## 与主表的关系

- Full ≡ B1 在粗类视图下同样成立（四折 retained 均为 8/8）。
- B0 与 B1 的差值在粗类视图下仍然远小于 n 能分辨的量级（唯一有分辨率的是 Shanghai：−.002）。
- 结论仍以机制（关系矩是否产生剪枝/增量）为主，粗类视图只用来判断「是不是评价样本太小的错觉」。
