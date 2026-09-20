# E01 第一轮执行状态（2026-09-21 00:15，会话 d0b3b3）

来源：网页端 ChatGPT 交付的 `E01_first_batch_v1.zip`（本地 `E:/Docms/无监督域适应/translated/0917/e01_first_batch/`，
SHA256SUMS 36/36 校验通过、包内 21 个单元测试本地全过、`verify_manifest.py` 通过：329 产品 / 658 极化行 /
adapt 161 · eval 168，且 `port_acquisition_split.csv` blob SHA 与包固定值一致）。

## 已按规格完成的数据层（服务器 /root/autodl-tmp）

| 件 | 状态 | 实测 |
|---|---|---|
| chips（P0 去重实例，VV+VH 同窗 128×128 native，uint8） | ✅ | 329 分片 / **408,981 实例（= E00 P0 数）**，完整性校验 0 损坏；有效比例≥0.95 的比例按产品报告 |
| `relations.npz`（包自带 `prepare_relations.py`） | ✅ | P0=408,981 原样；valid_counts = D0 408,981 / D1 144,131 / D2 109,798 / D3 408,981；`labels_read:false` |
| features/训练（Rotterdam 折） | 🔄 进行中 | 源 fit 188 产品 / meta 48 / cal 82 / 目标 adapt 5 + eval 6；词表 **5 类**（container_ship / dredger / fishing_vessel / product_chemical_tanker / tug_towing）；源 fit 有标签 **2,918** 实例；encoder 30 epoch（loss 1.64→1.28），随后冻结编码 408,981 实例 |

## 执行端决策（需 ChatGPT 确认或否决）

1. **词表只取 fine 且排除 `*_coarse`/`ship_untyped`**：E00 分类层里 `cargo_coarse`(4,286)/`tanker_coarse`(866) 的
   `class_level` 字段写作 fine，但它与 E00 `fine_class_resolved` 口径冲突，故本批按「严格 fine」处理。
2. **增广仅 180° 旋转(p=.5) + ±4px 平移（VV/VH 同步）**，无其它增广；归一化 uint8/255 后仅用 source-fit 统计
   （VV mean/std = 0.216/0.230，VH = 0.240/0.239）。
3. **按产品分块的批采样**（每 epoch 产品序打乱、产品内打乱）：npz 分片随机访问会逐样本解压，分块后每批只驻留一个分片。
   统计上仍是随机 SGD；若要求严格全局 shuffle，需要先把分片转成内存映射格式。
4. 目标端口标签只在最后的**隔离评价器**里读取；filter/校准输入 npz 均不含任何标签键（包内工具会校验）。

## 待办（链路自动跑）

fit_groups → fit_cached_heads(H=8) → fit_cached_relations(J=3) → glue(prep) → calibrate → filter → apply_retained → metrics，
Rotterdam 折跑完后按同配置顺序跑 Shanghai → Port Klang → Fujairah。
