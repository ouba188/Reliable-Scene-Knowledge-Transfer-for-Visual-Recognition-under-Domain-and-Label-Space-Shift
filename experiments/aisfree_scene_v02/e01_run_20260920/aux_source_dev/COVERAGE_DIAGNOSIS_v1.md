# (a) 覆盖/可分性诊断 v1（源端，逐港×逐类）

**数据**：源端开发台（`aux_source_dev`），冻结头前向。每折同一批源港，故各类 support 跨折基本一致
（fishing_vessel ≈ 2,880、container_ship ≈ 1,107、tug_towing ≈ 550–820）。

## 一、逐类召回（B1 = 8 头等权；union = 任一单头答对）

| 类别 | 出现折数 | B1 召回范围 | 单头并集范围 | 并集−B1 差 |
|---|---:|---|---|---|
| container_ship | 5 | **.33 – .72** | **.38 – .87** | +.05 ~ +.15 |
| fishing_vessel | 6 | **.38 – .89** | **.42 – .94** | +.04 ~ +.11 |
| tug_towing | 6 | .44 – .88 | .57 – .91 | +.05 ~ +.13 |
| product_chemical_tanker | 4 | .39 – .69 | .45 – .81 | +.06 ~ +.12 |
| dredger | 4 | .48 – .57 | .62 – .80 | +.14 ~ +.23 |
| passenger_ship | 1 | .58 | .69 | +.11 |

## 二、关键观察：**同一类别的召回随源港剧烈变化**，而类别清单本身没问题

| 类别 | 好折（港） | 差折（港） | 落差 |
|---|---|---|---|
| container_ship | Port Klang .72 / union .87 | **Port Said .33 / union .38**；Fujairah .34/.39 | **≈ 2.2×** |
| product_chemical_tanker | Port Klang .69 / .81 | Port Said .39 / .46；Fujairah .40 / .45 | ≈ 1.8× |
| fishing_vessel | Fujairah .82 / .91；Jebel Ali .89 / .94 | **Rotterdam .38 / .42** | **≈ 2.3×** |
| tug_towing | Shanghai .88 / .91 | Fujairah .44 / .57；Rotterdam .82/.88 | ≈ 2.0× |

**读法**：同一套头、同一类定义，在源端不同港之间就有 2 倍量级的召回差；且**连单头并集上限也一样低**
（Port Said container 只有 .38 —— 八个头里没有一个能答对多数样本）。
⇒ **瓶颈不是类别覆盖不足或头部多样性不足，而是港/域相关的表示可分性。**
而上海（目标港之一）的词表恰好只剩 `fishing_vessel` + `tug_towing` 两类——正是「源端可分性最好」的两类，
这解释了 Shanghai 折 B1 为何能到 .71，而 Rotterdam（含 container/tanker/fishing）只有 .44。

## 三、两个方向的可用空间（源端量化）

| 方向 | 量级 | 证据 |
|---|---|---|
| **(b) 决策粒度**（逐对象/条件选择） | 并集−固定子集 = **+.03 ~ +.08**（对应目标端 +.07） | 五折一致 |
| **(a) 域相关表示** | 同类跨港召回差 **≈ 2×**；差折的并集上限仍低 | container/fishing/tanker 均如此 |
| 提高类别数/放宽词表 | **没有证据支持** | 差折的问题在**已存在的类**上答不对，不是缺类 |

**结论**：不需要先重训 encoder 就能推进 (b)；(a) 的可行抓手是**域相关可分性**（而非补类别）。
若后续要动表示，目标应是「提升差港上已存在类别的可分性」，可用**源港留一**作为开发协议（不碰目标真值）。

## 四、文件

- 数据：`aux_source_dev/{fold}_source_summary.json`（含 `per_class` 与 `per_port_BA_by_head`）
- 脚本：`e01_source_dev.py`
