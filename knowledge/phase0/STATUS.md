# Phase 0 状态（19c027 自动同步）

更新时间：2026-09-23（自动生成，仅在数据变化时提交）

| 项 | 值 |
|---|---:|
| 大景框标注 | 430 景 / 448,080 框 |
| 地理回写 | 658 景 → `geo/<港>.geojson`（本地 + 服务器 `phase0/geo`，217 MB） |
| offshore mask | 658 景（1/8、80 m/px，4 值 + 500 m 近岸带）· 平均 陆 27.1% / 近岸 1.4% / **离岸 45.4%** / 无效 26.0% |
| 采集级 split | 24 港（adapt/eval 按采集日期前后半）· LOPO 24 折 · 标签空间设置 {'partial': 24} |
| 每港每类支撑 | 带 AIS 类别对象 79,059 个 |

关键发现：**LOPO 全折为 partial-set，target_private 天然为 0** → I3 的目标私有类必须走 episodic 模拟（与计划 Phase 4 一致）。

大文件（不入库）：本地 `E:/临时会话/knowledge_set_841/{geo,masks}`、服务器 `westd:/root/autodl-tmp/phase0/`（755 文件 / 418 MB，已回读核对）。
