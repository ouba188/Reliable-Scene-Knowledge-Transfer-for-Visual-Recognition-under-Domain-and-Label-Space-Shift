# E01 执行接口复核包（不改变 H/J/B/D）

本包依据仓库提交 `7b4a46bc452c3978ec64dec7adf3c82140442787` 的执行端脚本与原E01.1规范。

不是完整修复后的生产训练器。包含只读数组检查、可显式调用的辅助函数及合成单元测试。不会上传、删除或自动覆盖任何实验文件，不读取目标标签数组。

## 运行

```bash
cd /你的路径/e01_execution_review_v1
python -m unittest -v test_checks

python e01_execution_checks.py \
  --features /root/autodl-tmp/e01_runs/Rotterdam/features.npz \
  --relations /root/autodl-tmp/e01_runs/relations/relations.npz \
  --manifest /root/autodl-tmp/knowledge_841_20260920/e01_first_batch/split_manifest.json \
  --fold Rotterdam \
  --out /root/autodl-tmp/e01_runs/Rotterdam/EXECUTION_ID_AUDIT.json
```

ID审计只读取NPZ中的sample_id/product_id/port元数据，不加载z大数组；失败返回退出码2。第一次报告已存在时拒绝覆盖。它不能证明chip中心正确、知识可靠、标签正确或模型无泄漏。

## 辅助函数

- `strings_exact` / `strict_index`：动态宽度Unicode保存和禁止重复键。
- `calibration_products`：从产品manifest构建校准列表，不查错层级对象字典。
- `port_class_probabilities` / `balanced_epoch_draw`：源港口×类别平衡抽样。
- `classification_metrics` / `equal_port_ba`：真正的逐类recall宏平均。仅用于有权读标签的meta/evaluator。
- `augment_pair_reflect`：两极化同步reflect平移和180°旋转，显式接收epoch/draw。
- `centered_window`：正确的Rasterio row/col顺序；不代替CRS和双极化grid检查。

## 测试范围

12项本地合成测试通过，包括：U64截断复现、字符串无损往返、重复键拒绝、BA/Acc区分、校准映射、采样概率、增广一致性、Rasterio内存影像非对角目标窗口、双通道OR/AND反例。未执行真实SAR训练或服务器数据扫描。

## 外部技术核验（官方文档）

- Rasterio index returns row,col：https://rasterio.readthedocs.io/en/stable/api/rasterio.transform.html
- Window参数为col_off,row_off：https://rasterio.readthedocs.io/en/stable/api/rasterio.windows.html
- NumPy定宽字符串会截断：https://numpy.org/doc/2.2/user/basics.types.html
- np.roll为周期环绕：https://numpy.org/doc/stable/reference/generated/numpy.roll.html

详细执行决定见 `EXECUTION_DECISION.md`。不自动应用补丁：先核对服务器真正运行的代码版本，避免对已经修复的本地版本重复操作。
