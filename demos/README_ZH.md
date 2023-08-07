# 演示

此文件夹包含一些演示样例，帮助用户轻松体验 Data-Juicer 的各种功能和工具。

## 用法

使用 `demos` 子目录下的 `app.py` 来执行演示样例。

```shell
cd <subdir_of_demos>
streamlit run app.py
```

## 可用的演示

- 数据集样例 (`data`)
  - 该文件夹包含一些样例数据集。

- 数据处理回路 (`data_process_loop`)
  - 该示例用来分析和处理数据集，并给出处理前后数据集的统计信息比对。

- 词法多样性可视化 (`data_visualization_diversity`)
  - 该示例可以用来分析 SFT 数据集的动词-名词结构, 并绘制成sunburst层级环形图表。

- 算子效果可视化 (`data_visualization_op_effect`)
  - 该示例可以分析数据集的统计信息，并根据这些统计信息可以显示出每个 `Filter` 算子在不同阈值下的效果。

- 统计信息可视化 (`data_visualization_statistics`)
  - 示例可以分析数据集，并获得多达13种统计信息。

- 文本质量打分器 (`tool_quality_classifier`)
  - 该示例提供了3种文本质量打分器， 对数据集进行打分评估。

- 多语言分割数据集 (`tool_dataset_splitting_by_language`)
  - 该示例按照语言将数据集拆分为不同的子数据集。

## 即将上线的的演示
- Overview scan ｜ 初体验
- Auto evaluation helm ｜ 自动HELM评测
- Data mixture  ｜ 数据混合
- SFT data zh   ｜ 中文指令微调数据处理
- Process sci data ｜ 科学文献数据处理
- Process code data ｜ 代码数据处理
- Data process hpo  ｜ 数据混合超参自动优化

