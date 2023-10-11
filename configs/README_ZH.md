# 配置文件

此文件夹包含一些配置文件，帮助用户轻松理解各种功能的配置方法，并快速复现开源数据集的处理流程。

## 用法

```shell
＃处理数据集
python tools/process_data.py --config xxx.yaml

＃分析数据集
python tools/analyze_data.py --config xxx.yaml
```

## 分类

配置文件分为以下几类。

### Demo

Demo 配置文件用于帮助用户快速熟悉 Data-Juicer 的基本功能，请参阅 [demo](demo) 文件夹以获取详细说明。


### 复现的Redpajama

我们已经复现了部分 Redpajama 数据集的处理流程，请参阅 [reproduced_redpajama](reproduced_redpajama) 文件夹以获取详细说明。

### 复现的BLOOM

我们已经重现了部分 BLOOM 数据集的处理流程，请参阅 [reproduced_bloom](reproduced_bloom) 文件夹以获取详细说明。

### Data-Juicer 菜谱
我们使用 Data-Juicer 更细致地处理了一些开源数据集（包含 CFT 数据集），并提供了处理流程的配置文件。请参阅 [data_juicer_recipes](data_juicer_recipes) 文件夹以获取详细说明。