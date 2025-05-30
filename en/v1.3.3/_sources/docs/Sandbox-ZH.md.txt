# 用户指南
## 应用和成果
我们利用Data-Juicer沙盒实验室套件，通过数据与模型间的系统性研发工作流，调优数据和模型，相关工作请参考[论文](http://arxiv.org/abs/2407.11784)。在本工作中，我们在[VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)文生视频排行榜取得了新的榜首。
![top-1_in_vbench](https://img.alicdn.com/imgextra/i1/O1CN01I9wHW91UNnX9wtCWu_!!6000000002506-2-tps-1275-668.png)

模型已在ModelScope和HuggingFace平台发布，训练模型的数据集也已开源。

| 开源模型或数据集 | 链接 | 说明 |
| ------------ | --- | --- |
| Data-Juicer (T2V, 147k) |  [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) | 对应榜单中 Data-Juicer (T2V-Turbo) 模型 |
| Data-Juicer (DJ, 228k) | [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) | 对应榜单中 Data-Juicer (2024-09-23, T2V-Turbo) 模型 |
| data_juicer_t2v_optimal_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-optimal-data-pool)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/data-juicer-t2v-optimal-data-pool) | Data-Juicer (T2V, 147k) 的训练集 |
| data_juicer_t2v_evolution_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool_s2.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-evolution-data-pool) | Data-Juicer (2024-09-23, T2V-Turbo) 的训练集 |

Data-Juicer (DJ, 228k)模型输出样例如下表所示。
  | 文本提示 | 生成视频 |
  | --- | --- |
  | A beautiful coastal beach in spring, waves lapping on sand, zoom out | [![Case 0](https://img.alicdn.com/imgextra/i1/O1CN01KuJeOE1Ylqnk9zYkc_!!6000000003100-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case0.mp4) |
  | a boat accelerating to gain speed | [![Case 1](https://img.alicdn.com/imgextra/i2/O1CN01i1iMFE1TKlIUlqE8d_!!6000000002364-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case1.mp4) |
  | A boat sailing leisurely along the Seine River with the Eiffel Tower in background by Hokusai, in the style of Ukiyo | [![Case 2](https://img.alicdn.com/imgextra/i2/O1CN01u2cjJE1RBwRFeCFuo_!!6000000002074-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case2.mp4) |
  | a bottle on the left of a wine glass, front view | [![Case 3](https://img.alicdn.com/imgextra/i4/O1CN01vdMm6Q1xWc1CoJZW6_!!6000000006451-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case3.mp4) |
  | A corgi's head depicted as an explosion of a nebula | [![Case 4](https://img.alicdn.com/imgextra/i2/O1CN014oPB8Q1IrJg0AbUUg_!!6000000000946-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case4.mp4) |
  | A graceful ballerina doing a pirouette on a dimly lit stage, with soft spotlight highlighting her movements. | [![Case 5](https://img.alicdn.com/imgextra/i4/O1CN01yNlsVu1ymvkJgkvY8_!!6000000006622-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case5.mp4) |

复现论文实验请参考下面的sandbox使用指南，下图的实验流程，[初始数据集](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_init_data_pool.zip)，以及该流程的工作流的配置文件demo：[1_single_op_pipeline.yaml](../configs/demo/bench/1_single_op_pipeline.yaml)、[2_multi_op_pipeline.yaml](../configs/demo/bench/2_multi_op_pipeline.yaml)、[3_duplicate_pipeline.yaml](../configs/demo/bench/3_duplicate_pipeline.yaml)。
![bench_bottom_up](https://img.alicdn.com/imgextra/i2/O1CN01xvu2fo1HU80biR6Q5_!!6000000000760-2-tps-7756-3693.png)

## 什么是沙盒实验室（DJ-Sandbox）？
在Data-Juicer中，数据沙盒实验室为用户提供了持续生产数据菜谱的最佳实践，其具有低开销、可迁移、有指导性等特点，用户在沙盒中基于一些小规模数据集、模型对数据菜谱进行快速实验、迭代、优化，再迁移到更大尺度上，大规模生产高质量数据以服务大模型。

用户在沙盒中，除了Data-Juicer基础的数据优化与数据菜谱微调功能外，还可以便捷地使用数据洞察与分析、沙盒模型训练与评测、基于数据和模型反馈优化数据菜谱等可配置组件，共同组成完整的一站式数据-模型研发流水线。
## 快速上手
### 依赖准备
在使用沙盒实验室前，你可能需要使用如下命令安装沙盒相关的第三方依赖：
```shell
pip install -v -e .[sandbox]
```

**注意**：一些沙盒的依赖还需要额外的领域依赖。

1. 如果用户想要在沙盒中训练一个 ModelScope 平台的NLP模型，那可能需要为 `modelscope` 库
安装额外的 `nlp` 领域依赖（参考其[安装文档](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85) ）。

2. 要使用[EasyAnimate](https://github.com/aigc-apps/EasyAnimate)时需要执行如下安装脚本：
```shell
cd thirdparty/models/
bash setup_easyanimate.sh
cd ../../
```

3. 使用VBench测评视频时需要安装Detectron2，推荐安装如下分支。
```shell
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@b7c7f4ba82192ff06f2bbb162b9f67b00ea55867
```
因此如果使用沙盒过程中，这些第三方依赖抛出了一些"未找到模块（Module-Not-Found）"的报错时，用户需要先检查这些库的文档以寻求帮助。

### 准备沙盒配置文件
沙河实验总共会依次执行四类任务：数据/模型洞察（`probe_job_configs`）、基于洞察结果的数据菜谱微调迭代（`refine_recipe_job_configs`）、数据处理与模型训练（`execution_job_configs`）和数据/模型评估（`evaluation_job_configs`）。每类任务中，任务按照配置的任务列表依次执行。每个任务需要指定：挂载这个任务的钩子（`hook`），记录中间结果的标记名(`meta_name`)，Data-Juicer数据处理参数（`dj_configs`），以及该任务其他的特定参数（`extra_configs`）。这些参数中`hook`是必须指定的，其他允许置空。`dj_configs`可以参考完整的Data-Juicer数据处理参数 [config_all.yaml](https://github.com/modelscope/data-juicer/blob/main/configs/config_all.yaml)。`extra_configs`为任务特定的参数，没有限定，可以是模型训练、推理、评测等参数，比如用`path_k_sigma_recipe`指定利用k-sigma方法微调后的数据菜谱保存路径。一个sandbox的配置文件示例可参考`configs/demo/sandbox/sandbox.yaml`：
```yaml
# Sandbox config example

# global parameters
project_name: 'demo-sandbox'
experiment_name: 'demo-sandbox-run0'              # for wandb tracer name
hpo_config: null                                  # path to a configuration file when using auto-HPO tool.

# configs for each job, the jobs will be executed according to the order in the list
probe_job_configs:
  - hook: 'ProbeViaAnalyzerHook'
    meta_name: 'analysis_ori_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:

refine_recipe_job_configs:
  - hook: 'RefineRecipeViaKSigmaHook'
    meta_name: 'analysis_ori_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:
      path_k_sigma_recipe: './outputs/demo-process/k_sigma_new_recipe.yaml'

execution_job_configs:
  - hook: 'ProcessDataHook'
    meta_name:
    dj_configs: './outputs/demo-process/k_sigma_new_recipe.yaml'
    extra_configs:
  - hook: 'TrainModelHook'
    meta_name:
    dj_configs:
    extra_configs: 'configs/demo/sandbox/gpt3_extra_train_config.json'

evaluation_job_configs:
  - hook: 'ProbeViaAnalyzerHook'
    meta_name: 'analysis_processed_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:
  - hook: 'EvaluateDataHook'
    meta_name: 'eval_data'
    dj_configs:
    extra_configs: 'configs/demo/sandbox/gpt3_data_quality_eval_config.yaml'
```
根据这个配置文件，sandbox：

1. 先执行Data-Juicer数据分析功能，计算每条数据的指定指标，比如`configs/demo/process.yaml`中，指定`language_id_score_filter`计算了语言分。

2. 利用Data-Juicer数据分析的结果，用k-sigma方法微调数据菜谱。注意这里需要设置`meta_name`与数据分析时的`meta_name`相同才能利用到分析结果。

3. 用k-sigma方法微调后的菜谱执行Data-Juicer的数据筛选功能。

4. 用筛选后的数据训练模型。

5. 分析筛选后的数据。

6. 用打分器给筛选后的数据打分。

### 运行沙盒
沙盒的运行入口为`tools/sandbox_starter.py`，使用方法和数据处理与分析工具类似，需要指定沙盒配置文件：
```yaml
python tools/sandbox_starter.py --config configs/demo/sandbox/sandbox.yaml
```
运行开始后，沙盒会根据预定义好的流水线以及配置文件依次运行各个步骤。流水线默认的单次运行主要包括4个大步骤：

1. **数据/模型洞察**：该步骤会对输入的原始数据集/模型进行洞察，如对数据集进行分析或者对模型推理产出的数据进行分析，作为后续步骤的指导
2. **基于洞察结果的数据菜谱微调迭代**：该步骤会基于数据/模型的洞察结果，对输入的数据菜谱进行超参微调优化，如根据数据洞察结果可以使用k-sigma方法调整数据菜谱中的算子超参
3. **数据处理与模型训练**：该步骤会基于微调后的数据菜谱对输入的原始数据集进行处理清洗，如沙盒中配置了训练模型步骤，则还会使用处理后的数据集对配置好的模型进行训练
4. **数据/模型评估**：该步骤针对处理后的数据和前一步中训练好的模型（如有）进行评估，评估方法根据配置可包括数据集二次分析，指定benchmark评估等

如此便完成了一轮沙盒流水线运行，最终用户只需比较数据菜谱微调以及数据集处理前后的洞察结果和评估结果，即可验证该轮实验对于数据生产的有效性。

如果在配置文件里设置了`hpo_config`，并在其中配置了合适的优化目标以及待优化的算子超参，则沙盒会以HPO的形式进行多轮的流水线运行，并自动进行算子超参的多轮迭代微调优化。该配置文件的准备可参考 [hpo工具](https://github.com/modelscope/data-juicer/tree/main/tools/hpo) 。
## 组件工厂
在沙盒流水线的单次运行中，包括了四个大的步骤，其中涉及到如下一些可配置组件，他们分别对应了一个用于初始化这些组件的工厂类：

- **数据处理（DataExecutor）**：数据处理的执行器，即Data-Juicer的executor
- **数据分析（DataAnalyzer）**：数据分析器，即Data-Juicer的analyzer
- **数据评估（DataEvaluator）**：数据集质量的评估器
- **模型数据评估（ModelInferEvaluator）**：利用模型推理结果的数据集质量的评估器
- **模型训练（ModelTrainExecutor）**：模型训练执行器
- **模型推理（ModelInferExecutor）**：模型推理执行器
- **模型评估（ModelEvaluator）**：模型性能的评估器

除了DataExecutor和DataAnalyzer，其余组件均可在配置文件中指定`type`参数来选择具体的执行或者评估类型，如数据评估组件支持`type`为`"dj_text_quality_classifier"`来使用Data-Juicer的质量分类器工具来对数据集进行评估，而模型训练组件`type`为`"modelscope"`来训练来自于ModelScope平台的模型。

目前支持的组件工厂以及工厂中支持的组件包括：

- 数据处理工厂 -- DataExecutorFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `DJExecutor` | Data-Juicer数据处理模块 | - | - |

- 数据分析工厂 -- DataAnalyzerFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `DJAnalyzer` | Data-Juicer数据分析模块 | - | - |

- 数据评估工厂 -- DataEvaluatorFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `Gpt3QualityEvaluator` | 使用Data-Juicer复现的GPT-3文本质量分类器对数据集进行质量评估 | <br />- `eval_type`：该评估器评估对象类型，目前只支持`"data"`<br />- `eval_obj`：未使用的参数<br />- 返回值：待评估数据集样本质量打分均值<br /> | [Data-Juicer质量分类器工具集](https://github.com/modelscope/data-juicer/tree/main/tools/quality_classifier) |
| `VBenchEvaluator` | 使用VBench对基于prompt生成的视频进行多维度的评估 | <br />- `eval_type`：该评估器评估对象类型，目前只支持`"data"`<br />- `eval_obj`：未使用的参数<br />- 返回值：待评生成视频集各维度打分均值<br /> | [VBench论文](https://arxiv.org/abs/2311.17982) |
| `InceptionEvaluator` | 通过视频分类模型抽取特征测评生成的视频 | <br />- `eval_type`：该评估器评估对象类型，目前只支持`"data"`<br />- `eval_obj`：未使用的参数<br />- 返回值：根据给定的metric返回对应的字典<br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |

- 模型数据评估工厂 -- ModelInferEvaluatorFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `ModelscopeInferProbeExecutor` | 用数据集对ModelScope平台上的模型进行推理，并返回推理结果 | <br />- `run_type`：推理类型。需要在组件配置文件中设置`type`参数为`"modelscope"`来激活该组件<br />- `run_obj`：需要送入模型推理的采样数据集<br /> | [ModelScope模型推理文档](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline) |

- 模型训练工厂 -- ModelTrainExecutorFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `ModelscopeTrainExecutor` | 用Data-Juicer产出的数据集训练ModelScope平台上的模型，并监测loss变化信息 | <br />- `run_type`：训练模型类型。需要在组件配置文件中设置`type`参数为`"modelscope"`来激活该组件<br />- `run_obj`：未使用的参数<br /> | [ModelScope模型训练文档](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train) |
| `EasyAnimateTrainExecutor` | 用Data-Juicer产出的数据集训练文生视频模型EasyAnimate的LoRA模型，并监测loss变化信息  | <br />- `run_type`：训练模型类型。需要在组件配置文件中设置`type`参数为`"easyanimate"`来激活该组件<br />- `run_obj`：未使用的参数<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) |

- 模型推理工厂 -- ModelInferExecutorFactory

| 组件 | 功能 | `run`方法说明 | 参考材料 |
| --- | --- | --- | --- |
| `EasyAnimateInferExecutor` | 用VBench的prompt数据集对EasyAnimate模型进行推理，并存储生成的视频 | <br />- `run_type`：推理类型。需要在组件配置文件中设置`type`参数为`"easyanimate"`来激活该组件<br />- `run_obj`：未使用的参数<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) |

- 模型评估工厂 -- ModelEvaluatorFactory
   - TBD

详细定义可参考`data_juicer/core/sandbox/factories.py`。
# 开发者指南
正如上一章节所说，开发者可开发更多的可配置组件并将它们添加到对应的工厂类中，并用参数`type`进行实例化方法分配。实现了组件后，开发者可以将它们封装为钩子，并将钩子注册到工作列表中，工作列表在流水线中进行编排后，沙盒流水线执行时，会依次在每个步骤执行每个工作列表中的工作。这其中的每一个部分：组件、组件工厂、钩子、工作列表、流水线注册与执行流程编排，都可以由开发者自定义。各个部分的关系由下图示意。
![sandbox-pipeline](https://img.alicdn.com/imgextra/i3/O1CN01ERmGre1uz3luKOn4n_!!6000000006107-2-tps-4655-1918.png)

## 组件内部实现
目前组件主要分为两个大类：

- **执行器（Executor）**：由于数据执行器已经由Data-Juicer的Executor承担，因此此处的执行器特指模型的执行器，包括模型训练、推理、评估等执行器。代码位于`data_juicer/core/sandbox/model_executors.py`
- **评估器（Evaluator）**：用于对数据集或者模型进行质量以及性能的评估。代码位于`data_juicer/core/sandbox/evaluators.py`

### 执行器
模型执行器核心功能为对配置文件中指定的模型用指定的数据集进行训练、推理或评测。模型执行器需继承`BaseModelExecutor`并实现若干核心方法：

- 模型执行器的具体行为（训练、推理、评测等）需要在`_run`方法中进行定义
- 模型执行器无返回值，执行时需要进行监测的关键指标通常从模型执行产出的日志中进行解析（如loss、评测结果等），解析与监测过程需要由`_watch_run`方法定义
- 模型执行器在执行时需要数据集输入，因此需要实现`data_connector`方法将数据集由沙盒中的格式转为该模型依赖的框架或者模型库所需要的格式

需要注意的是，为了在模型训练过程中及时监控训练指标（如loss）的变化，需要同时对训练时产生的日志进行监控。因此，执行模型训练的`_run`方法以及监控日志的`watch_run`方法都需要为异步方法，即被关键字`async`修饰。在`run`方法中，我们在训练开始前将标准输出流（stdout）和标准错误流（stderr）都重定向到指定的日志文件，并创建两个异步任务分别执行上述两个方法，它们分别进行以下任务：

- `_run`方法：读入数据集后，根据模型训练配置开始进行模型训练，训练结束后向标准输出流（已重定向到指定的日志文件）输出一个预定义的任务执行结束标识符
- `watch_run`方法：监控指定的日志文件，逐行读取，并调用根据模型训练框架自定义的`_watch_run`方法解析最新的日志内容行，提取关键指标并进行监测，直到读取到预定义的任务结束标识符
### 评估器
评估器核心功能为对待评估对象使用某种方法进行质量、性能等维度的评估，并最终返回一个评估结果，通常为数值型结果。评估器需继承基类`BaseEvaluator`并实现`run`方法。`run`方法默认接受两个必要参数：

- `eval_type`：评估类型，用于在某种评估器内部进行评估类型选择
- `eval_obj`：待评估的对象

用户也可根据自己的实现方式对这两个参数进行扩展使用。
## 流水线钩子
正如章节开始部分所说，在流水线中，我们需要实现若干钩子将组件与流水线执行步骤通过工作列表连接起来。被激活的钩子会在流水线的工作列表中进行注册，然后在流水线执行时依次对各个步骤工作列表中的钩子执行。四个步骤对应的工作列表分别如下：

1. **数据/模型洞察**：洞察工作列表 -- probe_jobs
2. **基于洞察结果的数据菜谱微调迭代**：菜谱微调工作列表 -- refine_recipe_jobs
3. **数据处理与模型训练**：执行工作列表 -- execution_jobs
4. **数据/模型评估**：评估工作列表 -- evaluation_jobs

通常情况下，我们只需要为一类组件工厂实现一种钩子函数即可。而除了依赖于组件的钩子外，还有一些依赖于Data-Juicer已有功能或工具以及其他第三方库的钩子。这些钩子与依赖的组件、工具以及工作列表的对应关系如下：

| 钩子 | 功能 | 依赖的组件工厂 | 依赖的工具或库 | 注册工作列表 |
| --- | --- | --- | --- | --- |
| `ProbeViaAnalyzerHook` | 分析与洞察数据集质量、多样性等维度分布 | 数据分析工厂（DataAnalyzerFactory） | Data-Juicer分析器Analyzer | 洞察工作列表（probe_jobs）<br />评估工作列表（evaluation_jobs） |
| `ProbeViaModelInferHook` | 分析与洞察数据集对于模型的影响，挖掘与洞察“难”数据与“脏”数据 | 数据处理工厂（DataExecutorFactor）<br />模型数据评估工厂（ModelInferEvaluatorFactory） | Data-Juicer数据处理器Executor | 洞察工作列表（probe_jobs）<br />评估工作列表（evaluation_jobs） |
| `RefineRecipeViaKSigmaHook` | 根据数据集洞察结果，利用k-sigma方法对数据菜谱超参进行微调 | - | Data-Juicer超参优化工具HPO中的k-sigma菜谱微调工具 | 菜谱微调工作列表（refine_recipe_jobs） |
| `RefineRecipeViaModelFeedbackHook` | 利用模型洞察与反馈结果对数据菜谱超参进行微调 | TODO | - | 菜谱微调工作列表（refine_recipe_jobs） |
| `ProcessDataHook` | 基于当前数据菜谱对数据集进行处理与清洗 | 数据处理工厂（DataExecutorFactor） | Data-Juicer数据处理器Executor | 执行工作列表（execution_jobs） |
| `TrainModelHook` | 基于当前数据集训练一个模型 | 模型训练工厂（ModelTrainExecutorFactory） | [EasyAnimate](../thirdparty//easy_animate/README.md) | 执行工作列表（execution_jobs） |
| `InferModelHook` | 模型基于给定输入让模型产生输出 | 模型推理工厂（ModelInferExecutorFactory） | [EasyAnimate](../thirdparty//easy_animate/README.md) | 执行工作列表（execution_jobs） |
| `EvaluateDataHook` | 对当前数据集进行数据质量等维度的评估 | 数据评估工厂（DataEvaluatorFactory） | 图像或视频的[inception metrics](../tools/mm_eval/inception_metrics/README_ZH.md)，如FID、FVD <br /> [VBench](../tools/mm_eval/vbench_metrics/README_ZH.md) | 评估工作列表（evaluation_jobs） |
| `EvaluateModelHook` | 对当前训练后的模型进行评估 | 模型评估工厂（ModelEvaluatorFactory） | - | 评估工作列表（evaluation_jobs） |

值得注意的是，一个钩子可以在多个工作列表进行注册，因为这个钩子在不同的流水线阶段可以扮演不同的角色，比如我们可以对处理前后的数据集都进行分析，以比较数据集处理前后的质量、多样性等维度的变化情况。

## 自定义沙盒流水线
用户直接在参数配置文件中修改任务配置列表即可实现任务修改和编排。

## 监测器
在上述章节中，反复提到“监测”这个概念。流水线会对各个步骤中产生的若干指标都进行监测，这些监测过程都依靠沙盒监测器`SandboxWatcher`实现的。

`SandboxWatcher`基于wandb实现，主要包括4个方法：

- `setup_sweep`：在多轮HPO模式下会调用，多轮HPO由wandb中的sweep支持，因此需要额外传入`hpo_config`配置文件对其进行初始化
- `watch_cfgs`：对sandbox实验以及各个组件的配置文件进行监测与更新
- `watch`：对某个具体指标或实验结果进行监测，并记录到wandb日志
- `query`：对某个具体指标或实验结果从wandb日志中进行查询
