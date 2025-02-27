# User Guide
## Applications and Achievements
Leveraging the Data-Juicer Sandbox Laboratory Suite, we systematically fine-tuned data and models through a dedicated research and development workflow between data and models. For more detailed information, please refer to our [paper](http://arxiv.org/abs/2407.11784). In our work, we have secured a new leading position on the [VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) text-to-video leaderboard.
![top-1_in_vbench](https://img.alicdn.com/imgextra/i1/O1CN01I9wHW91UNnX9wtCWu_!!6000000002506-2-tps-1275-668.png)

The model is now publicly available on the ModelScope and HuggingFace platforms, and the training dataset has also been available.

| Open-source model or dataset | Link | Description |
| ------------ | --- | --- |
| Data-Juicer (T2V, 147k) |  [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) | Corresponding to Data-Juicer (T2V-Turbo) model in VBench leaderboard |
| Data-Juicer (DJ, 228k) | [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) | Corresponding to Data-Juicer (2024-09-23, T2V-Turbo) model in VBench leaderboard |
| data_juicer_t2v_optimal_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-optimal-data-pool)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/data-juicer-t2v-optimal-data-pool) | The training dataset of Data-Juicer (T2V, 147k) |
| data_juicer_t2v_evolution_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool_s2.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-evolution-data-pool) | The training dataset of Data-Juicer (2024-09-23, T2V-Turbo) |

Following is the case study for Data-Juicer (DJ, 228k) outputs.
  | Prompt | Generated Video |
  | --- | --- |
  | A beautiful coastal beach in spring, waves lapping on sand, zoom out | [![Case 0](https://img.alicdn.com/imgextra/i1/O1CN01KuJeOE1Ylqnk9zYkc_!!6000000003100-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case0.mp4) |
  | a boat accelerating to gain speed | [![Case 1](https://img.alicdn.com/imgextra/i2/O1CN01i1iMFE1TKlIUlqE8d_!!6000000002364-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case1.mp4) |
  | A boat sailing leisurely along the Seine River with the Eiffel Tower in background by Hokusai, in the style of Ukiyo | [![Case 2](https://img.alicdn.com/imgextra/i2/O1CN01u2cjJE1RBwRFeCFuo_!!6000000002074-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case2.mp4) |
  | a bottle on the left of a wine glass, front view | [![Case 3](https://img.alicdn.com/imgextra/i4/O1CN01vdMm6Q1xWc1CoJZW6_!!6000000006451-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case3.mp4) |
  | A corgi's head depicted as an explosion of a nebula | [![Case 4](https://img.alicdn.com/imgextra/i2/O1CN014oPB8Q1IrJg0AbUUg_!!6000000000946-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case4.mp4) |
  | A graceful ballerina doing a pirouette on a dimly lit stage, with soft spotlight highlighting her movements. | [![Case 5](https://img.alicdn.com/imgextra/i4/O1CN01yNlsVu1ymvkJgkvY8_!!6000000006622-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case5.mp4) |

To reproduce the paper's experiments, please refer to the sandbox usage guide below, the experimental process in the following figure, the [initial dataset](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_init_data_pool.zip), and the configuration file demos for the process: [1_single_op_pipeline.yaml](../configs/demo/bench/1_single_op_pipeline.yaml), [2_multi_op_pipeline.yaml](../configs/demo/bench/2_multi_op_pipeline.yaml), [3_duplicate_pipeline.yaml](../configs/demo/bench/3_duplicate_pipeline.yaml).
![bench_bottom_up](https://img.alicdn.com/imgextra/i2/O1CN01xvu2fo1HU80biR6Q5_!!6000000000760-2-tps-7756-3693.png)

## What is DJ-Sandbox?
In Data-Juicer, the data sandbox laboratory provides users with the best practices for continuously producing data recipes. It features low overhead, portability, and guidance. In the sandbox, users can quickly experiment, iterate, and refine data recipes based on small-scale datasets and models, before scaling up to produce high-quality data to serve large-scale models.

In addition to the basic data optimization and recipe refinement features offered by Data-Juicer, users can seamlessly use configurable components such as data probe and analysis, model training and evaluation, and data and model feedback-based recipe refinement to form a complete one-stop data-model research and development pipeline.
## Quick Start
### Requirements
Before using sandbox, you might need to install sandbox-related third-party dependencies by running the command below:
```shell
pip install -v -e .[sandbox]

```

**NOTICE**: some sandbox-related dependencies require extra domain dependencies. 

1. If users want to train an NLP model from ModelScope
in the sandbox, you might need to install extra `nlp` dependencies for `modelscope` library (see the [installation docs](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)).

2. To use [EasyAnimate](https://github.com/aigc-apps/EasyAnimate), you need to execute the following installation script:
```shell
cd thirdparty/models/
bash setup_easyanimate.sh
cd ../../
```

3. When using VBench to benchmark videos, it is necessary to install Detectron2. The following branch is recommended for installation.
```shell
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@b7c7f4ba82192ff06f2bbb162b9f67b00ea55867
```
So if some Module-Not-Found errors are raised by these third-party libraries when running the sandbox, users need to check their docs first.

### Prepare Configuration Files for Sandbox
The sandbox will sequentially execute four types of jobs: Data/Model Probe (`probe_job_configs`), Iterative Recipe Refinement based on Probe Results(`refine_recipe_job_configs`), Dataset Processing and Model Training (`execution_job_configs`) and Data/Model Evaluation (`evaluation_job_configs`). Within each category of jobs, jobs are carried out in the order specified by the configured job list. Each task requires specifying: the hook for mounting this job (`hook`), the tag name for recording intermediate results (`meta_name`), Data-Juicer data processing parameters (`dj_configs`), as well as other specific parameters for the job (`extra_configs`). Among these parameters, hook is required, while others may be left empty. dj_configs can refer to the full Data-Juicer data processing parameters available in [config_all.yaml](https://github.com/modelscope/data-juicer/blob/main/configs/config_all.yaml). The `extra_configs` are task-specific parameters without restrictions. They can include parameters for model training, inference, evaluation, etc. For example, `path_k_sigma_recipe` can be used to specify the path for saving the data recipe refined using the k-sigma method. An example of a sandbox configuration file can be found at `configs/demo/sandbox/sandbox.yaml`:
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
Based on this configuration file, sandbox:

1. Execute the Data-Juicer data analysis function to calculate specified metrics for each piece of data, for example, in `configs/demo/process.yaml`, the `language_id_score_filter` is designated to calculate language scores.

2. With the results from Data-Juicer data analysis, fine-tune the data recipe using the k-sigma method. Note that the `meta_name` here must be set the same as the `meta_name` used during data analysis to utilize the results.

3. Execute Data-Juicer's data filtering function with the data recipe fine-tuned by the k-sigma method.

4. Train the model with the filtered data.

5. Analyze the data after filtering.

6. Score the data after filtering with a scorer.

### Start Sandbox
The entry point for running the sandbox is `tools/sandbox_starter.py`. The usage is similar to the data processing and analysis tool, requiring specifying the sandbox configuration file:
```yaml
python tools/sandbox_starter.py --config configs/demo/sandbox/sandbox.yaml
```
Once the run is started, the sandbox will sequentially execute each of the predefined pipeline steps according to the configuration file. The default one trial of the pipeline mainly includes four major steps:

1. **Data/Model Probe**: This step provides probes into the input dataset/model, such as analysing the dataset or analysing the data produced by model inference, to guide the subsequent steps.
2. **Iterative Recipe Refinement based on Probe Results**: This step refines and optimizes the recipe hyperparameters based on the data/model probes. For example, the operator (OP) hyperparameters in the data recipe can be adjusted using the k-sigma method based on the data probes.
3. **Dataset Processing and Model Training**: This step processes and cleans the input dataset based on the refined recipe. If model training is configured in the sandbox, the processed dataset will also be used to train the configured model.
4. **Data/Model Evaluation**: This step evaluates the processed dataset and the model trained in the previous step (if applicable). The evaluation methods may include analysis of the processed dataset and specified benchmark evaluations based on the configuration.

Once this completes one trial of the sandbox pipeline run, the user can validate the effectiveness of the experiment in data production by comparing the probes and evaluation results before and after recipe refinement and dataset processing.

If the `hpo_config` is set in the configuration file and appropriate optimization objectives and OP hyperparameters to be refined are configured within it, the sandbox will perform multiple trials of pipeline runs in the form of Hyperparameter Optimization (HPO) and automatically conduct iterative refinement and optimization of the operator hyperparameters. The preparation of this configuration file can be referenced from the [HPO tool](https://github.com/modelscope/data-juicer/tree/main/tools/hpo).
## Component Factory
In a single trial of the sandbox pipeline, four major steps involve various configurable components. Each of these components corresponds to a factory class used to initialize them:

- **Data Processing (DataExecutor)**: Executor for dataset processing, i.e., the executor of Data-Juicer
- **Data Analyzing（DataAnalyzer）**: Analyzer for dataset, i.e., the analyzer of Data-Juicer
- **Data Evaluation (DataEvaluator)**: Evaluator on the quality of the dataset
- **Model-Data Evaluation（ModelInferEvaluator）**: Evaluator of dataset quality using the model's inference results
- **Model Training (ModelTrainExecutor)**: Executor for model training
- **Model Inference (ModelInferExecutor)**: Executor for model inference
- **Model Evaluation (ModelEvaluator)**: Evaluator on the performance of the model

Except for DataExecutor and DataAnalyzer, the rest of the components can be specified in the configuration file using the `type` parameter to choose a specific execution or evaluation type. For example, the data evaluation component supports a `type` of `"dj_text_quality_classifier"` to utilize Data-Juicer's text quality classifier tool for evaluating the dataset, while the model training component `type` can be set to `"modelscope"` to train a model from the ModelScope platform.

The currently supported component factories and the components supported within each factory are as follows:

- DataExecutorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `DJExecutor` | The data process module of Data-Juicer | - | - |

- DataAnalyzerFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `DJAnalyzer` | The data analysis module of Data-Juicer | - | - |

- DataEvaluatorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `Gpt3QualityEvaluator` | Evaluate the quality of a dataset using the GPT-3 text quality classifier reproduced by Data-Juicer. | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`.<br />- `eval_obj`: An useless parameter.<br /> | [Data-Juicer Quality Classifier Toolkit](https://github.com/modelscope/data-juicer/tree/main/tools/quality_classifier) |
| `VBenchEvaluator` | Evaluate the generated videos according to given prompts in multi dimensions | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: An useless parameter.<br />- Return: The average score of generated videos in multi dimensions.<br /> | [VBench paper](https://arxiv.org/abs/2311.17982) |
| `InceptionEvaluator` | Evaluate the generated videos by features extracted from video classification models. | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: An useless parameter.<br />- Return: A dictionary of scores in the given metric. <br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |

- ModelInferEvaluatorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `ModelscopeInferExecutor` | Perform inference on a model from the ModelScope platform using a specified sampled dataset, and return the inference results. | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"modelscope"` in the component configuration file to activate this component.<br />- `run_obj`: Sampled dataset to be fed into model inference.<br /> | [ModelScope Docs of Model Inference](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline) |

- ModelTrainExecutorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `ModelscopeTrainExecutor` | Perform a training task on a model from the ModelScope platform using specified datasets, and monitor the change in training loss. | <br />- `run_type`: Type of model training. We need to set `type`  arg as `"modelscope"` in the component configuration file to activate this component.<br />- `run_obj`: An useless parameter.<br /> | [ModelScope Docs of Model Training](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train) |
| `EasyAnimateTrainExecutor` | Perform a LoRA training task on EasyAnimate text-to-video model, and monitor the change in training loss. | <br />- `run_type`: Type of model training. We need to set `type`  arg as `"easyanimate"` in the component configuration file to activate this component.<br />- `run_obj`: An useless parameter.<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) |

- ModelInferExecutorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `EasyAnimateInferExecutor` | Perform inference on EasyAnimate text-to-video model with the prompts from VBench, and save the generated videos. | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"easyanimate"` in the component configuration file to activate this component.<br />- `run_obj`: An useless parameter.<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) |

- ModelEvaluatorFactory
   - TBD

Please refer to `data_juicer/core/sandbox/factories.py` for detailed definitions.
# Developer Guide
As mentioned in the previous section, developers can develop customized configurable components and add them to the corresponding factory classes, then route to appropriate instantiation methods using the `type` parameter. Once the components are implemented, developers can encapsulate them as hooks and register the hooks into the job list. After the job list is orchestrated in the pipeline, when the sandbox pipeline is executed, each job in the job list will be executed in sequence at each step. Each of these parts - components, component factory, hooks, job lists, and the registration and execution orchestration of the pipeline - can be customized by the developer. The relationship among these parts is illustrated in the diagram below.
![sandbox-pipeline](https://img.alicdn.com/imgextra/i3/O1CN01ERmGre1uz3luKOn4n_!!6000000006107-2-tps-4655-1918.png)

## The Internal Implementation of Components
Currently, components are mainly divided into two major categories:

- **Executor**: Since the data executor is already handled by the Data-Juicer's Executor, the executor here specifically refers to the model executor, including model training, inference, evaluation, etc. The code is located in `data_juicer/core/sandbox/model_executors.py`.
- **Evaluator**: Used for evaluating the quality and performance of datasets or models. The code is located in `data_juicer/core/sandbox/evaluators.py`.

### Executor
The core function of the model executor is to train, infer, or evaluate the model specified in the configuration file with the specified dataset. The model executor needs to inherit from `BaseModelExecutor` and implement several core methods:

- The specific behavior of the model executor (training, inference, evaluation, etc.) needs to be defined in the `_run` method.
- The model executor does not return any value. Key metrics that need to be monitored during execution are usually parsed from the logs produced by the model executor (such as loss, evaluation results, etc.). The parsing and monitoring process needs to be defined by the `_watch_run` method.
- Model executor requires input from a dataset, so the `data_connector` method needs to be implemented to convert the dataset from the sandbox's format to the format required by the model framework or library.

It is important to note that, to monitor the change of training metrics (e.g., loss) promptly during the model training process, logs generated during training need to be monitored. Therefore, both the `_run` method for executing model training and the `watch_run` method for monitoring logs need to be asynchronous methods, indicated by the `async` keyword. In the `run` method, we redirect the standard output stream (stdout) and standard error stream (stderr) to a designated log file before the training starts. Then, we create two asynchronous tasks to execute the aforementioned two methods, each performing the following tasks:

- `_run` method: After loading the dataset, it starts model training based on the model training configuration. Upon completion of training, it outputs a predefined task completion identifier to the standard output stream, which has been redirected to the designated log file.
- `watch_run` method: It monitors the designated log file, reads it line by line, and calls the `_watch_run` method. The called method is customized based on the model training framework and used to parse the latest log content line, extract key metrics, and monitor them until the predefined task completion identifier is read.
### Evaluator
The core function of the evaluator is to evaluate the quality and performance of the target using some specific methods and return the evaluation result, usually a numerical value. The evaluator needs to inherit from the base class `BaseEvaluator` and implement the `run` method. The `run` method typically takes two required parameters:

- `eval_type`: The type of evaluation, used for internal evaluation type routine within a certain evaluator.
- `eval_obj`: The object to be evaluated.

Users can also extend the usage of these two parameters based on their implementation.
## Pipeline Hook
As mentioned at the start of this section, in the pipeline, we need to implement several hooks to connect components with the pipeline execution steps through the job list. Activated hooks will be registered in the pipeline's job list and then executed one by one during the pipeline execution at each step. The job lists for the four corresponding steps are as follows:

1. **Data/Model Probe**: Probe job list -- probe_jobs
2. **Iterative Recipe Refinement based on Probe Results**: Refinement job list -- refine_recipe_jobs
3. **Data Processing and Model Training**: Execution job list - execution_jobs
4. **Data/Model Evaluation**: Evaluation job list - evaluation_jobs

In general, we only need to implement one type of hook function for a type of component factory. In addition to hooks that depend on components, some hooks depend on the existing functionality and tools of Data-Juicer or other third-party libraries. The correspondence among these hooks, dependent components, tools, and job lists is as follows:

| Hook | Function | Dependent Component Factory | Dependent Tool or Library | Registered Job List |
| --- | --- | --- | --- | --- |
| `ProbeViaAnalyzerHook` | Analyze and probe the quality and diversity distribution of the dataset | DataAnalyzerFactory | Data-Juicer Analyzer | - probe_jobs<br />- evaluation_jobs |
| `ProbeViaModelInferHook` | Analyze and understand the impact of the dataset on the model, explore and probe "difficult" and "dirty" data | DataExecutorFactor <br />ModelInferEvaluatorFactory | Data-Juicer Executor | - probe_jobs<br />- evaluation_jobs |
| `RefineRecipeViaKSigmaHook` | Refine data recipe hyperparameters using the k-sigma method based on the probe results of the dataset | - | k-sigma recipe refinement tool of Data-Juicer Hyperparameter Optimization (HPO) toolkit | - refine_recipe_jobs |
| `RefineRecipeViaModelFeedbackHook` | Refine data recipe hyperparameters using model probe and feedback results | TODO | - | - refine_recipe_jobs |
| `ProcessDataHook` | Process and clean the dataset based on the current data recipe | DataExecutorFactor | Data-Juicer Executor | - execution_jobs | Always |
| `TrainModelHook` | Train a model based on the current dataset | ModelTrainExecutorFactory | [EasyAnimate](../thirdparty//easy_animate/README.md) | - execution_jobs |
| `InferModelHook` | The model generates output based on the given input | ModelInferExecutorFactory | [EasyAnimate](../thirdparty//easy_animate/README.md) | - execution_jobs |
| `EvaluateDataHook` | Evaluate the dataset in terms of data quality and other dimensions | DataEvaluatorFactory | [inception metrics](../tools/mm_eval/inception_metrics/README.md) for images and videos, such as FID and FVD <br /> [VBench](../tools/mm_eval/vbench_metrics/README.md) | - evaluation_jobs |
| `EvaluateModelHook` | Evaluate the trained model | ModelEvaluatorFactory | - | - evaluation_jobs |

It is worth noting that a hook can be registered in multiple job lists, as this hook can play different roles in different steps of the pipeline. For example, we can analyze and probe both the pre-processed and post-processed datasets to compare the changes in quality, diversity, and other dimensions before and after data processing.

## Customized Sandbox Pipeline
Users can directly modify the job configuration list in the parameter configuration file to achieve task modification and orchestration.

## Watcher
In the above sections, the concept of "monitoring" is repeatedly mentioned. The pipeline will monitor several metrics produced in each step, and these monitoring processes are implemented by `SandboxWatcher`.

`SandboxWatcher` is based on wandb library and mainly includes four methods:

- `setup_sweep`: This method is called in the multi-trial HPO mode, which is supported by the sweep module in wandb library. Therefore, the additional `hpo_config` configuration for sweep initialization is required to be passed into the sandbox configuration file.
- `watch_cfgs`: This method monitors and updates the sandbox experiments and configuration files of various components.
- `watch`: This method monitors a specific metric or experiment result and records it in the wandb log.
- `query`: This method queries a specific metric or experiment result from the wandb log.
