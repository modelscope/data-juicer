# User Guide
## What is DJ-Sandbox?
In Data-Juicer, the data sandbox laboratory provides users with the best practices for continuously producing data recipes. It features low overhead, portability, and guidance. In the sandbox, users can quickly experiment, iterate, and refine data recipes based on small-scale datasets and models, before scaling up to produce high-quality data to serve large-scale models.

In addition to the basic data optimization and recipe refinement features offered by Data-Juicer, users can seamlessly use configurable components such as data probe and analysis, model training and evaluation, and data and model feedback-based recipe refinement to form a complete one-stop data-model research and development pipeline.
## Quick Start
### Requirements
Before using sandbox, you might need to install sandbox-related third-party dependencies by running the command below:
```shell
pip install -v -e .[sandbox]

pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@b7c7f4ba82192ff06f2bbb162b9f67b00ea55867
```

**NOTICE**: some sandbox-related dependencies require extra domain dependencies. For example, if users want to train an NLP model from ModelScope
in the sandbox, you might need to install extra `nlp` dependencies for `modelscope` library (see the [installation docs](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)).
So if some Module-Not-Found errors are raised by these third-party libraries when running the sandbox, users need to check their docs first.

### Prepare Configuration Files for Sandbox
The configuration file of the sandbox includes several additional parameters in addition to the configuration of Data-Juicer. These parameters are used to specify the configuration information for model training, inference, evaluation, and other steps that may run in the sandbox pipeline. For the complete set of additional parameters, please refer to the "for sandbox or hpo" section in the [config_all.yaml](https://github.com/modelscope/data-juicer/blob/main/configs/config_all.yaml). An example of a sandbox configuration file can be found in `configs/demo/sandbox/sandbox.yaml`:
```yaml
# Sandbox config example for dataset

# global parameters
project_name: 'demo-sandbox'
dataset_path: './demos/data/demo-dataset.jsonl'  # path to your dataset directory or file
np: 4  # number of subprocess to process your dataset

export_path: './outputs/demo-sandbox/demo-sandbox.jsonl'

# sandbox configs
# for refining recipe using k-sigma rules
path_k_sigma_recipe: './outputs/demo-sandbox/k_sigma_new_recipe.yaml'

# for gpt3 quality classifier as data evaluator
data_eval_config: 'configs/demo/sandbox/gpt3_data_quality_eval_config.yaml'
#data_eval_config:
#  type: dj_text_quality_classifier

# for gpt3 model training
model_train_config: 'configs/demo/sandbox/gpt3_extra_train_config.json'

# process schedule
# a list of several process operators with their arguments
process:
  - language_id_score_filter:
      lang: 'zh'
      min_score: 0.5
```
In the example configuration file, in addition to the Data-Juicer data processing related configurations, there are three additional parameters:

- `path_k_sigma_recipe`: Used to specify the save path for the refined recipe using the k-sigma method.
- `data_eval_config`: Used to specify the configuration file path for the data evaluation step. This part of the configuration can also be directly added as a dictionary under this field.
- `model_train_config`: Used to specify the configuration file path for training models using the processed data.

Additional configuration files can support both YAML and JSON formats, and their contents need to be specifically defined based on the implementation of each component used in each step, as well as the models and evaluation support. The specific configuration contents of the steps involved in this example above can be referred to as configuration file contents in the corresponding path.
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
- **Data Evaluation (DataEvaluator)**: Evaluator on the quality of the dataset
- **Model Training (ModelTrainExecutor)**: Executor for model training
- **Model Inference (ModelInferExecutor)**: Executor for model inference
- **Model Evaluation (ModelEvaluator)**: Evaluator on the performance of the model

Except for DataExecutor, the rest of the components can be specified in the configuration file using the `type` parameter to choose a specific execution or evaluation type. For example, the data evaluation component supports a `type` of `"dj_text_quality_classifier"` to utilize Data-Juicer's text quality classifier tool for evaluating the dataset, while the model training component `type` can be set to `"modelscope"` to train a model from the ModelScope platform.

The currently supported component factories and the components supported within each factory are as follows:

- DataEvaluatorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `Gpt3QualityEvaluator` | Evaluate the quality of a dataset using the GPT-3 text quality classifier reproduced by Data-Juicer. | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`.<br />- `eval_obj`: The path to the dataset to be evaluated.<br />- Return: The average quality score of the dataset samples.<br /> | [Data-Juicer Quality Classifier Toolkit](https://github.com/modelscope/data-juicer/tree/main/tools/quality_classifier) |
| `VBenchEvaluator` | Evaluate the generated videos according to given prompts in multi dimensions | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter<br />- Return: The average score of generated videos in multi dimensions.<br /> | [VBench paper](https://arxiv.org/abs/2311.17982) |
| `InceptionEvaluator` | Evaluate the generated videos by features extracted from video classification models. | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter<br />- Return: A dictionary of scores in the given metric. <br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |

- ModelTrainExecutorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `ModelscopeTrainExecutor` | Perform a training task on a model from the ModelScope platform using specified datasets, and monitor the change in training loss. | <br />- `run_type`: Type of model training. We need to set `type`  arg as `"modelscope"` in the component configuration file to activate this component.<br />- `run_obj`: Additional training configurations. Apart from the component configuration, this includes the dataset paths and the working directory for storing the training output. As they may change during the pipeline run, they are set dynamically within the pipeline.<br /> | [ModelScope Docs of Model Training](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train) |

- ModelInferExecutorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `ModelscopeInferExecutor` | Perform inference on a model from the ModelScope platform using a specified sampled dataset, and return the inference results. | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"infer_on_data"` in the component configuration file to activate this component.<br />- `run_obj`: Sampled dataset to be fed into model inference.<br /> | [ModelScope Docs of Model Inference](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline) |

- ModelEvaluatorFactory
   - TBD

Please refer to `data_juicer/core/sandbox/factories.py` for detailed definitions.
# Developer Guide
As mentioned in the previous section, developers can develop customized configurable components and add them to the corresponding factory classes, then route to appropriate instantiation methods using the `type` parameter. Once the components are implemented, developers can encapsulate them as hooks and register the hooks into the job list. After the job list is orchestrated in the pipeline, when the sandbox pipeline is executed, each job in the job list will be executed in sequence at each step. Each of these parts - components, component factory, hooks, job lists, and the registration and execution orchestration of the pipeline - can be customized by the developer. The relationship among these parts is illustrated in the diagram below.
![sandbox-pipeline](https://img.alicdn.com/imgextra/i1/O1CN01JsgSuu22ycGdJFRdc_!!6000000007189-2-tps-3640-2048.png)

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
## Pipeline Hook and Job List
As mentioned at the start of this section, in the pipeline, we need to implement several hooks to connect components with the pipeline execution steps through the job list. Activated hooks will be registered in the pipeline's job list and then executed one by one during the pipeline execution at each step. The job lists for the four corresponding steps are as follows:

1. **Data/Model Probe**: Probe job list -- probe_jobs
2. **Iterative Recipe Refinement based on Probe Results**: Refinement job list -- refine_recipe_jobs
3. **Data Processing and Model Training**: Execution job list - execution_jobs
4. **Data/Model Evaluation**: Evaluation job list - evaluation_jobs

In general, we only need to implement one type of hook function for a type of component factory. In addition to hooks that depend on components, some hooks depend on the existing functionality and tools of Data-Juicer or other third-party libraries. The correspondence among these hooks, dependent components, tools, and job lists is as follows:

| Hook | Function | Dependent Component Factory | Dependent Tool or Library | Registered Job List | Activation Method<br />(Default Pipeline Orchestration) |
| --- | --- | --- | --- | --- | --- |
| `hook_probe_via_analyzer` | Analyse and probe the quality and diversity distribution of the dataset | - | Data-Juicer Analyser | <br />- probe_jobs<br />- evaluation_jobs<br /> | Always |
| `hook_probe_via_model_infer` | Analyze and understand the impact of the dataset on the model, explore and probe "difficult" and "dirty" data | ModelInferExecutorFactory | - | <br />- probe_jobs<br />- evaluation_jobs<br /> | There are valid `model_infer_config` parameters in the sandbox configuration |
| `hook_refine_recipe_via_k_sigma` | Refine data recipe hyperparameters using the k-sigma method based on the probe results of the dataset | - | k-sigma recipe refinement tool of Data-Juicer Hyperparameter Optimization (HPO) toolkit | <br />- refine_recipe_jobs<br /> | There are valid `path_k_sigma_recipe` parameters in the sandbox configuration to specify the path to save the refined recipe |
| `hook_refine_recipe_via_model_feedback` | Refine data recipe hyperparameters using model probe and feedback results | TODO | - | <br />- refine_recipe_jobs<br /> | There are valid `path_model_feedback_recipe` parameters in the sandbox configuration to specify the path to save the refined recipe |
| `hook_process_data` | Process and clean the dataset based on the current data recipe | - | Data-Juicer Executor | <br />- execution_jobs<br /> | Always |
| `hook_train_model` | Train a model based on the current dataset | ModelTrainExecutorFactory | - | <br />- execution_jobs<br /> | There are valid `model_train_config` parameters in the sandbox configuration |
| `hook_evaluate_data` | Evaluate the dataset in terms of data quality and other dimensions | DataEvaluatorFactory | - | <br />- evaluation_jobs<br /> | There are valid `data_eval_config` parameters in the sandbox configuration |
| `hook_evaluate_model` | Evaluate the trained model | ModelEvaluatorFactory | - | <br />- evaluation_jobs<br /> | There are valid `model_eval_config` parameters in the sandbox configuration |

It is worth noting that a hook can be registered in multiple job lists, as this hook can play different roles in different steps of the pipeline. For example, we can analyze and probe both the pre-processed and post-processed datasets to compare the changes in quality, diversity, and other dimensions before and after data processing.
## Customized Sandbox Pipeline
In addition to the default sandbox pipeline, developers can also implement customized pipeline orchestration in `data_juicer/core/sandbox/pipelines.py`. Combining the concepts discussed in previous sections, implementing a customized pipeline orchestration by developers generally involves the following steps:

1. **Implementing customized components**: Developers can create new components based on existing factories, or create new categories of factories and their components.
2. **Encapsulate the hooks to call the customized components**: For example, reference the code in the method `hook_evaluate_data`, which calls the data evaluation component to evaluate the quality of datasets.
3. **Register the customized hooks into the job list**: Developers can implement customized job lists and registration methods. Reference the code in the `register_default_jobs` method for guidance.
4. **Implement a customized pipeline orchestration**: Based on the customized hooks and job lists, developers can customize, arrange, and build the pipeline execution process according to their specific requirements, as illustrated in the `one_trial` method in the pipeline.
## Watcher
In the above sections, the concept of "monitoring" is repeatedly mentioned. The pipeline will monitor several metrics produced in each step, and these monitoring processes are implemented by `SandboxWatcher`.

`SandboxWatcher` is based on wandb library and mainly includes four methods:

- `setup_sweep`: This method is called in the multi-trial HPO mode, which is supported by the sweep module in wandb library. Therefore, the additional `hpo_config` configuration for sweep initialization is required to be passed into the sandbox configuration file.
- `watch_cfgs`: This method monitors and updates the sandbox experiments and configuration files of various components.
- `watch`: This method monitors a specific metric or experiment result and records it in the wandb log.
- `query`: This method queries a specific metric or experiment result from the wandb log.
