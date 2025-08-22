# yapf: disable
import asyncio
from copy import deepcopy

from jsonargparse import dict_to_namespace
from loguru import logger

from data_juicer.config import get_init_configs, prepare_side_configs
from data_juicer.core.data.dj_dataset import nested_query
from data_juicer.core.sandbox.factories import (
    data_analyzer_factory,
    data_evaluator_factory,
    data_executor_factory,
    data_pool_manipulator_factory,
    general_data_executor_factory,
    general_probe_factory,
    mode_infer_evaluator_factory,
    model_evaluator_factory,
    model_infer_executor_factory,
    model_train_executor_factory,
)
from data_juicer.utils.constant import JobRequiredKeys
from tools.hpo.execute_hpo_3sigma import modify_recipe_k_sigma


class BaseHook:

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        self.job_cfg = job_cfg
        self.watcher = watcher

        self.hook_type = job_cfg.get(JobRequiredKeys.hook.value)

        # hook inputs mapping for config update
        self.input_mapping = job_cfg.get(JobRequiredKeys.input.value, {})
        # names of hook output for record the results of hook running
        self.output_keys = job_cfg.get(JobRequiredKeys.output.value, None)
        if self.output_keys and not isinstance(self.output_keys, list):
            self.output_keys = [self.output_keys]
        # hook local settings for configs
        self.local_settings = job_cfg.get(JobRequiredKeys.local.value, {})

        # unique meta name for this job
        self.meta_name = job_cfg.get(JobRequiredKeys.meta_name.value, 'anonymous_meta_name')
        # data-juicer config for some jobs based on Data-Juicer
        self.dj_cfg = job_cfg.get(JobRequiredKeys.dj_configs.value, None)
        self.inited_dj_cfg = None
        # extra config for some other specific jobs
        self.extra_cfg = job_cfg.get(JobRequiredKeys.extra_configs.value, None)

    def run(self, **context_infos):
        self._input_updating_hook(**context_infos)

        outputs = self.hook(**context_infos)

        if outputs:
            return self._output_recording_hook(outputs)
        else:
            return context_infos

    def _input_updating_hook(self, **context_infos):
        self.specify_dj_and_extra_configs(allow_fail=True)

        prev_dj_cfg = deepcopy(self.dj_cfg) if self.dj_cfg else None
        prev_extra_cfg = deepcopy(self.extra_cfg) if self.extra_cfg else None

        # update configs according to local settings
        for key, value in self.local_settings.items():
            key_to_updated_parts = key.split('.')
            cfg_type = key_to_updated_parts[0]
            # check which config group to update
            if cfg_type == JobRequiredKeys.dj_configs.value:
                target_config = self.dj_cfg
            elif cfg_type == JobRequiredKeys.extra_configs.value:
                target_config = self.extra_cfg
            else:
                raise ValueError(f'The key {cfg_type} to update is not supported.')
            key_to_updated_parts = key_to_updated_parts[1:]
            # update the target key
            if len(key_to_updated_parts) > 0:
                if len(key_to_updated_parts) > 1:
                    target_config = nested_query(target_config, '.'.join(key_to_updated_parts[:-1]))
                target_config[key_to_updated_parts[-1]] = value
            else:
                if cfg_type == JobRequiredKeys.dj_configs.value:
                    self.dj_cfg = value
                elif cfg_type == JobRequiredKeys.extra_configs.value:
                    self.extra_cfg = value

        # update configs according to input mapping and context_infos
        for key_to_updated, key_in_history in self.input_mapping.items():
            key_to_updated_parts = key_to_updated.split('.')
            key_in_history_parts = key_in_history.split('.')
            # check if it's -1
            history_job_infos = context_infos
            if key_in_history_parts[0] == '-1':
                # get the latest infos
                if len(key_in_history_parts) <= 1:
                    raise ValueError(f'Need to specify the job result keys precisely for inputs to '
                                     f'find the target values. Only got [{key_in_history}].')
                # find the last non-empty job_infos
                pipeline_keys = list(context_infos.keys())
                if len(pipeline_keys) == 0:
                    raise ValueError(f'Cannot find the previous non-empty job infos for [{key_in_history}].')
                last_idx = len(pipeline_keys) - 1
                history_job_infos = context_infos[pipeline_keys[last_idx]]
                while len(history_job_infos) == 0:
                    last_idx -= 1
                    if last_idx < 0:
                        raise ValueError(f'Cannot find the previous non-empty job infos for [{key_in_history}].')
                    history_job_infos = context_infos[pipeline_keys[last_idx]]
                # get the last job_infos
                history_job_infos = history_job_infos[-1]
                key_in_history_parts = key_in_history_parts[1:]
            else:
                # get the target job_infos according to pipeline_name and job meta_name
                # get the latest infos
                if len(key_in_history_parts) <= 2:
                    raise ValueError(f'Need to specify the job result keys precisely for inputs to '
                                     f'find the target values in addition to the pipeline name and meta name of jobs.'
                                     f'Only got [{key_in_history}].')
                pipeline_name = key_in_history_parts[0]
                meta_name = key_in_history_parts[1]
                job_info_list = context_infos[pipeline_name]
                for job_info in job_info_list:
                    if job_info['meta_name'] == meta_name:
                        history_job_infos = job_info
                        key_in_history_parts = key_in_history_parts[2:]
                        break
            # check which config group to update
            cfg_type = key_to_updated_parts[0]
            if cfg_type == JobRequiredKeys.dj_configs.value:
                target_config = self.dj_cfg
            elif cfg_type == JobRequiredKeys.extra_configs.value:
                target_config = self.extra_cfg
            else:
                raise ValueError(f'The key {key_to_updated_parts[0]} to update is not supported.')
            key_to_updated_parts = key_to_updated_parts[1:]
            # query target values
            target_value = nested_query(history_job_infos, '.'.join(key_in_history_parts))
            # update the target key
            if len(key_to_updated_parts) > 0:
                if len(key_to_updated_parts) > 1:
                    target_config = nested_query(target_config, '.'.join(key_to_updated_parts[:-1]))
                target_config[key_to_updated_parts[-1]] = target_value
            else:
                if cfg_type == JobRequiredKeys.dj_configs.value:
                    self.dj_cfg = target_value
                elif cfg_type == JobRequiredKeys.extra_configs.value:
                    self.extra_cfg = target_value

        if self.dj_cfg != prev_dj_cfg or self.extra_cfg != prev_extra_cfg:
            logger.info('Configs are updated according to input and local settings. Re-initializing them...')
            self.specify_dj_and_extra_configs()

    def _output_recording_hook(self, outputs):
        if isinstance(outputs, tuple):
            # multiple outputs
            outputs = list(outputs)
        else:
            # single output
            if isinstance(outputs, list) and len(outputs) == 1:
                # if there is only 1 ele in the returned list, unpack it
                outputs = outputs[0]
            outputs = [outputs]
        if self.output_keys is None:
            self.output_keys = list(range(len(outputs)))
        if len(outputs) != len(self.output_keys):
            raise ValueError(
                f'## HOOK [{self.hook_type}]: The number of outputs does not match the number of output keys. '
                f'Expected {len(self.output_keys)} but got {len(outputs)}'
            )
        curr_job_infos = {'meta_name': self.meta_name}
        for key, ret in zip(self.output_keys, outputs):
            curr_job_infos[key] = ret
        return curr_job_infos

    def hook(self, **kwargs):
        raise NotImplementedError

    def specify_dj_and_extra_configs(self, allow_fail=False):
        # prepare data-juicer config
        if self.dj_cfg:
            prev_dj_cfg = self.dj_cfg
            prev_inited_dj_cfg = self.inited_dj_cfg
            try:
                logger.info('Parsing Data-Juicer configs in the job.')
                self.dj_cfg = prepare_side_configs(self.dj_cfg)
                # require Data-Juicer data process in some jobs
                # so we need to init the Data-Juicer data process configs
                self.inited_dj_cfg = get_init_configs(self.dj_cfg)
                self.dj_cfg = dict_to_namespace(self.dj_cfg)
            except Exception as e:
                if allow_fail:
                    self.dj_cfg = prev_dj_cfg
                    self.inited_dj_cfg = prev_inited_dj_cfg
                else:
                    raise e

        # prepare extra configs
        if self.extra_cfg:
            prev_extra_cfg = self.extra_cfg
            try:
                logger.info('Parsing extra configs in the job.')
                self.extra_cfg = prepare_side_configs(self.extra_cfg)
                self.extra_cfg = dict_to_namespace(self.extra_cfg)
            except Exception as e:
                if allow_fail:
                    self.extra_cfg = prev_extra_cfg
                else:
                    raise e


class ProbeViaAnalyzerHook(BaseHook):
    """
    The hook to probe dataset via Data-Juicer Analyzer.

    Input:
        - A data-juicer config.
    Output:
        - the path to export the analyzed dataset.
    """

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for probing the data via Analyzer

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(ProbeViaAnalyzerHook, self).__init__(job_cfg, watcher, *args,
                                                   **kwargs)

    def hook(self, **kwargs):
        analyzer = data_analyzer_factory(self.inited_dj_cfg)
        # probe the data via Analyzer
        logger.info('Begin to analyze data')
        analyzer.run()
        analyzer_res = analyzer.overall_result
        # drop string rows to avoid unaligned dtypes
        string_rows = ['unique', 'top', 'freq']
        for row_name in string_rows:
            if row_name in analyzer_res.index:
                analyzer_res = analyzer_res.drop(row_name)
        self.watcher.watch(analyzer_res, self.meta_name)
        return analyzer.exporter.export_path


class ProbeViaModelInferHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for probing the data via Model Infer

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(ProbeViaModelInferHook, self).__init__(job_cfg, watcher, *args,
                                                     **kwargs)

    def hook(self, **kwargs):
        data_executor = data_executor_factory(self.inited_dj_cfg)
        model_infer_executor = mode_infer_evaluator_factory(self.extra_cfg)
        # TODO
        # probe the model (calling inference sub-pipeline) based on
        # original data, such that we know what is the "hard" data for
        # the model and how to process the data accordingly
        sampled_data = data_executor.sample_data(
            sample_ratio=self.inited_dj_cfg.data_probe_ratio,
            sample_algo=self.inited_dj_cfg.data_probe_algo,
        )
        res_type, infer_res = model_infer_executor.run(
            model_infer_executor.model_config['type'], sampled_data)
        self.watcher.watch(infer_res, self.meta_name)
        return infer_res


class GeneralProbeHook(BaseHook):
    def __init__(self, job_cfg, watcher, *args, **kwargs):
        super(GeneralProbeHook, self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        data_probe = general_probe_factory(self.extra_cfg)
        logger.info('Begin to probe data.')
        ret = data_probe.run()
        return ret


class RefineRecipeViaKSigmaHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for refining the recipe via K Sigma

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(RefineRecipeViaKSigmaHook,
              self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        path_k_sigma_recipe = self.extra_cfg.path_k_sigma_recipe
        # use k-sigma strategy to modify the data recipe
        modify_recipe_k_sigma(self.dj_cfg, self.watcher.query(self.meta_name),
                              path_k_sigma_recipe)
        return path_k_sigma_recipe


class RefineRecipeViaModelFeedbackHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for refining the recipe via Model Feedback

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(RefineRecipeViaModelFeedbackHook,
              self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        # TODO
        # use model-feedback-based strategy to modify the data recipe,
        # e.g., more mapper on the "hard" or "sensitive" data, those were
        # ranked by user-interested measurement after model inference
        if self.sandbox_cfg.path_model_feedback_recipe is not None:
            # modify_recipe_model_feedback(
            #     self.sandbox_cfg,
            #     self.watcher.query("measure_on_infer_res"),
            #     self.sandbox_cfg.path_model_feedback_recipe)
            raise NotImplementedError('Not implemented yet.')
        return None


class ProcessDataHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for processing the data via Data-Juicer

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(ProcessDataHook, self).__init__(job_cfg, watcher, *args,
                                              **kwargs)

    def hook(self, **kwargs):
        data_executor = data_executor_factory(self.inited_dj_cfg)
        # basic routine to process data, users can customize this freely
        logger.info('Begin to process the data with given dj recipe')
        data_executor.run()
        return self.inited_dj_cfg.export_path


class DataPoolManipulationHook(BaseHook):
    """
    Hook for data pool manipulation, including construction, combination, ranking, etc.
    """

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        super(DataPoolManipulationHook, self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        data_pool_manipulator = data_pool_manipulator_factory(self.extra_cfg)
        logger.info('Begin to manipulate data pools.')
        ret = data_pool_manipulator.run()
        return ret


class GeneralDataExecutorHook(BaseHook):
    def __init__(self, job_cfg, watcher, *args, **kwargs):
        super(GeneralDataExecutorHook, self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        data_executor = general_data_executor_factory(self.extra_cfg)
        logger.info('Begin to execute general data executor.')
        ret = data_executor.run()
        return ret


class TrainModelHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for model training

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(TrainModelHook, self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        # try to update train dataset
        model_trainer = model_train_executor_factory(self.extra_cfg,
                                                     watcher=self.watcher)
        # basic routine to train model via the processed data,
        # users can customize this freely
        logger.info('Begin to train the model with given model config')
        # update training dataset path
        ret = asyncio.run(
            model_trainer.run(model_trainer.model_config['type'], **kwargs))
        return ret


class InferModelHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for model training

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(InferModelHook, self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        model_infer = model_infer_executor_factory(self.extra_cfg,
                                                   watcher=self.watcher)

        logger.info('Begin to infer the model with given model config')
        ret = asyncio.run(model_infer.run(model_infer.model_config['type'],
                                          **kwargs))
        return ret


class EvaluateDataHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for data evaluation

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(EvaluateDataHook, self).__init__(job_cfg, watcher, *args,
                                               **kwargs)

    def hook(self, **kwargs):
        data_evaluator = data_evaluator_factory(self.extra_cfg)
        # basic routine to evaluate the given data,
        # users can customize this freely
        logger.info('Begin to evaluate the data with given evaluator config')
        eval_res = data_evaluator.run(eval_type='data', **kwargs)
        self.watcher.watch(eval_res, self.meta_name)
        return eval_res


class EvaluateModelHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for model evaluation

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(EvaluateModelHook, self).__init__(job_cfg, watcher, *args,
                                                **kwargs)

    def hook(self, **kwargs):
        model_evaluator = model_evaluator_factory(self.extra_cfg)
        # basic routine to evaluate the given model,
        # users can customize this freely
        logger.info('Begin to evaluate the model with given evaluator config')
        eval_res = model_evaluator.run(kwargs)
        self.watcher.watch(eval_res, self.meta_name)
        return eval_res


HOOK_MAPPING = {
    # Data/Model Probe hooks
    'ProbeViaAnalyzerHook': ProbeViaAnalyzerHook,
    'ProbeViaModelInferHook': ProbeViaModelInferHook,
    'GeneralProbeHook': GeneralProbeHook,

    # Data-Recipe Refinement hooks
    'RefineRecipeViaKSigmaHook': RefineRecipeViaKSigmaHook,
    'RefineRecipeViaModelFeedbackHook': RefineRecipeViaModelFeedbackHook,

    # Data/Model Execution hooks
    'ProcessDataHook': ProcessDataHook,
    'DataPoolManipulationHook': DataPoolManipulationHook,
    'GeneralDataExecutorHook': GeneralDataExecutorHook,
    'TrainModelHook': TrainModelHook,
    'InferModelHook': InferModelHook,

    # Evaluation hooks
    'EvaluateDataHook': EvaluateDataHook,
    'EvaluateModelHook': EvaluateModelHook,
}


def register_hook(job_cfg, watcher):
    if job_cfg.hook not in HOOK_MAPPING:
        raise ValueError('Undefined hook: [{job_cfg.hook}].')
    return HOOK_MAPPING[job_cfg.hook](job_cfg, watcher)
