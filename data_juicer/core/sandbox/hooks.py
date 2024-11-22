# yapf: disable
import asyncio

from jsonargparse import dict_to_namespace
from loguru import logger

from data_juicer.config import get_init_configs, prepare_side_configs
from data_juicer.core.sandbox.factories import (data_analyzer_factory,
                                                data_evaluator_factory,
                                                data_executor_factory,
                                                mode_infer_evaluator_factory,
                                                model_evaluator_factory,
                                                model_infer_executor_factory,
                                                model_train_executor_factory)
from data_juicer.utils.constant import JobRequiredKeys
from tools.hpo.execute_hpo_3sigma import modify_recipe_k_sigma


class BaseHook:

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        self.job_cfg = job_cfg
        self.watcher = watcher
        self.meta_name = job_cfg[JobRequiredKeys.meta_name.value]
        self.dj_cfg = job_cfg[JobRequiredKeys.dj_configs.value]
        self.other_cfg = job_cfg[JobRequiredKeys.extra_configs.value]

    def hook(self, **kwargs):
        raise NotImplementedError

    def specify_dj_and_extra_configs(self):
        if self.dj_cfg:
            logger.info('Parsing Data-Juicer configs in the job.')
            self.dj_cfg = prepare_side_configs(self.dj_cfg)
            # require Data-Juicer data process in some jobs
            # so we need to init the Data-Juicer data process configs
            self.inited_dj_cfg = get_init_configs(self.dj_cfg)
            self.dj_cfg = dict_to_namespace(self.dj_cfg)
        else:
            self.inited_dj_cfg = get_init_configs({})
        if self.other_cfg:
            logger.info('Parsing other configs in the job.')
            self.other_cfg = prepare_side_configs(self.other_cfg)
            self.other_cfg = dict_to_namespace(self.other_cfg)


class ProbeViaAnalyzerHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for probing the data via Analyzer

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(ProbeViaAnalyzerHook, self).__init__(job_cfg, watcher, *args,
                                                   **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_extra_configs()
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
        return kwargs


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
        self.specify_dj_and_extra_configs()
        data_executor = data_executor_factory(self.inited_dj_cfg)
        model_infer_executor = mode_infer_evaluator_factory(self.other_cfg)
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
        return kwargs


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
        self.specify_dj_and_extra_configs()
        path_k_sigma_recipe = self.other_cfg.path_k_sigma_recipe
        # use k-sigma strategy to modify the data recipe
        modify_recipe_k_sigma(self.dj_cfg, self.watcher.query(self.meta_name),
                              path_k_sigma_recipe)
        return kwargs


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
        self.specify_dj_and_extra_configs()
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
        return kwargs


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
        self.specify_dj_and_extra_configs()
        data_executor = data_executor_factory(self.inited_dj_cfg)
        # basic routine to process data, users can customize this freely
        logger.info('Begin to process the data with given dj recipe')
        data_executor.run()
        kwargs['dataset_path'] = self.inited_dj_cfg.export_path
        return kwargs


class TrainModelHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for model training

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(TrainModelHook, self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_extra_configs()
        # try to update train dataset
        if 'dataset_path' in kwargs:
            self.other_cfg['train_dataset'] = kwargs['dataset_path']
        model_trainer = model_train_executor_factory(self.other_cfg,
                                                     watcher=self.watcher)
        # basic routine to train model via the processed data,
        # users can customize this freely
        logger.info('Begin to train the model with given model config')
        # update training dataset path
        asyncio.run(
            model_trainer.run(model_trainer.model_config['type'], **kwargs))
        return kwargs


class InferModelHook(BaseHook):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hook for model training

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(InferModelHook, self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_extra_configs()
        model_infer = model_infer_executor_factory(self.other_cfg,
                                                   watcher=self.watcher)

        logger.info('Begin to infer the model with given model config')
        asyncio.run(model_infer.run(model_infer.model_config['type'],
                                    **kwargs))
        return kwargs


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
        self.specify_dj_and_extra_configs()
        data_evaluator = data_evaluator_factory(self.other_cfg)
        # basic routine to evaluate the given data,
        # users can customize this freely
        logger.info('Begin to evaluate the data with given evaluator config')
        eval_res = data_evaluator.run(eval_type='data', **kwargs)
        self.watcher.watch(eval_res, self.meta_name)
        return kwargs


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
        self.specify_dj_and_extra_configs()
        model_evaluator = model_evaluator_factory(self.other_cfg)
        # basic routine to evaluate the given model,
        # users can customize this freely
        logger.info('Begin to evaluate the model with given evaluator config')
        model_evaluator.run(kwargs)
        return kwargs


HOOK_MAPPING = {
    'ProbeViaAnalyzerHook': ProbeViaAnalyzerHook,
    'ProbeViaModelInferHook': ProbeViaModelInferHook,
    'RefineRecipeViaKSigmaHook': RefineRecipeViaKSigmaHook,
    'RefineRecipeViaModelFeedbackHook': RefineRecipeViaModelFeedbackHook,
    'ProcessDataHook': ProcessDataHook,
    'TrainModelHook': TrainModelHook,
    'InferModelHook': InferModelHook,
    'EvaluateDataHook': EvaluateDataHook,
    'EvaluateModelHook': EvaluateModelHook,
}


def register_hook(job_cfg, watcher):
    if job_cfg.hook not in HOOK_MAPPING:
        raise ValueError('Undefined hook: [{job_cfg.hook}].')
    return HOOK_MAPPING[job_cfg.hook](job_cfg, watcher)
