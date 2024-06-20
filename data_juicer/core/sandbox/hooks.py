import asyncio
import os

from jsonargparse import dict_to_namespace
from loguru import logger

from data_juicer.config import get_init_configs, prepare_side_configs
from data_juicer.core import Analyser
from data_juicer.core import Executor as DjExecutor
from data_juicer.core.sandbox.factories import (data_evaluator_factory,
                                                mode_infer_executor_factory,
                                                model_evaluator_factory,
                                                model_train_executor_factory)
from data_juicer.utils.constant import JobRequiredKeys
from tools.hpo.execute_hpo_3sigma import modify_recipe_k_sigma


class BaseHooker:

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        self.job_cfg = job_cfg
        self.watcher = watcher
        self.res_name = job_cfg[JobRequiredKeys.res_name.value]
        self.dj_cfg = job_cfg[JobRequiredKeys.dj_configs.value]
        self.other_cfg = job_cfg[JobRequiredKeys.other_configs.value]

    def hook(self, **kwargs):
        raise NotImplementedError

    def specify_dj_and_other_configs(self):
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


class ProbeViaAnalyserHooker(BaseHooker):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hooker for probing the data via Analyser

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(ProbeViaAnalyserHooker, self).__init__(job_cfg, watcher, *args,
                                                     **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_other_configs()
        analyser = Analyser(self.inited_dj_cfg)
        # probe the data via Analyser
        logger.info('Begin to analyze data')
        analyser.run()
        analyser_res = analyser.overall_result
        # drop string rows to avoid unaligned dtypes
        string_rows = ['unique', 'top', 'freq']
        for row_name in string_rows:
            if row_name in analyser_res.index:
                analyser_res = analyser_res.drop(row_name)
        self.watcher.watch(analyser_res, self.res_name)
        return kwargs


class ProbeViaModelInferHooker(BaseHooker):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hooker for probing the data via Model Infer

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(ProbeViaModelInferHooker, self).__init__(job_cfg, watcher, *args,
                                                       **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_other_configs()
        data_executor = DjExecutor(self.inited_dj_cfg)
        model_infer_executor = mode_infer_executor_factory(self.other_cfg)
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
        self.watcher.watch(infer_res, self.res_name)
        return kwargs


class RefineRecipeViaKSigmaHooker(BaseHooker):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hooker for refining the recipe via K Sigma

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(RefineRecipeViaKSigmaHooker,
              self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_other_configs()
        path_k_sigma_recipe = self.other_cfg.path_k_sigma_recipe
        # use k-sigma strategy to modify the data recipe
        modify_recipe_k_sigma(self.dj_cfg, self.watcher.query(self.res_name),
                              path_k_sigma_recipe)
        return kwargs


class RefineRecipeViaModelFeedbackHooker(BaseHooker):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hooker for refining the recipe via Model Feedback

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(RefineRecipeViaModelFeedbackHooker,
              self).__init__(job_cfg, watcher, *args, **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_other_configs()
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


class ProcessDataHooker(BaseHooker):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hooker for processing the data via Data-Juicer

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(ProcessDataHooker, self).__init__(job_cfg, watcher, *args,
                                                **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_other_configs()
        data_executor = DjExecutor(self.inited_dj_cfg)
        # basic routine to process data, users can customize this freely
        logger.info('Begin to process the data with given dj recipe')
        data_executor.run()
        return kwargs


class TrainModelHooker(BaseHooker):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hooker for model training

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(TrainModelHooker, self).__init__(job_cfg, watcher, *args,
                                               **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_other_configs()
        model_trainer = model_train_executor_factory(self.other_cfg,
                                                     watcher=self.watcher)
        # basic routine to train model via the processed data,
        # users can customize this freely
        logger.info('Begin to train the model with given model config')
        # update training dataset path
        training_args = {
            'train_dataset':
            self.other_cfg.dataset_path,
            'work_dir':
            os.path.join(self.other_cfg.work_dir, 'model_trainer_outputs'),
        }
        asyncio.run(
            model_trainer.run(model_trainer.model_config['type'],
                              training_args, **kwargs))
        return kwargs


class EvaluateDataHooker(BaseHooker):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hooker for data evaluation

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(EvaluateDataHooker, self).__init__(job_cfg, watcher, *args,
                                                 **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_other_configs()
        data_evaluator = data_evaluator_factory(self.other_cfg)
        # basic routine to evaluate the given data,
        # users can customize this freely
        logger.info('Begin to evaluate the data with given evaluator config')
        processed_dataset = self.other_cfg.dataset_path
        eval_res = data_evaluator.run(eval_type='data',
                                      eval_obj=processed_dataset,
                                      **kwargs)
        self.watcher.watch(eval_res, self.res_name)
        return kwargs


class EvaluateModelHooker(BaseHooker):

    def __init__(self, job_cfg, watcher, *args, **kwargs):
        """
        Initialize the hooker for model evaluation

        :param job_cfg: the job configs
        :param watcher: for watching the result
        """
        super(EvaluateModelHooker, self).__init__(job_cfg, watcher, *args,
                                                  **kwargs)

    def hook(self, **kwargs):
        self.specify_dj_and_other_configs()
        model_evaluator = model_evaluator_factory(self.other_cfg)
        # basic routine to evaluate the given model,
        # users can customize this freely
        logger.info('Begin to evaluate the model with given evaluator config')
        model_evaluator.run(kwargs)
        return kwargs


HOOKER_DICT = {
    'ProbeViaAnalyserHooker': ProbeViaAnalyserHooker,
    'ProbeViaModelInferHooker': ProbeViaModelInferHooker,
    'RefineRecipeViaKSigmaHooker': RefineRecipeViaKSigmaHooker,
    'RefineRecipeViaModelFeedbackHooker': RefineRecipeViaModelFeedbackHooker,
    'ProcessDataHooker': ProcessDataHooker,
    'TrainModelHooker': TrainModelHooker,
    'EvaluateDataHooker': EvaluateDataHooker,
    'EvaluateModelHooker': EvaluateModelHooker,
}


def regist_hooker(job_cfg, watcher):
    if job_cfg.hooker not in HOOKER_DICT:
        raise ValueError('Undefined hooker: [{job_cfg.hooker}].')
    return HOOKER_DICT[job_cfg.hooker](job_cfg, watcher)