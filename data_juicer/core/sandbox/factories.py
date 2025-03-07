from data_juicer.core import Analyzer as DJAnalyzer
from data_juicer.core.executor import DefaultExecutor as DJExecutor
from data_juicer.core.sandbox.evaluators import (Gpt3QualityEvaluator,
                                                 InceptionEvaluator,
                                                 VBenchEvaluator)
from data_juicer.core.sandbox.model_executors import (
    EasyAnimateInferExecutor, EasyAnimateTrainExecutor,
    ModelscopeInferProbeExecutor, ModelscopeTrainExecutor)


class DataExecutorFactory(object):
    """
    Factory for Data-Juicer executor.
    """

    def __call__(self, dj_cfg: dict = None, *args, **kwargs):
        if dj_cfg is None:
            return None

        return DJExecutor(dj_cfg)


data_executor_factory = DataExecutorFactory()


class DataAnalyzerFactory(object):
    """
    Factory for Data-Juicer analyzer.
    """

    def __call__(self, dj_cfg: dict = None, *args, **kwargs):
        if dj_cfg is None:
            return None

        return DJAnalyzer(dj_cfg)


data_analyzer_factory = DataAnalyzerFactory()


class DataEvaluatorFactory(object):
    """
        Factory for data evaluators, whose input is expected to be an instance
            of data-juicer's dataset.
            It will evaluate these data with specified measurements.
    """

    def __call__(self, eval_cfg: dict = None, *args, **kwargs):
        if eval_cfg is None:
            return None

        evaluator = None
        if eval_cfg.type == 'vbench_video_evaluator':
            evaluator = VBenchEvaluator(eval_cfg)
        elif eval_cfg.type == 'video_inception_evaluator':
            evaluator = InceptionEvaluator(eval_cfg)
        elif eval_cfg.type == 'dj_text_quality_classifier':
            evaluator = Gpt3QualityEvaluator(eval_cfg)

        return evaluator


data_evaluator_factory = DataEvaluatorFactory()


class ModelEvaluatorFactory(object):
    """
    Factory for model evaluators, whose input is expected to be a loaded model
        and an (optional) instance of data-juicer's dataset.
        It will evaluate the model with specified measurements.
    """

    def __call__(self, eval_cfg: dict = None, *args, **kwargs):
        if eval_cfg is None:
            return None

        pass


model_evaluator_factory = ModelEvaluatorFactory()


class ModelInferEvaluatorFactory(object):

    def __call__(self, model_cfg: dict = None, *args, **kwargs):
        if model_cfg is None:
            return None

        if model_cfg.type == 'modelscope':
            return ModelscopeInferProbeExecutor(model_cfg)

        # add more model inference here freely


mode_infer_evaluator_factory = ModelInferEvaluatorFactory()


class ModelTrainExecutorFactory(object):

    def __call__(self, model_cfg: dict = None, *args, **kwargs):
        if model_cfg is None:
            return None

        if model_cfg.type == 'modelscope':
            return ModelscopeTrainExecutor(model_cfg, **kwargs)
        elif model_cfg.type == 'easyanimate':
            return EasyAnimateTrainExecutor(model_cfg, **kwargs)

        # add more model trainer here freely


model_train_executor_factory = ModelTrainExecutorFactory()


class ModelInferExecutorFactory(object):

    def __call__(self, generate_cfg: dict = None, *args, **kwargs):
        if generate_cfg is None:
            return None

        if generate_cfg.type == 'easyanimate':
            return EasyAnimateInferExecutor(generate_cfg, **kwargs)

        # add more data generation here freely


model_infer_executor_factory = ModelInferExecutorFactory()
