from data_juicer.core.sandbox.evaluators import (Gpt3QualityEvaluator,
                                                 InceptionEvaluator,
                                                 VBenchEvaluator)
from data_juicer.core.sandbox.model_executors import (
    EasyAnimateInferExecutor, EasyAnimateTrainExecutor,
    ModelscopeInferProbeExecutor, ModelscopeTrainExecutor)


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


class ModelInferProberFactory(object):

    def __call__(self, model_cfg: dict = None, *args, **kwargs):
        if model_cfg is None:
            return None

        if model_cfg.type == 'modelscope':
            return ModelscopeInferProbeExecutor(model_cfg)

        # add more model inference here freely


mode_infer_prober_factory = ModelInferProberFactory()


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


class DataProcessorFactory(object):

    def __call__(self, dj_executor, process_type: str = None, *args, **kwargs):
        if process_type is None:
            return None

        if process_type == 'data_juicer_run':
            return dj_executor.run
        elif process_type == 'data_sample':
            return dj_executor.sample_data
        elif process_type == 'divide_by_percentiles':
            return dj_executor.divide_by_percentiles

        # add more data processor here freely


data_processor_factory = DataProcessorFactory()
