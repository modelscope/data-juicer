from data_juicer.core import Analyzer as DJAnalyzer
from data_juicer.core.executor import DefaultExecutor as DJExecutor
from data_juicer.core.sandbox.data_pool_manipulators import (
    DataPoolCombination,
    DataPoolConstruction,
    DataPoolDownsampling,
    DataPoolDuplication,
    DataPoolRanking,
)
from data_juicer.core.sandbox.evaluators import Gpt3QualityEvaluator, InceptionEvaluator
from data_juicer.core.sandbox.model_executors import (
    ModelscopeInferProbeExecutor,
    ModelscopeTrainExecutor,
)


class DataExecutorFactory(object):
    """
    Factory for Data-Juicer executor. Require configs for Data-Juicer and return a Data-Juicer executor.
    """

    def __call__(self, dj_cfg: dict = None, *args, **kwargs):
        if dj_cfg is None:
            return None

        return DJExecutor(dj_cfg)


data_executor_factory = DataExecutorFactory()


class DataAnalyzerFactory(object):
    """
    Factory for Data-Juicer analyzer. Require configs for Data-Juicer and return a Data-Juicer analyzer.
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
        if eval_cfg.type == "vbench_video_evaluator":
            from data_juicer.core.sandbox.specific_hooks.text_to_video.model_hooks import (
                VBenchEvaluator,
            )

            evaluator = VBenchEvaluator(eval_cfg)
        elif eval_cfg.type == "video_inception_evaluator":
            evaluator = InceptionEvaluator(eval_cfg)
        elif eval_cfg.type == "dj_text_quality_classifier":
            evaluator = Gpt3QualityEvaluator(eval_cfg)

        return evaluator


data_evaluator_factory = DataEvaluatorFactory()


class GeneralProbeFactory(object):
    def __call__(self, probe_cfg: dict = None, *args, **kwargs):
        if probe_cfg is None:
            return None

        probe = None
        if probe_cfg.type == "data_pool_ranking":
            probe = DataPoolRanking(probe_cfg)

        return probe


general_probe_factory = GeneralProbeFactory()


class DataPoolManipulatorFactory(object):
    def __call__(self, data_pool_cfg: dict = None, *args, **kwargs):
        if data_pool_cfg is None:
            return None

        manipulator = None
        if data_pool_cfg.type == "data_pool_construction":
            manipulator = DataPoolConstruction(data_pool_cfg)
        elif data_pool_cfg.type == "data_pool_combination":
            manipulator = DataPoolCombination(data_pool_cfg)
        elif data_pool_cfg.type == "data_pool_duplication":
            manipulator = DataPoolDuplication(data_pool_cfg)
        elif data_pool_cfg.type == "data_pool_downsampling":
            manipulator = DataPoolDownsampling(data_pool_cfg)

        return manipulator


data_pool_manipulator_factory = DataPoolManipulatorFactory()


class GeneralDataExecutorFactory(object):
    def __call__(self, data_exec_cfg: dict = None, *args, **kwargs):
        if data_exec_cfg is None:
            return None

        executor = None
        if data_exec_cfg.type == "coco_caption_to_dj_conversion":
            from data_juicer.core.sandbox.specific_hooks.intervl_coco_captioning.preparation_hooks import (
                COCOCaptionToDJConversion,
            )

            executor = COCOCaptionToDJConversion(data_exec_cfg)
        elif data_exec_cfg.type == "coco_caption_meta_generation":
            from data_juicer.core.sandbox.specific_hooks.intervl_coco_captioning.preparation_hooks import (
                COCOCaptionMetaGeneration,
            )

            executor = COCOCaptionMetaGeneration(data_exec_cfg)
        elif data_exec_cfg.type == "dj_to_easyanimate_video_dataset_conversion":
            from data_juicer.core.sandbox.specific_hooks.text_to_video.preparation_hooks import (
                DJToEasyAnimateVideoConversion,
            )

            executor = DJToEasyAnimateVideoConversion(data_exec_cfg)

        return executor


general_data_executor_factory = GeneralDataExecutorFactory()


class ModelEvaluatorFactory(object):
    """
    Factory for model evaluators, whose input is expected to be a loaded model
        and an (optional) instance of data-juicer's dataset.
        It will evaluate the model with specified measurements.
    """

    def __call__(self, eval_cfg: dict = None, *args, **kwargs):
        if eval_cfg is None:
            return None

        evaluator = None
        if eval_cfg.type == "internvl_coco_caption":
            from data_juicer.core.sandbox.specific_hooks.intervl_coco_captioning.model_hooks import (
                InternVLCOCOCaptionEvaluator,
            )

            evaluator = InternVLCOCOCaptionEvaluator(eval_cfg)

        return evaluator


model_evaluator_factory = ModelEvaluatorFactory()


class ModelInferEvaluatorFactory(object):
    def __call__(self, model_cfg: dict = None, *args, **kwargs):
        if model_cfg is None:
            return None

        if model_cfg.type == "modelscope":
            return ModelscopeInferProbeExecutor(model_cfg)

        # add more model inference here freely


mode_infer_evaluator_factory = ModelInferEvaluatorFactory()


class ModelTrainExecutorFactory(object):
    def __call__(self, model_cfg: dict = None, *args, **kwargs):
        if model_cfg is None:
            return None

        trainer = None
        if model_cfg.type == "modelscope":
            trainer = ModelscopeTrainExecutor(model_cfg, **kwargs)
        elif model_cfg.type == "easyanimate":
            from data_juicer.core.sandbox.specific_hooks.text_to_video.model_hooks import (
                EasyAnimateTrainExecutor,
            )

            trainer = EasyAnimateTrainExecutor(model_cfg, **kwargs)
        elif model_cfg.type == "trinity-rft":
            from data_juicer.core.sandbox.specific_hooks.rft.model_hooks import (
                TrinityRFTTrainExecutor,
            )

            trainer = TrinityRFTTrainExecutor(model_cfg, **kwargs)
        elif model_cfg.type == "internvl_coco_caption":
            from data_juicer.core.sandbox.specific_hooks.intervl_coco_captioning.model_hooks import (
                InternVLCOCOCaptionTrainExecutor,
            )

            trainer = InternVLCOCOCaptionTrainExecutor(model_cfg, **kwargs)

        return trainer


model_train_executor_factory = ModelTrainExecutorFactory()


class ModelInferExecutorFactory(object):
    def __call__(self, generate_cfg: dict = None, *args, **kwargs):
        if generate_cfg is None:
            return None

        if generate_cfg.type == "easyanimate":
            from data_juicer.core.sandbox.specific_hooks.text_to_video.model_hooks import (
                EasyAnimateInferExecutor,
            )

            return EasyAnimateInferExecutor(generate_cfg, **kwargs)

        # add more data generation here freely


model_infer_executor_factory = ModelInferExecutorFactory()
