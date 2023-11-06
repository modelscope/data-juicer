from data_juicer.core import Executor
from tools.quality_classifier.predict import predict_score


def get_hpo_objective(obj_name):
    if obj_name == 'quality_score':
        return obj_quality_score
    # elif obj_name == "model_loss":
    #     return obj_model_loss
    # elif obj_name == "downstream_task":
    #     return obj_downstream_task
    # elif obj_name == "synergy_metric":
    #     return obj_synergy_metric
    else:
        raise NotImplementedError(
            f'unsupported objective type in HPO: {obj_name}. '
            f'Please implement it first.')


def obj_quality_score(dj_cfg):
    """
        HPO loop:  cfg --> data --> data score --> cfg --> data --> ...

    :param dj_cfg: specified data recipe (as a search point)
    :return: a data score, after
        1. processing data according to the dj_cfg;
        2. applying a quality classifier
    """

    if dj_cfg.executor_type == 'default':
        executor = Executor(dj_cfg)
    elif dj_cfg.executor_type == 'ray':
        from data_juicer.core.ray_executor import RayExecutor
        executor = RayExecutor(dj_cfg)
    else:
        raise NotImplementedError(
            f'unsupported executor_type: {dj_cfg.executor_type}, '
            f'expected in [`default`, `ray`]', )
    executor.run()

    # calculate and aggregate data score

    # feel free to customize the quality scorer, via the following args
    # [--model <model_path>] \
    # [--tokenizer <tokenizer_type>] \
    # [--keep_method <keep_method>] \
    # [--text_key <text_key>] \
    overall_quality_stats = predict_score(dj_cfg.export_path,
                                          dj_cfg.export_path,
                                          overall_stats=True)

    # by default, using the mean quality score of processed data as final score
    return overall_quality_stats.loc['mean']
