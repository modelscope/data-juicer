from utils import (
    call_data_juicer_api,
    init_op_config,
    execute_analyzer,
    show_analyzed_results,
    execute_filter
)

from agentscope.service import (
    ServiceResponse,
    ServiceExecStatus,
)

_model_dir = ''
aesthetics_model_path = _model_dir + 'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'
nsfw_model_path = _model_dir + 'Falconsai/nsfw_image_detection'


def execute_alphabet_or_numeric_filter(dataset_path: str) -> ServiceResponse:
    """
    Filter text with alphabet/numeric ratio out of specific range.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'alphanumeric_filter')
        analyzed_result_path = execute_analyzer(dj_config)
        min_th, max_th = show_analyzed_results(analyzed_result_path)
        dj_config = init_op_config(analyzed_result_path, 'alphanumeric_filter', min_ratio=min_th, max_ratio=max_th)
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Filtered dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)   

    
def execute_text_length_filter(dataset_path: str) -> ServiceResponse:
    """
    Filter text with length out of specific range.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'text_length_filter')
        analyzed_result_path = execute_analyzer(dj_config)
        min_th, max_th = show_analyzed_results(analyzed_result_path)
        dj_config = init_op_config(analyzed_result_path, 'text_length_filter', min_len=int(min_th), max_len=int(max_th))
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Filtered dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_image_aesthetics_filter(dataset_path: str) -> ServiceResponse:
    """
    Filter samples according to the aesthetics score of images.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'image_aesthetics_filter', hf_scorer_model=aesthetics_model_path)
        analyzed_result_path = execute_analyzer(dj_config)
        min_th, max_th = show_analyzed_results(analyzed_result_path)
        dj_config = init_op_config(analyzed_result_path, 'image_aesthetics_filter', min_score=min_th, max_score=max_th, hf_scorer_model=aesthetics_model_path)
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Filtered dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_video_aesthetics_filter(dataset_path: str) -> ServiceResponse:
    """
    Filter samples according to the aesthetics score of frame images extracted from videos.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'video_aesthetics_filter', hf_scorer_model=aesthetics_model_path)
        analyzed_result_path = execute_analyzer(dj_config)
        min_th, max_th = show_analyzed_results(analyzed_result_path)
        dj_config = init_op_config(analyzed_result_path, 'video_aesthetics_filter', min_score=min_th, max_score=max_th, hf_scorer_model=aesthetics_model_path)
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Filtered dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_image_nsfw_filter(dataset_path: str) -> ServiceResponse:
    """
    Filter samples according to the nsfw scores of images in them.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'image_nsfw_filter', hf_nsfw_model=nsfw_model_path)
        analyzed_result_path = execute_analyzer(dj_config)
        min_th, max_th = show_analyzed_results(analyzed_result_path, require_min=False)
        dj_config = init_op_config(analyzed_result_path, 'image_nsfw_filter', score_threshold=max_th, hf_nsfw_model=nsfw_model_path)
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Filtered dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_video_nsfw_filter(dataset_path: str) -> ServiceResponse:
    """
    Filter samples according to the nsfw scores of videos in them.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'video_nsfw_filter', hf_nsfw_model=nsfw_model_path)
        analyzed_result_path = execute_analyzer(dj_config)
        min_th, max_th = show_analyzed_results(analyzed_result_path, require_min=False)
        dj_config = init_op_config(analyzed_result_path, 'video_nsfw_filter', score_threshold=max_th, hf_nsfw_model=nsfw_model_path, frame_sampling_method='uniform')
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Filtered dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)