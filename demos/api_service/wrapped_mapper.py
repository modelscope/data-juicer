from utils import (
    call_data_juicer_api,
    init_op_config,
    execute_filter
)

from agentscope.service import (
    ServiceResponse,
    ServiceExecStatus,
)

_model_dir = ''
diffusion_model_path = _model_dir + 'CompVis/stable-diffusion-v1-4'
img2seq_model_path = _model_dir + 'Salesforce/blip2-opt-2.7b'
video_blip_model_path = _model_dir + 'kpyu/video-blip-opt-2.7b-ego4d'


def execute_image_caption_mapper(dataset_path: str) -> ServiceResponse:
    """
    Give captions to the images in the data.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'image_captioning_mapper', hf_img2seq=img2seq_model_path, keep_original_sample=False, mem_required='16GB')
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_image_diffusion_mapper(dataset_path: str) -> ServiceResponse:
    """
    Produce images according texts in the data.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'image_diffusion_mapper', hf_diffusion=diffusion_model_path, keep_original_sample=False, caption_key='text', mem_required='8GB')
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_image_face_blur_mapper(dataset_path: str) -> ServiceResponse:
    """
    Blur faces detected in images of the data.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'image_face_blur_mapper', keep_original_sample=False)
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_video_caption_mapper(dataset_path: str) -> ServiceResponse:
    """
    Give captions to the videos in the data.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'video_captioning_from_video_mapper', keep_original_sample=False, hf_video_blip=video_blip_model_path, mem_required='20GB')
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)


def execute_video_face_blur_mapper(dataset_path: str) -> ServiceResponse:
    """
    Give captions to the videos in the data.

    Args:
        dataset_path (`str`):
            The input dataset path.
    """
    try:
        dj_config = init_op_config(dataset_path, 'video_face_blur_mapper', keep_original_sample=False)
        result_path = execute_filter(dj_config)
        return ServiceResponse(ServiceExecStatus.SUCCESS, f"Mapped dataset path: {result_path}")
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, e)
