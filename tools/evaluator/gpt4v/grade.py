import configparser
import json
from typing import Any, Optional

import fire

from data_juicer.utils.mm_utils import (image_path_to_base64,
                                        video_path_to_base64)

from .lib import (grade_frames_to_caption, grade_image_to_caption,
                  grade_prompt_to_frames, grade_prompt_to_image)


def _construct_text_eval(score, rationale):
    return {'score': score, 'rationale': rationale}


def _construct_image_eval(score, rationale):
    return {
        'relevance': {
            'score': score,
            'rationale': rationale
        },
        'clarity': {
            'score': score,
            'rationale': rationale
        },
        'accuracy': {
            'score': score,
            'rationale': rationale
        },
        'overall': {
            'score': score,
            'rationale': rationale
        },
    }


def _construct_video_eval(score, rationale):
    return {
        'relevance': {
            'score': score,
            'rationale': rationale
        },
        'clarity': {
            'score': score,
            'rationale': rationale
        },
        'coherence': {
            'score': score,
            'rationale': rationale
        },
        'accuracy': {
            'score': score,
            'rationale': rationale
        },
        'overall': {
            'score': score,
            'rationale': rationale
        },
    }


def _check_missing_keys(json_dict, expected_keys):
    missing = []
    for key in expected_keys:
        if key not in json_dict:
            missing.append(key)
    return missing


def _parse_ini(result_raw):
    try:
        config = configparser.ConfigParser()
        config.read_string(result_raw)
        result = {sec: dict(config[sec]) for sec in config.sections()}
    except Exception:
        pass
    finally:
        result = json.dumps(result_raw)
    return result


# image --> text
def image_to_caption(input: str, output: str, **kwargs: Any):
    expected_keys = ['image', 'caption']

    with open(input) as fin, open(output, 'w') as fout:
        for line in fin:
            line = line.strip()
            entry = json.loads(line)
            missing = _check_missing_keys(entry, expected_keys)
            if missing:
                result = _construct_text_eval(
                    0, f"Missing keys: {' '.join(missing)}")
            else:
                image = image_path_to_base64(entry['image'], mime_type='auto')
                caption = entry['caption'].strip()
                result_raw = grade_image_to_caption(image, caption, **kwargs)
                if not result_raw:
                    result = _construct_text_eval(0, 'Request error')
                else:
                    result = _parse_ini(result_raw)
            fout.write(result + '\n')


# text --> image
def prompt_to_image(input: str, output: str, **kwargs: Any):
    expected_keys = ['prompt', 'image']

    with open(input) as fin, open(output, 'w') as fout:
        for line in fin:
            line = line.strip()
            entry = json.loads(line)
            missing = _check_missing_keys(entry, expected_keys)
            if missing:
                result = _construct_image_eval(
                    0, f"Missing keys: {' '.join(missing)}")
            else:
                prompt = entry['prompt'].strip()
                image = image_path_to_base64(entry['image'], mime_type='auto')
                result_raw = grade_prompt_to_image(prompt, image, **kwargs)
                if not result_raw:
                    result = _construct_image_eval(0, 'Request error')
                else:
                    result = _parse_ini(result_raw)
            fout.write(result + '\n')


# video --> text
def video_to_caption(
    input: str,
    output: str,
    *,
    frame_num: Optional[int] = None,
    sampling_fps: Optional[float] = None,
    **kwargs: Any,
):
    if frame_num is not None and sampling_fps is not None:
        raise ValueError("The parameters 'frame_num' and 'sampling_fps' are \
                mutually exclusive; only one may be set.")

    expected_keys = ['video', 'caption']

    with open(input) as fin, open(output, 'w') as fout:
        for line in fin:
            line = line.strip()
            entry = json.loads(line)
            missing = _check_missing_keys(entry, expected_keys)
            if missing:
                result = _construct_text_eval(
                    0, f"Missing keys: {' '.join(missing)}")
            else:
                frames = video_path_to_base64(entry['video'],
                                              frame_num=frame_num,
                                              sampling_fps=sampling_fps,
                                              mime_type='auto')
                caption = entry['caption'].strip()
                result_raw = grade_frames_to_caption(frames, caption, **kwargs)
                if not result_raw:
                    result = _construct_text_eval(0, 'Request error')
                else:
                    result = _parse_ini(result_raw)
            fout.write(result + '\n')


# text --> video
def prompt_to_video(
    input: str,
    output: str,
    *,
    frame_num: Optional[int] = None,
    sampling_fps: Optional[float] = None,
    **kwargs: Any,
):
    if frame_num is not None and sampling_fps is not None:
        raise ValueError("The parameters 'frame_num' and 'sampling_fps' are \
                mutually exclusive; only one may be set.")

    expected_keys = ['prompt', 'video']

    with open(input) as fin, open(output, 'w') as fout:
        for line in fin:
            line = line.strip()
            entry = json.loads(line)
            missing = _check_missing_keys(entry, expected_keys)
            if missing:
                result = _construct_video_eval(
                    0, f"Missing keys: {' '.join(missing)}")
            else:
                prompt = entry['prompt'].strip()
                frames = video_path_to_base64(entry['video'],
                                              frame_num=frame_num,
                                              sampling_fps=sampling_fps,
                                              mime_type='auto')
                result_raw = grade_prompt_to_frames(prompt, frames, **kwargs)
                if not result_raw:
                    result = _construct_video_eval(0, 'Request error')
                else:
                    result = _parse_ini(result_raw)

            fout.write(result + '\n')


if __name__ == '__main__':
    fire.Fire()
