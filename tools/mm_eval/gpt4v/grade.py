import configparser
import json
from typing import Any, Optional

import fire
import tqdm

from data_juicer.utils.mm_utils import (image_path_to_base64,
                                        video_path_to_base64)
from tools.mm_eval.gpt4v.lib import (grade_image_to_text, grade_text_to_image,
                                     grade_text_to_video, grade_video_to_text)


def _construct_text_eval(score, rationale):
    result = {'score': score, 'rationale': rationale}
    return json.dumps(result)


def _construct_image_eval(score, rationale):
    result = {
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
    return json.dumps(result)


def _construct_video_eval(score, rationale):
    result = {
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
    return json.dumps(result)


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
        result = result_raw

    return json.dumps(result)


def image_to_text(input: str, output: str, **kwargs: Any):
    expected_keys = ['image', 'text']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm.tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = _check_missing_keys(entry, expected_keys)
            if missing:
                result = _construct_text_eval(
                    0, f"Missing keys: {' '.join(missing)}")
            else:
                image = image_path_to_base64(entry['image'], mime_type='auto')
                text = entry['text'].strip()
                result_raw = grade_image_to_text(image, text, **kwargs)
                if not result_raw:
                    result = _construct_text_eval(0, 'Request error')
                else:
                    result = _parse_ini(result_raw)
            fout.write(result + '\n')


def text_to_image(input: str, output: str, **kwargs: Any):
    expected_keys = ['text', 'image']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm.tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = _check_missing_keys(entry, expected_keys)
            if missing:
                result = _construct_image_eval(
                    0, f"Missing keys: {' '.join(missing)}")
            else:
                text = entry['text'].strip()
                image = image_path_to_base64(entry['image'], mime_type='auto')
                result_raw = grade_text_to_image(text, image, **kwargs)
                if not result_raw:
                    result = _construct_image_eval(0, 'Request error')
                else:
                    result = _parse_ini(result_raw)
            fout.write(result + '\n')


def video_to_text(
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

    expected_keys = ['video', 'text']

    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm.tqdm(lines):
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
                text = entry['text'].strip()
                result_raw = grade_video_to_text(frames, text, **kwargs)
                if not result_raw:
                    result = _construct_text_eval(0, 'Request error')
                else:
                    result = _parse_ini(result_raw)
            fout.write(result + '\n')


# text --> video
def text_to_video(
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

    expected_keys = ['text', 'video']
    with open(input) as fin:
        lines = fin.readlines()

    with open(output, 'w') as fout:
        for line in tqdm.tqdm(lines):
            line = line.strip()
            entry = json.loads(line)
            missing = _check_missing_keys(entry, expected_keys)
            if missing:
                result = _construct_video_eval(
                    0, f"Missing keys: {' '.join(missing)}")
            else:
                text = entry['text'].strip()
                frames = video_path_to_base64(entry['video'],
                                              frame_num=frame_num,
                                              sampling_fps=sampling_fps,
                                              mime_type='auto')
                result_raw = grade_text_to_video(text, frames, **kwargs)
                if not result_raw:
                    result = _construct_video_eval(0, 'Request error')
                else:
                    result = _parse_ini(result_raw)

            fout.write(result + '\n')


if __name__ == '__main__':
    fire.Fire()
