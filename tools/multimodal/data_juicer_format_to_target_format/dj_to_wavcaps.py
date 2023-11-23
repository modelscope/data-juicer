# This tool is used to convert multimodal dataset in Data-Juicer format to a
# target dataset in WavCaps format.
#
# Data-Juicer format:
# {'audios': ['./path/to/audio/2219.flac'],
#  'text': '<__dj__audio>\n'
#          'An airplane is landing. <|__dj__eoc|>',
#  '__dj__meta__': {
#       'num_captions_per_audio': 1,
#       'title': 'Airplane Landing Airport',
#       'description': 'Large commercial airplane landing at an airport runway.',  # noqa: E501
#       'author': 'Daniel Simion',
#       'href': '2219-Airplane-Landing-Airport.html',
#       'caption': 'An airplane is landing.',
#       'id': '2219',
#       'duration': 14.1424375,
#       'audio': 'wav_path',
#       'download_link': 'http://soundbible.com/grab.php?id=2219&type=wav',
#       'category': '',
#       'tags': '' }}
# {'audios': ['./path/to/audio/2218.flac'],
#  'text': '<__dj__audio>\n'
#          'Someone is ringing a bell. <|__dj__eoc|>',
#  '__dj__meta__': {
#       'num_captions_per_audio': 1,
#       'title': 'Service Bell Help',
#       'description': 'Customer ringing service bell in need of help in a store.',  # noqa: E501
#       'author': 'Daniel Simion',
#       'href': '2218-Service-Bell-Help.html',
#       'caption': 'Someone is ringing a bell.',
#       'id': '2218',
#       'duration': 1.5698125,
#       'audio': 'wav_path',
#       'download_link': 'http://soundbible.com/grab.php?id=2218&type=wav',
#       'category': '',
#       'tags': '' }}
#
# Corresponding WavCps format:
# { 'num_captions_per_audio': 1,
#   'data': [{
#       'title': 'Airplane Landing Airport',
#       'description': 'Large commercial airplane landing at an airport runway.',  # noqa: E501
#       'author': 'Daniel Simion',
#       'href': '2219-Airplane-Landing-Airport.html',
#       'caption': 'An airplane is landing.',
#       'id': '2219',
#       'duration': 14.1424375,
#       'audio': 'wav_path',
#       'download_link': 'http://soundbible.com/grab.php?id=2219&type=wav'
#   },  {
#       'title': 'Service Bell Help',
#       'description': 'Customer ringing service bell in need of help in a store.',  # noqa: E501
#       'author': 'Daniel Simion',
#       'href': '2218-Service-Bell-Help.html',
#       'caption': 'Someone is ringing a bell.',
#       'id': '2218',
#       'duration': 1.5698125,
#       'audio': 'wav_path',
#       'download_link': 'http://soundbible.com/grab.php?id=2218&type=wav'
#   },
#   ...]
# }

import json
import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.constant import Fields


@logger.catch
def main(dj_ds_path: str, target_wavcaps_ds_path: str):
    """
    Convert a Data-Juicer-format dataset to a WavCaps-like dataset.

    :param dj_ds_path: path to the input dataset in Data-Juicer format.
    :param target_wavcaps_ds_path: path to store the converted dataset in
        WavCaps format.
    """

    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(
            f'Input dataset [{dj_ds_path}] can not be found.')
    if not target_wavcaps_ds_path.endswith('.json'):
        raise ValueError(
            'Only support "json" target dataset file for WavCaps now.')
    if os.path.dirname(target_wavcaps_ds_path) \
            and not os.path.exists(os.path.dirname(target_wavcaps_ds_path)):
        logger.info(
            f'Create directory [{os.path.dirname(target_wavcaps_ds_path)}] '
            f'for the target dataset.')
        os.makedirs(os.path.dirname(target_wavcaps_ds_path))

    logger.info('Start to convert.')
    samples = {'num_captions_per_audio': 1, 'data': []}
    with jl.open(dj_ds_path, 'r') as reader:
        for sample in tqdm(reader):
            if Fields.meta not in sample:
                logger.warning(f'{Fields.meta} does not exist in this sample.')
                continue
            else:
                samples['num_captions_per_audio'] = sample[
                    Fields.meta]['num_captions_per_audio']
                del sample[Fields.meta]['num_captions_per_audio']
                samples['data'].append(sample[Fields.meta])

    logger.info(f'Start to write the converted dataset to '
                f'[{target_wavcaps_ds_path}]...')
    json.dump(samples, open(target_wavcaps_ds_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    fire.Fire(main)
