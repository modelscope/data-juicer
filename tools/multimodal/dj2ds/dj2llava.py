# This tool is used to convert multimodal dataset in LLaVA format to a target
# dataset in Data-Juicer format.
#
# Corresponding Data-Juicer format:
#   - multi-chunk interleaved image-text sequence
#   - in jsonl
# {'id': '000000033471',
#  'images': ['coco/train2017/000000033471.jpg'],
#  'text': '[[human]]: <image>\n'
#          'What are the colors of the bus in the image?\n'
#          '[[gpt]]: The bus in the image is white and red.\n'
#          '[[human]]: What feature can be seen on the back of the bus?\n'
#          '[[gpt]]: The back of the bus features an advertisement.\n'
#          '[[human]]: Is the bus driving down the street or pulled off to'
#          'the side?\n'
#          '[[gpt]]: The bus is driving down the street, which is crowded '
#          'with people and other vehicles. <|__dj__eoc|>'}
#
# LLaVA format:
#   - single/multi-turn conversation
#   - in json
# [
#   {
#     "id": "000000033471",
#     "image": "coco/train2017/000000033471.jpg",
#     "conversations": [
#       {
#         "from": "human",
#         "value": "<image>\nWhat are the colors of the bus in the image?"
#       },
#       {
#         "from": "gpt",
#         "value": "The bus in the image is white and red."
#       },
#       {
#         "from": "human",
#         "value": "What feature can be seen on the back of the bus?"
#       },
#       {
#         "from": "gpt",
#         "value": "The back of the bus features an advertisement."
#       },
#       {
#         "from": "human",
#         "value": "Is the bus driving down the street or pulled off to the side?"
#       },
#       {
#         "from": "gpt",
#         "value": "The bus is driving down the street, which is crowded with people and other vehicles."
#       }
#     ]
#   },
#   ...
# ]
#
# Reference:
# https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md

import os
import fire
import json
import jsonlines as jl
import regex as re

from tqdm import tqdm
from loguru import logger

from data_juicer.utils.mm_utils import SpecialTokens

@logger.catch
def main(
        dj_ds_path: str,
        target_llava_ds_path: str,
        keep_only_first_image: bool = True,
        eoc_special_token: str = SpecialTokens.eoc,
        image_special_token: str = '<image>',
        sent_seperator: str = '\n'
):
    """
    Convert a Data-Juicer-format dataset to a LLaVA-like dataset.

    :param dj_ds_path: path to the input dataset in Data-Juicer format.
    :param target_llava_ds_path: path to store the converted dataset in LLaVA
        format.
    :param keep_only_first_image: whether to only keep the image token in the
        first conversation round. Default: True.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split conversation chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param image_special_token: the special token for images. It's used to
        locate the images in the conversation. In typical LLaVA-like datasets,
        this token always be "<image>". You can change it to align with your
        own LLaVA-like datasets but should be careful of possible compatibility
        problems that come from this change. Default: <image>.
    :param sent_seperator: seperator to split different sentences. Default: \n.
    """
    # ----- Constant settings. Better not to change them. -----
    # default key of field to store the sample text
    text_key = 'text'
    # default key of field to store the image list
    image_key = 'images'
    # default pattern for the conversation role
    from_pattern = re.compile(r'\[\[([a-zA-Z]*?)\]\]: ')
    # ----- Constant settings. Better not to change them. -----
    # check arguments
    # check paths
    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(
            f'Input dataset [{dj_ds_path}] can not be found.')
    if not target_llava_ds_path.endswith('.json'):
        raise ValueError(
            f'Only support "json" target dataset file for LLaVA now.')
    if os.path.dirname(target_llava_ds_path) \
            and not os.path.exists(os.path.dirname(target_llava_ds_path)):
        logger.info(
            f'Create directory [{os.path.dirname(target_llava_ds_path)}] for '
            f'the target dataset.')
        os.makedirs(os.path.dirname(target_llava_ds_path))

    # check if the default image special token is changed
    if image_special_token != '<image>':
        logger.warning(f'The image_special_token used in the original LLaVA '
                       f'dataset is "<image>". It\'s better to align the this '
                       f'token. There might be some compatibility problem if '
                       f'you change it.')

    logger.info('Start to convert.')
    samples = []
    with jl.open(dj_ds_path, 'r') as reader:
        for sample in tqdm(reader):
            id = sample['id']
            images = list(set(sample.get(image_key, [])))
            text = sample[text_key]

            if len(images) > 1:
                raise ValueError(f'There are more than 1 distinct images in '
                                 f'the sample with id [{id}], which is not '
                                 f'compatible with LLaVA dataset format. '
                                 f'Please check and fix it and retry.')

            # convert dj text format to LLaVA conversation format
            # split the text into a list of:
            # [role1, sent1, role2, sent2, role1, sent3, role2, sent4, ...]
            parts = from_pattern.split(text)
            if parts[0] == '':
                parts = parts[1:]
            if len(parts) % 4 != 0:
                raise ValueError(f'The conversations in the sample text with '
                                 f'id [{id}] contains unbalance (human, '
                                 f'robot) conversation round (number of '
                                 f'conversation is [{len(parts)}]). Please '
                                 f'check and fix the dataset and retry.')

            conversations = []
            # the number of sentences
            num_sent = len(parts) // 2
            for i in range(num_sent):
                # get role and its sentence
                role = parts[2 * i]
                sent = parts[2 * i + 1].strip()

                # remove sentence seperator
                if sent.endswith(sent_seperator):
                    sent = sent[:-len(sent_seperator)].strip()
                # remove possible eoc_special_tokens
                if sent.endswith(eoc_special_token):
                    sent = sent[:-len(eoc_special_token)].strip()
                # remove possible image_special_tokens when only keeping it in
                # the first conversation round
                if i > 0 and keep_only_first_image:
                    if sent.startswith(image_special_token):
                        sent = sent[len(image_special_token):].strip()
                        if sent.startswith(sent_seperator):
                            sent = sent[len(sent_seperator):].strip()
                    if sent.endswith(image_special_token):
                        sent = sent[:-len(image_special_token)].strip()
                        if sent.endswith(sent_seperator):
                            sent = sent[:-len(sent_seperator)].strip()

                conversation = {
                    'from': role,
                    'value': sent
                }
                conversations.append(conversation)

            # make up the new sample
            new_sample = {
                'id': id,
                'conversations': conversations
            }
            if len(images) == 1:
                new_sample['image'] = images[0]
            samples.append(new_sample)

    logger.info(f'Start to write the converted dataset to '
                f'[{target_llava_ds_path}]...')
    json.dump(samples, open(target_llava_ds_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    fire.Fire(main)
