# This tool is used to split datasets to sub-datasets
# by fast-text lanuage model.

import os

import fire
import pandas as pd
from jsonargparse import Namespace
from loguru import logger

from data_juicer.format import load_formatter
from data_juicer.ops.filter.language_id_score_filter import \
    LanguageIDScoreFilter


def keep_by_lang(sample, lang):
    """
    Keep samples with the specified language.
    :param sample: a sample in dataset
    :param lang: the specified language
    :return: True to keep,  False to discard
    """
    if sample['stats']['lang'] == lang:
        return True
    return False


def main(src_dir,
         target_dir,
         text_keys_to_load=None,
         text_key_to_process='text',
         suffixes=[],
         num_proc=1):
    """
    Load dataset from the source directory, then apply language identification
    using the operation filter called `LanguageIDScoreFilter`,
    finally, split the dataset by language and save it.
    :param src_dir: path to store the dataset.
    :param target_dir: path to store subset files(`jsonl` format)
    :param text_key: key name of field that stores sample text, default `text`:
    :param suffixes: files with suffixes to be loaded, default None
    :param num_proc: number of processes to process dataset, default 1.
    """
    if text_keys_to_load is None:
        text_keys_to_load = ['text']
    # check if the source directory exists.
    if not os.path.exists(src_dir):
        raise ValueError('The raw source data directory does not exist,'
                         ' Please check and retry.')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Note:
    # key name of `"keys_to_load"` in sample will be rename to "text"
    formatter = load_formatter(src_dir,
                               keys_to_load=text_keys_to_load,
                               suffixes=suffixes)
    tmp_cfg = Namespace({'text_key_to_process': text_key_to_process})
    dataset = formatter.load_dataset(num_proc, tmp_cfg)

    op = LanguageIDScoreFilter(text_key=tmp_cfg['text_key_to_process'])

    if 'stats' not in dataset.features:
        # TODO:
        # this is a temp solution,
        # only add stats when calling filter op
        dataset = dataset.add_column(name='stats',
                                     column=[{}] * dataset.num_rows)

    # identify language
    dataset = dataset.map(op.compute_stats, num_proc=num_proc)

    langs = pd.DataFrame(dataset['stats'])['lang']
    unique_langs = list(set(langs))

    logger.info(f'There are {len(dataset)} in dataset')
    logger.info(f'Languages in dataset are {unique_langs}')

    # split and save subset of dataset by language
    for lang in unique_langs:
        ds = dataset.filter(keep_by_lang,
                            num_proc=num_proc,
                            fn_kwargs=dict(lang=lang))

        logger.info(f'There are {len(ds)} with language [{lang}]')
        jsonl_fp = os.path.join(target_dir, lang + '.jsonl')
        ds.to_json(jsonl_fp, force_ascii=False)


if __name__ == '__main__':
    fire.Fire(main)