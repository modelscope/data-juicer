# This tool is used for converting the raw Alpaca-Cot data downloaded
# from Huggingface (ref: https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
# to several jsonl files.

import os
import pathlib
from multiprocessing import Pool

import fire
from datasets import load_dataset
from loguru import logger

meta_dict = {
    'Chain-of-Thought': {  # sub directory
        'Task': 'MT',  # Alpaca-Cot original Task
        'Gen': 'HG',  # Alpaca-Cot original Gen
        'Lang': 'EN/CN',  # Alpaca-Cot original Language
        'Dataset': 'Chain-of-Thought',  # sub directory
        'Multi-round Dialog':
        False,  # whether is Multi-round Dialog data, added by Data-Juicer
        'IFT': True,  # whether is IFT data, added by Data-Juicer
        'SFT': False,  # whether is SFT data, added by Data-Juicer
        'Preference':
        False,  # whether is Preference data, added by Data-Juicer
    },
    'GPT4all': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN',
        'Dataset': 'GPT4all',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': True,
        'Preference': False,
    },
    'GPTeacher': {
        'Task': 'MT',
        'Gen': 'SI',
        'Lang': 'EN',
        'Dataset': 'GPTeacher',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'Guanaco': {
        'Task': 'MT',
        'Gen': 'SI',
        'Lang': 'ML',
        'Dataset': 'Guanaco',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'HC3': {
        'Task': 'TS',
        'Gen': 'MIX',
        'Lang': 'EN/CN',
        'Dataset': 'HC3',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': True,
    },
    'alpaca': {
        'Task': 'MT',
        'Gen': 'SI',
        'Lang': 'EN',
        'Dataset': 'alpaca',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'Natural-Instructions': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'ML',
        'Dataset': 'Natural-Instructions',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'belle_cn': {
        'Task': 'TS/MT',
        'Gen': 'SI',
        'Lang': 'CN',
        'Dataset': 'belle_cn',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'instinwild': {
        'Task': 'MT',
        'Gen': 'SI',
        'Lang': 'EN/CN',
        'Dataset': 'instinwild',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'prosocial-dialog': {
        'Task': 'TS',
        'Gen': 'MIX',
        'Lang': 'EN',
        'Dataset': 'prosocial-dialog',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'finance': {
        'Task': 'TS',
        'Gen': 'COL',
        'Lang': 'EN',
        'Dataset': 'finance',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'xP3': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'ML',
        'Dataset': 'xP3',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'firefly': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'CN',
        'Dataset': 'firefly',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'instruct': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN',
        'Dataset': 'instruct',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'CodeAlpaca': {
        'Task': 'TS',
        'Gen': 'SI',
        'Lang': 'EN',
        'Dataset': 'CodeAlpaca',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'alpacaGPT4': {
        'Task': 'MT',
        'Gen': 'SI',
        'Lang': 'EN/CN',
        'Dataset': 'alpacaGPT4',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': True,
    },
    'webGPT': {
        'Task': 'TS',
        'Gen': 'MIX',
        'Lang': 'EN',
        'Dataset': 'webGPT',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': True,
    },
    'dolly': {
        'Task': 'TS',
        'Gen': 'HG',
        'Lang': 'EN',
        'Dataset': 'dolly',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'baize': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN',
        'Dataset': 'baize',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'hh-rlhf': {
        'Task': 'TS',
        'Gen': 'MIX',
        'Lang': 'EN',
        'Dataset': 'hh-rlhf',
        'Multi-round Dialog': True,
        'IFT': False,
        'SFT': True,
        'Preference': True,
    },
    'OIG': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN',
        'Dataset': 'OIG',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'GAOKAO': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'CN',
        'Dataset': 'GAOKAO',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'camel': {
        'Task': 'MT',
        'Gen': 'SI',
        'Lang': 'EN',
        'Dataset': 'camel',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'FLAN-Muffin': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN',
        'Dataset': 'FLAN-Muffin',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'COIG': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'CN',
        'Dataset': 'COIG',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'gpt4tools': {
        'Task': 'MT',
        'Gen': 'SI',
        'Lang': 'EN',
        'Dataset': 'gpt4tools',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'ShareGPT': {
        'Task': 'MT',
        'Gen': 'MIX',
        'Lang': 'EN',
        'Dataset': 'ShareGPT',
        'Multi-round Dialog': True,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'Auto-CoT': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN',
        'Dataset': 'Auto-CoT',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'MOSS': {
        'Task': 'TS',
        'Gen': 'SI',
        'Lang': 'EN/CN',
        'Dataset': 'MOSS',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'ultrachat': {
        'Task': 'TS',
        'Gen': 'SI',
        'Lang': 'EN',
        'Dataset': 'ultrachat',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'Chinese-medical': {
        'Task': 'TS',
        'Gen': 'COL',
        'Lang': 'CN',
        'Dataset': 'Chinese-medical',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': False,
    },
    'CSL': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'CN',
        'Dataset': 'CSL',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'pCLUE': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'CN',
        'Dataset': 'pCLUE',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'news_commentary': {
        'Task': 'TS',
        'Gen': 'COL',
        'Lang': 'CN',
        'Dataset': 'news_commentary',
        'Multi-round Dialog': False,
        'IFT': True,
        'SFT': False,
        'Preference': False,
    },
    'StackExchange': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN',
        'Dataset': 'StackExchange',
        'Multi-round Dialog': False,
        'IFT': False,
        'SFT': True,
        'Preference': True,
    },
    "ConvAI2": {
        "Task": "TS",
        "Gen": "HG",
        "Lang": "EN",
        "Dataset": "ConvAI2",
        "Multi-round Dialog": False,
        "IFT": False,
        "SFT": True,
        "Preference": False,
    },
    "FastChat": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "EN",
        "Dataset": "FastChat",
        "Multi-round Dialog": False,
        "IFT": False,
        "SFT": True,
        "Preference": False,
    },
    'Tabular-LLM-Data': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN/CN',
        "Dataset": "Tabular-LLM-Data",
        "Multi-round Dialog": False,
        "IFT": True,
        "SFT": False,
        "Preference": False,
    },
    'ThoughtSource': {
        'Task': 'MT',
        'Gen': 'COL',
        'Lang': 'EN',
        "Dataset": "ThoughtSource",
        "Multi-round Dialog": False,
        "IFT": True,
        "SFT": False,
        "Preference": False,
    }
}


def merge_and_add_meta(filename, target_dir):
    """
    Merge `instruction`/`input`/`output` to `text` for process,
    and add meta info.
    :param filename: input dataset file
    :param target_dir: path to save updated dataset
    """

    ds = load_dataset('json', data_files=[filename], split='train')

    if 'instruction' in ds.features and \
       'input' in ds.features and \
       'output' in ds.features:
        for column_name in ds.column_names:
            if column_name not in ['instruction', 'input', 'output']:
                ds = ds.remove_columns(column_name)
    else:
        logger.warning(f'Can not find ["instruction", "input", "output"] in \
             {filename}, do nothing.')
        return

    meta = None
    for key in meta_dict.keys():
        if key in filename:
            meta = meta_dict[key]

    if meta is None:
        logger.warning(f'Can not find meta in {filename}, do nothing.')
        return

    def _merge_and_add_meta(sample, path, meta):
        """
        Merge `instruction`/`input`/`output` to `text` for process,
        and add meta info.
        :param sample: a dict sample in dataset
        :param path: sample in which file
        :param meta: meta added to sample
        :return: updated sample
        """
        sample['text'] = ' '.join(
            [sample['instruction'], sample['input'], sample['output']])
        sample['meta'] = meta
        sample['meta']['origin_path'] = path
        return sample

    path = ''.join(['Alpaca-CoT', filename.split('Alpaca-CoT')[1]])
    ds = ds.map(_merge_and_add_meta,
                num_proc=48,
                fn_kwargs={
                    'path': path,
                    'meta': meta
                })

    if len(ds) > 0:
        out_file = ''.join([target_dir, filename.split('Alpaca-CoT')[1]])
        out_file = out_file.replace('.json', '.jsonl')
        dir_name = os.path.dirname(out_file)
        os.makedirs(dir_name, exist_ok=True)
        ds.to_json(out_file, force_ascii=False)


def fp_iter(src_dir):
    """
    Find all tar files in the source directory.
    :param src_dir: path to source dataset directory
    :return: iterator over json files
    """
    for fp in pathlib.Path(src_dir).rglob('*.json'):
        yield fp


def main(src_dir, target_dir, num_proc=4):
    """
    Load dataset from the source directory, then apply language identification
    using the operation filter called `LanguageIDScoreFilter`,
    finally, split the dataset by language and save it.
    :param src_dir: path thats store dataset directory
    :param target_dir: path to store subset files(`jsonl` format)
    :param num_proc: number of processes to process dataset, default 1.
    """

    # check if the source directory exists.
    if not os.path.exists(src_dir):
        raise ValueError('The raw source data directory does not exist,'
                         ' Please check and retry.')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    pool = Pool(num_proc)

    for fp in fp_iter(src_dir):
        pool.apply_async(merge_and_add_meta, args=(str(fp), target_dir))

    pool.close()
    pool.join()


if __name__ == '__main__':
    fire.Fire(main)
