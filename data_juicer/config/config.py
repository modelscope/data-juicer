import copy
import json
import os
import shutil
import tempfile
import time
from argparse import ArgumentError, Namespace
from typing import Dict, List, Union

import yaml
from jsonargparse import (ActionConfigFile, ArgumentParser, dict_to_namespace,
                          namespace_to_dict)
from jsonargparse.typehints import ActionTypeHint
from jsonargparse.typing import ClosedUnitInterval, NonNegativeInt, PositiveInt
from loguru import logger

from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.logger_utils import setup_logger
from data_juicer.utils.mm_utils import SpecialTokens

global_cfg = None
global_parser = None


def init_configs(args=None):
    """
    initialize the jsonargparse parser and parse configs from one of:
        1. POSIX-style commands line args;
        2. config files in yaml (json and jsonnet supersets);
        3. environment variables
        4. hard-coded defaults

    :param args: list of params, e.g., ['--conifg', 'cfg.yaml'], defaut None.
    :return: a global cfg object used by the Executor or Analyzer
    """
    parser = ArgumentParser(default_env=True, default_config_files=None)

    parser.add_argument('--config',
                        action=ActionConfigFile,
                        help='Path to a dj basic configuration file.',
                        required=True)

    parser.add_argument(
        '--hpo_config',
        type=str,
        help='Path to a configuration file when using auto-HPO tool.',
        required=False)
    parser.add_argument(
        '--data_probe_algo',
        type=str,
        default='uniform',
        help='Sampling algorithm to use. Options are "uniform", '
        '"frequency_specified_field_selector", or '
        '"topk_specified_field_selector". Default is "uniform". Only '
        'used for dataset sampling',
        required=False)
    parser.add_argument(
        '--data_probe_ratio',
        type=ClosedUnitInterval,
        default=1.0,
        help='The ratio of the sample size to the original dataset size. '
        'Default is 1.0 (no sampling). Only used for dataset sampling',
        required=False)

    # basic global paras with extended type hints
    # e.g., files can be mode include flags
    # "fr": "path to a file that exists and is readable")
    # "fc": "path to a file that can be created if it does not exist")
    # "dw": "path to a directory that exists and is writeable")
    # "dc": "path to a directory that can be created if it does not exist")
    # "drw": "path to a directory that exists and is readable and writeable")
    parser.add_argument('--project_name',
                        type=str,
                        default='hello_world',
                        help='Name of your data process project.')
    parser.add_argument(
        '--executor_type',
        type=str,
        default='default',
        choices=['default', 'ray'],
        help='Type of executor, support "default" or "ray" for now.')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='',
        help='Path to datasets with optional weights(0.0-1.0), 1.0 as '
        'default. Accepted format:<w1> dataset1-path <w2> dataset2-path '
        '<w3> dataset3-path ...')
    parser.add_argument(
        '--generated_dataset_config',
        type=Dict,
        default=None,
        help='Configuration used to create a dataset. '
        'The dataset will be created from this configuration if provided. '
        'It must contain the `type` field to specify the dataset name.')
    parser.add_argument(
        '--export_path',
        type=str,
        default='./outputs/hello_world.jsonl',
        help='Path to export and save the output processed dataset. The '
        'directory to store the processed dataset will be the work '
        'directory of this process.')
    parser.add_argument(
        '--export_shard_size',
        type=NonNegativeInt,
        default=0,
        help='Shard size of exported dataset in Byte. In default, it\'s 0, '
        'which means export the whole dataset into only one file. If '
        'it\'s set a positive number, the exported dataset will be split '
        'into several sub-dataset shards, and the max size of each shard '
        'won\'t larger than the export_shard_size')
    parser.add_argument(
        '--export_in_parallel',
        type=bool,
        default=False,
        help='Whether to export the result dataset in parallel to a single '
        'file, which usually takes less time. It only works when '
        'export_shard_size is 0, and its default number of processes is '
        'the same as the argument np. **Notice**: If it\'s True, '
        'sometimes exporting in parallel might require much more time '
        'due to the IO blocking, especially for very large datasets. '
        'When this happens, False is a better choice, although it takes '
        'more time.')
    parser.add_argument(
        '--keep_stats_in_res_ds',
        type=bool,
        default=False,
        help='Whether to keep the computed stats in the result dataset. If '
        'it\'s False, the intermediate fields to store the stats '
        'computed by Filters will be removed. Default: False.')
    parser.add_argument(
        '--keep_hashes_in_res_ds',
        type=bool,
        default=False,
        help='Whether to keep the computed hashes in the result dataset. If '
        'it\'s False, the intermediate fields to store the hashes '
        'computed by Deduplicators will be removed. Default: False.')
    parser.add_argument('--np',
                        type=PositiveInt,
                        default=4,
                        help='Number of processes to process dataset.')
    parser.add_argument(
        '--text_keys',
        type=Union[str, List[str]],
        default='text',
        help='Key name of field where the sample texts to be processed, e.g., '
        '`text`, `text.instruction`, `text.output`, ... Note: currently, '
        'we support specify only ONE key for each op, for cases '
        'requiring multiple keys, users can specify the op multiple '
        'times.  We will only use the first key of `text_keys` when you '
        'set multiple keys.')
    parser.add_argument(
        '--image_key',
        type=str,
        default='images',
        help='Key name of field to store the list of sample image paths.')
    parser.add_argument(
        '--image_special_token',
        type=str,
        default=SpecialTokens.image,
        help='The special token that represents an image in the text. In '
        'default, it\'s "<__dj__image>". You can specify your own special'
        ' token according to your input dataset.')
    parser.add_argument(
        '--audio_key',
        type=str,
        default='audios',
        help='Key name of field to store the list of sample audio paths.')
    parser.add_argument(
        '--audio_special_token',
        type=str,
        default=SpecialTokens.audio,
        help='The special token that represents an audio in the text. In '
        'default, it\'s "<__dj__audio>". You can specify your own special'
        ' token according to your input dataset.')
    parser.add_argument(
        '--video_key',
        type=str,
        default='videos',
        help='Key name of field to store the list of sample video paths.')
    parser.add_argument(
        '--video_special_token',
        type=str,
        default=SpecialTokens.video,
        help='The special token that represents a video in the text. In '
        'default, it\'s "<__dj__video>". You can specify your own special'
        ' token according to your input dataset.')
    parser.add_argument(
        '--eoc_special_token',
        type=str,
        default=SpecialTokens.eoc,
        help='The special token that represents the end of a chunk in the '
        'text. In default, it\'s "<|__dj__eoc|>". You can specify your '
        'own special token according to your input dataset.')
    parser.add_argument(
        '--suffixes',
        type=Union[str, List[str]],
        default=[],
        help='Suffixes of files that will be find and loaded. If not set, we '
        'will find all suffix files, and select a suitable formatter '
        'with the most files as default.')
    parser.add_argument(
        '--turbo',
        type=bool,
        default=False,
        help='Enable Turbo mode to maximize processing speed. Stability '
        'features like fault tolerance will be disabled.')
    parser.add_argument(
        '--use_cache',
        type=bool,
        default=True,
        help='Whether to use the cache management of huggingface datasets. It '
        'might take up lots of disk space when using cache')
    parser.add_argument(
        '--ds_cache_dir',
        type=str,
        default=None,
        help='Cache dir for HuggingFace datasets. In default it\'s the same '
        'as the environment variable `HF_DATASETS_CACHE`, whose default '
        'value is usually "~/.cache/huggingface/datasets". If this '
        'argument is set to a valid path by users, it will override the '
        'default cache dir. Modifying this arg might also affect the other two'
        ' paths to store downloaded and extracted datasets that depend on '
        '`HF_DATASETS_CACHE`')
    parser.add_argument(
        '--cache_compress',
        type=str,
        default=None,
        help='The compression method of the cache file, which can be'
        'specified in ["gzip", "zstd", "lz4"]. If this parameter is'
        'None, the cache file will not be compressed.')
    parser.add_argument(
        '--use_checkpoint',
        type=bool,
        default=False,
        help='Whether to use the checkpoint management to save the latest '
        'version of dataset to work dir when processing. Rerun the same '
        'config will reload the checkpoint and skip ops before it. Cache '
        'will be disabled when it is true . If args of ops before the '
        'checkpoint are changed, all ops will be rerun from the '
        'beginning.')
    parser.add_argument(
        '--temp_dir',
        type=str,
        default=None,
        help='Path to the temp directory to store intermediate caches when '
        'cache is disabled. In default it\'s None, so the temp dir will '
        'be specified by system. NOTICE: you should be caution when '
        'setting this argument because it might cause unexpected program '
        'behaviors when this path is set to an unsafe directory.')
    parser.add_argument(
        '--open_tracer',
        type=bool,
        default=False,
        help='Whether to open the tracer to trace samples changed during '
        'process. It might take more time when opening tracer.')
    parser.add_argument(
        '--op_list_to_trace',
        type=List[str],
        default=[],
        help='Which ops will be traced by tracer. If it\'s empty, all ops in '
        'cfg.process will be traced. Only available when open_tracer is '
        'true.')
    parser.add_argument(
        '--trace_num',
        type=int,
        default=10,
        help='Number of samples extracted by tracer to show the dataset '
        'difference before and after a op. Only available when '
        'open_tracer is true.')
    parser.add_argument(
        '--op_fusion',
        type=bool,
        default=False,
        help='Whether to fuse operators that share the same intermediate '
        'variables automatically. Op fusion might reduce the memory '
        'requirements slightly but speed up the whole process.')
    parser.add_argument(
        '--process',
        type=List[Dict],
        default=[],
        help='List of several operators with their arguments, these ops will '
        'be applied to dataset in order')
    parser.add_argument(
        '--percentiles',
        type=List[float],
        default=[],
        help='Percentiles to analyze the dataset distribution. Only used in '
        'Analysis.')
    parser.add_argument(
        '--export_original_dataset',
        type=bool,
        default=False,
        help='whether to export the original dataset with stats. If you only '
        'need the stats of the dataset, setting it to false could speed '
        'up the exporting..')
    parser.add_argument(
        '--save_stats_in_one_file',
        type=bool,
        default=False,
        help='Whether to save all stats to only one file. Only used in '
        'Analysis.')
    parser.add_argument('--ray_address',
                        type=str,
                        default='auto',
                        help='The address of the Ray cluster.')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Whether to run in debug mode.')

    # add all parameters of the registered ops class to the parser,
    # and these op parameters can be modified through the command line,
    ops_sorted_by_types = sort_op_by_types_and_names(OPERATORS.modules.items())
    _collect_config_info_from_class_docs(ops_sorted_by_types, parser)

    try:
        cfg = parser.parse_args(args=args)
        cfg = init_setup_from_cfg(cfg)
        cfg = update_op_process(cfg, parser)

        # copy the config file into the work directory
        config_backup(cfg)

        # show the final config tables before the process started
        display_config(cfg)

        global global_cfg, global_parser
        global_cfg = cfg
        global_parser = parser

        if cfg.debug:
            logger.debug('In DEBUG mode.')

        return cfg
    except ArgumentError:
        logger.error('Config initialization failed')


def update_ds_cache_dir_and_related_vars(new_ds_cache_path):
    from pathlib import Path

    from datasets import config

    # update the HF_DATASETS_CACHE
    config.HF_DATASETS_CACHE = Path(new_ds_cache_path)
    # and two more PATHS that depend on HF_DATASETS_CACHE
    # - path to store downloaded datasets (e.g. remote datasets)
    config.DEFAULT_DOWNLOADED_DATASETS_PATH = os.path.join(
        config.HF_DATASETS_CACHE, config.DOWNLOADED_DATASETS_DIR)
    config.DOWNLOADED_DATASETS_PATH = Path(
        config.DEFAULT_DOWNLOADED_DATASETS_PATH)
    # - path to store extracted datasets (e.g. xxx.jsonl.zst)
    config.DEFAULT_EXTRACTED_DATASETS_PATH = os.path.join(
        config.DEFAULT_DOWNLOADED_DATASETS_PATH, config.EXTRACTED_DATASETS_DIR)
    config.EXTRACTED_DATASETS_PATH = Path(
        config.DEFAULT_EXTRACTED_DATASETS_PATH)


def init_setup_from_cfg(cfg):
    """
    Do some extra setup tasks after parsing config file or command line.

    1. create working directory and a log directory
    2. update cache directory
    3. update checkpoint and `temp_dir` of tempfile

    :param cfg: an original cfg
    :param cfg: an updated cfg
    """

    cfg.export_path = os.path.abspath(cfg.export_path)
    cfg.work_dir = os.path.dirname(cfg.export_path)
    export_rel_path = os.path.relpath(cfg.export_path, start=cfg.work_dir)
    log_dir = os.path.join(cfg.work_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    cfg.timestamp = timestamp
    logfile_name = f'export_{export_rel_path}_time_{timestamp}.txt'
    setup_logger(save_dir=log_dir,
                 filename=logfile_name,
                 level='DEBUG' if cfg.debug else 'INFO',
                 redirect=cfg.executor_type == 'default')

    # check and get dataset dir
    if cfg.get('dataset_path', None) and os.path.exists(cfg.dataset_path):
        cfg.dataset_path = os.path.abspath(cfg.dataset_path)
        if os.path.isdir(cfg.dataset_path):
            cfg.dataset_dir = cfg.dataset_path
        else:
            cfg.dataset_dir = os.path.dirname(cfg.dataset_path)
    elif cfg.dataset_path == '':
        logger.warning('dataset_path is empty by default.')
        cfg.dataset_dir = ''
    else:
        logger.warning(f'dataset_path [{cfg.dataset_path}] is not a valid '
                       f'local path. Please check and retry, otherwise we '
                       f'will treat it as a remote dataset or a mixture of '
                       f'several datasets.')
        cfg.dataset_dir = ''

    # check number of processes np
    sys_cpu_count = os.cpu_count()
    if cfg.np > sys_cpu_count:
        logger.warning(f'Number of processes `np` is set as [{cfg.np}], which '
                       f'is larger than the cpu count [{sys_cpu_count}]. Due '
                       f'to the data processing of Data-Juicer is a '
                       f'computation-intensive task, we recommend to set it to'
                       f' a value <= cpu count. Set it to [{sys_cpu_count}] '
                       f'here.')
        cfg.np = sys_cpu_count

    # whether or not to use cache management
    # disabling the cache or using checkpoint explicitly will turn off the
    # cache management.
    if not cfg.use_cache or cfg.use_checkpoint:
        logger.warning('Cache management of datasets is disabled.')
        from datasets import disable_caching
        disable_caching()
        cfg.use_cache = False

        # disabled cache compression when cache is disabled
        if cfg.cache_compress:
            logger.warning('Disable cache compression due to disabled cache.')
            cfg.cache_compress = None

        # when disabling cache, enable the temp_dir argument
        logger.warning(f'Set temp directory to store temp files to '
                       f'[{cfg.temp_dir}].')
        import tempfile
        if cfg.temp_dir is not None and not os.path.exists(cfg.temp_dir):
            os.makedirs(cfg.temp_dir, exist_ok=True)
        tempfile.tempdir = cfg.temp_dir

    # The checkpoint mode is not compatible with op fusion for now.
    if cfg.op_fusion:
        cfg.use_checkpoint = False

    # update huggingface datasets cache directory only when ds_cache_dir is set
    from datasets import config
    if cfg.ds_cache_dir:
        logger.warning(f'Set dataset cache directory to {cfg.ds_cache_dir} '
                       f'using the ds_cache_dir argument, which is '
                       f'{config.HF_DATASETS_CACHE} before based on the env '
                       f'variable HF_DATASETS_CACHE.')
        update_ds_cache_dir_and_related_vars(cfg.ds_cache_dir)
    else:
        cfg.ds_cache_dir = str(config.HF_DATASETS_CACHE)

    # if there is suffix_filter op, turn on the add_suffix flag
    cfg.add_suffix = False
    for op in cfg.process:
        op_name, _ = list(op.items())[0]
        if op_name == 'suffix_filter':
            cfg.add_suffix = True
            break

    # update special tokens
    SpecialTokens.image = cfg.image_special_token
    SpecialTokens.eoc = cfg.eoc_special_token

    # Apply text_key modification during initializing configs
    # users can freely specify text_key for different ops using `text_key`
    # otherwise, set arg text_key of each op to text_keys
    if isinstance(cfg.text_keys, list):
        text_key = cfg.text_keys[0]
    else:
        text_key = cfg.text_keys
    for op in cfg.process:
        for op_name in op:
            args = op[op_name]
            if args is None:
                args = {
                    'text_key': text_key,
                    'image_key': cfg.image_key,
                    'audio_key': cfg.audio_key,
                    'video_key': cfg.video_key,
                    'num_proc': cfg.np,
                    'turbo': cfg.turbo,
                }
            else:
                if 'text_key' not in args or args['text_key'] is None:
                    args['text_key'] = text_key
                if 'image_key' not in args or args['image_key'] is None:
                    args['image_key'] = cfg.image_key
                if 'audio_key' not in args or args['audio_key'] is None:
                    args['audio_key'] = cfg.audio_key
                if 'video_key' not in args or args['video_key'] is None:
                    args['video_key'] = cfg.video_key
                if 'num_proc' not in args or args['num_proc'] is None:
                    args['num_proc'] = cfg.np
                if 'turbo' not in args or args['turbo'] is None:
                    args['turbo'] = cfg.turbo
            op[op_name] = args

    return cfg


def _collect_config_info_from_class_docs(configurable_ops, parser):
    """
    Add ops and its params to parser for command line.

    :param configurable_ops: a list of ops to be added, each item is
        a pair of op_name and op_class
    :param parser: jsonargparse parser need to update
    :return: all params of each OP in a dictionary
    """

    op_params = {}
    for op_name, op_class in configurable_ops:
        params = parser.add_class_arguments(
            theclass=op_class,
            nested_key=op_name,
            fail_untyped=False,
            instantiate=False,
        )
        op_params[op_name] = params
    return op_params


def sort_op_by_types_and_names(op_name_classes):
    """
    Split ops items by op type and sort them to sub-ops by name, then concat
    together.

    :param op_name_classes: a list of op modules
    :return: sorted op list , each item is a pair of op_name and
        op_class
    """

    mapper_ops = [(name, c) for (name, c) in op_name_classes
                  if 'mapper' in name]
    filter_ops = [(name, c) for (name, c) in op_name_classes
                  if 'filter' in name]
    deduplicator_ops = [(name, c) for (name, c) in op_name_classes
                        if 'deduplicator' in name]
    selector_ops = [(name, c) for (name, c) in op_name_classes
                    if 'selector' in name]
    ops_sorted_by_types = sorted(mapper_ops) + sorted(filter_ops) + sorted(
        deduplicator_ops) + sorted(selector_ops)
    return ops_sorted_by_types


def update_op_process(cfg, parser):
    op_keys = list(OPERATORS.modules.keys())
    args = [
        arg.split('--')[1] for arg in parser.args
        if arg.startswith('--') and arg.split('--')[1].split('.')[0] in op_keys
    ]
    option_in_commands = list(set([''.join(arg.split('.')[0])
                                   for arg in args]))
    full_option_in_commands = list(
        set([''.join(arg.split('=')[0]) for arg in args]))

    if cfg.process is None:
        cfg.process = []

    # check and update every op params in `cfg.process`
    # e.g.
    # `python demo.py --config demo.yaml
    #  --language_id_score_filter.lang en`
    temp_cfg = cfg
    for i, op_in_process in enumerate(cfg.process):
        op_in_process_name = list(op_in_process.keys())[0]

        if op_in_process_name not in option_in_commands:

            # update op params to temp cfg if set
            if op_in_process[op_in_process_name]:
                temp_cfg = parser.merge_config(
                    dict_to_namespace(op_in_process), temp_cfg)
        else:

            # args in the command line override the ones in `cfg.process`
            for full_option_in_command in full_option_in_commands:

                key = full_option_in_command.split('.')[1]
                if op_in_process[op_in_process_name] and key in op_in_process[
                        op_in_process_name].keys():
                    op_in_process[op_in_process_name].pop(key)

            if op_in_process[op_in_process_name]:
                temp_cfg = parser.merge_config(
                    dict_to_namespace(op_in_process), temp_cfg)

        # update op params of cfg.process
        internal_op_para = temp_cfg.get(op_in_process_name)

        cfg.process[i] = {
            op_in_process_name:
            None if internal_op_para is None else
            namespace_to_dict(internal_op_para)
        }

    # check the op params via type hint
    temp_parser = copy.deepcopy(parser)
    recognized_args = set([
        action.dest for action in parser._actions
        if hasattr(action, 'dest') and isinstance(action, ActionTypeHint)
    ])

    temp_args = namespace_to_arg_list(temp_cfg,
                                      includes=recognized_args,
                                      excludes=['config'])
    temp_args = ['--config', temp_cfg.config[0].absolute] + temp_args
    temp_parser.parse_args(temp_args)
    return cfg


def namespace_to_arg_list(namespace, prefix='', includes=None, excludes=None):
    arg_list = []

    for key, value in vars(namespace).items():

        if issubclass(type(value), Namespace):
            nested_args = namespace_to_arg_list(value, f'{prefix}{key}.')
            arg_list.extend(nested_args)
        elif value is not None:
            concat_key = f'{prefix}{key}'
            if includes is not None and concat_key not in includes:
                continue
            if excludes is not None and concat_key in excludes:
                continue
            arg_list.append(f'--{concat_key}')
            arg_list.append(f'{value}')

    return arg_list


def config_backup(cfg):
    cfg_path = cfg.config[0].absolute
    work_dir = cfg.work_dir
    target_path = os.path.join(work_dir, os.path.basename(cfg_path))
    logger.info(f'Back up the input config file [{cfg_path}] into the '
                f'work_dir [{work_dir}]')
    if not os.path.exists(target_path):
        shutil.copyfile(cfg_path, target_path)


def display_config(cfg):
    import pprint

    from tabulate import tabulate
    table_header = ['key', 'values']

    # remove ops outside the process list for better displaying
    shown_cfg = cfg.clone()
    for op in OPERATORS.modules.keys():
        _ = shown_cfg.pop(op)

    # construct the table as 2 columns
    config_table = [(k, pprint.pformat(v, compact=True))
                    for k, v in shown_cfg.items()]
    table = tabulate(config_table, headers=table_header, tablefmt='fancy_grid')

    logger.info('Configuration table: ')
    print(table)


def export_config(cfg,
                  path,
                  format='yaml',
                  skip_none=True,
                  skip_check=True,
                  overwrite=False,
                  multifile=True):
    """
    Save the config object, some params are from jsonargparse

    :param cfg: cfg object to save (Namespace type)
    :param path: the save path
    :param format: 'yaml', 'json', 'json_indented', 'parser_mode'
    :param skip_none: Whether to exclude entries whose value is None.
    :param skip_check: Whether to skip parser checking.
    :param overwrite: Whether to overwrite existing files.
    :param multifile: Whether to save multiple config files
        by using the __path__ metas.

    :return:
    """
    # remove ops outside the process list for better displaying
    cfg_to_export = cfg.clone()
    for op in OPERATORS.modules.keys():
        _ = cfg_to_export.pop(op)

    global global_parser
    if not global_parser:
        init_configs()  # enable the customized type parser
    if isinstance(cfg_to_export, Namespace):
        cfg_to_export = namespace_to_dict(cfg_to_export)
    global_parser.save(cfg=cfg_to_export,
                       path=path,
                       format=format,
                       skip_none=skip_none,
                       skip_check=skip_check,
                       overwrite=overwrite,
                       multifile=multifile)

    logger.info(f'Saved the configuration in {path}')


def merge_config(ori_cfg, new_cfg: Dict):
    """
    Merge configuration from new_cfg into ori_cfg

    :param ori_cfg: the original configuration object, whose type is
        expected as namespace from jsonargparse
    :param new_cfg: the configuration object to be merged, whose type is
        expected as dict or namespace from jsonargparse

    :return: cfg_after_merge
    """
    try:
        ori_specified_op_names = set()
        ori_specified_op_idx = {}  # {op_name: op_order}

        for op_order, op_in_process in enumerate(ori_cfg.process):
            op_name = list(op_in_process.keys())[0]
            ori_specified_op_names.add(op_name)
            ori_specified_op_idx[op_name] = op_order

        for new_k, new_v in new_cfg.items():
            # merge parameters other than `cfg.process` and DJ-OPs
            if new_k in ori_cfg and new_k != 'process' and '.' not in new_k:
                logger.info('=' * 15)
                logger.info(f'Before merging, the cfg item is: '
                            f'{new_k}: {ori_cfg[new_k]}')
                ori_cfg[new_k] = new_v
                logger.info(f'After merging,  the cfg item is: '
                            f'{new_k}: {new_v}')
                logger.info('=' * 15)
            else:
                # merge parameters of DJ-OPs into cfg.process
                # for nested style, e.g., `remove_table_text_mapper.min_col: 2`
                key_as_groups = new_k.split('.')
                if len(key_as_groups) > 1 and \
                        key_as_groups[0] in ori_specified_op_names:
                    op_name, para_name = key_as_groups[0], key_as_groups[1]
                    op_order = ori_specified_op_idx[op_name]
                    ori_cfg_val = ori_cfg.process[op_order][op_name][para_name]
                    logger.info('=' * 15)
                    logger.info(f'Before merging, the cfg item is: '
                                f'{new_k}: {ori_cfg_val}')
                    ori_cfg.process[op_order][op_name][para_name] = new_v
                    logger.info(f'After merging,  the cfg item is: '
                                f'{new_k}: {new_v}')
                    logger.info('=' * 15)

        ori_cfg = init_setup_from_cfg(ori_cfg)

        # copy the config file into the work directory
        config_backup(ori_cfg)

        return ori_cfg

    except ArgumentError:
        logger.error('Config merge failed')


def prepare_side_configs(ori_config):
    """
    parse the config if ori_config is a string of a config file path with
        yaml, yml or json format

    :param ori_config: a config dict or a string of a config file path with
        yaml, yml or json format

    :return: a config dict
    """

    if isinstance(ori_config, str):
        # config path
        if ori_config.endswith('.yaml') or ori_config.endswith('.yml'):
            with open(ori_config) as fin:
                config = yaml.safe_load(fin)
        elif ori_config.endswith('.json'):
            with open(ori_config) as fin:
                config = json.load(fin)
        else:
            raise TypeError(f'Unrecognized config file type: [{ori_config}]. '
                            f'Should be one of the types [".yaml", ".yml", '
                            f'".json"].')
    elif isinstance(ori_config, dict) or isinstance(ori_config, Namespace):
        config = ori_config
    else:
        raise TypeError(
            f'Unrecognized side config type: [{type(ori_config)}].')

    return config


def get_init_configs(cfg):
    """
    set init configs of datajucer for cfg
    """
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, 'job_dj_config.json')
    if isinstance(cfg, Namespace):
        cfg = namespace_to_dict(cfg)
    # create an temp config file
    with open(temp_file, 'w') as f:
        json.dump(cfg, f)
    inited_dj_cfg = init_configs(['--config', temp_file])
    return inited_dj_cfg
