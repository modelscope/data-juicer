import os
import time
from argparse import ArgumentError
from typing import Dict, List, Optional, Tuple, Union

from jsonargparse import (ActionConfigFile, ArgumentParser, dict_to_namespace,
                          namespace_to_dict)
from jsonargparse.typing import NonNegativeInt, PositiveInt
from loguru import logger

from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.logger_utils import setup_logger


def init_configs(args=None):
    """
    initialize the jsonargparse parser and parse configs from one of:
        1. POSIX-style commands line args;
        2. config files in yaml (json and jsonnet supersets);
        3. environment variables
        4. hard-coded defaults

    :param args: list of params, e.g., ['--conifg', 'cfg.yaml'], defaut None.
    :return: a global cfg object used by the Executor or Analyser
    """
    parser = ArgumentParser(default_env=True, default_config_files=None)

    parser.add_argument('--config',
                        action=ActionConfigFile,
                        help='Path to a configuration file.',
                        required=True)

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
        '--dataset_path',
        type=str,
        help='Path to datasets with optional weights(0.0-1.0), '
        '1.0 as default. Accepted format:'
        '<w1> dataset1-path <w2> dataset2-path '
        '<w3> dataset3-path ...')
    parser.add_argument(
        '--export_path',
        type=str,
        default='./outputs/hello_world.jsonl',
        help='Path to export and save the output processed dataset.'
        ' The directory to store the processed dataset will be the '
        'work directory of this process.')
    parser.add_argument('--export_shard_size',
                        type=NonNegativeInt,
                        default=0,
                        help='Shard size of exported dataset in Byte. In '
                        'default, it\'s 0, which means export the whole '
                        'dataset into only one file. If it\'s set a '
                        'positive number, the exported dataset will be '
                        'split into several sub-dataset shards, and the max '
                        'size of each shard won\'t larger than the '
                        'export_shard_size')
    parser.add_argument('--np',
                        type=PositiveInt,
                        default=4,
                        help='Number of processes to process dataset.')
    parser.add_argument('--text_key_to_process',
                        type=Optional[str],
                        default='text',
                        help='Key name of field where the sample '
                        'texts to be processed, e.g., '
                        '`text`, `text.instruction`, `text.output`, ...'
                        'Note: currently, we support specify only ONE key for '
                        'each op, for cases requiring multiple keys, users can'
                        ' specify the op multiple times')
    parser.add_argument('--text_keys_to_load',
                        type=Union[List[str], Tuple[str], str],
                        default='deprecated',
                        help='Key name of field where the sample '
                        'texts stored in the original data')
    parser.add_argument('--suffixes',
                        type=Union[str, List[str], Tuple[str]],
                        default=[],
                        help='Suffixes of files that will be find and loaded. '
                        'If not set, we will find all suffix files, and select'
                        'a suitable formatter with the most files as default.')
    parser.add_argument('--use_cache',
                        type=bool,
                        default=True,
                        help='Whether to use the cache management of hugging'
                        'face datasets. It might take up lots of disk '
                        'space when using cache')
    parser.add_argument('--ds_cache_dir',
                        type=str,
                        default='~/.cache/huggingface/datasets',
                        help='Cache dir for HuggingFace datasets. In default '
                        'it\'s the default cache dir "~/.cache/huggingface/dat'
                        'asets". If this argument is reset by users, it will '
                        'override the default cache dir.')
    parser.add_argument('--use_checkpoint',
                        type=bool,
                        default=False,
                        help='Whether to use the checkpoint management to '
                        'save the latest version of dataset to work dir when '
                        'processing. Rerun the same config will reload the '
                        'checkpoint and skip ops before it. Cache will be '
                        'disabled when it is true . If args of ops '
                        'before the checkpoint are changed, all ops will be '
                        'rerun from the beginning.')
    parser.add_argument('--temp_dir',
                        type=str,
                        default=None,
                        help='Path to the temp directory to store '
                        'intermediate caches when cache is disabled. In '
                        'default it\'s None, so the temp dir will be '
                        'specified by system. NOTICE: you should be caution '
                        'when setting this argument because it might cause '
                        'unexpected program behaviors when this path is set '
                        'to an unsafe directory.')
    parser.add_argument('--open_tracer',
                        type=bool,
                        default=False,
                        help='Whether to open the tracer to trace samples'
                        'changed during process. It might take more '
                        'time when opening tracer.')
    parser.add_argument(
        '--op_list_to_trace',
        type=List[str],
        default=[],
        help='Which ops will be traced by tracer. '
        'If it\'s empty, all ops in cfg.process will be traced. Only '
        'available when open_tracer is true.')
    parser.add_argument('--trace_num',
                        type=int,
                        default=10,
                        help='Number of samples extracted by tracer to show '
                        'the dataset difference before and after a op. '
                        'Only available when open_tracer is true.')
    parser.add_argument(
        '--process',
        type=List[Dict],
        help='List of several operators with their '
        'arguments, these ops will be applied to dataset in order')
    parser.add_argument('--save_stats_in_one_file',
                        type=bool,
                        default=False,
                        help='Whether to save all stats to only one file. Only'
                        ' used in Analysis.')

    # add all parameters of the registered ops class to the parser,
    # and these op parameters can be modified through the command line,
    ops_sorted_by_types = sort_op_by_types_and_names(OPERATORS.modules.items())
    _collect_config_info_from_class_docs(ops_sorted_by_types, parser)

    try:
        cfg = parser.parse_args(args=args)
        option_in_commands = [
            ''.join(arg.split('--')[1].split('.')[0]) for arg in parser.args
            if '--' in arg and 'config' not in arg
        ]

        full_option_in_commands = list(
            set([
                ''.join(arg.split('--')[1].split('=')[0])
                for arg in parser.args if '--' in arg and 'config' not in arg
            ]))

        if cfg.process is None:
            cfg.process = []

        # check and update every op params in `cfg.process`
        # e.g.
        # `python demo.py --config demo.yaml
        #  --language_id_score_filter.lang en`
        for i, op_in_process in enumerate(cfg.process):
            op_in_process_name = list(op_in_process.keys())[0]

            temp_cfg = cfg
            if op_in_process_name not in option_in_commands:

                # update op params to temp cfg if set
                if op_in_process[op_in_process_name]:
                    temp_cfg = parser.merge_config(
                        dict_to_namespace(op_in_process), cfg)
            else:

                # args in the command line override the ones in `cfg.process`
                for full_option_in_command in full_option_in_commands:

                    key = full_option_in_command.split('.')[1]
                    if op_in_process[
                            op_in_process_name] and key in op_in_process[
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

        cfg = init_setup_from_cfg(cfg)
        return cfg
    except ArgumentError:
        logger.error('Config initialization failed')


def init_setup_from_cfg(cfg):
    """
    Do some extra setup tasks after parsing config file or command line.

    1. create working directory and a log directory
    2. update cache directory
    3. update checkpoint and `temp_dir` of tempfile

    :param cfg: a original cfg
    :param cfg: a updated cfg
    """

    export_path = cfg.export_path
    cfg.work_dir = os.path.dirname(export_path)
    log_dir = os.path.join(cfg.work_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logfile_name = time.strftime('%Y%m%d%H%M%S', time.localtime(
        time.time())) + '.txt'
    setup_logger(save_dir=log_dir, filename=logfile_name)

    # whether or not to use cache management
    # disabling the cache or using checkpoint explicitly will turn off the
    # cache management.
    if not cfg.use_cache or cfg.use_checkpoint:
        logger.warning('Cache management of datasets is disabled.')
        from datasets import disable_caching
        disable_caching()

        # when disabling cache, enable the temp_dir argument
        logger.warning(f'Set temp directory to store temp files to '
                       f'[{cfg.temp_dir}].')
        import tempfile
        if cfg.temp_dir is not None and not os.path.exists(cfg.temp_dir):
            os.makedirs(cfg.temp_dir, exist_ok=True)
        tempfile.tempdir = cfg.temp_dir

    # reset huggingface datasets cache directory
    from datasets import config
    config.HF_DATASETS_CACHE = cfg.ds_cache_dir

    # if there is suffix_filter op, turn on the add_suffix flag
    cfg.add_suffix = False
    for op in cfg.process:
        op_name, _ = list(op.items())[0]
        if op_name == 'suffix_filter':
            cfg.add_suffix = True
            break

    return cfg


def _collect_config_info_from_class_docs(configurable_ops, parser):
    """
    Add ops and its params to parser for command line.

    :param configurable_ops: a list of ops to be to added, each item is
        a pair of op_name and op_class
    :param parser: jsonargparse parser need to update
    """

    for op_name, op_class in configurable_ops:
        parser.add_class_arguments(
            theclass=op_class,
            nested_key=op_name,
            fail_untyped=False,
            instantiate=False,
        )


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
