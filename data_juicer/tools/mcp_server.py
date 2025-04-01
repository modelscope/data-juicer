import datetime
import os
import sys
from typing import Dict

from loguru import logger
from mcp.server.fastmcp import FastMCP

from data_juicer.config import get_init_configs
from data_juicer.core import Executor

DEFAULT_OUTPUT_DIR = './output'
mcp = FastMCP('Data-Juicer Server')


def add_extra_cfg(dj_cfg: Dict) -> Dict:
    """ Add extra dj config. """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dj_cfg['export_path'] = os.path.join(DEFAULT_OUTPUT_DIR, timestamp,
                                         'processed_data.jsonl')

    # Problem: It will holding when use multi threads/procs
    # Can't multithreading and multiprocessing be used in a coroutine?
    dj_cfg['np'] = 1  # set num proc to be 1
    dj_cfg['open_monitor'] = False  # unable monitor to avoid multi proc

    return dj_cfg


def execute_op(dj_cfg: Dict):

    try:
        dj_cfg = add_extra_cfg(dj_cfg)
        logger.info(f'DJ config in MCP server: {str(dj_cfg)}')
        dj_cfg = get_init_configs(dj_cfg)
        executor = Executor(dj_cfg)
        executor.run()
        return f"Result dataset is saved in: {dj_cfg['export_path']}"
    except Exception as e:
        return f'Occur error when executing Data-Juicer: {e}'


@mcp.tool()
def text_length_filter(
    dataset_path: str,
    min_len: int = 10,
    max_len: int = sys.maxsize,
) -> str:
    """
    Filter to keep samples with total text length within a specific range.

    # :param dataset_path: Path to the dataset to be processed.
    # :param min_len: The min text length in the filtering. samples
    #         will be filtered if their text length is below this
    #         parameter.
    # :param max_len: The max text length in the filtering. samples
    #         will be filtered if their text length exceeds this
    #         parameter.
    """

    args_dict = dict(locals())
    args_dict.pop('dataset_path')
    dj_cfg = {
        'dataset_path': dataset_path,
        'process': [{
            'text_length_filter': args_dict
        }]
    }
    return execute_op(dj_cfg)


if __name__ == '__main__':
    mcp.run()
