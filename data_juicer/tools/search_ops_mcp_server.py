import datetime
import json
import os
from typing import Dict

from loguru import logger
from mcp.server.fastmcp import FastMCP

from data_juicer.config import get_init_configs
from data_juicer.core import DefaultExecutor

# Server configuration
DEFAULT_OUTPUT_DIR = './outputs'
mcp = FastMCP('Data-Juicer Server')

# Load operator modality mapping
abs_path = os.path.abspath(os.path.dirname(__file__))
ops_mapping_path = os.path.join(abs_path, 'ops_modality_mapping.json')

with open(ops_mapping_path, 'r') as f:
    op_modality_mapping = json.load(f)


@mcp.prompt()
def data_processing(requirements: str) -> list:

    user_message = f"""请帮我处理以下数据任务：
需求描述：{requirements}

请按以下步骤处理：
1. 分析需求需要何种类型的ops
2. 使用get_data_processing_ops工具获取可用的ops列表
3. 选择 ops 组合：
    尝试在可用 ops 列表中寻找完全匹配需求的一组 ops。
    如果未找到完全匹配的 ops，则尝试使用 Unknown 模态再次寻找。 这可能意味着某些算子属于通用类，或暂未进行明确的模态标识，因此在 Unknown 模态下可能存在符合需求的 ops。
    如果最终未找到完全匹配需求的 ops，则跳过该需求。
4. 使用run_data_recipe工具执行"""

    return [
        # {"role": "system", "content": system_message},
        {
            'role': 'user',
            'content': user_message
        },
    ]


# Operator Management
@mcp.tool()
def get_data_processing_ops(ops_type: str = '', modality: str = '') -> dict:
    """
    Retrieves a list of available data processing operators based on
    the specified type and data modality, with specific parameters.

    Operators are a collection of basic processes that assist in data modification,
    cleaning, filtering, deduplication, etc.
    The following `ops_type` values are supported:
    - aggregator: Aggregate for batched samples, such as summary or conclusion.
    - deduplicator: Detects and removes duplicate samples.
    - filter: Filters out low-quality samples.
    - formatter: Discovers, loads, and canonicalizes source data.
    - grouper: Group samples to batched samples.
    - mapper: Edits and transforms samples.
    - selector: Selects top samples based on ranking.
    The `modality` parameter specifies the type of data being processed.
    The following values are supported:
    - text: process text data specifically.
    - image: process image data specifically.
    - video: process video data specifically.
    - audio: process audio data specifically.
    - multimodal: process multimodal data.
    - unknow: Used for data types that are not clearly defined or do not fit into the other modality categories.
    If a matching operator is not found under other modality modes,
    consider checking this category to see if a suitable general-purpose operator is available.

    :param ops_type: The type of data processing operator to retrieve.
                     If empty, all operators under the specified modality are returned.
                     If specified, must be one of the values listed above with their descriptions. Defaults to None.
    :param modality: The modality of the data. If empty, all operators under the specified ops_type are returned.
                     If specified, must be one of the values listed above with their descriptions. Defaults to None.
    :returns: A dict of dictionaries, where each dictionary represents an available data processing operator.
    """

    if modality:
        ops_dict = op_modality_mapping[modality]
    else:
        ops_dict = {}
        for _, ops in op_modality_mapping.items():
            ops_dict.update(ops)

    if ops_type:
        ops_dict = {
            ops_name: ops_info
            for ops_name, ops_info in ops_dict.items() if ops_type in ops_name
        }

    return ops_dict


@mcp.tool()
def run_data_recipe(dataset_path: str, process: list[Dict]) -> str:
    """
    Run data recipe.

    :param dataset_path: Path to the dataset to be processed.
    :param process: List of process to be executed,
                    dictionary containing operator names as keys and operator parameter dictionaries as values
    """
    args_dict = dict(locals())
    args_dict.pop('dataset_path')
    dj_cfg = {
        'dataset_path': dataset_path,
        'process': process,
    }
    return execute_op(dj_cfg)


# Execution Pipeline
def add_extra_cfg(dj_cfg: Dict) -> Dict:
    """Add extra dj config."""
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
        executor = DefaultExecutor(dj_cfg)
        executor.run()
        return f"Result dataset is saved in: {dj_cfg['export_path']}"
    except Exception as e:
        return f'Occur error when executing Data-Juicer: {e}'


if __name__ == '__main__':
    mcp.run()
