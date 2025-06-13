from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from data_juicer.tools.mcp_tool import execute_op
from data_juicer.tools.op_search import OPSearcher

# Server configuration
mcp = FastMCP('Data-Juicer Server')

# Operator Management
searcher = OPSearcher()


@mcp.prompt()
def data_processing(requirements: str) -> list:

    user_message = f"""请帮我处理以下数据任务：
需求描述：{requirements}

请按以下步骤处理：
1. 分析需求需要何种类型的ops
2. 使用get_data_processing_ops工具获取可用的ops列表
3. 选择 ops 组合：
    尝试在可用 ops 列表中寻找完全匹配需求的一组 ops。
    如果最终未找到完全匹配需求的 ops，则跳过该需求。
4. 使用run_data_recipe工具执行"""

    return [
        {
            'role': 'user',
            'content': user_message
        },
    ]


# Operator Management
@mcp.tool()
def get_data_processing_ops(
    op_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    match_all: bool = True,
) -> dict:
    """
    Retrieves a list of available data processing operators based on the specified type and tags.
    Operators are a collection of basic processes that assist in data modification,
    cleaning, filtering, deduplication, etc.

    Should be used with run_data_recipe.

    If both tags and ops_type are None, return a list of all operators.

    The following `op_type` values are supported:
    - aggregator: Aggregate for batched samples, such as summary or conclusion.
    - deduplicator: Detects and removes duplicate samples.
    - filter: Filters out low-quality samples.
    - grouper: Group samples to batched samples.
    - mapper: Edits and transforms samples.
    - selector: Selects top samples based on ranking.

    The `tags` parameter specifies the characteristics of the data or the required resources.
    Available tags are:

    Modality Tags:
        - text: process text data specifically.
        - image: process image data specifically.
        - audio: process audio data specifically.
        - video: process video data specifically.
        - multimodal: process multimodal data.

    Resource Tags:
        - cpu: only requires CPU resource.
        - gpu: requires GPU/CUDA resource as well.

    Model Tags:
        - api: equipped with API-based models (e.g. ChatGPT, GPT-4o).
        - vllm: equipped with models supported by vLLM.
        - hf: equipped with models from HuggingFace Hub.

    Tags are used to refine the search for suitable operators based on specific data processing needs.

    :param op_type: The type of data processing operator to retrieve.
                     If None, no ops_type-based filtering is applied.
                     If specified, must be one of the values listed. Defaults to None.
    :param tags: An optional list of tags to filter operators.  See the tag list above for options.
                 If None, no tag-based filtering is applied. Defaults to None.
    :param match_all: If True, only operators matching all specified tags are returned.
                      If False, operators matching any of the specified tags are returned. Defaults to True.
    :returns: A dict containing detailed information about the available operators
    """
    op_results = searcher.search(tags=tags,
                                 op_type=op_type,
                                 match_all=match_all)

    ops_dict = dict()
    for op in op_results:
        ops_dict[op['name']] = '\n'.join([
            op['description'], op['param_desc'], 'Parameters: ',
            str(op['signature'])
        ])

    return ops_dict


@mcp.tool()
def run_data_recipe(
    dataset_path: str,
    process: list[Dict],
    export_path: Optional[str] = None,
    np: int = 1,
) -> str:
    """
    Run data recipe.
    It is recommended to use get_data_processing_ops to obtain a list of
    operators related to your data processing requirements, organized into
    your data processing flow.

    :param dataset_path: Path to the dataset to be processed.
    :param process: List of processing operations to be executed sequentially.
                Each element is a dictionary with operator name as key and
                its configuration as value. Multiple operators can be chained.
    :param export_path: Path to export the processed dataset. Defaults to None,
                       which exports to './outputs' directory.
    :param np: Number of processes to use. Defaults to 1.

    Example:
        Run a data recipe to filter samples with text length outside 10-50 range:
        >>> run_data_recipe(
        ...     "/path/to/dataset.jsonl",
        ...     [
        ...         {
        ...             "text_length_filter": {
        ...                 "min_len": 10,
        ...                 "max_len": 50
        ...             }
        ...         }
        ...     ]
        ... )
    """
    dj_cfg = dict(locals())
    return execute_op(dj_cfg)


if __name__ == '__main__':
    mcp.run()
