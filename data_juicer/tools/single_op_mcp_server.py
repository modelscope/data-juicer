import datetime
import inspect
import os
import re
import sys
from typing import Annotated, Dict, Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from data_juicer.config import get_init_configs
from data_juicer.core import DefaultExecutor
from data_juicer.ops import OPERATORS

# Global Configuration
DEFAULT_OUTPUT_DIR = './outputs'
mcp = FastMCP('Data-Juicer Server')

# Operator Management
ops_list_path = os.getenv('DJ_OPS_LIST_PATH', None)
if ops_list_path:
    with open(ops_list_path, 'r', encoding='utf-8') as file:
        ops_list = [line.strip() for line in file if line.strip()]
else:
    ops_list = list(OPERATORS.modules.keys())


# Dynamic MCP Tool Creation
def extract_param_docstring(docstring):
    """
    Extract:param from docstring of__init__method.
    """
    params = []
    if not docstring:
        return params
    param_pattern = re.compile(r'(:param\s+(?!args|kwargs)\w+:\s+([^:]*))')
    matches = param_pattern.findall(docstring)
    for match in matches:
        params.append(match[0])
    return params


def process_parameter(name: str,
                      param: inspect.Parameter) -> inspect.Parameter:
    """
    Processes a function parameter:
    - Converts jsonargparse.typing.ClosedUnitInterval to a local equivalent annotation.
    """
    ClosedUnitInterval = Annotated[
        float,
        Field(ge=0.0, le=1.0, description='float restricted to be ≥0 and ≤1')]
    if param.annotation == getattr(sys.modules.get('jsonargparse.typing'),
                                   'ClosedUnitInterval', None):
        return param.replace(annotation=ClosedUnitInterval)
    return param


def create_operator_function(op_name, op_cls):
    """Creates a callable function for a Data-Juicer operator class.

    This function dynamically creates a function that can be registered as an MCP tool,
    with proper signature and documentation based on the operator's __init__ method.
    """
    sig = inspect.signature(op_cls.__init__)
    docstring = op_cls.__doc__ or ''
    init_docstring = op_cls.__init__.__doc__ or ''
    param_docs = extract_param_docstring(init_docstring)

    # Build :param docstring section
    param_docstring = ''
    if param_docs:
        for param in param_docs:
            param_docstring += f'    {param}\n'

    # Create new function signature with dataset_path as first parameter
    # Consider adding other common parameters later, such as export_psth
    new_parameters = [
        inspect.Parameter('dataset_path',
                          inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=str),
        inspect.Parameter(
            'export_path',
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Optional[str],
            default=None,
        ),
    ] + [
        process_parameter(name, param)
        for name, param in sig.parameters.items()
        if name not in ('args', 'kwargs', 'self')
    ]
    new_signature = sig.replace(parameters=new_parameters,
                                return_annotation=str)

    def func(*args, **kwargs):
        args_dict = {}
        bound_arguments = new_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        export_path = bound_arguments.arguments.pop('export_path')
        dataset_path = bound_arguments.arguments.pop('dataset_path')
        args_dict = {k: v for k, v in bound_arguments.arguments.items() if v}

        dj_cfg = {
            'dataset_path': dataset_path,
            'export_path': export_path,
            'process': [{
                op_name: args_dict
            }],
        }
        return execute_op(dj_cfg)

    func.__signature__ = new_signature
    func.__doc__ = f"""{docstring}\n\n{param_docstring}\n"""
    func.__name__ = op_name

    decorated_func = mcp.tool()(func)

    return decorated_func


# Register all operators as MCP tools
for op_name in ops_list:
    op_cls = OPERATORS.get(op_name)
    if op_cls is None:
        continue
    operator_function = create_operator_function(op_name, op_cls)


# Execution Pipeline
def add_extra_cfg(dj_cfg: Dict) -> Dict:
    """Add extra dj config."""
    if not dj_cfg.get('export_path'):
        logger.info('export_path is not set, use default export_path')
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
