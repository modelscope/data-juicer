import datetime
import os
import yaml
import requests
import json
import importlib

from PIL import Image
from loguru import logger

from agentscope.service import ServiceResponse, ServiceToolkit
from agentscope.message import Msg

api_url = 'http://localhost:8000'


def call_data_juicer_api(sub_url, params, js=None):
    url = api_url + sub_url

    if js is not None:
        response = requests.post(url, params=params, json=js)
    else:
        response = requests.get(url, params=params)
    
    return json.loads(response.text)


dj_default_config_path = './configs/dj_default_configs.yaml'
export_dir = './outputs'


def init_op_config(dataset_path: str, op_name: str, **op_args):
    """
    Init the data-juicer config with operator `op_name`.

    Args:
        dataset_path (`str`):
            The input dataset path.
        op_name: the name of operator to be inited.
        op_args: args of the op
    """
    with open(dj_default_config_path) as fin:
        dj_config = yaml.safe_load(fin)
    dj_config['dataset_path'] = dataset_path
    current_time = datetime.datetime.now()
    export_path = f"{current_time.strftime('%Y%m%d%H%M%S')}/processed_data.jsonl"
    dj_config['export_path'] = os.path.join(export_dir, export_path)
    dj_config['process'].append({op_name: op_args})
    try:
        res = call_data_juicer_api('/data_juicer/config/get_init_configs', {"cfg": json.dumps(dj_config)})
    except Exception as e:
        error_msg = f'An unexpected error occurred in Data-Juicer init: {e}'
        raise RuntimeError(error_msg)
    return res['result']


def execute_analyzer(dj_config: dict):
    """
    Execute data-juicer analyzer.

    Args:
        dj_config: configs of data-juicer
    """
    logger.chat(Msg(name="system", content="Analyzing data...", role="system"))
    try:
        res = call_data_juicer_api('/data_juicer/core/Analyzer/run', {"skip_return": True}, js={"cfg": json.dumps(dj_config)})
        assert res['status'] == 'success'
        return dj_config['export_path']
    except Exception as e:
        error_msg = f'An unexpected error occurred in Data-Juicer: {e}'
        raise RuntimeError(error_msg)

    
def show_analyzed_results(analyzed_result_path: str, require_min=True, require_max=True):
    """
    Show the analyzed results to the users and get the specified thresholds.

    Args:
        analyzed_result_path (`str`):
            The analyzed result path.
    """
    result_dir = os.path.dirname(analyzed_result_path)
    result_dir = os.path.join(result_dir, 'analysis')
    dist_path = None
    for root, dirs, files in os.walk(result_dir):
        for file_name in files:
            if file_name.endswith("hist.png"):
                dist_path = os.path.join(root, file_name)
    if dist_path is not None:
        img = Image.open(dist_path)
        img.show()
        min_threshold, max_threshold = 0, 0
        if require_min:
            min_threshold = float(input("Based on above analyzed results, enter the minimum threshold value for filter: "))
        if require_max:
            max_threshold = float(input("Based on above analyzed results, enter the maximum threshold value for filter: "))
        return min_threshold, max_threshold
    else:
        error_msg = f'Error in showing analyzed results: {analyzed_result_path}'
        raise RuntimeError(error_msg)


def execute_filter(dj_config: dict):
    """
    Execute data-juicer data process.

    Args:
        dj_config: configs of data-juicer
    """
    logger.chat(Msg(name="system", content="Processing data...", role="system"))
    try:
        res = call_data_juicer_api('/data_juicer/core/Executor/run', {"skip_return": True}, js={"cfg": json.dumps(dj_config)})
        assert res['status'] == 'success'
        return dj_config['export_path']
    except Exception as e:
        error_msg = f'An unexpected error occurred in Data-Juicer: {e}'
        raise RuntimeError(error_msg)


def add_to_service_toolkit(module_name, service_toolkit=None):
    if service_toolkit is None:
        service_toolkit = ServiceToolkit()
    module = importlib.import_module(module_name)
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj):
            service_toolkit.add(obj)
    return service_toolkit
