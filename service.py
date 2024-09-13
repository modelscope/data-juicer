import importlib
import inspect
import json
import logging
import os
import time
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from jsonargparse import Namespace
from pydantic import validate_call

from data_juicer.core.exporter import Exporter
from data_juicer.format.load import load_formatter
from data_juicer.utils.file_utils import add_suffix_to_filename

logger = logging.getLogger('uvicorn.error')
app = FastAPI()


def register_objects_from_init(directory: str):
    """
    Traverse the specified directory for __init__.py files and
    register objects defined in their __all__ list.
    """
    for dirpath, _, filenames in os.walk(os.path.normpath(directory)):
        if '__init__.py' in filenames:
            module_path = dirpath.replace(os.sep, '.')
            module = importlib.import_module(module_path)

            if hasattr(module, '__all__'):
                for object_name in module.__all__:
                    obj = getattr(module, object_name)
                    if inspect.isclass(obj):
                        register_class(obj, object_name, dirpath)
                    elif callable(obj):
                        register_function(obj, object_name, dirpath)


def register_class(cls, class_name: str, module_path: str):
    """Register class and its methods as endpoints."""

    def create_class_call(method_name):

        async def class_call(request: Request):
            try:
                init_args = await request.json() if await request.body(
                ) else {}
                cls.__init__ = validate_call(
                    cls.__init__, config=dict(arbitrary_types_allowed=True))
                instance = cls(**init_args)
                method = validate_call(getattr(instance, method_name))
                q_params = parse_qs(request.url.query, keep_blank_values=True)
                d_params = dict((k, v if len(v) > 1 else v[0])
                                for k, v in q_params.items())
                exporter = None
                if dataset_path := d_params.get('dataset'):
                    if isinstance(dataset_path, str):
                        dataset = load_formatter(dataset_path).load_dataset()
                        d_params['dataset'] = dataset
                        export_path = add_suffix_to_filename(
                            dataset_path, f'_{int(time.time())}')
                        exporter = Exporter(export_path,
                                            keep_stats_in_res_ds=True,
                                            keep_hashes_in_res_ds=True,
                                            export_stats=False)
                # TODO: Traverse method's signature and convert any arguments \
                #  that should be Namespace but are passed as str
                if cfg_path := d_params.get('cfg'):
                    if isinstance(cfg_path, str):
                        cfg = Namespace(**json.loads(cfg_path))
                        d_params['cfg'] = cfg
                skip_return = d_params.pop('skip_return', False)
                result = method(**d_params)
                if exporter:
                    exporter.export(result)
                    result = export_path
                if skip_return:
                    result = ''
                return {'status': 'success', 'result': result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return class_call

    for method_name in get_public_methods(cls):
        api_path = f'/{module_path}/{class_name}/{method_name}'
        class_call = create_class_call(method_name)
        app.add_api_route(api_path,
                          class_call,
                          methods=['POST'],
                          tags=['POST'])
        logger.debug(f'Registered {api_path}')


def register_function(func, func_name: str, module_path: str):
    """Register a function as an endpoint."""
    api_path = f'/{module_path}/{func_name}'

    async def func_call(request: Request):
        try:
            nonlocal func
            func = validate_call(func)
            result = func(**request.query_params)
            return {'status': 'success', 'result': result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    app.add_api_route(api_path, func_call, methods=['GET'], tags=['GET'])
    logger.debug(f'Registered {api_path}')


def get_public_methods(cls):
    """Get public methods of a class."""
    selected = [
        'run', 'process', 'compute_stats', 'compute_hash', 'analyze', 'compute'
    ]
    return [
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith('_') and name in selected
    ]


# Specify the directories to search
directories_to_search = [
    'data_juicer',
    # "tools",  # Uncomment to add more directories
]

# Register objects from each specified directory
for directory in directories_to_search:
    register_objects_from_init(directory)
