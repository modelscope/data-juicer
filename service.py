import importlib
import inspect
import logging
import os

from fastapi import FastAPI, HTTPException, Request

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
    for method_name in get_public_methods(cls):
        api_path = f'/{module_path}/{class_name}/{method_name}'

        async def class_call(request: Request):
            try:
                init_args = await request.json() if await request.body(
                ) else {}
                instance = cls(**init_args)
                result = getattr(instance, method_name)(**request.query_params)
                return {'status': 'success', 'result': result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        app.add_api_route(api_path, class_call, methods=['POST'])
        logger.debug(f'Registered {api_path}')


def register_function(func, func_name: str, module_path: str):
    """Register a function as an endpoint."""
    api_path = f'/{module_path}/{func_name}'

    async def func_call(request: Request):
        try:
            result = func(**request.query_params)
            return {'status': 'success', 'result': result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    app.add_api_route(api_path, func_call, methods=['GET'])
    logger.debug(f'Registered {api_path}')


def get_public_methods(cls):
    """Get public methods of a class."""
    return [
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith('_')
    ]


# Specify the directories to search
directories_to_search = [
    'data_juicer',
    # "tools",  # Uncomment to add more directories
]

# Register objects from each specified directory
for directory in directories_to_search:
    register_objects_from_init(directory)
