from typing import Union

from .default_executor import DefaultExecutor
from .ray_executor import RayExecutor


class ExecutorFactory:

    @staticmethod
    def create_executor(
            executor_type: str) -> Union[DefaultExecutor, RayExecutor]:
        if executor_type in ('local', 'default'):
            return DefaultExecutor()
        elif executor_type == 'ray':
            return RayExecutor()
        # TODO: add nemo support
        #  elif executor_type == "nemo":
        #    return NemoExecutor()
        # TODO: add dask support
        #  elif executor_type == "dask":
        #    return DaskExecutor()
        else:
            raise ValueError('Unsupported executor type')
