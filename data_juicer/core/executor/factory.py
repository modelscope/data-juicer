from typing import Union

from .local_executor import LocalExecutor
from .ray_executor import RayExecutor


class ExecutorFactory:

    @staticmethod
    def create_executor(
            executor_type: str) -> Union[LocalExecutor, RayExecutor]:
        if executor_type == 'local':
            return LocalExecutor()
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
