import asyncio
import os.path
import re
import sys
import time

from data_juicer.utils.file_utils import follow_read


class BaseModelExecutor(object):
    """
    Base abstraction for model executor within the DataJuicer's sandbox
    """

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.executor = None
        self.watcher = None
        # log to tell the end of model execution
        self.END_OF_MODEL_EXEC = \
            "<DJ-Sandbox> End of ModelExecutor's running <DJ-Sandbox>"

    async def run(self, run_type, run_obj, **kwargs):
        """
        conduct some model-related execution tasks
            given specified run_type and run_obj
        """
        run_task = asyncio.create_task(self._run(run_type, run_obj, **kwargs))
        watch_task = asyncio.create_task(
            self.watch_run(run_type, run_obj, **kwargs))

        if self.watcher is None:
            await run_task
            return None
        else:
            original_stdout = sys.stdout
            try:
                timestamp = time.strftime('%Y%m%d%H%M%S',
                                          time.localtime(time.time()))
                log_f_name = os.path.join(
                    self.watcher.dj_cfg.work_dir,
                    f'model_exe_{run_type}_{timestamp}.log')
                self.watcher.model_exe_log_file = log_f_name
                summarized_watched_res = await watch_task
                with open(log_f_name, 'w') as log_f:
                    sys.stdout = log_f
                    run_task = asyncio.create_task(
                        self._run(run_type, run_obj, **kwargs))
                    await run_task
                    print(self.END_OF_MODEL_EXEC)
            finally:
                sys.stdout = original_stdout
            return summarized_watched_res

    async def _run(self, run_type, run_obj, **kwargs):
        raise NotImplementedError

    async def watch_run(self, run_type, run_obj, **kwargs):
        """
        watch the running process in an online manner, and
            return the summarized results
        """
        while True:
            for line in follow_read(self.watcher.model_exe_log_file):
                if self.END_OF_MODEL_EXEC in line:
                    break
                else:
                    self._watch_run(line, **kwargs)

    def _watch_run(self, line, **kwargs):
        """
        customized log watcher, depending on specific model executor
        """
        raise NotImplementedError

    def data_connector(self, input_data, **kwargs):
        """
        convert input_data (usually in Data-Juicer's Dataset format) into
            the appropriate format for specific model executor
        """
        raise NotImplementedError


class ModelScopeExecutor(BaseModelExecutor):

    def data_connector(self, input_data, **kwargs):
        try:
            from modelscope.msdatasets import MsDataset
            return MsDataset.to_ms_dataset(input_data)
        except ModuleNotFoundError:
            raise ModuleNotFoundError('modelscope package not installed')

    def _watch_run(self, line, **kwargs):
        # e.g., "2023-07-02 17:26:50,324 - modelscope - INFO - epoch
        # [1][100/4953]\tlr: 4.825e-05, memory: 8742, loss: 5.6125\n"
        pattern = r'loss:\s*([0-9.]+)'

        if self.watcher is not None:
            match = re.search(pattern, line)
            if match:
                loss_value = match.group(1)
                self.watcher.watch(loss_value, 'loss')


class ModelscopeInferExecutor(ModelScopeExecutor):

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            from modelscope.pipelines import pipeline
            self.executor = pipeline(**model_config)
        except ModuleNotFoundError:
            raise ModuleNotFoundError('modelscope package not installed')

    def run(self, run_type, run_obj, **kwargs):
        if run_type == 'infer_on_data':
            return self.executor(self.data_connector(run_obj), **kwargs)
        else:
            raise ValueError(f'run_type {run_type} not supported')


class ModelscopeTrainExecutor(ModelScopeExecutor):

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.model_config = model_config
        try:
            from modelscope.trainers import build_trainer
            self.executor = build_trainer(**model_config)
        except ModuleNotFoundError:
            raise ModuleNotFoundError('modelscope package not installed')

    def run(self, run_type, run_obj, **kwargs):
        if run_type == 'has_configured':
            self.executor.train()
        elif run_type == 'update_config':
            # training cfg updated, such as datasets and training parameters
            if issubclass(type(run_obj), dict):
                builder_kwargs = {**run_obj, **kwargs}
            else:
                builder_kwargs = kwargs
            if 'train_dataset' in builder_kwargs:
                builder_kwargs['train_dataset'] = self.data_connector(
                    builder_kwargs['train_dataset'])
            if 'eval_dataset' in builder_kwargs:
                builder_kwargs['eval_dataset'] = self.data_connector(
                    builder_kwargs['eval_dataset'])
            self.__init__(**builder_kwargs)
            self.executor.train()
        else:
            raise ValueError(f'run_type {run_type} not supported')


# TODO: add watcher for the re-directed output streams and log parsers
class EasySoraExecutor(BaseModelExecutor):

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError('To be implemented from easysora.')


class LLaVAExecutor(BaseModelExecutor):

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError("To be refactored, from DJ's refinement for "
                                  'LLaVA related experiments.')


class LLaMAFactoryExecutor(BaseModelExecutor):

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError('To be refactored, which is used in '
                                  'data-juicer competition.')


class MegatronExecutor(BaseModelExecutor):

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError("To be refactored from dj's `thirdparty`.")
