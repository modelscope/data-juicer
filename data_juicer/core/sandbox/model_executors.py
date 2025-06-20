import asyncio
import os
import re
import subprocess
import sys
import time

from jsonargparse import namespace_to_dict

from data_juicer.utils.file_utils import follow_read


class BaseModelExecutor(object):
    """
    Base abstraction for model executor within the DataJuicer's sandbox
    """

    def __init__(self, model_config: dict, watcher=None):
        self.model_config = model_config
        self.executor = None
        self.watcher = watcher
        # log to tell the end of model execution
        self.END_OF_MODEL_EXEC = "<DJ-Sandbox> End of ModelExecutor's running <DJ-Sandbox>"

    async def run(self, run_type, run_obj=None, **kwargs):
        """
        conduct some model-related execution tasks
            given specified run_type and run_obj
        """
        watch_task = asyncio.create_task(self.watch_run(run_type, run_obj, **kwargs))
        if self.watcher is None:
            run_task = asyncio.create_task(self._run(run_type, run_obj, **kwargs))
            ret = await run_task
            return ret
        else:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
                log_f_name = os.path.join(self.watcher.sandbox_cfg.work_dir, f"model_exe_{run_type}_{timestamp}.log")
                self.watcher.model_exe_log_file = log_f_name
                with open(log_f_name, "w") as log_f:
                    sys.stdout = log_f
                    sys.stderr = log_f
                    run_task = asyncio.create_task(self._run(run_type, run_obj, **kwargs))
                    ret = await run_task
                    print(self.END_OF_MODEL_EXEC, flush=True)
                    await watch_task
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            return ret

    def run_subprocess(self, script_path, run_args, working_dir, cmd="bash"):
        run_args = [str(arg) for arg in run_args]
        args = [cmd, script_path] + run_args
        subprocess.run(args, cwd=working_dir)

    async def _run(self, run_type, run_obj=None, **kwargs):
        raise NotImplementedError

    async def watch_run(self, run_type, run_obj=None, **kwargs):
        """
        watch the running process in an online manner, and
            return the summarized results
        """
        met_eof = False
        while not met_eof:
            if os.path.exists(self.watcher.model_exe_log_file):
                async for line in follow_read(self.watcher.model_exe_log_file):
                    if self.END_OF_MODEL_EXEC in line:
                        met_eof = True
                        break
                    else:
                        self._watch_run(line, **kwargs)
            else:
                await asyncio.sleep(0.1)

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
    def data_connector(self, input_data, split="train", key_remapping=None, **kwargs):
        try:
            from modelscope.msdatasets import MsDataset

            if isinstance(input_data, str):
                # dataset path
                ds = MsDataset.load(input_data, split=split).ds_instance
            else:
                ds = input_data
            if key_remapping:
                features = ds.features
                for key in key_remapping:
                    if key in features:
                        ds = ds.rename_column(key, key_remapping[key])
            return ds
        except ModuleNotFoundError:
            raise ModuleNotFoundError("modelscope package not installed")

    def _watch_run(self, line, **kwargs):
        # e.g., "2023-07-02 17:26:50,324 - modelscope - INFO - epoch
        # [1][100/4953]\tlr: 4.825e-05, memory: 8742, loss: 5.6125\n"
        pattern = r"loss:\s*([0-9.]+)"

        if self.watcher is not None:
            match = re.search(pattern, line)
            if match:
                loss_value = float(match.group(1))
                self.watcher.watch(loss_value, "loss")


class ModelscopeInferProbeExecutor(ModelScopeExecutor):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            from modelscope.pipelines import pipeline

            self.executor = pipeline(**model_config)
        except ModuleNotFoundError:
            raise ModuleNotFoundError("modelscope package not installed")

    async def _run(self, run_type, run_obj=None, **kwargs):
        if run_type == "infer_on_data":
            return self.executor(self.data_connector(run_obj), **kwargs)
        else:
            raise ValueError(f"run_type {run_type} not supported")


class ModelscopeTrainExecutor(ModelScopeExecutor):
    def __init__(self, model_config, watcher=None):
        super().__init__(model_config, watcher)
        self.model_config = namespace_to_dict(model_config)
        self.executor = None

    def cfg_modify_fn(self, cfg):
        cfg.merge_from_dict(self.model_config)
        return cfg

    def build_executor(self, model_name, trainer_name, work_dir, train_dataset=None, eval_dataset=None):
        try:
            from modelscope.trainers import build_trainer

            kwargs = dict(
                model=model_name,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                work_dir=work_dir,
                cfg_modify_fn=self.cfg_modify_fn,
            )
            self.executor = build_trainer(
                name=trainer_name,
                default_args=kwargs,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError("modelscope package not installed")

    async def _run(self, run_type, run_obj=None, **kwargs):
        # training cfg updated, such as datasets and training parameters
        builder_kwargs = {
            "model_name": self.model_config["model_name"],
            "trainer_name": self.model_config["trainer_name"],
        }
        if "key_remapping" in self.model_config:
            key_remapping = self.model_config["key_remapping"]
        else:
            key_remapping = None
        if "train_dataset" in self.model_config:
            builder_kwargs["train_dataset"] = self.data_connector(
                self.model_config["train_dataset"], split="train", key_remapping=key_remapping
            )
        if "eval_dataset" in self.model_config:
            builder_kwargs["eval_dataset"] = self.data_connector(
                self.model_config["eval_dataset"], split="val", key_remapping=key_remapping
            )
        if "work_dir" in self.model_config:
            builder_kwargs["work_dir"] = self.model_config["work_dir"]
        self.work_dir = builder_kwargs["work_dir"]
        self.build_executor(**builder_kwargs)
        self.executor.train()
        return self.work_dir


class LLaVAExecutor(BaseModelExecutor):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError("To be refactored, from DJ's refinement for " "LLaVA related experiments.")


class LLaMAFactoryExecutor(BaseModelExecutor):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError("To be refactored, which is used in " "data-juicer competition.")


class MegatronExecutor(BaseModelExecutor):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError("To be refactored from dj's `thirdparty`.")
