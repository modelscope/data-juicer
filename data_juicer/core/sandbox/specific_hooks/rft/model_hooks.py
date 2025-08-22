import json
import re

import yaml

from data_juicer.core.sandbox.env_manager import ENV_ROUTER
from data_juicer.core.sandbox.model_executors import BaseModelExecutor


class TrinityRFTTrainExecutor(BaseModelExecutor):
    """
    A training executor for [Trinity-RFT](https://github.com/modelscope/Trinity-RFT).

    The config file for this executor should at least include the following items:
    1. `type`: must be "trinity-rft". Otherwise, this executor won't be returned.
    2. `env_name`: the name of the environment for Trinity-RFT
    3. `env_manager`: the environment manager. Should be one of {"conda", "mamba", "venv", "virtualenv", "uv"}.
    4. `env_params`: a dict for other parameters of environments. Only works for conda-like environment. The
        `env_config_path` for creating the env and `env_py_version` to specify the Python version can be added.
    5. `trinity_home`: the home path of Trinity-RFT. Used to install Trinity-RFT if it's not installed in the
        environment.
    6. `trinity_config_path`: the config file path to train with Trinity-RFT. Refer to this
        [doc](https://modelscope.github.io/Trinity-RFT/tutorial/trinity_configs.html) for more details.
    """

    def __init__(self, model_config: dict, watcher=None):
        super().__init__(model_config, watcher)
        trinity_env = self.model_config.get("env_name", None)
        trinity_env_manager = self.model_config.get("env_manager", "conda")
        trinity_env_params = self.model_config.get("env_params", {})
        trinity_home = self.model_config.get("trinity_home", None)
        # prepare trinity environment
        self.env = ENV_ROUTER[trinity_env_manager](
            env_name=trinity_env, env_manager=trinity_env_manager, **trinity_env_params
        )
        self.env.create()
        if trinity_home:
            self.env.install_py_deps(trinity_home)

    async def _run(self, run_type, run_obj=None, **kwargs):
        """
        run_obj here is used to pass the trinity config path.
        """
        if run_obj is not None and isinstance(run_obj, str):
            trinity_config_path = run_obj
        else:
            trinity_config_path = self.model_config.get("trinity_config_path", None)
        cmd = f"trinity run --config {trinity_config_path}"
        self.env.run_cmd(cmd)
        trinity_cfg = yaml.safe_load()
        return trinity_cfg.get("checkpoint_root_dir", None)

    async def _watch_run(self, line, **kwargs):
        # e.g. "(Trainer pid=3839503) INFO 05-08 11:11:13 monitor.py:96] Step 12: {<a dict record>}"
        pattern = r"^\(Trainer pid=(\d+)\) INFO (.*?) monitor.py:(.*?)] Step (\d*?): (.*?)$"
        if self.watcher:
            match = re.match(pattern, line.strip())
            if match:
                pid, time, lino, step, record = match.groups()
                try:
                    record_dict = json.loads(record.replace("'", '"'))
                except:  # noqa: E722
                    record_dict = {}
                if "perf/throughput" in record_dict:
                    for key, val in record_dict.items():
                        self.watcher.watch(val, key)
