import json
import os
import re
import stat

from loguru import logger

from data_juicer.core.sandbox.env_manager import ENV_ROUTER
from data_juicer.core.sandbox.evaluators import BaseEvaluator
from data_juicer.core.sandbox.model_executors import BaseModelExecutor
from data_juicer.utils.resource_utils import cuda_device_count


class EasyAnimateTrainExecutor(BaseModelExecutor):
    """
    A training executor for text-to-video generation based on EasyAnimate.
    The home path of EasyAnimate is set to <data-juicer>/thirdparty/models/EasyAnimate in default.

    The config file for this executor should at least include the following items:
    1. `type`: must be "easyanimate".
    2. `env_name`: the name of the environment for EasyAnimate
    3. `env_manager`: the environment manager. Should be one of {"conda", "mamba", "venv", "virtualenv", "uv"}.
    4. `env_params`: a dict for other parameters of environments. Only works for conda-like environment. The
        `env_config_path` for creating the env and `env_py_version` to specify the Python version can be added.
    5. other items can be referred to configs/data_juicer_recipes/sandbox/easyanimate_text_to_video/model_train.yaml.
    """

    def __init__(self, model_config: dict, watcher=None):
        super().__init__(model_config, watcher)
        # env related
        easyanimate_env = self.model_config.get("env_name", None)
        easyanimate_env_manager = self.model_config.get("env_manager", "conda")
        easyanimate_env_params = self.model_config.get("env_params", {})
        cur_working_dir = os.getcwd()
        self.easyanimate_home = os.path.join(cur_working_dir, "thirdparty/models/EasyAnimate")
        self.env = ENV_ROUTER[easyanimate_env_manager](
            env_name=easyanimate_env, env_manager=easyanimate_env_manager, **easyanimate_env_params
        )
        self.env.create()
        # setup EasyAnimate
        cmd = f'cd {self.easyanimate_home.replace("EasyAnimate", "")} && bash setup_easyanimate.sh'
        self.env.run_cmd(cmd)
        # install requirements
        cmd = f"cd {self.easyanimate_home} && python install.py"
        self.env.run_cmd(cmd)
        # install extra deepspeed and func_timeout
        self.env.install_py_deps(["deepspeed", "func_timeout", "wandb"])

        self.script_path = os.path.join(self.easyanimate_home, "train_lora.sh")
        # make sure executable
        current_permissions = os.stat(self.script_path).st_mode
        os.chmod(self.script_path, current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    async def _run(self, run_type, run_obj=None, **kwargs):
        config = self.model_config.train
        run_args = [
            config.model_path.pretrained_model_name_or_path,
            config.model_path.transformer_path,
            config.dataset_path.dataset_name,
            config.dataset_path.dataset_meta_name,
            config.training_config.sample_size,
            config.training_config.mixed_precision,
            config.training_config.batch_size_per_gpu,
            config.training_config.gradient_accumulation_steps,
            config.training_config.num_train_epochs,
            config.training_config.dataloader_num_workers,
            config.training_config.seed,
            config.saving_config.output_dir,
            config.tracker_config.project_name,
            config.tracker_config.experiment_name,
        ]
        str_run_args = [f'"{str(arg)}"' for arg in run_args]
        cmd = f'cd {self.easyanimate_home} && bash {self.script_path} {" ".join(str_run_args)}'
        self.env.run_cmd(cmd)
        return os.path.abspath(config.saving_config.output_dir)


class EasyAnimateInferExecutor(BaseModelExecutor):
    """
    A inference executor for text-to-video generation based on EasyAnimate.
    The home path of EasyAnimate is set to <data-juicer>/thirdparty/models/EasyAnimate in default.

    The config file for this executor should at least include the following items:
    1. `type`: must be "easyanimate".
    2. `env_name`: the name of the environment for EasyAnimate
    3. `env_manager`: the environment manager. Should be one of {"conda", "mamba", "venv", "virtualenv", "uv"}.
    4. `env_params`: a dict for other parameters of environments. Only works for conda-like environment. The
        `env_config_path` for creating the env and `env_py_version` to specify the Python version can be added.
    5. other items can be referred to configs/data_juicer_recipes/sandbox/easyanimate_text_to_video/model_train.yaml.
    """

    def __init__(self, model_config: dict, watcher=None):
        super().__init__(model_config, watcher)
        # env related
        easyanimate_env = self.model_config.get("env_name", None)
        easyanimate_env_manager = self.model_config.get("env_manager", "conda")
        easyanimate_env_params = self.model_config.get("env_params", {})
        cur_working_dir = os.getcwd()
        self.easyanimate_home = os.path.join(cur_working_dir, "thirdparty/models/EasyAnimate")
        self.env = ENV_ROUTER[easyanimate_env_manager](
            env_name=easyanimate_env, env_manager=easyanimate_env_manager, **easyanimate_env_params
        )
        self.env.create()
        # setup EasyAnimate
        cmd = f'cd {self.easyanimate_home.replace("EasyAnimate", "")} && bash setup_easyanimate.sh'
        self.env.run_cmd(cmd)
        # install requirements
        cmd = f"cd {self.easyanimate_home} && python install.py"
        self.env.run_cmd(cmd)
        # install extra deepspeed and func_timeout
        self.env.install_py_deps(["deepspeed", "func_timeout", "wandb"])

        self.script_path = os.path.join(self.easyanimate_home, "infer_lora.sh")

    async def _run(self, run_type, run_obj=None, **kwargs):
        config = self.model_config.train
        run_args = [
            config.model_path.pretrained_model_name_or_path,
            config.model_path.transformer_path,
            config.model_path.lora_path,
            config.infer_config.image_size,
            config.infer_config.prompt_info_path,
            config.infer_config.gpu_num,
            config.infer_config.batch_size,
            config.infer_config.mixed_precision,
            config.infer_config.video_num_per_prompt,
            config.infer_config.seed,
            config.saving_config.output_video_dir,
        ]
        str_run_args = [f'"{str(arg)}"' for arg in run_args]
        cmd = f'cd {self.easyanimate_home} && bash {self.script_path} {" ".join(str_run_args)}'
        self.env.run_cmd(cmd)
        return os.path.abspath(config.saving_config.output_video_dir)


class VBenchEvaluator(BaseEvaluator):
    """
    An evaluator for VBench.

    The config for this executor should at least include the following items:
    1. `type`: must be "vbench_video_evaluator".
    2. `env_name`: the name of the environment for VBench.
    3. `env_manager`: the environment manager. Should be one of {"conda", "mamba", "venv", "virtualenv", "uv"}.
    4. `env_params`: a dict for other parameters of environments. Only works for conda-like environment. The
        `env_config_path` for creating the env and `env_py_version` to specify the Python version can be added.
    5. `num_gpus`: the number of GPUs used to evaluate. If it's not set, use all gpus.
    6. `eval_name`: the specified name for this evaluation hook.
    6. `full_json_dir`: path to save the json file that contains the prompt and dimension information.
    7. `output_path`: output path to save the evaluation results.
    8. `videos_path`: folder that contains the sampled videos.
    9. `dimension_list`: list of evaluation dimensions.
    10. `load_ckpt_from_local`: whether load checkpoints from local default paths.
    """

    def __init__(self, eval_config: dict):
        super().__init__(eval_config)
        # env related
        vbench_env = self.eval_config.get("env_name", None)
        vbench_env_manager = self.eval_config.get("env_manager", "conda")
        vbench_env_params = self.eval_config.get("env_params", {})
        self.env = ENV_ROUTER[vbench_env_manager](
            env_name=vbench_env, env_manager=vbench_env_manager, **vbench_env_params
        )
        self.env.create()
        # install vbench
        self.env.install_py_deps(
            [
                "vbench",
                "detectron2@git+https://github.com/facebookresearch/detectron2.git@b7c7f4ba82192ff06f2bbb162b9f67b00ea55867",
            ]
        )

        # eval gpus
        self.num_gpus = self.eval_config.get("num_gpus", cuda_device_count())
        if self.num_gpus <= 0:
            raise RuntimeError("No available GPUs.")
        if self.num_gpus > cuda_device_count():
            logger.warning(
                f"GPUs are not enough for {self.num_gpus}. Fallback to the number of all the GPUs "
                f"({cuda_device_count()}) on this machine."
            )
            self.num_gpus = cuda_device_count()

        # eval arguments
        self.eval_name = self.eval_config.get("eval_name", "vbench_evaluator")
        self.full_json_dir = self.eval_config.get("full_json_dir", None)
        self.output_path = self.eval_config.get("output_path", None)
        self.videos_path = self.eval_config.get("videos_path", None)
        self.dimension_list = self.eval_config.get("dimension_list", [])
        self.load_ckpt_from_local = self.eval_config.get("load_ckpt_from_local", False)

        if self.full_json_dir is None:
            raise ValueError("Please specify the full_json_dir.")
        if self.output_path is None:
            raise ValueError("Please specify the output_path.")
        if self.videos_path is None:
            raise ValueError("Please specify the videos_path.")
        if not isinstance(self.dimension_list, list) or len(self.dimension_list) == 0:
            raise ValueError("Please specify the dimension_list.")

        os.makedirs(self.output_path, exist_ok=True)

    def run(self, eval_type, eval_obj=None, **kwargs):
        if eval_type == "data":
            result_dict = {"mean_score": 0, "detail": {}}
            scores = []
            for dimension in self.dimension_list:
                eval_log_path = os.path.join(self.output_path, f"eval_log_{dimension}.txt")
                logger.info(f"Evaluating for {dimension}")
                cmd = (
                    f"vbench evaluate --ngpus {self.num_gpus} --full_json_dir {self.full_json_dir} "
                    f"--output_path {self.output_path} --videos_path {self.videos_path} "
                    f"--dimension {dimension} --load_ckpt_from_local {self.load_ckpt_from_local} 2>&1"
                    f' | tee -a "{eval_log_path}"'
                )
                self.env.run_cmd(cmd)
                # read eval log to find the result json file
                with open(eval_log_path, "r") as fin:
                    content = fin.read()
                result_name_pattern = r"Evaluation results saved to (.*?).json"
                res = re.findall(result_name_pattern, content)
                if len(res) > 0:
                    result_name = res[0]
                else:
                    raise RuntimeError("Cannot find the result json file from the evaluation log.")
                results = json.load(open(f"{result_name}.json", "r"))
                score = results[dimension][0]
                result_dict["detail"][dimension] = score
                scores.append(score)
            result_dict["mean_score"] = sum(scores) / len(scores)

            with open(os.path.join(self.output_path, f"{self.eval_name}_merged_results.json"), "w") as f:
                json.dump(result_dict, f)

            return float(result_dict["mean_score"])
        else:
            raise NotImplementedError("Unsupported evaluation type: {}".format(eval_type))
