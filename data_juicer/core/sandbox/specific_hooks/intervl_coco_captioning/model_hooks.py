import os
import re

from loguru import logger

from data_juicer.core.sandbox.data_pool_manipulators import check_io_paths
from data_juicer.core.sandbox.env_manager import ENV_ROUTER
from data_juicer.core.sandbox.evaluators import BaseEvaluator
from data_juicer.core.sandbox.model_executors import BaseModelExecutor
from data_juicer.utils.resource_utils import cuda_device_count


class InternVLCOCOCaptionTrainExecutor(BaseModelExecutor):
    """
    A training executor for InternVL COCO Caption Fine-Tuning.

    The config file for this executor should at least include the following items:
    1. `type`: must be "internvl_coco_caption".
    2. `env_name`: the name of the environment for InternVL
    3. `env_manager`: the environment manager. Should be one of {"conda", "mamba", "venv", "virtualenv", "uv"}.
    4. `env_params`: a dict for other parameters of environments. Only works for conda-like environment. The
        `env_config_path` for creating the env and `env_py_version` to specify the Python version can be added.
    5. `internvl_home`: the home path of InternVL. Used to install InternVL if it's not installed in the environment.
    6. `num_gpus`: the number of GPUs used to train. If it's not set, use all gpus.
    7. `batch_size_per_device`: the batch size on each gpu. If it's not set, use the default 4 that is aligned to the
        official script.
    8. `work_dir`: the working directory to store the outputs of training. Each experiment will create a subdirectory in
        it with the name of the basename of the input meta file.
    9. `model_name_or_path`: the model path as the initial model. The training is based on this model. According to the
        official doc, it's usually a path like `<internvl_home>/pretrained/xxx_model`.
    10. `conv_style_or_scale`: the conv style or scale of the base model. The mapping of scale to conv_style is:
        - 1B, 40B -> Hermes-2
        - 2B, 8B, 26B, 76B -> internlm2-chat
        - 4B -> phi3-chat
        Users can specify either conv_style or scale.
    11. `meta_paths`: the meta paths to be trained in sequence. Refer to:
        https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/data/coco_caption.json.
    12. `script_dir`: the path to store the training scripts.
    """

    SCALE_MAPPING = {
        "1B": "Hermes-2",
        "2B": "internlm2-chat",
        "4B": "phi3-chat",
        "8B": "internlm2-chat",
        "26B": "internlm2-chat",
        "40B": "Hermes-2",
        "76B": "internlm2-chat",
    }
    TOTAL_BATCH_SIZE = 512

    SCRIPT_TEMPLATE = """
    set -x

    cd %s

    GPUS=%d
    BATCH_SIZE=${BATCH_SIZE:-512}
    PER_DEVICE_BATCH_SIZE=%d
    GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export MASTER_PORT=34229
    export TF_CPP_MIN_LOG_LEVEL=3
    export LAUNCHER=pytorch

    OUTPUT_DIR='%s'

    if [ ! -d "$OUTPUT_DIR" ]; then
      mkdir -p "$OUTPUT_DIR"
    fi

    # total batch size: 512
    # epoch: 1
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      internvl/train/internvl_chat_finetune.py \
      --model_name_or_path "%s" \
      --conv_style "%s" \
      --output_dir ${OUTPUT_DIR} \
      --meta_path "%s" \
      --overwrite_output_dir True \
      --force_image_size 448 \
      --max_dynamic_patch 6 \
      --down_sample_ratio 0.5 \
      --drop_path_rate 0.0 \
      --freeze_llm True \
      --freeze_mlp True \
      --freeze_backbone True \
      --use_llm_lora 128 \
      --vision_select_layer -1 \
      --dataloader_num_workers 4 \
      --bf16 True \
      --num_train_epochs 1 \
      --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
      --gradient_accumulation_steps ${GRADIENT_ACC} \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 200 \
      --save_total_limit 1 \
      --learning_rate 4e-5 \
      --weight_decay 0.01 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --max_seq_length 4096 \
      --do_train True \
      --grad_checkpoint True \
      --group_by_length True \
      --dynamic_image_size True \
      --use_thumbnail True \
      --ps_version 'v2' \
      --deepspeed "zero_stage1_config.json" \
      --report_to "tensorboard" \
      2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
    """

    def __init__(self, model_config: dict, watcher=None):
        super().__init__(model_config, watcher)
        # env related
        internvl_env = self.model_config.get("env_name", None)
        internvl_env_manager = self.model_config.get("env_manager", "conda")
        internvl_env_params = self.model_config.get("env_params", {})
        self.internvl_home = self.model_config.get("internvl_home", None)
        self.env = ENV_ROUTER[internvl_env_manager](
            env_name=internvl_env, env_manager=internvl_env_manager, **internvl_env_params
        )
        self.env.create()
        if self.internvl_home is None:
            raise ValueError("Please specify the `internvl_home` in the config.")

        # following https://internvl.readthedocs.io/en/latest/get_started/installation.html
        # install requirements
        self.env.install_py_deps(os.path.join(self.internvl_home, "requirements.txt"))
        # install flash-attn
        self.env.install_py_deps(["flash-attn==2.3.6 --no-build-isolation", "datasets", "deepspeed==0.15.4"])

        # training related
        num_gpus = self.model_config.get("num_gpus", cuda_device_count())
        batch_size_per_device = self.model_config.get("batch_size_per_device", 4)
        work_dir = self.model_config.get("work_dir", None)
        model_name_or_path = self.model_config.get("model_name_or_path", None)
        conv_style_or_scale = self.model_config.get("conv_style_or_scale", None)
        meta_paths = self.model_config.get("meta_paths", [])
        script_dir = self.model_config.get("script_dir", None)
        # check training params
        if num_gpus <= 0:
            raise RuntimeError("No available GPUs.")
        if num_gpus > cuda_device_count():
            logger.warning(
                f"GPUs are not enough for {num_gpus}. Fallback to the number of all the GPUs "
                f"({cuda_device_count()}) on this machine."
            )
            num_gpus = cuda_device_count()
        if batch_size_per_device <= 0:
            raise ValueError("batch_size_per_device should be greater than 0.")
        if num_gpus * batch_size_per_device > self.TOTAL_BATCH_SIZE:
            raise ValueError(
                f"Total batch size on all devices in 1 step (num_gpus * batch_size_per_device) should be "
                f"less than {self.TOTAL_BATCH_SIZE}."
            )
        if self.TOTAL_BATCH_SIZE % (num_gpus * batch_size_per_device) != 0:
            raise ValueError(
                f"Total batch size on all devices in 1 step (num_gpus * batch_size_per_device) should be "
                f"divisible by {self.TOTAL_BATCH_SIZE}."
            )
        # check I/O paths
        existing_meta_paths, work_dir = check_io_paths(meta_paths, work_dir)
        if script_dir is None:
            raise ValueError("script_dir is not specified.")
        os.makedirs(script_dir, exist_ok=True)
        if conv_style_or_scale in self.SCALE_MAPPING:
            conv_style = self.SCALE_MAPPING[conv_style_or_scale]
        elif conv_style_or_scale in self.SCALE_MAPPING.values():
            conv_style = conv_style_or_scale
        else:
            raise ValueError(f"conv_style_or_scale [{conv_style_or_scale}] is not supported.")

        self.running_scripts, self.output_paths = self._prepare_training_script(
            script_dir, num_gpus, batch_size_per_device, work_dir, model_name_or_path, conv_style, existing_meta_paths
        )

        self.run_meta_name = "init"

    def _prepare_training_script(
        self, script_dir, num_gpus, batch_size_per_device, work_dir, model_name_or_path, conv_style, meta_paths
    ):
        script_paths = []
        output_paths = []

        for meta_path in meta_paths:
            meta_basename = os.path.splitext(os.path.basename(meta_path))[0]
            base_model_name = os.path.splitext(os.path.basename(model_name_or_path))[0]
            meta_basename = f"{meta_basename}_{base_model_name}"
            output_dir = os.path.join(work_dir, meta_basename)
            script_running_home = os.path.join(self.internvl_home, "internvl_chat")
            script_path = os.path.join(script_dir, f"{meta_basename}.sh")
            with open(script_path, "w") as f:
                f.write(
                    self.SCRIPT_TEMPLATE
                    % (
                        script_running_home,
                        num_gpus,
                        batch_size_per_device,
                        output_dir,
                        model_name_or_path,
                        conv_style,
                        meta_path,
                    )
                )

            script_paths.append(script_path)
            output_paths.append(output_dir)

        return script_paths, output_paths

    async def _run(self, run_type, run_obj=None, **kwargs):
        """
        run_type and run_obj is not used here.

        The home to run these scripts should be in "<internvl_home>/internvl_chat"
        """
        for script_path in self.running_scripts:
            # run the script
            logger.info(f"Training with script: {script_path}")
            cmd = f"bash {script_path}"
            # use system stdout and stderr because they are redirected by this hook
            self.env.run_cmd(cmd, use_sys_stdio=True)
        return self.output_paths

    def _watch_run(self, line, **kwargs):
        # e.g. "{'loss': 2.055, 'learning_rate': 3.0000000000000004e-05, 'epoch': 0.02}"
        pattern = r"^\{\'loss\': (.*?), \'learning_rate\': (.*?), \'epoch\': (.*?)\}$"
        run_meta_pattern = r"^run_name=\/(.*)\/(.*?),$"
        if self.watcher:
            match = re.match(pattern, line.strip())
            match_run_meta = re.match(run_meta_pattern, line.strip())
            if match:
                loss, lr, epoch = match.groups()
                self.watcher.watch({"loss": float(loss), "lr": float(lr), "epoch": float(epoch)}, self.run_meta_name)
            if match_run_meta:
                self.run_meta_name = match_run_meta.groups()[1]


class InternVLCOCOCaptionEvaluator(BaseEvaluator):
    """
    An evaluator for InternVL COCO Caption task.

    The config file for this executor should at least include the following items:
    1. `type`: must be "internvl_coco_caption".
    2. `env_name`: the name of the environment for InternVL
    3. `env_manager`: the environment manager. Should be one of {"conda", "mamba", "venv", "virtualenv", "uv"}.
    4. `env_params`: a dict for other parameters of environments. Only works for conda-like environment. The
        `env_config_path` for creating the env and `env_py_version` to specify the Python version can be added.
    5. `internvl_home`: the home path of InternVL. Used to install InternVL if it's not installed in the environment.
    6. `num_gpus`: the number of GPUs used to evaluate. If it's not set, use all gpus.
    7. `ckpt_paths`: the paths to the trained checkpoints.
    """

    DIMENSIONS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]

    def __init__(self, eval_config: dict):
        super().__init__(eval_config)
        # env related
        internvl_env = self.eval_config.get("env_name", None)
        internvl_env_manager = self.eval_config.get("env_manager", "conda")
        internvl_env_params = self.eval_config.get("env_params", {})
        self.internvl_home = self.eval_config.get("internvl_home", None)
        self.env = ENV_ROUTER[internvl_env_manager](
            env_name=internvl_env, env_manager=internvl_env_manager, **internvl_env_params
        )
        self.env.create()
        if self.internvl_home is None:
            raise ValueError("Please specify the `internvl_home` in the config.")

        # following https://internvl.readthedocs.io/en/latest/get_started/installation.html
        # install requirements
        self.env.install_py_deps(os.path.join(self.internvl_home, "requirements.txt"))
        # install flash-attn
        self.env.install_py_deps(["flash-attn==2.3.6 --no-build-isolation", "datasets", "deepspeed==0.15.4"])

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

        # checkpoint paths
        ckpt_paths = self.eval_config.get("ckpt_paths", [])
        if isinstance(ckpt_paths, str):
            ckpt_paths = [ckpt_paths]
        self.existing_ckpt_paths = []
        missing_paths = []
        for p in ckpt_paths:
            if not os.path.exists(p):
                missing_paths.append(p)
            else:
                self.existing_ckpt_paths.append(p)
        if len(missing_paths) > 0:
            logger.error(f'Input paths [{",".join(missing_paths)}] does not exist. Skipped!')

    def run(self, eval_type, eval_obj=None, **kwargs):
        """
        The evaluated results will be exported to the file "eval_log.txt" in the corresponding checkpoint dir. Then the
        scores of each dimension will be extracted from the file, including: {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
        "METEOR", "ROUGE_L", "CIDEr"}
        """
        results = []
        for ckpt_path in self.existing_ckpt_paths:
            logger.info(f"Evaluating {ckpt_path}")
            eval_log_path = os.path.join(ckpt_path, "eval_log.txt")
            script_path = os.path.join(self.internvl_home, "internvl_chat")
            # need to convert to a relative path due to the script "evaluate.sh" will add the script path forcely
            ckpt_path = os.path.relpath(ckpt_path, script_path)
            cmd = f"cd {script_path}"
            cmd += (
                f" && GPUS={self.num_gpus} bash evaluate.sh {ckpt_path} caption-coco --dynamic 2>&1"
                f' | tee -a "{eval_log_path}"'
            )
            self.env.run_cmd(cmd)

            # parse results from eval_log_path
            with open(eval_log_path, "r") as fin:
                content = fin.read()
            result = {}
            for dim in self.DIMENSIONS:
                pattern = rf"{dim}: (.*?)\n"
                res = re.findall(pattern, content)
                if len(res) > 0:
                    result[dim] = float(res[0])
            result["avg_score"] = sum(result.values()) / len(result)
            results.append(result)

        return results
