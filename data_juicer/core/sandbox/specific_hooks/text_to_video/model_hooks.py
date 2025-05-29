import json
import os
import re

from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.sandbox.env_manager import ENV_ROUTER
from data_juicer.core.sandbox.evaluators import BaseEvaluator


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
        vbench_env = self.eval_config.get('env_name', None)
        vbench_env_manager = self.eval_config.get('env_manager', 'conda')
        vbench_env_params = self.eval_config.get('env_params', {})
        self.env = ENV_ROUTER[vbench_env_manager](
            env_name=vbench_env,
            env_manager=vbench_env_manager,
            **vbench_env_params)
        self.env.create()
        # install vbench
        self.env.install_py_deps([
            'vbench',
            'detectron2@git+https://github.com/facebookresearch/detectron2.git@b7c7f4ba82192ff06f2bbb162b9f67b00ea55867'
        ])

        # eval gpus
        self.num_gpus = self.eval_config.get('num_gpus', cuda_device_count())
        if self.num_gpus <= 0:
            raise RuntimeError('No available GPUs.')
        if self.num_gpus > cuda_device_count():
            logger.warning(
                f'GPUs are not enough for {self.num_gpus}. Fallback to the number of all the GPUs '
                f'({cuda_device_count()}) on this machine.')
            self.num_gpus = cuda_device_count()

        # eval arguments
        self.eval_name = self.eval_config.get('eval_name', 'vbench_evaluator')
        self.full_json_dir = self.eval_config.get('full_json_dir', None)
        self.output_path = self.eval_config.get('output_path', None)
        self.videos_path = self.eval_config.get('videos_path', None)
        self.dimension_list = self.eval_config.get('dimension_list', [])
        self.load_ckpt_from_local = self.eval_config.get(
            'load_ckpt_from_local', False)

        if self.full_json_dir is None:
            raise ValueError('Please specify the full_json_dir.')
        if self.output_path is None:
            raise ValueError('Please specify the output_path.')
        if self.videos_path is None:
            raise ValueError('Please specify the videos_path.')
        if not isinstance(self.dimension_list, list) or len(
                self.dimension_list) == 0:
            raise ValueError('Please specify the dimension_list.')

    def run(self, eval_type, eval_obj=None, **kwargs):
        if eval_type == 'data':
            result_dict = {'mean_score': 0, 'detail': {}}
            scores = []
            eval_log_path = os.path.join(self.output_path, 'eval_log.txt')
            logger.info(f'Evaluating for {self.dimension_list}')
            cmd = f'vbench evaluate --ngpus {self.num_gpus} --full_json_dir {self.full_json_dir} ' \
                  f'--output_path {self.output_path} --videos_path {self.videos_path} ' \
                  f'--dimension {self.dimension_list} --load_ckpt_from_local {self.load_ckpt_from_local} 2>&1' \
                  f' | tee -a "{eval_log_path}"'
            self.env.run_cmd(cmd)
            # read eval log to find the result json file
            with open(eval_log_path, 'r') as fin:
                content = fin.read()
            result_name_pattern = r'Evaluation results saved to (.*?).json'
            res = re.findall(result_name_pattern, content)
            if len(res) > 0:
                result_name = res[0]
            else:
                raise RuntimeError(
                    'Cannot find the result json file from the evaluation log.'
                )
            results = json.load(open(f'{result_name}.json', 'r'))
            for dimension in results:
                score = results[dimension][0]
                result_dict['detail'][dimension] = score
                scores.append(score)
            result_dict['mean_score'] = sum(scores) / len(scores)

            with open(
                    os.path.join(self.output_path,
                                 f'{self.eval_name}_merged_results.json'),
                    'w') as f:
                json.dump(result_dict, f)

            return float(result_dict['mean_score'])
        else:
            raise NotImplementedError(
                'Unsupported evaluation type: {}'.format(eval_type))
