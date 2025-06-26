import json
import os
import shutil
from typing import Any, Dict


from loguru import logger

from data_juicer.core.sandbox.env_manager import ENV_ROUTER
from tools.mm_eval.inception_metrics.calc_metrics_for_videos import \
    calc_metrics

# TODO: cannot import tools correctly if DJ is installed by pypi. Maybe we need
#       other importing methods.
from tools.quality_classifier.predict import predict_score


class BaseEvaluator(object):
    def __init__(self, eval_config: dict):
        self.eval_config = eval_config

    def run(self, eval_type, eval_obj=None, **kwargs) -> dict:
        """
        conduct the evaluation given specified measurement
            on specified target object;
        return evaluated results in a dict: {res_name: res_val}
        """
        raise NotImplementedError


class Gpt3QualityEvaluator(BaseEvaluator):
    def run(self, eval_type, eval_obj=None, **kwargs):
        if eval_type == "data":
            input_data_path = self.eval_config.dataset_path
            tmp_res_export_path = input_data_path + ".tmp_res.jsonl"
            if os.path.exists(tmp_res_export_path):
                if os.path.isfile(tmp_res_export_path):
                    os.remove(tmp_res_export_path)
                if os.path.isdir(tmp_res_export_path):
                    shutil.rmtree(tmp_res_export_path)

            overall_quality_stats = predict_score(input_data_path, tmp_res_export_path, overall_stats=True)

            shutil.rmtree(tmp_res_export_path)

            # by default, using the mean quality score of processed data
            # as final score
            return float(overall_quality_stats.loc["mean"])
        else:
            raise NotImplementedError("Unsupported evaluation type: {}".format(eval_type))


class InceptionEvaluator(BaseEvaluator):
    def run(self, eval_type, eval_obj=None, **kwargs):
        if eval_type == "data":
            result_dict = calc_metrics(
                fake_data_path=self.eval_config.fake_data_path,
                real_data_path=self.eval_config.real_data_path,
                fake_mm_dir=self.eval_config.fake_mm_dir,
                real_mm_dir=self.eval_config.real_mm_dir,
                metric=self.eval_config.metric,
                detector_path=self.eval_config.detector_path,
                result_path=self.eval_config.result_path,
                num_runs=self.eval_config.num_runs,
                height=self.eval_config.height,
                width=self.eval_config.width,
                replace_cache=self.eval_config.replace_cache,
                verbose=self.eval_config.verbose,
            )

            return result_dict
        else:
            raise NotImplementedError("Unsupported evaluation type: {}".format(eval_type))


class HelmEvaluator(BaseEvaluator):
    def run(self, eval_type, eval_obj=None, **kwargs):
        raise NotImplementedError("To be refactored from dj's `thirdparty`.")


class GptEvaluator(BaseEvaluator):
    def run(self, eval_type, eval_obj=None, **kwargs):
        raise NotImplementedError("To be refactored from `tools.evaluator`,")


class VideoFvdEvaluator(BaseEvaluator):
    def run(self, eval_type, eval_obj=None, **kwargs):
        raise NotImplementedError("To be refactored from video fvd/isv related tools.")


class Gpt4VEvaluator(BaseEvaluator):
    def run(self, eval_type, eval_obj=None, **kwargs):
        raise NotImplementedError("To be refactored from gpt4v related operators/tools.")


class LmHarnessEvaluator(BaseEvaluator):
    def run(self, eval_type, eval_obj=None, **kwargs):
        raise NotImplementedError("To be refactored from, used in data-juicer competition.")


class EvalscopeEvaluator(BaseEvaluator):
    """
    Evaluator using the EvalScope framework for LLM evaluation.

    Evalscope: https://github.com/modelscope/evalscope

    Args:
    Refer to https://evalscope.readthedocs.io/zh-cn/latest/index.html
    Configuration dictionary with the following items:
    - Required config:
        - `type`: muse be "evalscope_evaluator"
        - `env_name`: the name of the environment for evalscope
        - `env_manager`: the environment manager.
            Should be one of {"conda", "venv", "virtualenv", "uv"}.
        - `env_params`: a dict for other parameters of environments. Only works
            for conda-like environment. The `env_config_path` for creating the
            env and `env_py_version` to specify the Python version can be added.
        - `output_path`: output directory path
        - `evalscope_type`: execution mode ('config' or 'command')
    - evalscope_type-config:
        - `config_path`: path to configuration file
    - evalscope_type-command:
        - `eval_service`: Service type (default 'checkpoint' or 'service')
        - `model`: model's hf-id, local path or vllm model-id
        - `datasets`: dataset names for evaluation
        - `limits`: evaluation limits (optional)
        - `api_url`: API endpoint URL (service mode)
        - `api_key`: API authentication key (service mode, default 'EMPTY')
    """

    def __init__(self, eval_config: dict):
        super().__init__(eval_config)

        # output path
        self.output_path = eval_config.get('output_path')
        if not self.output_path:
            raise ValueError('output_path must be provided in eval_config')
        os.makedirs(self.output_path, exist_ok=True)

        # env related
        evalscope_env = self.eval_config.get('env_name', None)
        self.evalscope_env_manager = self.eval_config.get(
            'env_manager', 'conda')
        if self.evalscope_env_manager in ('venv', 'virtualenv', 'uv'):
            raise RuntimeError('To be implemented...')
        evalscope_env_params = self.eval_config.get('env_params', {})
        self.env = ENV_ROUTER[self.evalscope_env_manager](
            env_name=evalscope_env,
            env_manager=self.evalscope_env_manager,
            **evalscope_env_params)
        self.env.create()
        self.env.install_py_deps(['evalscope', 'evalscope[perf]'])

        # eval arguments
        self.model = self.eval_config.get('model')
        self.datasets = self.eval_config.get('datasets', [])
        if isinstance(self.datasets, str):
            self.datasets = self.datasets.split()
        self.limits = self.eval_config.get('limits')
        self.eval_service = self.eval_config.get('eval_service', 'checkpoint')
        self.evalscope_type = self.eval_config.get('evalscope_type', 'config')
        self.config_path = self.eval_config.get('config_path')

    def run(self, eval_type, eval_obj=None, **kwargs):
        work_dir = os.path.join(self.output_path, 'outputs')
        log_file = os.path.join(self.output_path, 'exe_eval.log')

        if self.evalscope_type == 'config':
            if not self.config_path:
                raise ValueError(
                    'config_path must be provided for config mode')
            cmd = f'python {self.config_path} --work_dir {work_dir} 2>&1 | tee "{log_file}"'

        else:
            if not all([self.model, self.datasets]):
                raise ValueError('model and datasets must be provided')

            cmd_parts = [
                'evalscope eval', f'--model "{self.model}"',
                f'--work-dir {work_dir}', f'--eval-type {self.eval_service}'
            ]
            if self.datasets:
                cmd_parts.append(f'--datasets {" ".join(self.datasets)}')
            if self.limits:
                cmd_parts.append(f'--limit {self.limits}')

            if self.eval_service == 'service':
                api_url = self.eval_config.get('api_url')
                if not api_url:
                    raise ValueError(
                        'api_url must be provided for service mode')

                api_key = self.eval_config.get('api_key', 'EMPTY')
                cmd_parts.extend(
                    [f'--api-url "{api_url}"', f'--api-key "{api_key}"'])

            cmd_parts.append(f'2>&1 | tee "{log_file}"')
            cmd = ' '.join(cmd_parts)

        logger.info(f'Running evalscope evaluation command: {cmd}')
        self.env.run_cmd(cmd)

        result_dict, mean_score = self.parse_results(work_dir, log_file)
        return result_dict

    def parse_results(self, work_dir: str,
                      log_file: str) -> tuple[Dict[str, Any], float]:
        try:
            latest_folder = self._get_latest_folder(work_dir)
        except Exception as e:
            raise RuntimeError(f'Failed to find latest result folder: {e}')

        reports_path = os.path.join(latest_folder, 'reports')
        if not os.path.exists(reports_path):
            logger.warning(f'Reports directory not found in {latest_folder}')

            result_dict = {
                'result': [{
                    'model': 'unknown',
                    'dataset': 'unknown',
                    'score': 0.0
                }],
                'mean_score':
                0.0,
                'error':
                f'Reports directory not found in {latest_folder}'
            }

            merged_result_path = os.path.join(self.output_path,
                                              'eval_results.json')
            with open(merged_result_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            return result_dict, 0.0

        result_dict = {'result': [], 'mean_score': 0.0}
        scores = []

        for model_name in os.listdir(reports_path):
            model_path = os.path.join(reports_path, model_name)
            if not os.path.isdir(model_path):
                continue
            for file_name in os.listdir(model_path):
                if file_name.endswith('.json'):
                    json_path = os.path.join(model_path, file_name)
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        score = data.get('score')
                        if score is None:
                            logger.warning(f'Score not found in {json_path}')
                            continue
                        scores.append(score)
                        result_dict['result'].append({
                            'model':
                            model_name,
                            'dataset':
                            data.get('dataset_name', file_name),
                            'score':
                            score
                        })
                    except Exception as e:
                        logger.error(f'Failed to parse {json_path}: {e}')
                        continue

        if not scores:
            logger.warning('No scores found in the evaluation results.')
            result_dict = {
                'result': [{
                    'model': 'unknown',
                    'dataset': 'unknown',
                    'score': 0.0
                }],
                'mean_score':
                0.0,
                'error':
                'No scores found in the evaluation results'
            }
        else:
            mean_score = sum(scores) / len(scores)
            result_dict['mean_score'] = mean_score

        merged_result_path = os.path.join(self.output_path,
                                          'eval_results.json')
        with open(merged_result_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        return result_dict, result_dict.get('mean_score', 0.0)

    def _get_latest_folder(self, base_path: str) -> str:
        if not os.path.exists(base_path):
            raise FileNotFoundError(f'Path does not exist: {base_path}')
        folders = [
            f for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))
        ]
        if not folders:
            raise RuntimeError(f'No subdirectories found in {base_path}')
        folders.sort(
            key=lambda x: os.path.getmtime(os.path.join(base_path, x)),
            reverse=True)
        return os.path.join(base_path, folders[0])
