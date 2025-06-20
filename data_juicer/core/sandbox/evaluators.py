import os
import shutil

from tools.mm_eval.inception_metrics.calc_metrics_for_videos import calc_metrics

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


class ModelscopeEvaluator(BaseEvaluator):
    def run(self, eval_type, eval_obj=None, **kwargs):
        raise NotImplementedError("To be implemented from https://github.com/modelscope/eval-scope.")
