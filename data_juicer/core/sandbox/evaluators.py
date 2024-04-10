import os
import shutil

from datasets import Dataset

from tools.quality_classifier.predict import predict_score


class BaseEvaluator(object):

    def __init__(self, eval_config: dict):
        self.eval_config = eval_config

    def run(self, eval_type, eval_obj, **kwargs) -> dict:
        """
            conduct the evaluation given specified measurement
                on specified target object;
            return evaluated results in a dict: {res_name: res_val}
        """
        raise NotImplementedError


class Gpt3QualityEvaluator(BaseEvaluator):

    def run(self, eval_type, eval_obj, export_path='', **kwargs):
        if eval_type == 'data':
            assert issubclass(type(eval_obj), Dataset)
            tmp_data_export_path = export_path + '.tmp_data.jsonl'
            eval_obj.save_to_disk(export_path=tmp_data_export_path)
            tmp_res_export_path = tmp_data_export_path + '.tmp_res.jsonl'
            if os.path.exists(tmp_res_export_path):
                if os.path.isfile(tmp_res_export_path):
                    os.remove(tmp_res_export_path)
                if os.path.isdir(tmp_res_export_path):
                    shutil.rmtree(tmp_res_export_path)

            overall_quality_stats = predict_score(tmp_data_export_path,
                                                  tmp_res_export_path,
                                                  overall_stats=True)

            os.remove(tmp_res_export_path)

            # by default, using the mean quality score of processed data
            # as final score
            return overall_quality_stats.loc['mean']
        else:
            raise NotImplementedError(
                'Unsupported evaluation type: {}'.format(eval_type))


class HelmEvaluator(BaseEvaluator):

    def run(self, eval_type, eval_obj, **kwargs):
        raise NotImplementedError("To be refactored from dj's `thirdparty`.")


class GptEvaluator(BaseEvaluator):

    def run(self, eval_type, eval_obj, **kwargs):
        raise NotImplementedError('To be refactored from `tools.evaluator`,')


class VideoFvdEvaluator(BaseEvaluator):

    def run(self, eval_type, eval_obj, **kwargs):
        raise NotImplementedError(
            'To be refactored from video fvd/isv related tools.')


class Gpt4VEvaluator(BaseEvaluator):

    def run(self, eval_type, eval_obj, **kwargs):
        raise NotImplementedError(
            'To be refactored from gpt4v related operators/tools.')


class LmHarnessEvaluator(BaseEvaluator):

    def run(self, eval_type, eval_obj, **kwargs):
        raise NotImplementedError(
            'To be refactored from, used in data-juicer competition.')


class ModelscopeEvaluator(BaseEvaluator):

    def run(self, eval_type, eval_obj, **kwargs):
        raise NotImplementedError(
            'To be implemented from https://github.com/modelscope/eval-scope.')
