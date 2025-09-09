import os
import shutil

from loguru import logger

from data_juicer.core.data.dj_dataset import nested_query
from data_juicer.core.sandbox.data_pool_manipulators import load_data_pool
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


class AccuracyEvaluator(BaseEvaluator):
    """
    A simple evaluator to compare the labels in the predicted ones and ground truth.

    The config file for this evaluator should at least include the following items:
    1. `type`: must be "accuracy".
    2. `predicted_dataset_path`: Required. The dataset path to the data that stores the predicted labels.
    3. `ground_truth_dataset_path`:  The dataset path to the data that stores the ground truth labels. If it's None,
        we assume that the ground truth labels are already in the predicted_dataset_path.
    4. `predicted_label_key`: the key name to store the predicted labels. '.' operator is allowed.
    5. `ground_truth_label_key`: the key name to store the ground truth labels. '.' operator is allowed.
    """

    def __init__(self, eval_config: dict):
        super(AccuracyEvaluator, self).__init__(eval_config)
        self.predicted_dataset_path = self.eval_config.get("predicted_dataset_path", [])
        self.ground_truth_dataset_path = self.eval_config.get("ground_truth_dataset_path", [])
        self.predicted_label_key = self.eval_config.get("predicted_label_key", None)
        self.ground_truth_label_key = self.eval_config.get("ground_truth_label_key", None)

        if isinstance(self.predicted_dataset_path, str):
            self.predicted_dataset_path = [self.predicted_dataset_path]
        if isinstance(self.ground_truth_dataset_path, str):
            self.ground_truth_dataset_path = [self.ground_truth_dataset_path]
        assert len(self.ground_truth_dataset_path) == 0 or len(self.predicted_dataset_path) == len(
            self.ground_truth_dataset_path
        )

        existing_predicted_dataset_paths, existing_ground_truth_dataset_paths = [], []
        if len(self.ground_truth_dataset_path) == 0:
            logger.warning(
                "The ground truth dataset path is not specified. Assume the ground truth labels are already "
                "in the predicted dataset."
            )
            self.ground_truth_dataset_path = self.predicted_dataset_path[:]
        for pred_path, gt_path in zip(self.predicted_dataset_path, self.ground_truth_dataset_path):
            if os.path.exists(pred_path) and os.path.exists(gt_path):
                existing_predicted_dataset_paths.append(pred_path)
                existing_ground_truth_dataset_paths.append(gt_path)

        if len(existing_predicted_dataset_paths) == 0:
            raise ValueError("Please specify a valid predicted dataset path")
        if self.predicted_label_key is None:
            raise ValueError("Please specify the predicted label key")
        if self.ground_truth_label_key is None:
            raise ValueError("Please specify the ground truth label key")

        self.predicted_dataset_path = existing_predicted_dataset_paths
        self.ground_truth_dataset_path = existing_ground_truth_dataset_paths

    def run(self, eval_type, eval_obj=None, **kwargs):
        results = []
        for pred_path, gt_path in zip(self.predicted_dataset_path, self.ground_truth_dataset_path):
            pred_ds = load_data_pool(pred_path)
            if gt_path == pred_path:
                gt_ds = pred_ds
            else:
                gt_ds = load_data_pool(gt_path)

            result = {}
            result["pred_path"] = pred_path
            result["gt_path"] = gt_path
            total = 0
            hit = 0
            for pred_sample, gt_sample in zip(pred_ds, gt_ds):
                pred_label = str(nested_query(pred_sample, self.predicted_label_key))
                gt_label = str(nested_query(gt_sample, self.ground_truth_label_key))
                total += 1
                if pred_label == gt_label:
                    hit += 1
            result["accuracy"] = hit * 1.0 / total
            results.append(result)
        max_accuracy = max([result["accuracy"] for result in results])
        return results, max_accuracy


class MSEEvaluator(BaseEvaluator):
    """
    A simple evaluator to compute the MSE between the predicted values and ground truth.

    The config file for this evaluator should at least include the following items:
    1. `type`: must be "mse".
    2. `predicted_dataset_path`: Required. The dataset path to the data that stores the predicted labels.
    3. `ground_truth_dataset_path`:  The dataset path to the data that stores the ground truth labels. If it's None,
        we assume that the ground truth labels are already in the predicted_dataset_path.
    4. `predicted_value_key`: the key name to store the predicted values. '.' operator is allowed.
    5. `ground_truth_value_key`: the key name to store the ground truth values. '.' operator is allowed.
    """

    def __init__(self, eval_config: dict):
        super(MSEEvaluator, self).__init__(eval_config)
        self.predicted_dataset_path = self.eval_config.get("predicted_dataset_path", [])
        self.ground_truth_dataset_path = self.eval_config.get("ground_truth_dataset_path", [])
        self.predicted_value_key = self.eval_config.get("predicted_value_key", None)
        self.ground_truth_value_key = self.eval_config.get("ground_truth_value_key", None)

        if isinstance(self.predicted_dataset_path, str):
            self.predicted_dataset_path = [self.predicted_dataset_path]
        if isinstance(self.ground_truth_dataset_path, str):
            self.ground_truth_dataset_path = [self.ground_truth_dataset_path]
        assert len(self.ground_truth_dataset_path) == 0 or len(self.predicted_dataset_path) == len(
            self.ground_truth_dataset_path
        )

        existing_predicted_dataset_paths, existing_ground_truth_dataset_paths = [], []
        if len(self.ground_truth_dataset_path) == 0:
            logger.warning(
                "The ground truth dataset path is not specified. Assume the ground truth labels are already "
                "in the predicted dataset."
            )
            self.ground_truth_dataset_path = self.predicted_dataset_path[:]
        for pred_path, gt_path in zip(self.predicted_dataset_path, self.ground_truth_dataset_path):
            if os.path.exists(pred_path) and os.path.exists(gt_path):
                existing_predicted_dataset_paths.append(pred_path)
                existing_ground_truth_dataset_paths.append(gt_path)

        if len(existing_predicted_dataset_paths) == 0:
            raise ValueError("Please specify a valid predicted dataset path")
        if self.predicted_value_key is None:
            raise ValueError("Please specify the predicted value key")
        if self.ground_truth_value_key is None:
            raise ValueError("Please specify the ground truth value key")

        self.predicted_dataset_path = existing_predicted_dataset_paths
        self.ground_truth_dataset_path = existing_ground_truth_dataset_paths

    def run(self, eval_type, eval_obj=None, **kwargs):
        results = []
        for pred_path, gt_path in zip(self.predicted_dataset_path, self.ground_truth_dataset_path):
            pred_ds = load_data_pool(pred_path)
            if gt_path == pred_path:
                gt_ds = pred_ds
            else:
                gt_ds = load_data_pool(gt_path)

            result = {}
            result["pred_path"] = pred_path
            result["gt_path"] = gt_path
            total = 0
            mse = 0
            fmt_err = 0
            for pred_sample, gt_sample in zip(pred_ds, gt_ds):
                try:
                    pred_value = float(nested_query(pred_sample, self.predicted_value_key))
                    gt_value = float(nested_query(gt_sample, self.ground_truth_value_key))
                except ValueError as e:
                    logger.warning(f"{e}")
                    fmt_err += 1
                    continue
                total += 1
                mse += (pred_value - gt_value) ** 2
            result["mse"] = mse / total
            result["format_error"] = fmt_err
            results.append(result)
        min_mse = min([result["mse"] for result in results])
        return results, min_mse


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
