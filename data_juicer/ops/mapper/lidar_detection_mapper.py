from typing import Dict

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

mmdet3d = LazyLoader("mmdet3d")

OP_NAME = "lidar_detection_mapper"


@OPERATORS.register_module(OP_NAME)
class LiDARDetectionMapper(Mapper):
    """Mapper to detect ground truth from LiDAR data."""

    _batched_op = True
    _accelerator = "cuda"

    def __init__(self, model_name="centerpoint", *args, **kwargs):
        """
        Initialization method.

        :param mode:
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.model_name = model_name

        if self.model_name == "centerpoint":
            self.deploy_cfg_path = "voxel-detection_onnxruntime_dynamic.py"
            self.model_cfg_path = "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
            self.backend_files = [
                "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.onnx"
            ]
        else:
            raise NotImplementedError(f'Only support "centerpoint" for now, but got {self.model_name}')

        self.model_key = prepare_model(
            "mmlab",
            model_cfg=self.model_cfg_path,
            deploy_cfg=self.deploy_cfg_path,
            backend_files=self.backend_files,
        )

    #  Maybe should include model name, timestamp, filename, image info etc.
    def pred2dict(self, data_sample: mmdet3d.structures.Det3DDataSample) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.

        Returns:
            dict: Prediction results.
        """
        result = {}
        if "pred_instances_3d" in data_sample:
            pred_instances_3d = data_sample.pred_instances_3d.numpy()
            result = {
                "labels_3d": pred_instances_3d.labels_3d.tolist(),
                "scores_3d": pred_instances_3d.scores_3d.tolist(),
                "bboxes_3d": pred_instances_3d.bboxes_3d.tensor.cpu().tolist(),
            }

        if "pred_pts_seg" in data_sample:
            pred_pts_seg = data_sample.pred_pts_seg.numpy()
            result["pts_semantic_mask"] = pred_pts_seg.pts_semantic_mask.tolist()

        if data_sample.box_mode_3d == mmdet3d.structures.Box3DMode.LIDAR:
            result["box_type_3d"] = "LiDAR"
        elif data_sample.box_mode_3d == mmdet3d.structures.Box3DMode.CAM:
            result["box_type_3d"] = "Camera"
        elif data_sample.box_mode_3d == mmdet3d.structures.Box3DMode.DEPTH:
            result["box_type_3d"] = "Depth"

        return result

    def process_batched(self, samples, rank=None):
        model = get_model(self.model_key, rank, self.use_cuda())
        lidars = samples[self.lidar_key]

        results = [model(lidar) for lidar in lidars]
        results = [self.pred2dict(result[0]) for result in results]
        samples["lidar_detections"] = results

        return samples
