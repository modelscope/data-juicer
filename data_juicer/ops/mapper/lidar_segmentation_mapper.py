from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

mmdet3d = LazyLoader("mmdet3d")

OP_NAME = "lidar_segmentation_mapper"


@OPERATORS.register_module(OP_NAME)
class LiDARSegmentationMapper(Mapper):
    """Mapper to do segmentation from LiDAR data."""

    _batched_op = True
    _accelerator = "cuda"

    def __init__(self, model_name="cylinder3d", model_cfg_path="", model_path="", *args, **kwargs):
        """
        Initialization method.
        :param mode:
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.model_name = model_name

        if self.model_name == "cylinder3d":
            self.model_cfg_path = model_cfg_path
            self.model_path = model_path
        else:
            raise NotImplementedError(f'Only support "cylinder3d" for now, but got {self.model_name}')

        self.model_key = prepare_model(
            "mmlab", model_cfg=self.model_cfg_path, model_path=self.model_path, task="LiDARSegmentation"
        )

    def process_batched(self, samples, rank=None):
        model = get_model(self.model_key, rank, self.use_cuda())

        # lidars = []
        # for temp_sample in samples[self.lidar_key]:
        #     lidars.append(dict(points=temp_sample))

        results = [model(dict(points=lidar)) for lidar in samples[self.lidar_key]]
        samples["lidar_segmentations"] = results

        return samples
