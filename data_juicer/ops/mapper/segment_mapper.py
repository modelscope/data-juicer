import copy

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import load_image
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = 'segment_mapper'

with AvailabilityChecking(['torch', 'ultralytics'], OP_NAME):
    import torch
    import ultralytics  # noqa: F401

    # avoid hanging when calling model in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class SegmentMapper(Mapper):
    """Perform segment-anything on images and return the bounding boxes."""

    _accelerator = 'cuda'
    _batched_op = True

    def __init__(self,
                 fastsam_path='FastSAM-x.pt',
                 imgsz=1024,
                 conf=0.05,
                 iou=0.5,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param fastsam_path: location of FastSAM
        :param imgsz: image resolution after image resizing
        :param conf: confidence score threshold
        :param iou: IoU (Intersection over Union) score threshold

        """
        super().__init__(*args, **kwargs)

        self.model_key = prepare_model(
            model_type='fastsam', pretrained_model_name_or_path=fastsam_path)

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def process(self, ori_sample, rank=None):

        # there is no image in this sample
        if self.image_key not in ori_sample or \
                not ori_sample[self.image_key]:
            return []

        generated_samples = copy.deepcopy(ori_sample)

        loaded_image_keys = ori_sample[self.image_key]
        images = {}
        for loaded_image_key in loaded_image_keys:
            if loaded_image_key not in images:
                # avoid loading the same images
                image = load_image(loaded_image_key)
                images[loaded_image_key] = image

        model = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())

        generated_samples[Fields.bbox_tag] = []

        for image in images:
            masks = model([image],
                          retina_masks=True,
                          imgsz=self.imgsz,
                          conf=self.conf,
                          iou=self.iou,
                          verbose=False)[0]

            if len(masks.boxes.xyxy) == 0:
                generated_samples[Fields.bbox_tag].append([])
            else:
                generated_samples[Fields.bbox_tag].append(masks.boxes.xyxy)

        return generated_samples
