import copy

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.mm_utils import load_image
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = 'segment_mapper'

with AvailabilityChecking(['torch', 'transformers', 'simhash-pybind'],
                          OP_NAME):
    import simhash  # noqa: F401
    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling model in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class SegmentMapper(Mapper):
    """Perform segment-anything on images and return the bounding boxes."""

    _accelerator = 'cuda'
    _batched_op = True

    def __init__(self,
                 fastsam_path='./FastSAM-x.pt',
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

    def _process_single_sample(self, ori_sample, rank=None):
        """

        :param ori_sample: a single data sample before applying generation
        :return: batched results after generation
        """
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
        masks = model([image],
                      retina_masks=True,
                      imgsz=self.imgsz,
                      conf=self.conf,
                      iou=self.iou,
                      verbose=False)[0]

        if len(masks.boxes.xyxy) == 0:
            return []

        generated_samples['bboxes'] = masks.boxes.xyxy

        return generated_samples

    def process(self, samples, rank=None):
        """
        Note:
            This is a batched_OP, whose input and output type are
            both list. Suppose there are $N$ input sample list with batch
            size as $b$, and denote caption_num as $M$.
            the number of total samples after generation is $2Nb$
            for 'random_any' and 'similar_one' mode,
            and $(1+M)Nb$ for 'all' mode.

        :param samples:
        :return:
        """
        # reconstruct samples from "dict of lists" to "list of dicts"
        reconstructed_samples = []
        for i in range(len(samples['images'])):
            reconstructed_samples.append(
                {key: samples[key][i]
                 for key in samples})
        samples_after_generation = []
        # do generation for each sample within the batch
        for ori_sample in reconstructed_samples:
            generated_samples = self._process_single_sample(ori_sample,
                                                            rank=rank)
            if len(generated_samples) != 0:
                samples_after_generation.append(generated_samples)
        # reconstruct samples from "list of dicts" to "dict of lists"
        keys = samples_after_generation[0].keys()
        res_samples = {}
        for key in keys:
            res_samples[key] = [s[key] for s in samples_after_generation]

        return res_samples
