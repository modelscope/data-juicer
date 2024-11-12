import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

torch = LazyLoader('torch', 'torch')

OP_NAME = 'image_nsfw_filter'


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageNSFWFilter(Filter):
    """Filter to keep samples whose images have low nsfw scores."""

    _accelerator = 'cuda'

    def __init__(self,
                 hf_nsfw_model: str = 'Falconsai/nsfw_image_detection',
                 trust_remote_code: bool = False,
                 score_threshold: float = 0.5,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_nsfw_model: nsfw detection model name on huggingface.
        :param score_threshold: the nsfw score threshold for samples.
            range from 0 to 1. Samples with nsfw score less than this threshold
            will be kept.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.score_threshold = score_threshold
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
        self.model_key = prepare_model(
            model_type='huggingface',
            pretrained_model_name_or_path=hf_nsfw_model,
            trust_remote_code=trust_remote_code)

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.image_nsfw_score in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_nsfw_score] = np.array(
                [], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        model, processor = get_model(self.model_key, rank, self.use_cuda())

        images = [images[key] for key in images]
        inputs = processor(images=images, return_tensors='pt').to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        nsfw_scores = [
            float(scores[1]) for scores in torch.softmax(logits, dim=-1)
        ]

        sample[Fields.stats][StatsKeys.image_nsfw_score] = nsfw_scores

        return sample

    def process_single(self, sample, rank=None):
        itm_scores = sample[Fields.stats][StatsKeys.image_nsfw_score]
        if len(itm_scores) <= 0:
            return True

        keep_bools = np.array(
            [itm_score < self.score_threshold for itm_score in itm_scores])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
