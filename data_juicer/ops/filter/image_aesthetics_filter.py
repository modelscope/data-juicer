import numpy as np
from jsonargparse.typing import ClosedUnitInterval
from loguru import logger

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ...utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_aesthetics_filter'
CHECK_PKGs = ['torch', 'transformers', 'simple-aesthetics-predictor']

with AvailabilityChecking(CHECK_PKGs, OP_NAME):

    import aesthetics_predictor  # noqa: F401
    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling clip in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageAestheticsFilter(Filter):
    """Filter to keep samples with aesthetics scores within a specific range.
    """

    _accelerator = 'cuda'

    def __init__(self,
                 hf_scorer_model='',
                 trust_remote_code=False,
                 min_score: ClosedUnitInterval = 0.5,
                 max_score: ClosedUnitInterval = 1.0,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_scorer_model: Huggingface model name for the aesthetics
            predictor. By default, we will use
            'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE',
            refer to pypi.org/project/simple-aesthetics-predictor
        :param min_score: Min score for the predicted aesthetics in an image.
        :param max_score: Max score for the predicted aesthetics in an image.
        :param any_or_all: Keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: Extra positional arguments.
        :param kwargs: Extra keyword arguments.
        """

        super().__init__(*args, **kwargs)
        if hf_scorer_model == '':
            hf_scorer_model = \
                'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'
        self.min_score = min_score
        self.max_score = max_score

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

        self.model_key = prepare_model(
            model_type='simple_aesthetics',
            pretrained_model_name_or_path=hf_scorer_model,
            trust_remote_code=trust_remote_code)
        # the original score predicted by laion-ai's scorer is within [0, 10]
        self.need_normalized_by_ten = ('shunk031/aesthetics-predictor'
                                       in hf_scorer_model)

    def compute_stats(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.image_aesthetics_scores in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_aesthetics_scores] = np.array(
                [], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        # compute aesthetics_scores
        model, processor = get_model(self.model_key, rank, self.use_cuda())
        inputs = processor(images=list(images.values()),
                           return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        if self.need_normalized_by_ten:
            aesthetics_scores = outputs.logits / 10.0
        else:
            aesthetics_scores = outputs.logits

        aesthetics_scores = [
            aesthetics_score.item() for aesthetics_score in aesthetics_scores
        ]

        logger.debug(f'aesthetics_scores: {aesthetics_scores}')

        sample[Fields.stats][StatsKeys.image_aesthetics_scores] =\
            aesthetics_scores
        return sample

    def process(self, sample):
        aesthetics_scores = (
            sample)[Fields.stats][StatsKeys.image_aesthetics_scores]
        if len(aesthetics_scores) <= 0:
            return True

        keep_bools = np.array([
            self.min_score <= aesthetics_score <= self.max_score
            for aesthetics_score in aesthetics_scores
        ])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
