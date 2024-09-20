import numpy as np
from jsonargparse.typing import ClosedUnitInterval

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = 'image_pair_similarity_filter'

with AvailabilityChecking(['torch', 'transformers'], OP_NAME):

    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling clip in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImagePairSimilarityFilter(Filter):
    """Filter to keep image pairs with similarities between images
    within a specific range."""

    _accelerator = 'cuda'

    def __init__(self,
                 hf_clip='openai/clip-vit-base-patch32',
                 trust_remote_code=False,
                 min_score: ClosedUnitInterval = 0.1,
                 max_score: ClosedUnitInterval = 1.0,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
    Initialization method.

        :param hf_clip: clip model name on huggingface to compute
            the similarity between image and text.
        :param min_score: The min similarity to keep samples.
        :param max_score: The max similarity to keep samples.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
        self.model_key = prepare_model(model_type='huggingface',
                                       pretrained_model_name_or_path=hf_clip,
                                       trust_remote_code=trust_remote_code)

    def compute_stats(self, sample, rank=None, context=False):

        # check if it's computed already
        if StatsKeys.image_pair_similarity in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if (self.image_key not in sample
                or not len(sample[self.image_key]) == 2
                or sample[self.image_key][0] == sample[self.image_key][1]):
            raise ValueError('Each sample must include two images.')

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        similarity = []
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        image_list = []
        for temp_key in images.keys():
            image_list.append(images[temp_key])
        image_tensors = processor.image_processor(
            image_list, return_tensors='pt')['pixel_values']
        image1_batch_feature = model.get_image_features(
            image_tensors[0].unsqueeze(0).to(model.device))
        image2_batch_feature = model.get_image_features(
            image_tensors[1].unsqueeze(0).to(model.device))

        similarity = torch.cosine_similarity(image1_batch_feature,
                                             image2_batch_feature,
                                             dim=1)
        sample[Fields.stats][StatsKeys.image_pair_similarity] = similarity

        return sample

    def process(self, sample, rank=None):
        similarity = sample[Fields.stats][StatsKeys.image_pair_similarity]
        if len(similarity) <= 0:
            return True

        keep_bools = np.array([
            self.min_score <= sim_value <= self.max_score
            for sim_value in similarity
        ])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
