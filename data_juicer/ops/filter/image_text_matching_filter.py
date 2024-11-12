import numpy as np
from PIL import ImageOps

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (SpecialTokens, load_data_with_context,
                                        load_image, remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_text_matching_filter'


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageTextMatchingFilter(Filter):
    """Filter to keep samples those matching score between image and text
    within a specific range."""

    _accelerator = 'cuda'

    def __init__(self,
                 hf_blip: str = 'Salesforce/blip-itm-base-coco',
                 trust_remote_code: bool = False,
                 min_score: float = 0.003,
                 max_score: float = 1.0,
                 horizontal_flip: bool = False,
                 vertical_flip: bool = False,
                 any_or_all: str = 'any',
                 reduce_mode: str = 'avg',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_blip: blip model name on huggingface to compute
            the matching score between image and text.
        :param min_score: The min matching score to keep samples.
        :param max_score: The max matching score to keep samples.
        :param horizontal_flip: Flip image horizontally (left to right).
        :param vertical_flip: Flip image vertically (top to bottom).
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param reduce_mode: reduce mode when one text corresponds to
            multiple images in a chunk.
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        if reduce_mode not in ['avg', 'max', 'min']:
            raise ValueError(f'Reduce mode [{reduce_mode}] is not supported. '
                             f'Can only be one of ["avg", "max", "min"].')
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
        self.model_key = prepare_model(model_type='huggingface',
                                       pretrained_model_name_or_path=hf_blip,
                                       trust_remote_code=trust_remote_code)
        self.reduce_mode = reduce_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.image_text_matching_score in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][
                StatsKeys.image_text_matching_score] = np.array(
                    [], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        text = sample[self.text_key]
        offset = 0
        matching_scores = []
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        for chunk in text.split(SpecialTokens.eoc):
            count = chunk.count(SpecialTokens.image)

            # no image or no text
            if count == 0 or len(chunk) == 0:
                continue
            else:
                text_chunk = remove_special_tokens(chunk)
                image_chunk = []
                for image_key in loaded_image_keys[offset:offset + count]:
                    image = images[image_key]
                    if self.horizontal_flip:
                        image = ImageOps.mirror(image)
                    if self.vertical_flip:
                        image = ImageOps.flip(image)
                    image_chunk.append(image)

                inputs = processor(text=text_chunk,
                                   images=image_chunk,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=model.config.text_config.
                                   max_position_embeddings,
                                   padding=True).to(model.device)

                outputs = model(**inputs)
                itm_scores = outputs.itm_score.detach().cpu().softmax(
                    dim=-1)[:, 1]

                if self.reduce_mode == 'avg':
                    chunk_itm_score = itm_scores.mean()
                elif self.reduce_mode == 'max':
                    chunk_itm_score = itm_scores.max()
                else:
                    chunk_itm_score = itm_scores.min()

                matching_scores.append(float(chunk_itm_score))
            offset += count
        sample[Fields.stats][
            StatsKeys.image_text_matching_score] = matching_scores

        return sample

    def process_single(self, sample, rank=None):
        itm_scores = sample[Fields.stats][StatsKeys.image_text_matching_score]
        if len(itm_scores) <= 0:
            return True

        keep_bools = np.array([
            self.min_score <= itm_score <= self.max_score
            for itm_score in itm_scores
        ])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
