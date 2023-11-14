import numpy as np
import torch
from jsonargparse.typing import PositiveFloat

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import SpecialTokens, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

# avoid hanging when calling clip in multiprocessing
torch.get_num_threads()


@OPERATORS.register_module('clip_similarity_filter')
@LOADED_IMAGES.register_module('clip_similarity_filter')
class ClipSimilarityFilter(Filter):
    """Filter to keep samples those similarity between image and text
    within a specific range."""

    def __init__(self,
                 hf_clip='openai/clip-vit-base-patch32',
                 min_ratio: PositiveFloat = 0.1,
                 max_ratio: PositiveFloat = 1.0,
                 any_or_all: str = 'any',
                 reduce_mode: str = 'avg',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_clip: clip model name on huggingface to compute
            the similarity between image and text.
        :param min_ratio: The min similarity to keep samples.
        :param max_ratio: The max similarity to keep samples.
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
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        if reduce_mode not in ['avg', 'max', 'min']:
            raise ValueError(f'Reduce mode [{reduce_mode}] is not supported. '
                             f'Can only be one of ["avg", "max", "min"].')
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
        self.model_key = prepare_model(model_type='hf_clip', model_key=hf_clip)
        self.reduce_mode = reduce_mode

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.clip_image_text_similarity in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][
                StatsKeys.clip_image_text_similarity] = np.array(
                    [], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        images = {}
        for loaded_image_key in loaded_image_keys:
            if context and loaded_image_key in sample[Fields.context]:
                # load from context
                images[loaded_image_key] = sample[
                    Fields.context][loaded_image_key]
            else:
                if loaded_image_key not in images:
                    # avoid load the same images
                    image = load_image(loaded_image_key)
                    images[loaded_image_key] = image
                    if context:
                        # store the image data into context
                        sample[Fields.context][loaded_image_key] = image

        text = sample[self.text_key]
        special_token_dict = {
            key: value
            for key, value in SpecialTokens.__dict__.items()
            if not key.startswith('__')
        }
        offset = 0

        def remove_special_token(text):
            for key, value in special_token_dict.items():
                text = text.replace(value, '')
            return text

        similarity = []
        model, processor = get_model(self.model_key)

        for chunk in text.split(SpecialTokens.eoc):
            count = chunk.count(SpecialTokens.image)

            # no image or no text
            if count == 0 or len(chunk) == 0:
                continue
            else:
                text_chunk = remove_special_token(chunk)
                image_chunk = [
                    images[image_key]
                    for image_key in loaded_image_keys[offset:offset + count]
                ]

                inputs = processor(text=text_chunk,
                                   images=image_chunk,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=model.config.text_config.
                                   max_position_embeddings,
                                   padding=True)

                outputs = model(**inputs)
                chunk_logits = outputs.logits_per_text.detach().cpu() / 100.0

                if self.reduce_mode == 'avg':
                    chunk_similarity = chunk_logits.mean()
                elif self.reduce_mode == 'max':
                    chunk_similarity = chunk_logits.max()
                else:
                    chunk_similarity = chunk_logits.min()

                similarity.append(float(chunk_similarity))
            offset += count
        sample[Fields.stats][StatsKeys.clip_image_text_similarity] = similarity

        return sample

    def process(self, sample):
        similarity = sample[Fields.stats][StatsKeys.clip_image_text_similarity]
        if len(similarity) <= 0:
            return True

        keep_bools = np.array([
            self.min_ratio <= sim_value <= self.max_ratio
            for sim_value in similarity
        ])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
