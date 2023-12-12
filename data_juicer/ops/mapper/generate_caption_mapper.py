import random

import numpy as np
from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..deduplicator.document_simhash_deduplicator import \
    DocumentSimhashDeduplicator
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'generate_caption_mapper'

with AvailabilityChecking(['torch', 'transformers'], OP_NAME):
    import torch
    import transformers  # noqa: F401

    # avoid hanging when calling blip2 in multiprocessing
    torch.set_num_threads(1)


def jaccard_similarity_np(hash_val1, hash_val2):
    equal_hashes = np.sum(hash_val1 == hash_val2)
    total_hashes = len(hash_val1)  # Assume both vectors are the same length
    return equal_hashes / total_hashes


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class GenerateCaptionMapper(Mapper):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    def __init__(self,
                 hf_blip2='Salesforce/blip2-opt-2.7b',
                 caption_num: PositiveInt = 1,
                 keep_candidate_mode: str = 'random_any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_blip2: blip2 model name on huggingface to generate caption
        :param caption_num: how many candidate captions to generate
        for each image
        :param keep_candidate_mode: retain strategy for the generated
        $caption_num$ candidates.
            'random_any': Retain the random one from generated captions
            'similar_one': Retain the generated one that is most similar to the
                original caption
            'all': Retain all generated captions by concatenation
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if keep_candidate_mode not in [
                'random_any', 'similar_one_simhash', 'all'
        ]:
            raise ValueError(
                f'Keep strategy [{keep_candidate_mode}] is not supported. '
                f'Can only be one of '
                f'["random_any", "similar_one_simhash", "all"].')
        self.model_key = prepare_model(model_type='hf_blip',
                                       model_key=hf_blip2)
        self.model_in_ctx = None
        self.img_processor_in_ctx = None
        self.caption_num = caption_num
        self.keep_candidate_mode = keep_candidate_mode
        self.extra_args = kwargs

    def compute_stats(self, sample, context=True):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][
                StatsKeys.image_text_matching_score] = np.array(
                    [], dtype=np.float64)
            return sample

        # load images for context re-using
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

        # load model and processor for context re-using
        if context and self.model_in_ctx is None:
            model, img_processor = get_model(self.model_key)
            self.model_in_ctx = model
            self.img_processor_in_ctx = img_processor

        return sample

    def process(self, sample):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_sizes] = np.array(
                [], dtype=np.float64)
            return sample

        # 1. load image(s)
        loaded_image_keys = sample[self.image_key]
        images = {}
        for loaded_image_key in loaded_image_keys:
            assert loaded_image_key in sample[Fields.context], \
                "Image should has been loaded in 'compute_stats' calling"
            # load from context
            images[loaded_image_key] = sample[Fields.context][loaded_image_key]

        # 2. generate candidate caption(s) in batch manner
        generated_text_candidates = []
        for n in range(self.caption_num):
            inputs = self.img_processor_in_ctx(images=images.values(), )
            generated_ids = self.model_in_ctx.generate(**inputs)
            generated_text = self.img_processor_in_ctx.batch_decode(
                generated_ids, skip_special_tokens=True)
            generated_text_candidates.append(' '.join(generated_text))

        # 3. reduce the captions according to given mode
        if self.keep_candidate_mode == 'random_any':
            sample[self.text_key] = random.choice(generated_text_candidates)
        elif self.keep_candidate_mode == 'all':
            sample[self.text_key] = ' '.join(generated_text_candidates)
        elif self.keep_candidate_mode == 'similar_one_simhash':
            ori_text = sample[self.text_key]
            # using a simhash OP to calculate their similarity
            op_simhash = DocumentSimhashDeduplicator(**self.extra_args)
            ori_text_hash = op_simhash.compute_hash(ori_text)
            generated_text_hashes = [
                op_simhash.compute_hash(candidate_text)
                for candidate_text in generated_text_candidates
            ]
            similarity_scores = [
                jaccard_similarity_np(ori_text_hash, generated_text_hash)
                for generated_text_hash in generated_text_hashes
            ]
            max_index = max(range(len(similarity_scores)),
                            key=similarity_scores.__getitem__)
            sample[self.text_key] = generated_text_candidates[max_index]

        return sample
