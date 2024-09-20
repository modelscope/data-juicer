from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import HashKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Deduplicator
from ..op_fusion import LOADED_IMAGES
from .document_deduplicator import DocumentDeduplicator

OP_NAME = 'image_deduplicator'

with AvailabilityChecking(['imagededup'], OP_NAME):
    import imagededup  # noqa: F401

    HASH_METHOD = {'phash', 'dhash', 'whash', 'ahash'}

    def get_hash_method(method_name):
        from imagededup.methods import AHash, DHash, PHash, WHash

        mapping = {
            'phash': PHash,
            'dhash': DHash,
            'whash': WHash,
            'ahash': AHash
        }

        return mapping[method_name]


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageDeduplicator(Deduplicator):
    """
    Deduplicator to deduplicate samples at document-level using exact matching
    of images between documents.
    """

    def __init__(self,
                 method: str = 'phash',
                 consider_text: bool = False,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param method: hash method for image
        :param consider_text: whether to consider text hash together with image
            hash when applying deduplication.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if method not in HASH_METHOD:
            raise ValueError(f'Keep strategy [{method}] is not supported. '
                             f'Can only be one of {HASH_METHOD}.')
        self.hasher = get_hash_method(method)()
        self.consider_text = consider_text
        self.text_dedup_op = None
        if self.consider_text:
            self.text_dedup_op = DocumentDeduplicator(**kwargs)

    def compute_hash(self, sample, context=False):
        # get hash of text first
        if self.consider_text:
            sample = self.text_dedup_op.compute_hash(sample)
        # check if it's computed already
        if HashKeys.imagehash in sample:
            return sample

        # there is no image in this sample
        sample[HashKeys.imagehash] = ''
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        # compute hash
        for key in images:
            sample[HashKeys.imagehash] += self.hasher.encode_image(
                image_array=np.array(images[key]))
        return sample

    def process(self, dataset, show_num=0):
        """
        For doc-level, dataset --> dataset.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is
            open.
        :return: deduplicated dataset and the sampled duplicate pairs.
        """
        # no need to deduplicate because too few samples
        if len(dataset) <= 1:
            return dataset, {}

        dup_hashes = None
        if show_num > 0:
            # sample duplicate pairs
            if self.consider_text:
                hash2ids: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
                hashes = zip(dataset[HashKeys.imagehash],
                             dataset[HashKeys.hash])
            else:
                hash2ids: Dict[int, Set[int]] = defaultdict(set)
                hashes = dataset[HashKeys.imagehash]
            for sid, hash_val in enumerate(hashes):
                if hash_val:
                    hash2ids[hash_val].add(sid)
            dup_samples = sorted(list(hash2ids.items()),
                                 key=lambda x: len(x[1]),
                                 reverse=True)
            dup_hashes = set([
                item[0] for item in dup_samples if len(item[1]) > 1
            ][:show_num])

        def _filter_dup_helper(sample, hashes):
            if self.consider_text:
                hash = (sample[HashKeys.imagehash], sample[HashKeys.hash])
            else:
                hash = sample[HashKeys.imagehash]
            if not hash:
                return True
            if show_num > 0 and hash in dup_hashes \
                    and len(dup_pairs[hash]) < 2:
                # tracer is open and not enough duplicate sample pairs
                dup_pairs[hash].append(sample)
            if hash in hashes:
                return False
            else:
                hashes.add(hash)
                return True

        hashes = set()
        dup_pairs = {hash_v: [] for hash_v in dup_hashes} if dup_hashes else {}
        dataset = dataset.filter(
            _filter_dup_helper,
            fn_kwargs=dict(hashes=hashes),
            load_from_cache_file=False if show_num > 0 else True)  # num_proc=1
        return dataset, dup_pairs
