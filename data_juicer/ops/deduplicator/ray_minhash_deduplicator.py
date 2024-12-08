import random
import time
import uuid
from collections import defaultdict
from typing import Optional

import ray
import numpy as np
import pandas as pd
import pyarrow as pa
import regex
from loguru import logger
from pydantic import Field, PositiveInt
from typing_extensions import Annotated
from typing import Dict

from data_juicer.utils.constant import HashKeys, Fields
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import prepare_sentencepiece_model

from ..base_op import OPERATORS, Deduplicator
from ..common.helper_func import split_on_whitespace
from .document_minhash_deduplicator import (MAX_HASH, MERSENNE_PRIME,
                                            optimal_param, sha1_hash32)


@ray.remote
class UnionFindWithMerge:
    def __init__(self):
        """Initialization method."""
        self.parent: Dict[str | bytes, str] = {}

    @staticmethod
    def find_with_parent(parent, x):
        x_list = []
        while x in parent:
            x_list.append(x)
            x = parent[x]
        for xx in x_list:
            parent[xx] = x
        return x

    def find(self, x):
        return self.find_with_parent(self.parent, x)

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return
        if px > py:
            px, py = py, px
        self.parent[py] = px

    def union_list(self, x_list):
        px_list = [self.find(x) for x in x_list]
        p = min(px_list)
        for px in px_list:
            if p != px:
                self.parent[px] = p

    def is_ancestor(self, x):
        assert (x not in self.parent) == (x == self.parent.get(x, x)), f'{x}, {self.parent.get(x, x)}'
        return x not in self.parent

    def get_num(self):
        return len(self.parent)
    
    def get_nodes(self):
        return set(self.parent.keys())

    def get_parent(self):
        return self.parent

    def merge(self, union_find_set):
        union_find_set_parent = ray.get(union_find_set.get_parent.remote())
        union_find_set_nodes = set(union_find_set_parent.keys())
        parrent_nodes = set(self.parent.keys())
        for x in union_find_set_nodes:
            px = self.find_with_parent(union_find_set_parent, x)
            if x in parrent_nodes:
                py = self.find(x)
                self.union(px, py)
            else:
                self.parent[x] = px

    def merge_list(self, union_find_set_list):
        for union_find_set in union_find_set_list:
            self.merge(union_find_set)


OP_NAME = 'ray_minhash_deduplicator'


@OPERATORS.register_module(OP_NAME)
class RayMinhashDeduplicator(Deduplicator):
    """
    A basic exact matching deduplicator for RAY.
    Although its functionality is deduplication,
    it is implemented as Filter sub-class.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = 'EMPTY'
    _batched_op = True

    def __init__(
        self,
        tokenization: str = 'space',
        window_size: PositiveInt = 5,
        lowercase: bool = True,
        ignore_pattern: Optional[str] = None,
        num_permutations: PositiveInt = 256,
        jaccard_threshold: Annotated[float, Field(ge=0, le=1)] = 0.7,
        num_bands: Optional[PositiveInt] = None,
        num_rows_per_band: Optional[PositiveInt] = None,
        tokenizer_model: Optional[str] = None,
        union_find_parallel_num: Optional[int] = 16,
        union_find_merge_num: Optional[int] = 2,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param tokenization: tokenization method for sample texts. It
            should be one of [space, punctuation, character,
            sentencepiece]. For English-like languages, we recommend
            to use 'space', for Chinese-like languages, we recommend
            to use 'character', and for multiple languages, we recommend
            to use 'sentencepiece'. If using 'sentencepiece', please
            provided the model path in the 'tokenizer_model' field.
        :param window_size: window size of shingling
        :param lowercase: whether to convert text to lower case first
        :param ignore_pattern: whether to ignore sub-strings with
            specific pattern when computing minhash
        :param num_permutations: number of permutations in minhash
            computing
        :param jaccard_threshold: the min jaccard similarity threshold
            in near-duplicate detection. When the jaccard similarity of
            two sample texts is >= this threshold, they are regarded as
            similar samples and this op will only keep one of them after
            deduplication
        :param num_bands: number of bands in LSH. Default it's None, and
            it will be determined by an optimal params computation
            algorithm by minimize the weighted sum of probs of False
            Positives and False Negatives
        :param num_rows_per_band: number of rows in each band in LSH.
            Default it's None, and it will be determined by an optimal
            params computation algorithm
        :param tokenizer_model: path for the sentencepiece model, used for
            sentencepiece tokenization.
        """
        super().__init__(*args, **kwargs)
        # about minhash computation
        self.tokenization = tokenization
        self.window_size = window_size
        self.lowercase = lowercase
        self.ignore_pattern = ignore_pattern
        if self.ignore_pattern:
            self.ignore_pattern = regex.compile(self.ignore_pattern)

        # check parameters
        if self.ignore_pattern and self.tokenization == 'punctuation':
            logger.warning('Be careful that tokenization with punctuations '
                           'won\'t work if the ignore pattern includes '
                           'punctuations.')
        self.punctuation_pattern = regex.compile(r'\p{P}')

        if self.tokenization == 'sentencepiece':
            if tokenizer_model is None:
                raise ValueError("To use 'sentencepiece' tokenization, "
                                 "'tokenizer_model' is required.")
            self.tokenizer = prepare_sentencepiece_model(tokenizer_model)
        else:
            self.tokenizer = None

        # about deduplication
        self.num_permutation = num_permutations
        self.jaccard_threshold = jaccard_threshold
        self.num_bands = num_bands
        self.num_rows_per_band = num_rows_per_band

        # initialize deduplication parameters
        # check number of bands and rows
        if self.num_bands is None or self.num_rows_per_band is None:
            self.num_bands, self.num_rows_per_band = optimal_param(
                self.jaccard_threshold,
                self.num_permutation,
            )

        # compute hash ranges and create hash tables
        self.hash_ranges = [(i * self.num_rows_per_band,
                             (i + 1) * self.num_rows_per_band)
                            for i in range(self.num_bands)]
        self.hash_tables = [defaultdict(set) for _ in range(self.num_bands)]

        # generate permutations
        gen = np.random.RandomState(seed=42)
        self.perm_a, self.perm_b = np.array(
            [(
                gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            ) for _ in range(self.num_permutation)],
            dtype=np.uint64,
        ).T

        self.init_union_find(union_find_parallel_num, union_find_merge_num)


    def init_union_find(self, union_find_parallel_num, union_find_merge_num):
        self.union_find_parallel_num = union_find_parallel_num # 2 # 16
        self.union_find_merge_num = union_find_merge_num
        self.union_find_list = [
            UnionFindWithMerge.remote()
            for _ in range(self.union_find_parallel_num)
        ]

    def compute_stats(self, samples: pa.Table) -> pa.Table:
        samples_list = samples[self.text_key]
        uuid_list = [uuid.uuid4().bytes for _ in range(samples.num_rows)]
        all_hash_values = [[] for _ in range(self.num_bands)]

        for text in samples_list:
            text = text.as_py()
            if self.lowercase:
                text = text.lower()
            if self.ignore_pattern:
                text = self.ignore_pattern.sub('', text)

            # get tokens for different tokenization method
            tokens = set()
            if self.tokenization == 'character':
                tokens = {
                    str.encode(text[i:i + self.window_size])
                    for i in range(len(text) - self.window_size)
                }
            elif self.tokenization == 'punctuation':
                tokens = self.punctuation_pattern.split(text)
                tokens = {
                    str.encode(' '.join(tokens[i:i + self.window_size]))
                    for i in range(len(tokens) - self.window_size)
                }
            elif self.tokenization == 'space':
                tokens = split_on_whitespace(text)
                tokens = {
                    str.encode(' '.join(tokens[i:i + self.window_size]))
                    for i in range(len(tokens) - self.window_size)
                }
            elif self.tokenization == 'sentencepiece':
                tokens = self.tokenizer.encode(text, out_type=str)
                tokens = {
                    str.encode(''.join(tokens[i:i + self.window_size]))
                    for i in range(len(tokens) - self.window_size)
                }
            else:
                raise NotImplementedError(
                    f'Unimplemented tokenization method [{self.tokenization}]')

            if len(tokens) > 0:
                hv = np.array(
                    [sha1_hash32(token) for token in tokens],
                    dtype=np.uint64
                )
                phv = (
                    (hv[:, None] * self.perm_a[None, :] 
                     + self.perm_b) % MERSENNE_PRIME
                ).astype(np.uint32)
                hash_values = phv.min(axis=0)
            else:
                hash_values = np.full_like(self.perm_a, MAX_HASH, dtype=np.uint32)
            for i, (start, end) in enumerate(self.hash_ranges):
                all_hash_values[i].append(
                    i.to_bytes(4, 'big') + 
                    bytes(hash_values[start:end].byteswap().data)
                )

        samples = samples.append_column(HashKeys.uid, pa.array(uuid_list))
        for i, hash_values in enumerate(all_hash_values):
            samples = samples.append_column(HashKeys.minhash + f"_{i}", pa.array(hash_values))
        return samples

    def map_batched(self, samples: pa.Table) -> pa.Table:
        table = pa.Table.from_arrays(
            [
                pa.concat_arrays(
                    [samples[HashKeys.uid].combine_chunks()] * len(self.hash_ranges)
                ),
                # pa.array(
                #     [uid.as_py() for uid in samples[HashKeys.uid]] * len(self.hash_ranges)
                # ),
                pa.concat_arrays(
                    [
                        samples[HashKeys.minhash + f'_{i}'].combine_chunks()
                        for i in range(len(self.hash_ranges))
                    ]
                ),
            ],
            names=[HashKeys.uid, HashKeys.minhash]
        )
        return table

    def agg_func(self, group: pa.Table) -> pa.Table:
        if group.num_rows != 1:
            uuid_list = [uid.as_py() for uid in group[HashKeys.uid]]
            union_find_id = np.random.randint(0, self.union_find_parallel_num)
            union_find = self.union_find_list[union_find_id]
            ray.get(union_find.union_list.remote(uuid_list))
        return group
    
    def merge(self):
        union_find_list = self.union_find_list
        while len(union_find_list) > 1:
            new_union_find_list = []
            task_list = []
            buffer = []
            for union_find in union_find_list:
                buffer.append(union_find)
                if len(buffer) == self.union_find_merge_num:
                    new_union_find_list.append(buffer[0])
                    task_list.append(buffer[0].merge_list.remote(buffer[1:]))
                    buffer = []
            if len(buffer) > 0:
                new_union_find_list.append(buffer[0])
                if len(buffer) > 1:
                    task_list.append(buffer[0].merge_list.remote(buffer[1:]))
            ray.get(task_list)
            union_find_list = new_union_find_list
        self.parent = ray.get(union_find_list[0].get_nodes.remote())

    def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
        mask = [
            uid.as_py() not in self.parent
            for uid in samples[HashKeys.uid]
        ]
        return samples.filter(mask)

    def run(self, dataset):
        # import time
        # start_time = time.time()
        dataset = dataset.map_batches(
            self.compute_stats,
            batch_format='pyarrow',
        ).materialize()
        drop_columns = []
        for i in range(len(self.hash_ranges)):
            drop_column = HashKeys.minhash + f'_{i}'
            drop_columns.append(drop_column)
        # end_time = time.time()
        # print(f'minhash time = {end_time - start_time}')

        # start_time = time.time()
        dataset.map_batches(
            self.map_batched,
            batch_format='pyarrow',
        ).groupby(
            HashKeys.minhash
        ).map_groups(
            self.agg_func, batch_format='pyarrow'
        ).materialize()
        # end_time = time.time()
        # print(f'group time = {end_time - start_time}')
        # start_time = time.time()
        self.merge()
        # end_time = time.time()
        # print(f'merge time = {end_time - start_time}')
        result = dataset.drop_columns(
            drop_columns
        ).map_batches(
            self.filter_with_union_find,
            batch_format='pyarrow'
        ).drop_columns(
            HashKeys.uid
        ).materialize()
        logger.info(f'Keep {result.count()} samples after MinHash dedup.')
        return result
