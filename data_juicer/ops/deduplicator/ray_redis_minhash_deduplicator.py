import random
import time
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import regex
from loguru import logger
from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import prepare_sentencepiece_model

from ..base_op import OPERATORS, Deduplicator
from ..common.helper_func import split_on_whitespace
from .document_minhash_deduplicator import (MAX_HASH, MERSENNE_PRIME,
                                            optimal_param, sha1_hash32)

redis = LazyLoader('redis', 'redis')


def retry_on_busy(func):

    def wrapper(*args, **kwargs):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if 'BUSY' in str(e) and attempt < max_retries - 1:
                    time.sleep(random.uniform(0.1, 0.3) * (2**attempt))
                else:
                    raise

    return wrapper


class RedisUnionFind:

    def __init__(self,
                 prefix: str,
                 redis_address: str = 'redis://localhost:6379'):
        self.prefix = prefix
        self.redis_address = redis_address
        self.redis = redis.from_url(url=redis_address)
        self.set_key = f'{prefix}_UF_SET'
        self.rank_key = f'{prefix}_UF_RANK'
        self.incur_id_key = f'{prefix}_UF_INCURID'

        # Lua scripts
        self.union_script = self.redis.register_script("""
           local function find(x)
                local path = {}
                while true do
                    local parent = redis.call('HGET', KEYS[1], x)
                    if not parent then
                        return nil
                    end
                    if parent == x then
                        break
                    end
                    table.insert(path, x)
                    x = parent
                end
                for _, node in ipairs(path) do
                    redis.call('HSET', KEYS[1], node, x)
                end
                return x
            end

            local root_x = find(ARGV[1])
            local root_y = find(ARGV[2])
            if not root_x then
                redis.call('HSET', KEYS[1], ARGV[1], ARGV[1])
                redis.call('HSET', KEYS[2], ARGV[1], 0)
                root_x = ARGV[1]
            end
            if not root_y then
                redis.call('HSET', KEYS[1], ARGV[2], ARGV[2])
                redis.call('HSET', KEYS[2], ARGV[2], 0)
                root_y = ARGV[2]
            end
            if root_x == root_y then
                return root_x
            end
            local rank_x = tonumber(redis.call('HGET', KEYS[2], root_x))
            local rank_y = tonumber(redis.call('HGET', KEYS[2], root_y))
            if rank_x < rank_y then
                redis.call('HSET', KEYS[1], root_x, root_y)
                return root_y
            elseif rank_x > rank_y then
                redis.call('HSET', KEYS[1], root_y, root_x)
                return root_x
            else
                redis.call('HSET', KEYS[1], root_y, root_x)
                redis.call('HINCRBY', KEYS[2], root_x, 1)
                return root_x
            end
            """)

    def get_uid(self):
        return int(self.redis.incr(self.incur_id_key))

    @retry_on_busy
    def union(self, x, y):
        return self.union_script(keys=[self.set_key, self.rank_key],
                                 args=[x, y])

    def is_ancestor(self, x):
        ancestor = self.redis.hget(self.set_key, x)
        return ancestor is None or int(ancestor) == x

    def __reduce__(self):
        return (RedisUnionFind, (self.prefix, self.redis_address))

    def clean(self):
        self.redis.delete(self.set_key, self.rank_key, self.incur_id_key)


OP_NAME = 'ray_redis_minhash_deduplicator'


@OPERATORS.register_module(OP_NAME)
class RayRedisMinhashDeduplicator(Deduplicator):
    """
    A basic exact matching deduplicator for RAY.
    Although its functionality is deduplication,
    it is implemented as Filter sub-class.
    """

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
        redis_address: str = 'redis://localhost:6379',
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
        :param redis_address: address of your redis instance, e.g.
            'redis://localhost:6379'
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
        self.redis_address = redis_address

    def run(self, dataset):
        from ray.data.aggregate import AggregateFn

        union_find = RedisUnionFind(prefix=uuid.uuid4().hex[:8], redis_address=self.redis_address)

        def add_uid_column(table: pa.Table) -> pa.Table:
            new_column_data = [union_find.get_uid() for _ in range(len(table))]
            new_table = table.append_column(HashKeys.uid, [new_column_data])
            return new_table

        def calculate_minhash(table: pa.Table) -> pa.Table:
            ids = table.column(HashKeys.uid).to_pandas()
            texts = table.column(self.text_key).to_pandas()
            hashes = texts.apply(lambda x: self.compute_minhash(x))
            hashes = pa.Array.from_pandas(hashes).flatten()

            repeated_ids = pa.Array.from_pandas(ids.repeat(self.num_bands))

            return pa.Table.from_arrays([repeated_ids, hashes],
                                        names=[HashKeys.uid, HashKeys.minhash])

        def _is_null(r):
            return pd.isnull(r)

        class UnionFn(AggregateFn):

            def __init__(self, union_find):
                union_find = union_find

                def accumulate(cur, row):
                    if _is_null(row):
                        return cur
                    elif _is_null(cur):
                        return row[HashKeys.uid]
                    else:
                        root = union_find.union(row[HashKeys.uid], cur)
                        return int(root)

                def merge(a, b):
                    if _is_null(a):
                        return b
                    if _is_null(b):
                        return a
                    root = union_find.union(a, b)
                    return int(root)

                super().__init__(
                    init=lambda k: None,
                    accumulate_row=accumulate,
                    merge=merge,
                    name='union',
                )

        def filter_with_union_find(table: pa.Table) -> pa.Table:
            uids = table.column(HashKeys.uid).to_pandas()
            mask = pa.Array.from_pandas(
                uids.apply(lambda x: union_find.is_ancestor(x)))
            return table.filter(mask)

        dataset_with_id = dataset.map_batches(
            add_uid_column, batch_format='pyarrow').materialize()
        dataset_with_id.map_batches(calculate_minhash,
                                    batch_format='pyarrow').groupby(
                                        HashKeys.minhash).aggregate(
                                            UnionFn(union_find)).materialize()
        result = dataset_with_id.map_batches(filter_with_union_find,
                                             batch_format='pyarrow')
        logger.info(f'Keep {result.count()} samples after MinHash dedup.')
        union_find.clean()
        return result

    def compute_minhash(self, text):
        """
        Compute minhash values for the sample.

        :param sample: input sample
        :return: sample with minhash value.
        """
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

        # compute minhash value
        hv = np.array([sha1_hash32(token) for token in tokens],
                      dtype=np.uint64)
        phv = np.bitwise_and(
            ((hv * np.tile(self.perm_a,
                           (len(hv), 1)).T).T + self.perm_b) % MERSENNE_PRIME,
            MAX_HASH)
        hash_values = np.vstack([
            phv,
            np.ones(self.num_permutation, dtype=np.uint64) * MAX_HASH
        ]).min(axis=0)
        return [
            bytes(hash_values[start:end].byteswap().data) +
            start.to_bytes(8, byteorder='little')
            for start, end in self.hash_ranges
            # groupby minhash||brand_id
        ]
