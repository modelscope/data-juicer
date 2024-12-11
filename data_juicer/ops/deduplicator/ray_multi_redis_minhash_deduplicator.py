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
import concurrent

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
                 redis_address: str = 'redis://localhost:6380'):
        self.prefix = prefix
        self.redis_address = redis_address
        self.redis = redis.from_url(url=redis_address)
        self.set_key = f'{prefix}_UF_SET'
        self.incur_id_key = f'{prefix}_UF_INCURID'

        # Lua scripts
        self.union_script = self.redis.register_script("""
           local function find(x)
                local path = {}
                while true do
                    local parent = redis.call('HGET', KEYS[1], x)
                    if not parent then
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
            if root_x == root_y then
                return root_x
            end
            if root_x < root_y then
                redis.call('HSET', KEYS[1], root_y, root_x)
                return root_x
            else
                redis.call('HSET', KEYS[1], root_x, root_y)
                return root_y
            end
            """)

        self.merge_script = self.redis.register_script("""
           local function find(key, x)
                local path = {}
                while true do
                    local parent = redis.call('HGET', key, x)
                    if not parent then
                        break
                    end
                    table.insert(path, x)
                    x = parent
                end
                for _, node in ipairs(path) do
                    redis.call('HSET', key, node, x)
                end
                return x
            end

            local function merge(key)
                local nodes = redis.call('HKEYS', key)
                for _, node in ipairs(nodes) do
                    local root = find(key, node)
                    local root_x = find(KEYS[1], node)
                    local root_y = find(KEYS[1], root)
                    if root_x < root_y then
                        redis.call('HSET', KEYS[1], root_y, root_x)
                    elseif root_x > root_y then
                        redis.call('HSET', KEYS[1], root_x, root_y)
                    end
                end
            end

            for _, key in ipairs(ARGV) do
                merge(key)
            end
            """)

    def get_uid(self):
        return int(self.redis.incr(self.incur_id_key))

    @retry_on_busy
    def union(self, x, y) -> int:
        return int(self.union_script(keys=[self.set_key],
                                     args=[x, y]))

    @retry_on_busy
    def merge(self, set_keys):
        # self.merge_script(keys=[self.set_key] + set_keys, args=set_keys)
        for set_key in set_keys:
            for x, y in self.redis.hgetall(set_key).items():
                self.union(x, y)
            # for x in self.redis.hkeys(set_key):
            #     y = self.redis.hget(set_key, x)
            #     self.redis.

    def get_nodes(self):
        return set(int(x) for x in self.redis.hkeys(self.set_key))

    def get_data(self):
        result = {}
        for x in self.get_nodes():
            y = int(self.redis.hget(self.set_key, x))
            result[x] = y
        return result

    def is_ancestor(self, x):
        ancestor = self.redis.hget(self.set_key, x)
        return ancestor is None or int(ancestor) == x

    def __reduce__(self):
        return (RedisUnionFind, (self.prefix, self.redis_address))

    def clean(self):
        self.redis.delete(self.set_key, self.incur_id_key)


OP_NAME = 'ray_multi_redis_minhash_deduplicator'


@OPERATORS.register_module(OP_NAME)
class RayMultiRedisMinhashDeduplicator(Deduplicator):
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
        redis_address: str = 'redis://localhost:6380',
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
        self.union_find_parallel_num = union_find_parallel_num
        self.union_find_merge_num = union_find_merge_num

    def run(self, dataset):
        from ray.data.aggregate import AggregateFn

        # union_find = RedisUnionFind(prefix=uuid.uuid4().hex[:8],
        #                             redis_address=self.redis_address)
        union_find_list = [
            RedisUnionFind(prefix=uuid.uuid4().hex[:8] + f'_{i}', redis_address=self.redis_address)
            for i in range(self.union_find_parallel_num)
        ]

        def add_uid_column(table: pa.Table) -> pa.Table:
            new_column_data = [union_find_list[0].get_uid() for _ in range(len(table))]
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

        class UnionFn(AggregateFn):

            def __init__(self, union_find_list):
                # union_find = union_find
                union_find_num = len(union_find_list)

                def accumulate(cur, row):
                    if cur is None:
                        return int.from_bytes(row[HashKeys.minhash][:8], byteorder='big') % union_find_num, row[HashKeys.uid]
                    else:
                        assert cur[0] == int.from_bytes(row[HashKeys.minhash][:8], byteorder='big') % union_find_num
                        union_find = union_find_list[cur[0]]
                        root = union_find.union(row[HashKeys.uid], cur[1])
                        return cur[0], root

                def merge(a, b):
                    if a is None:
                        return b
                    if b is None:
                        return a
                    assert a[0] == b[0]
                    union_find = union_find_list[a[0]]
                    root = union_find.union(a[1], b[1])
                    # root = union_find.union(a, b)
                    return a[0], root

                super().__init__(
                    init=lambda k: None,
                    accumulate_row=accumulate,
                    merge=merge,
                    name='union',
                )

        dataset_with_id = dataset.map_batches(
            add_uid_column, batch_format='pyarrow').materialize()
        dataset_with_id.map_batches(
            calculate_minhash,
            batch_format='pyarrow'
        ).groupby(
            HashKeys.minhash
        ).aggregate(
            UnionFn(union_find_list)
        ).materialize()

        # results = []
        # for union_find in union_find_list:
        #     results.append(union_find.get_data())
        @ray.remote
        def merge(x, keys):
            x.merge(keys)

        merge_list = union_find_list
        while len(merge_list) > 1:
            new_merge_list, buffer = [], []
            task_list = []
            for union_find in merge_list:
                buffer.append(union_find)
                if len(buffer) == self.union_find_merge_num:
                    new_merge_list.append(buffer[0])
                    keys = [u.set_key for u in buffer[1:]]
                    task_list.append(
                        merge.remote(buffer[0], keys)
                    )
                    buffer = []
            if len(buffer) > 0:
                new_merge_list.append(buffer[0])
                if len(buffer) > 1:
                    keys = [u.set_key for u in buffer[1:]]
                    task_list.append(
                        merge.remote(buffer[0], keys)
                    )
            ray.get(task_list)
            merge_list = new_merge_list
            # for m in merge_list:
            #     results.append(m.get_data())

        # results.append(merge_list[0].get_data())
        # import json
        # with open(f'data_{len(results)}.json', 'w') as f:
        #     json.dump(results, f)
        dup_ids = merge_list[0].get_nodes()

        def filter_with_union_find(table: pa.Table) -> pa.Table:
            uids = table.column(HashKeys.uid).to_pandas()
            mask = pa.Array.from_pandas(
                uids.apply(lambda x: x not in dup_ids))
            return table.filter(mask)

        result = dataset_with_id.map_batches(
            filter_with_union_find,
            batch_format='pyarrow'
        ).materialize()
        logger.info(f'Keep {result.count()} samples after MinHash dedup.')
        for union_find in union_find_list:
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

        # # compute minhash value
        # hv = np.array([sha1_hash32(token) for token in tokens],
        #               dtype=np.uint64)
        # phv = np.bitwise_and(
        #     ((hv * np.tile(self.perm_a,
        #                    (len(hv), 1)).T).T + self.perm_b) % MERSENNE_PRIME,
        #     MAX_HASH)
        # hash_values = np.vstack([
        #     phv,
        #     np.ones(self.num_permutation, dtype=np.uint64) * MAX_HASH
        # ]).min(axis=0)
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
        return [
            bytes(hash_values[start:end].byteswap().data) +
            start.to_bytes(4, byteorder='little')
            for start, end in self.hash_ranges
            # groupby minhash||brand_id
        ]
