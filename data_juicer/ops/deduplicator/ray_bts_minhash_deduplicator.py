import os
import time
from typing import List, Optional, Union

import numpy as np
import pyarrow as pa
import ray
import regex
from loguru import logger
from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.model_utils import prepare_sentencepiece_model

from ..base_op import OPERATORS, Deduplicator
from ..common.helper_func import split_on_whitespace
from .document_minhash_deduplicator import (MAX_HASH, MERSENNE_PRIME,
                                            optimal_param, sha1_hash32)

BATCH_SIZE = 1000


@ray.remote
class IdGenerator:

    def __init__(self, start_id=0):
        self.next_id = start_id

    @ray.method(num_returns=2)
    def get_next_id(self, count):
        current_id = self.next_id
        self.next_id += count
        return (current_id, self.next_id)


@ray.remote(scheduling_strategy='SPREAD')
class EdgeBuffer:

    def __init__(self):
        self.edge_dict = {}

    def clear(self):
        self.edge_dict = {}

    def set_edges(self, edge_dict):
        self.edge_dict = edge_dict

    def get_edges(self, key):
        return self.edge_dict.pop(key, [])


@ray.remote(scheduling_strategy='SPREAD')
class BTSUnionFind:
    """
    A distributed implementation of Union-Find with load balancing.

    The original paper on BTS Union-Find is available at:
    https://ieeexplore.ieee.org/document/10598116
    """

    def __init__(
        self,
        union_threshold,
        parallel_num,
        parallel_id,
        remote_edge_buffers,
        max_pending_edge_buffer_task,
        num_edge_buffer_task_returns,
    ):
        self.union_threshold = union_threshold
        self.parallel_num = parallel_num
        self.parallel_id = parallel_id
        self.hash_table = {}
        self.parent = {}
        self.old_parent = {}
        self.remote_edge_buffers = remote_edge_buffers
        self.edge_buffer = []
        self.edge_list_dict = {}
        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns

    def add_key_value_pairs(self, pairs):
        for key, value in pairs:
            if key not in self.hash_table:
                self.hash_table[key] = []
            self.hash_table[key].append(value)
            if len(self.hash_table[key]) > self.union_threshold:
                self.hash_table[key] = [self.union_list(self.hash_table[key])]

    def flush_key_value_pairs(self):
        for value in self.hash_table.values():
            if len(value) > 1:
                self.union_list(value)
        del self.hash_table

    def balanced_union_find(self):
        for x, y in self.edge_buffer:
            self.union(x, y)
        self.edge_buffer = []
        result_refs = []
        for remote_edge_buffer in self.remote_edge_buffers:
            if len(result_refs) > self.max_pending_edge_buffer_task:
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=self.num_edge_buffer_task_returns)
                edge_list = ray.get(ready_refs)
                for edges in edge_list:
                    for x, y in edges:
                        self.union(x, y)
                del ready_refs
            result_refs.append(
                remote_edge_buffer.get_edges.remote(self.parallel_id))
        edge_list = ray.get(result_refs)
        for edges in edge_list:
            for x, y in edges:
                self.union(x, y)
        del edge_list, result_refs
        self.rebalancing()
        return self.old_parent != self.parent

    def distribute_edge(self, u, v):
        hash_u = u // BATCH_SIZE % self.parallel_num
        hash_v = v // BATCH_SIZE % self.parallel_num
        if hash_u not in self.edge_list_dict:
            self.edge_list_dict[hash_u] = []
        self.edge_list_dict[hash_u].append((u, v))
        if hash_u != hash_v:
            if hash_v not in self.edge_list_dict:
                self.edge_list_dict[hash_v] = []
            self.edge_list_dict[hash_v].append((u, v))

    def set_edge_buffer(self):
        if self.parallel_id in self.edge_list_dict:
            self.edge_buffer = self.edge_list_dict[self.parallel_id]
            del self.edge_list_dict[self.parallel_id]
        else:
            self.edge_buffer = []
        ray.get(self.remote_edge_buffers[self.parallel_id].set_edges.remote(
            self.edge_list_dict))
        self.edge_list_dict = {}

    def edge_redistribution(self):
        self.flush_key_value_pairs()
        self.rebalancing()
        self.edge_list_dict = {}
        for u, v in self.parent.items():
            self.distribute_edge(u, v)
        self.parent = {}
        self.set_edge_buffer()

    def communication(self):
        self.edge_list_dict = {}
        del_list = []
        for u, v in self.parent.items():
            hash_u = u // BATCH_SIZE % self.parallel_num
            if self.parent[u] != self.old_parent.get(u, u) or (
                    hash_u != self.parallel_id and v not in self.parent):
                self.distribute_edge(u, v)
            if hash_u != self.parallel_id:
                del_list.append(u)
        self.old_parent = self.parent.copy()
        for u in del_list:
            del self.parent[u]
        self.set_edge_buffer()

    def find(self, x):
        if x not in self.parent:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

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
        return p

    def rebalancing(self):
        new_px_dict = {}
        for x in self.parent:
            hash_x = x // BATCH_SIZE % self.parallel_num
            px = self.find(x)
            key = (px, hash_x)
            if key not in new_px_dict:
                new_px_dict[key] = x
            else:
                new_px_dict[key] = min(new_px_dict[key], x)
        px_set = set(px for px, _ in new_px_dict)
        for px in px_set:
            hash_px = px // BATCH_SIZE % self.parallel_num
            key = (px, hash_px)
            if key not in new_px_dict:
                new_px_dict[key] = px
            else:
                new_px_dict[key] = min(new_px_dict[key], px)

        for x in self.parent:
            hash_x = x // BATCH_SIZE % self.parallel_num
            px = self.find(x)
            key = (px, hash_x)
            if x == new_px_dict[key]:
                continue
            self.parent[x] = new_px_dict[key]

    def squeeze(self):
        dup_keys = {
            x
            for x in self.parent
            if x // BATCH_SIZE % self.parallel_num == self.parallel_id
        }
        self.parent = dup_keys
        self.old_parent = {}
        self.edge_buffer = []
        ray.get(self.remote_edge_buffers[self.parallel_id].clear.remote())

    def dup_idx(self, queries):
        return [idx for uid, idx in queries if uid in self.parent]


OP_NAME = 'ray_bts_minhash_deduplicator'


@OPERATORS.register_module(OP_NAME)
class RayBTSMinhashDeduplicator(Deduplicator):
    """
    A MinhashLSH deduplicator based on RAY.
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
        union_find_parallel_num: Union[int, str] = 'auto',
        union_threshold: Optional[int] = 256,
        max_pending_edge_buffer_task: Optional[int] = 20,
        num_edge_buffer_task_returns: Optional[int] = 10,
        max_pending_filter_tasks: Optional[int] = 20,
        num_filter_task_returns: Optional[int] = 10,
        merge_batch_size: Optional[int] = 1000,
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
        :param union_find_parallel_num: number of parallel workers for
            union-find algorithm. Default it's 'auto', and it will be
            determined by half of the number of CPUs.
        :param union_threshold: threshold for minhash values group to
            perform union-find algorightm. Default it's 256.
        :param max_pending_edge_buffer_task: max number of pending edge buffer
            ray tasks. Default it's 20.
        :param num_edge_buffer_task_returns: number of edge buffer tasks for
            `ray.wait` to return. Default it's 10.
        :param max_pending_filter_tasks: max number of pending filter ray
            tasks. Default it's 20.
        :param num_filter_task_returns: number of filter tasks for `ray.wait`
            to return. Default it's 10.
        :param merge_batch_size: batch size for BTS operations. Default
            it's 1000.
        :param tmp_file_name: the temporary folder name for deduplication.
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

        if self.tokenization == 'character':

            def tokenization_func(text):
                return {
                    str.encode(text[i:i + self.window_size])
                    for i in range(len(text) - self.window_size)
                }
        elif self.tokenization == 'punctuation':

            def tokenization_func(text):
                tokens = self.punctuation_pattern.split(text)
                return {
                    str.encode(' '.join(tokens[i:i + self.window_size]))
                    for i in range(len(tokens) - self.window_size)
                }
        elif self.tokenization == 'space':

            def tokenization_func(text):
                tokens = split_on_whitespace(text)
                return {
                    str.encode(' '.join(tokens[i:i + self.window_size]))
                    for i in range(len(tokens) - self.window_size)
                }
        elif self.tokenization == 'sentencepiece':

            def tokenization_func(text):
                tokens = self.tokenizer.encode(text, out_type=str)
                return {
                    str.encode(''.join(tokens[i:i + self.window_size]))
                    for i in range(len(tokens) - self.window_size)
                }
        else:
            raise NotImplementedError(
                f'Unimplemented tokenization method [{self.tokenization}]')
        self.tokenization_func = tokenization_func

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

        # generate permutations
        gen = np.random.RandomState(seed=42)
        self.perm_a, self.perm_b = np.array(
            [(
                gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            ) for _ in range(self.num_permutation)],
            dtype=np.uint64,
        ).T

        if union_find_parallel_num == 'auto':
            union_find_parallel_num = int(ray.cluster_resources().get('CPU') /
                                          2)
        else:
            union_find_parallel_num = int(union_find_parallel_num)

        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns
        self.max_pending_filter_tasks = max_pending_filter_tasks
        self.num_filter_task_returns = num_filter_task_returns
        self.merge_batch_size = min(merge_batch_size, union_find_parallel_num)

        logger.info(f'union_find_parallel_num = {union_find_parallel_num}')
        self.union_find_parallel_num = union_find_parallel_num
        self.union_threshold = union_threshold
        self.remote_edge_buffers = [
            EdgeBuffer.remote() for _ in range(self.union_find_parallel_num)
        ]
        self.union_find_list = [
            BTSUnionFind.remote(
                self.union_threshold,
                self.union_find_parallel_num,
                i,
                self.remote_edge_buffers,  # TODO: fix this
                self.max_pending_edge_buffer_task,
                self.num_edge_buffer_task_returns,
            ) for i in range(self.union_find_parallel_num)
        ]

        empty_hash_value = np.full((self.num_rows_per_band, ),
                                   MAX_HASH,
                                   dtype=np.uint32)
        self.empty_hash_value = b'\x00\x00\x00\x00' \
            + empty_hash_value.tobytes()
        self.empty_hash_table_id = int(MAX_HASH % self.union_find_parallel_num)

    def calc_minhash(self, text_list: pa.Array, uid_list: List) -> pa.Table:
        pairs = {}

        for text, uid in zip(text_list, uid_list):
            text = text.as_py()
            if self.lowercase:
                text = text.lower()
            if self.ignore_pattern:
                text = self.ignore_pattern.sub('', text)

            tokens = self.tokenization_func(text)

            if len(tokens) > 0:
                hv = np.array([sha1_hash32(token) for token in tokens],
                              dtype=np.uint64)
                phv = ((hv[:, None] * self.perm_a[None, :] + self.perm_b) %
                       MERSENNE_PRIME).astype(np.uint32)
                hash_values = phv.min(axis=0)
                for i, (start, end) in enumerate(self.hash_ranges):
                    hash_value = i.to_bytes(4, 'big') \
                        + hash_values[start:end].tobytes()
                    hash_table_id = hash_values[start] \
                        % self.union_find_parallel_num
                    if hash_table_id not in pairs:
                        pairs[hash_table_id] = []
                    pairs[hash_table_id].append((hash_value, uid))
            else:
                if self.empty_hash_table_id not in pairs:
                    pairs[self.empty_hash_table_id] = []
                pairs[self.empty_hash_table_id].append(
                    (self.empty_hash_value, uid))
        result_refs = []
        for i, p in pairs.items():
            if len(result_refs) > self.max_pending_filter_tasks:
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=self.num_filter_task_returns)
                ray.get(ready_refs)
            result_refs.append(
                self.union_find_list[i].add_key_value_pairs.remote(p))
        ray.get(result_refs)

    def merge_op_batch(self, object_refs):
        results = []
        while object_refs:
            ready_refs, object_refs = ray.wait(object_refs,
                                               num_returns=min(
                                                   self.merge_batch_size,
                                                   len(object_refs)))
            results.extend(ray.get(ready_refs))
        return results

    def merge(self):
        self.merge_op_batch([
            union_find.edge_redistribution.remote()
            for union_find in self.union_find_list
        ])
        while any(
                self.merge_op_batch([
                    union_find.balanced_union_find.remote()
                    for union_find in self.union_find_list
                ])):
            self.merge_op_batch([
                union_find.communication.remote()
                for union_find in self.union_find_list
            ])
        self.merge_op_batch([
            union_find.squeeze.remote() for union_find in self.union_find_list
        ])

    def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
        query_dict = {}
        for idx, uid in enumerate(samples[HashKeys.uid]):
            uid = uid.as_py()
            hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
            if hash_id not in query_dict:
                query_dict[hash_id] = []
            query_dict[hash_id].append((uid, idx))
        mask = np.ones(len(samples), dtype=np.bool_)
        result_refs = []
        for hash_id, query in query_dict.items():
            if len(result_refs) > self.max_pending_filter_tasks:
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=self.num_filter_task_returns)
                results = ray.get(ready_refs)
                for result in results:
                    mask[result] = False
                del ready_refs
            result_refs.append(
                self.union_find_list[hash_id].dup_idx.remote(query))
        results = ray.get(result_refs)
        for result in results:
            mask[result] = False
        del query_dict, results
        columns_to_keep = [
            name for name in samples.column_names if name != HashKeys.uid
        ]
        return samples.select(columns_to_keep).filter(mask)

    def run(self, dataset):
        start_time = time.time()
        id_generator = IdGenerator.remote()

        def minhash_with_uid(table: pa.Table) -> pa.Table:
            num_rows = len(table)
            min_id, max_id = ray.get(id_generator.get_next_id.remote(num_rows))
            uid_list = range(min_id, max_id)
            self.calc_minhash(table[self.text_key], uid_list)
            new_table = table.append_column(HashKeys.uid,
                                            pa.array(list(uid_list)))
            return new_table

        tmp_dir = os.path.join(self.work_dir, '.tmp',
                               ray.get_runtime_context().get_job_id())
        dataset.map_batches(
            minhash_with_uid,
            batch_format='pyarrow',
            zero_copy_batch=True,
        ).write_parquet(tmp_dir)
        dataset = ray.data.read_parquet(tmp_dir)
        end_time = time.time()
        logger.info(f'MinHash time = {end_time - start_time}')

        start_time = time.time()
        self.merge()
        end_time = time.time()
        logger.info(f'merge time = {end_time - start_time}')
        result = dataset.map_batches(
            self.filter_with_union_find,
            batch_format='pyarrow',
            zero_copy_batch=True,
        )
        return result
