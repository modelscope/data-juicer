"""Wrap native Ray operator execution in an independent recovery unit."""

from types import GeneratorType

import pyarrow as pa
from ray.data._internal.batcher import Batcher
from ray.data.block import BlockAccessor

from data_juicer.core.data.ray_dataset import (
    RayOperatorActor,
    filter_batch,
    operator_batch_columns,
    prepare_batch_columns,
)
from data_juicer.core.executor.unit_protocol import ProtocolError
from data_juicer.ops import Filter, Mapper
from data_juicer.ops.base_op import DEFAULT_BATCH_SIZE


def _arrow_batches(blocks, batch_size):
    batcher = Batcher(batch_size)
    for block in blocks:
        batcher.add(block)
        while batcher.has_batch():
            yield BlockAccessor.for_block(batcher.next_batch()).to_arrow()
    batcher.done_adding()
    if batcher.has_any():
        yield BlockAccessor.for_block(batcher.next_batch()).to_arrow()


class _OperatorTables:
    def __init__(self, operator_type, args, kwargs, batch_size):
        self.operator_type = operator_type
        self.args = args
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.actor = None

    def __call__(self, table: pa.Table):
        yield from self.process_unit((table,))

    def process_unit(self, tables):
        def compute():
            for table in _arrow_batches(tables, self.batch_size):
                if self.actor is None:
                    self.actor = RayOperatorActor(self.operator_type, self.args, self.kwargs)
                op = self.actor.operator
                table = prepare_batch_columns(table, operator_batch_columns(op))
                result = self.actor(table)
                batches = result if isinstance(result, GeneratorType) else (result,)
                for batch in batches:
                    yield BlockAccessor.for_block(BlockAccessor.batch_to_block(batch)).to_arrow()

        if issubclass(self.operator_type, Filter):
            for table in _arrow_batches(compute(), DEFAULT_BATCH_SIZE):
                op = self.actor.operator
                yield filter_batch(table, op.process, is_batched=op.is_batched_op())
        else:
            yield from compute()


def operator_unit_stage(op) -> _OperatorTables:
    if not isinstance(op, (Mapper, Filter)):
        raise TypeError("Recovery-unit adapter supports Mapper and Filter only")
    if "ray" not in op._supported_exec_modes:
        raise NotImplementedError(
            f"Operator '{op._name or type(op).__name__}' does not support "
            f"Ray mode. Supported modes: {op._supported_exec_modes}"
        )
    if getattr(op, "_prepare_for_ray_map_batches", None) is not None:
        raise ProtocolError("Operator with _prepare_for_ray_map_batches requires shared Ray state")
    if isinstance(op, Filter) and op.stats_export_path is not None:
        raise ProtocolError("Filter stats export needs its own transactional sink")
    batch_size = getattr(op, "batch_size", 1) if op.is_batched_op() else 1
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ProtocolError("Recovery-unit adapter requires a positive integer batch_size")
    return _OperatorTables(op.__class__, op._init_args, op._init_kwargs, batch_size)
