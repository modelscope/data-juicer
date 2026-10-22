import hashlib
import inspect
from dataclasses import dataclass
from types import GeneratorType

import pyarrow.fs as pafs
from ray.data import ActorPoolStrategy, Dataset
from ray.data._internal.block_batching import batch_blocks
from ray.data.block import BlockAccessor
from ray.data.datasource.path_util import _resolve_paths_and_filesystem


@dataclass
class BatchOperation:
    function: object
    options: dict


@dataclass
class StatsExport:
    path: str
    options: dict

    def publish(self, data, identity):
        prefix = "dj-stats-" + hashlib.sha256("\0".join(identity).encode()).hexdigest() + "_"
        destinations, filesystem = _resolve_paths_and_filesystem(self.path, self.options.get("filesystem"))
        for entry in filesystem.get_file_info(pafs.FileSelector(destinations[0], allow_not_found=True)):
            if entry.is_file and entry.base_name.startswith(prefix) and entry.base_name.endswith(".json"):
                filesystem.delete_file(entry.path)
        # A stable ownership prefix permits replay without deleting unrelated exports.
        data._set_uuid(prefix)
        data.write_json(self.path, **self.options)


class RayBatchPlan:
    def __init__(self):
        self.steps = []

    def map_batches(self, function, **options):
        batch_size = options.get("batch_size")
        if isinstance(batch_size, int) and batch_size < 1:
            raise ValueError("Batch size can't be negative or 0")
        if (options.get("num_gpus") or 0) > 0 and batch_size in (None, "auto"):
            raise ValueError("You must provide an explicit batch_size when requesting GPUs")
        self.steps.append(BatchOperation(function, options))
        return self

    def materialize(self):
        self.steps.append("materialize")
        return self

    def write_json(self, path, **options):
        self.steps.append(StatsExport(path, options))

    def apply(self, dataset, recovery=None):
        if recovery is None:
            for step in self.steps:
                if isinstance(step, BatchOperation):
                    dataset = dataset.map_batches(step.function, **step.options)
                elif isinstance(step, StatsExport):
                    dataset.write_json(step.path, **step.options)
                else:
                    dataset = dataset.materialize()
            return dataset

        from data_juicer.core.executor import ray_partition_runner
        from data_juicer.core.executor.partition_protocol import ProtocolError

        if "materialize" in self.steps:
            raise ProtocolError("A global materialization barrier cannot execute within a checkpointed batch")
        for step in self.steps:
            if isinstance(step, StatsExport):
                recovery.register_stats(step)
        if dataset is None:
            return None
        compute_step = next(
            step for step in self.steps if isinstance(step, BatchOperation) and "compute" in step.options
        )
        options = {
            key: compute_step.options[key]
            for key in ("compute", "num_cpus", "num_gpus", "runtime_env")
            if key in compute_step.options
        }
        if isinstance(options["compute"], ActorPoolStrategy):
            function = ray_partition_runner.CheckpointStageActor
            options.update(fn_constructor_args=(recovery.context, self))
        else:
            function = ray_partition_runner.run_checkpoint_task
            options.update(fn_args=(recovery.context, self))
        return map_whole_blocks(dataset, function, batch_format="pyarrow", zero_copy_batch=True, **options)


def map_whole_blocks(dataset, function, **options):
    """Hand every Ray block to ``function`` unmerged, also for GPU stages.

    An integer ``batch_size`` makes Ray merge smaller blocks into one task, and
    the public ``map_batches`` rejects ``batch_size=None`` together with GPUs.
    """
    method = dataset._map_batches_without_batch_size_validation
    accepted = inspect.signature(method).parameters
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(Dataset.map_batches).parameters.items()
        if name in accepted and parameter.default is not inspect.Parameter.empty
    }
    return method(function, **{**defaults, **options, "batch_size": None})


class BatchPlanWorker:
    def __init__(self, plan):
        self.plan = plan
        self.functions = {}

    def _map(self, tables, index, step):
        options = step.options
        batch_size = options.get("batch_size")
        if batch_size == "auto":
            from ray.data._internal.execution.operators.map_transformer import (
                _compute_auto_batch_size,
            )

            batch_size, tables = _compute_auto_batch_size(tables)
        zero_copy_batch = options.get("zero_copy_batch", True)
        batches = batch_blocks(
            iter(tables),
            batch_size=batch_size,
            batch_format=options.get("batch_format", "default"),
            ensure_copy=not zero_copy_batch,
        )
        for batch in batches:
            # Arrow-to-NumPy conversion can recreate read-only views after the block copy.
            if not zero_copy_batch and isinstance(batch, dict):
                batch = {key: column if column.flags.writeable else column.copy() for key, column in batch.items()}
            if index not in self.functions:
                function = step.function
                if isinstance(function, type):
                    function = function(
                        *(options.get("fn_constructor_args") or ()), **(options.get("fn_constructor_kwargs") or {})
                    )
                self.functions[index] = function
            result = self.functions[index](batch, *(options.get("fn_args") or ()), **(options.get("fn_kwargs") or {}))
            outputs = result if isinstance(result, GeneratorType) else (result,)
            for output in outputs:
                yield BlockAccessor.for_block(BlockAccessor.batch_to_block(output)).to_arrow()

    def process_partition(self, tables, *, write_stats=None):
        for index, step in enumerate(self.plan.steps):
            if isinstance(step, BatchOperation):
                tables = self._map(tables, index, step)
            elif isinstance(step, StatsExport):
                if write_stats is None:
                    raise RuntimeError("Statistics export requires an attempt-owned writer")
                tables = write_stats(tables)
            else:
                raise RuntimeError("A global materialization barrier cannot execute within a checkpointed batch")
        yield from tables
