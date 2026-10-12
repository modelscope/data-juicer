"""Recovery boundaries for the shared Ray Dataset execution plan."""

from dataclasses import dataclass

import pyarrow as pa
import ray

from data_juicer.core.data.ray_batch_plan import BatchPlanWorker
from data_juicer.core.executor.unit_protocol import (
    LocalUnitStore,
    PreparedOutput,
    ProtocolError,
)
from data_juicer.core.executor.unit_runner import execute_unit_attempt

_SOURCE_OFFSET = "__dj_unit_source_offset__"


def _ticket(unit, stage=-1, commit_id=""):
    return {
        "unit_id": unit.unit_id,
        "ordinal": unit.ordinal,
        "start": unit.start,
        "stop": unit.stop,
        "stage": stage,
        "commit_id": commit_id,
    }


def _initial_ticket(batch, unit_size, total_rows):
    if batch.column_names.count(_SOURCE_OFFSET) != 1:
        raise ProtocolError(f"Input contains reserved recovery column {_SOURCE_OFFSET}")
    offsets = batch[_SOURCE_OFFSET]
    table = batch.drop_columns([_SOURCE_OFFSET])
    rows = table.num_rows
    if not rows:
        raise ProtocolError("Recovery unit source cannot be empty")
    start = offsets[0].as_py()
    stop = offsets[rows - 1].as_py() + 1
    if (
        start % unit_size
        or stop - start != rows
        or any(offsets[index].as_py() != start + index for index in range(rows))
        or (rows != unit_size and stop != total_rows)
    ):
        raise ProtocolError("Ray repartition did not preserve an exact recovery unit")
    ordinal = start // unit_size
    return {
        "unit_id": f"{ordinal:012d}",
        "ordinal": ordinal,
        "start": start,
        "stop": stop,
        "stage": -1,
        "commit_id": "",
    }, table


@ray.remote(num_cpus=0)
class _UnitCoordinator:
    def __init__(self, directory, run_id, units):
        self.store = LocalUnitStore(directory, run_id)
        self.units = {unit.unit_id: unit for unit in units}

    def claim(self, item, stage):
        unit = self.units.get(item["unit_id"])
        if unit is None:
            raise ProtocolError("Ticket unit is absent from the recovery plan")
        upstream = self.store.get_commit(unit.unit_id, stage - 1) if stage else None
        upstream_id = upstream.attempt.attempt_id if upstream is not None else ""
        if (stage and upstream is None) or item != _ticket(unit, stage - 1, upstream_id):
            raise ProtocolError("Ticket does not reference the committed upstream result")
        committed = self.store.get_commit(unit.unit_id, stage)
        attempt = None if committed is not None else self.store.begin(unit.unit_id, stage)
        return unit, attempt, upstream, committed

    def finish(self, attempt, output):
        committed = self.store.get_commit(attempt.unit_id, attempt.stage)
        if committed is not None:
            return committed
        return self.store.commit(attempt, output)


@dataclass(frozen=True)
class UnitWorkerContext:
    directory: str
    run_id: str
    stage: int
    unit_size: int
    total_rows: int
    coordinator: object


class _UnitTicketActor:
    def __init__(self, context, plan):
        self.store = LocalUnitStore(context.directory, context.run_id)
        self.stage = context.stage
        self.unit_size = context.unit_size
        self.total_rows = context.total_rows
        self.coordinator = context.coordinator
        self.operator = BatchPlanWorker(plan)

    def __call__(self, batch):
        if self.stage == 0:
            work = (_initial_ticket(batch, self.unit_size, self.total_rows),)
        else:
            if _SOURCE_OFFSET in batch.column_names:
                raise ProtocolError("Unit source survived beyond the first recovery stage")
            work = ((item, None) for item in batch.to_pylist())
        for item, source_table in work:
            unit, attempt, upstream, committed = ray.get(self.coordinator.claim.remote(item, self.stage))
            if committed is None:
                source_reader = None
                if self.stage == 0:

                    def source_reader(start, stop):
                        if (start, stop) != (unit.start, unit.stop):
                            raise ProtocolError("Unit source does not match the requested unit")
                        yield source_table

                stats_files = ()

                def write_stats(tables):
                    nonlocal stats_files
                    stats_files = self.store.write_output(attempt, tables, stats=True).stats_files
                    return self.store.read_output(stats_files)

                output = execute_unit_attempt(
                    self.store, attempt, unit, source_reader, self.operator, upstream, write_stats=write_stats
                )
                output = PreparedOutput(output.files, stats_files)
                committed = ray.get(self.coordinator.finish.remote(attempt, output))
            yield pa.Table.from_pylist([_ticket(unit, self.stage, committed.attempt.attempt_id)])


def run_unit_task(batch, context, plan):
    yield from _UnitTicketActor(context, plan)(batch)


class UnitRecoveryStage:
    def __init__(self, recovery, stage):
        self.recovery = recovery
        self.stage = stage

    @property
    def complete(self):
        return not self.recovery.units or self.stage < self.recovery.first_stage

    @property
    def worker_context(self):
        recovery = self.recovery
        unit_size = recovery.units[0].stop - recovery.units[0].start if recovery.units else 1
        total_rows = recovery.units[-1].stop if recovery.units else 0
        return UnitWorkerContext(
            str(recovery.store.root), recovery.store.run_id, self.stage, unit_size, total_rows, recovery.coordinator
        )

    def register_stats(self, export):
        self.recovery.stats_exports[self.stage] = export


class UnitRecovery:
    def __init__(self, store, units, source, stage_count):
        self.store = store
        self.units = units
        self.source = source
        self.stage_count = stage_count
        self.first_stage = 0
        self.coordinator = None
        self.stats_exports = {}

    def open(self):
        items = [_ticket(unit) for unit in self.units]
        for stage in range(self.stage_count):
            commits = [self.store.get_commit(unit.unit_id, stage) for unit in self.units]
            if any(commit is None for commit in commits):
                break
            items = [_ticket(unit, stage, commit.attempt.attempt_id) for unit, commit in zip(self.units, commits)]
            self.first_stage = stage + 1
        if self.units and self.first_stage < self.stage_count:
            self.coordinator = _UnitCoordinator.remote(str(self.store.root), self.store.run_id, self.units)
        if not self.units or self.first_stage:
            self.source = None
            return ray.data.from_items(items)
        if _SOURCE_OFFSET in self.source.schema().names:
            raise ProtocolError(f"Input contains reserved recovery column {_SOURCE_OFFSET}")
        total_rows = self.units[-1].stop
        offsets = ray.data.range(total_rows).rename_columns({"id": _SOURCE_OFFSET})
        source = (
            self.source.zip(offsets)
            .repartition(target_num_rows_per_block=self.units[0].stop - self.units[0].start, strict=True)
            .materialize()
        )
        self.source = None
        return source

    def stage(self, index):
        return UnitRecoveryStage(self, index)

    def finish(self, dataset):
        if self.units and self.first_stage < self.stage_count:
            remaining = {unit.unit_id: unit for unit in self.units}
            for batch in dataset.iter_batches(batch_size=None, batch_format="pyarrow", prefetch_batches=0):
                for item in batch.to_pylist():
                    unit = remaining.pop(item["unit_id"], None)
                    if unit is None:
                        raise ProtocolError("Unexpected or duplicate final recovery ticket")
                    committed = self.store.get_commit(unit.unit_id, self.stage_count - 1)
                    if committed is None or item != _ticket(unit, self.stage_count - 1, committed.attempt.attempt_id):
                        raise ProtocolError("Final ticket does not reference a committed result")
            if remaining:
                raise ProtocolError("Missing final recovery tickets")
        return self.output_paths(self.stage_count - 1)

    def output_paths(self, stage, *, stats=False):
        paths = []
        for unit in self.units:
            commit = self.store.get_commit(unit.unit_id, stage)
            if commit is None:
                raise ProtocolError("Recovery unit has no committed output")
            paths.extend(str(self.store.root / item.path) for item in (commit.stats_files if stats else commit.files))
        return paths

    def close(self):
        if self.coordinator is not None:
            ray.kill(self.coordinator, no_restart=True)
            self.coordinator = None
