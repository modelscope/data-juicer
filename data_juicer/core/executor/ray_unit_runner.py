"""Ray Data ticket pipeline backed by a shared local POSIX checkpoint store."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pyarrow as pa
import ray

from data_juicer.core.executor.unit_protocol import LocalUnitStore, ProtocolError, Unit
from data_juicer.core.executor.unit_runner import TableOperator, TableSource, collect_final_units, execute_unit_attempt


@dataclass(frozen=True)
class RayStage:
    operator: TableOperator
    actors: int = 1
    num_cpus: float = 1
    num_gpus: float = 0
    runtime_env: Optional[dict] = None


def _ticket(unit, stage=-1, commit_id=""):
    return {
        "unit_id": unit.unit_id,
        "ordinal": unit.ordinal,
        "start": unit.start,
        "stop": unit.stop,
        "stage": stage,
        "commit_id": commit_id,
    }


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


class _UnitTicketActor:
    def __init__(self, directory, run_id, source, operator, stage, coordinator):
        self.store = LocalUnitStore(directory, run_id)
        self.source = source
        self.operator = operator
        self.stage = stage
        self.coordinator = coordinator

    def __call__(self, batch):
        for item in batch.to_pylist():
            unit, attempt, upstream, committed = ray.get(self.coordinator.claim.remote(item, self.stage))
            if committed is None:
                output = execute_unit_attempt(self.store, attempt, unit, self.source, self.operator, upstream)
                committed = ray.get(self.coordinator.finish.remote(attempt, output))
            yield pa.Table.from_pylist([_ticket(unit, self.stage, committed.attempt.attempt_id)])


def run_ray_units(
    store: LocalUnitStore,
    units: Sequence[Unit],
    source: TableSource,
    stages: Sequence[RayStage],
    *,
    materialize_output: bool = True,
) -> Optional[Tuple[pa.Table, ...]]:
    """Emit each downstream ticket only after its complete U output is committed."""
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before running recovery units")
    if not stages:
        raise ValueError("At least one stage is required")
    if any(spec.actors < 1 for spec in stages):
        raise ValueError("Each Ray stage needs at least one actor")
    if not units:
        return () if materialize_output else None

    first_stage = 0
    items = [_ticket(unit) for unit in units]
    for stage in range(len(stages)):
        commits = [store.get_commit(unit.unit_id, stage) for unit in units]
        if any(commit is None for commit in commits):
            break
        items = [_ticket(unit, stage, commit.attempt.attempt_id) for unit, commit in zip(units, commits)]
        first_stage = stage + 1
    if first_stage == len(stages):
        return collect_final_units(store, units, len(stages) - 1) if materialize_output else None

    dataset = ray.data.from_items(items)
    # A yielded ticket must not wait for Ray's business-data-sized output buffer.
    dataset.context.target_max_block_size = 1
    dataset.context.target_min_block_size = 1
    dataset.context.actor_task_retry_on_errors = False
    dataset.context.execution_options.preserve_order = False
    coordinator = _UnitCoordinator.remote(str(store.root), store.run_id, units)
    try:
        for stage in range(first_stage, len(stages)):
            spec = stages[stage]
            dataset = dataset.map_batches(
                _UnitTicketActor,
                batch_size=1,
                batch_format="pyarrow",
                compute=ray.data.ActorPoolStrategy(size=min(spec.actors, len(units)), max_tasks_in_flight_per_actor=1),
                fn_constructor_args=(str(store.root), store.run_id, source, spec.operator, stage, coordinator),
                num_cpus=spec.num_cpus,
                num_gpus=spec.num_gpus,
                runtime_env=spec.runtime_env,
                max_restarts=2,
                max_task_retries=2,
            )
        remaining = {unit.unit_id: unit for unit in units}
        for batch in dataset.iter_batches(batch_size=None, batch_format="pyarrow", prefetch_batches=0):
            for item in batch.to_pylist():
                unit = remaining.pop(item["unit_id"], None)
                if unit is None:
                    raise ProtocolError("Unexpected or duplicate final recovery ticket")
                committed = store.get_commit(unit.unit_id, len(stages) - 1)
                if committed is None or item != _ticket(unit, len(stages) - 1, committed.attempt.attempt_id):
                    raise ProtocolError("Final ticket does not reference a committed result")
        if remaining:
            raise ProtocolError("Missing final recovery tickets")
    finally:
        ray.kill(coordinator, no_restart=True)

    return collect_final_units(store, units, len(stages) - 1) if materialize_output else None
