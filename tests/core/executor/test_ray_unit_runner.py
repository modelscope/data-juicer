"""Real Ray actor and Data-Juicer operator recovery-unit tests."""

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor import ray_unit_runner
from data_juicer.core.executor.ray_unit_runner import UnitRecovery
from data_juicer.core.executor.unit_protocol import LocalUnitStore, ProtocolError
from data_juicer.ops.base_op import Filter, Mapper
from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper
from data_juicer.utils.constant import Fields


class TableMapper(Mapper):
    _batched_op = True

    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback

    def __call__(self, table):
        yield from self.callback(table)

    def process_batched(self, samples):
        raise AssertionError("TableMapper must execute its custom __call__")


class DuplicateMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        return {key: [value for item in values for value in (item, item)] for key, values in samples.items()}


class NullableMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        samples["text"] = [None if value % 2 == 0 else str(value) for value in samples["value"]]
        return samples


class KeepEvenFilter(Filter):
    _batched_op = True

    def compute_stats_batched(self, samples):
        for value, stats in zip(samples["value"], samples[Fields.stats]):
            stats["even"] = value % 2 == 0
        return samples

    def process_batched(self, samples):
        return [stats["even"] for stats in samples[Fields.stats]]


def _planned(tmp_path, rows, stages, unit_size=1):
    source_path = tmp_path / "source.parquet"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    store = LocalUnitStore(str(tmp_path / "recover"), "job-1")
    units = store.plan(
        input_fingerprint=source_hash,
        config_fingerprint="whitespace-then-alphanumeric-v1",
        input_rows=rows,
        unit_size=unit_size,
        stage_count=stages,
    )
    return store, units, ray.data.read_parquet(str(source_path))


def _process_units(store, units, source, ops):
    context = UnitRecovery(store, units, source, stage_count=len(ops))
    result = RayDataset(source, auto_op_parallelism=False).process(ops, unit_recovery=context)
    assert isinstance(result, RayDataset)
    return tuple(result.data.iter_batches(batch_format="pyarrow"))


def test_rejects_reserved_source_offset_column(tmp_path, local_ray):
    rows = [{"value": 1, "__dj_unit_source_offset__": "user data"}]
    source = ray.data.from_items(rows)
    store = LocalUnitStore(str(tmp_path / "recover"), "job-1")
    units = store.plan(
        input_fingerprint="input-v1",
        config_fingerprint="identity-v1",
        input_rows=1,
        unit_size=1,
        stage_count=1,
    )
    context = UnitRecovery(store, units, source, stage_count=1)
    with pytest.raises(ProtocolError, match="reserved recovery column"):
        RayDataset(source, auto_op_parallelism=False).process(
            [TableMapper(identity, num_proc=1, num_cpus=1, ray_execution_mode="actor")],
            unit_recovery=context,
        )


def test_source_blocks_are_repartitioned_into_exact_units_without_persisting_input(tmp_path, local_ray):
    rows = [{"value": value} for value in range(23)]
    source = ray.data.from_items(rows, override_num_blocks=4)
    store = LocalUnitStore(str(tmp_path / "recover"), "job-1")
    units = store.plan(
        input_fingerprint="input-v1",
        config_fingerprint="identity-v1",
        input_rows=len(rows),
        unit_size=5,
        stage_count=1,
    )
    op = TableMapper(identity, num_proc=1, num_cpus=1, ray_execution_mode="actor")
    result = _process_units(store, units, source, [op])
    assert [row for block in result for row in block.to_pylist()] == rows
    assert [store.get_commit(unit.unit_id, 0).rows for unit in units] == [5, 5, 5, 5, 3]


def test_completed_first_stage_does_not_read_source_again(tmp_path, local_ray):
    rows = [{"value": value} for value in range(8)]
    source = ray.data.from_items(rows, override_num_blocks=3)
    store = LocalUnitStore(str(tmp_path / "recover"), "job-1")
    units = store.plan(
        input_fingerprint="input-v1",
        config_fingerprint="identity-v1",
        input_rows=len(rows),
        unit_size=3,
        stage_count=1,
    )
    op = TableMapper(identity, num_proc=1, num_cpus=1, ray_execution_mode="actor")
    expected = _process_units(store, units, source, [op])

    def fail_if_read(batch):
        raise AssertionError("completed stage 0 reread the source")

    poisoned = source.map_batches(fail_if_read, batch_format="pyarrow")
    restored = _process_units(store, units, poisoned, [op])
    assert [block.to_pylist() for block in restored] == [block.to_pylist() for block in expected]


def test_actual_mapper_filter_complete_empty_unit_without_schema(tmp_path, local_ray):
    pq.write_table(pa.table({"text": [" A\tB ", "!!!", " C "]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 3, 2)
    ops = [
        WhitespaceNormalizationMapper(num_proc=1, num_cpus=1, ray_execution_mode="actor"),
        AlphanumericFilter(min_ratio=0.5, num_proc=1, num_cpus=1, ray_execution_mode="actor"),
    ]
    result = _process_units(store, units, source, ops)
    assert [text for block in result for text in block["text"].to_pylist()] == ["A B", "C"]
    assert store.get_commit(units[1].unit_id, 1).rows == 0
    assert store.get_commit(units[1].unit_id, 1).files == ()
    assert [text for block in _process_units(store, units, source, ops) for text in block["text"].to_pylist()] == [
        "A B",
        "C",
    ]


def with_worker_pid(table):
    yield table.append_column("worker_pid", pa.array([os.getpid()] * table.num_rows))


def identity(table):
    yield table


@dataclass
class FailSecondUnitOnce:
    marker: str

    def __call__(self, table):
        if table["value"][0].as_py() == 1 and not os.path.exists(self.marker):
            with open(self.marker, "w") as stream:
                stream.write("attempted")
            raise RuntimeError("worker failed inside unit")
        return (table,)


@dataclass
class WaitForDownstream:
    directory: str
    empty_first: bool

    def __call__(self, table):
        value = table["value"][0].as_py()
        if value == 0 and self.empty_first:
            return ()
        if value == 1:
            store = LocalUnitStore(self.directory, "job-1")
            deadline = time.monotonic() + 30
            while store.get_commit("000000000000", 1) is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("first U never reached downstream before the next U")
                time.sleep(0.05)
        return (table,)


@dataclass
class SlowFirstUnit:
    directory: str

    def __call__(self, table):
        if table["value"][0].as_py() == 0:
            store = LocalUnitStore(self.directory, "job-1")
            deadline = time.monotonic() + 30
            while store.get_commit("000000000001", 1) is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("A completed later U was blocked by input order")
                time.sleep(0.05)
        return (table,)


@dataclass
class RecordCalls:
    path: str
    crash_point: Optional[str] = None

    def __call__(self, table):
        value = table["value"][0].as_py()
        with open(self.path, "a") as stream:
            stream.write(f"{value}\n")
            stream.flush()
            os.fsync(stream.fileno())
        marker = self.path + ".crashed"
        should_crash = value == 1 and not os.path.exists(marker)
        if should_crash and self.crash_point == "before_write":
            with open(marker, "x"):
                pass
            os._exit(86)
        yield table
        if should_crash and self.crash_point == "during_write":
            with open(marker, "x"):
                pass
            os._exit(86)


class ExitAfterCommitOnce(ray_unit_runner._UnitTicketActor):
    def __call__(self, batch):
        for result in super().__call__(batch):
            marker = self.store.root / "exited-after-commit"
            if result["ordinal"][0].as_py() == 1 and not marker.exists():
                with marker.open("x"):
                    pass
                os._exit(86)
            yield result


@pytest.fixture
def local_ray():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


def test_ray_actor_reused_across_units_and_commits_in_input_order(tmp_path, local_ray):
    pq.write_table(pa.table({"value": [0, 1, 2, 3]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 4, 1)
    op = TableMapper(with_worker_pid, num_proc=1, num_cpus=1, ray_execution_mode="actor")
    result = _process_units(store, units, source, [op])
    assert [value for block in result for value in block["value"].to_pylist()] == [0, 1, 2, 3]
    assert len({pid for block in result for pid in block["worker_pid"].to_pylist()}) == 1
    assert all(store.get_commit(unit.unit_id, 0) is not None for unit in units)
    reopened, same_units, same_source = _planned(tmp_path, 4, 1)
    restored = _process_units(reopened, same_units, same_source, [op])
    assert [block.to_pylist() for block in restored] == [block.to_pylist() for block in result]


def test_ray_runner_with_real_mapper_and_filter(tmp_path, local_ray):
    pq.write_table(pa.table({"text": [" A\tB ", "!!!", " C "]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 3, 2)
    ops = [
        WhitespaceNormalizationMapper(num_proc=1, num_cpus=1, ray_execution_mode="actor"),
        AlphanumericFilter(min_ratio=0.5, num_proc=1, num_cpus=1, ray_execution_mode="actor"),
    ]
    result = _process_units(store, units, source, ops)
    assert [text for block in result for text in block["text"].to_pylist()] == ["A B", "C"]
    assert store.get_commit(units[1].unit_id, 1).rows == 0


def test_ray_mapper_expansion_and_filter_zero_output(tmp_path, local_ray):
    pq.write_table(pa.table({"value": [0, 1, 2]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 3, 2)
    ops = [
        DuplicateMapper(num_proc=1, num_cpus=1, ray_execution_mode="actor"),
        KeepEvenFilter(num_proc=1, num_cpus=1, ray_execution_mode="actor"),
    ]
    result = _process_units(store, units, source, ops)
    assert [value for block in result for value in block["value"].to_pylist()] == [0, 0, 2, 2]
    assert [store.get_commit(unit.unit_id, 0).rows for unit in units] == [2, 2, 2]
    assert [store.get_commit(unit.unit_id, 1).rows for unit in units] == [2, 0, 2]


def test_ray_mapper_nullable_schema_changes_within_unit_and_resumes(tmp_path, local_ray):
    pq.write_table(pa.table({"value": [1, 2]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 2, 1, unit_size=2)
    op = NullableMapper(batch_size=1, num_proc=1, num_cpus=1, ray_execution_mode="actor")
    result = _process_units(store, units, source, [op])
    expected = [{"value": 1, "text": "1"}, {"value": 2, "text": None}]
    assert [row for block in result for row in block.to_pylist()] == expected
    assert len(units) == 1
    committed = store.get_commit(units[0].unit_id, 0)
    assert committed is not None
    assert committed.rows == 2
    assert [entry.rows for entry in committed.files] == [1, 1]
    checkpoints = [pq.read_table(store.root / entry.path) for entry in committed.files]
    assert [table.schema.field("text").type for table in checkpoints] == [pa.string(), pa.null()]
    assert [row for table in checkpoints for row in table.to_pylist()] == expected
    reopened, same_units, same_source = _planned(tmp_path, 2, 1, unit_size=2)
    restored = _process_units(reopened, same_units, same_source, [op])
    assert [row for block in restored for row in block.to_pylist()] == expected
    assert reopened.get_commit(same_units[0].unit_id, 0) == committed


def test_ray_failure_replays_only_uncommitted_unit(tmp_path, local_ray):
    pq.write_table(pa.table({"value": [0, 1]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 2, 1)
    op = TableMapper(
        FailSecondUnitOnce(str(tmp_path / "failed-once")), num_proc=1, num_cpus=1, ray_execution_mode="actor"
    )
    with pytest.raises(ray.exceptions.RayTaskError, match="worker failed inside unit"):
        _process_units(store, units, source, [op])
    first = store.get_commit(units[0].unit_id, 0)
    assert first is not None
    assert store.get_commit(units[1].unit_id, 0) is None
    reopened, same_units, same_source = _planned(tmp_path, 2, 1)
    result = _process_units(reopened, same_units, same_source, [op])
    assert reopened.get_commit(units[0].unit_id, 0) == first
    assert [value for block in result for value in block["value"].to_pylist()] == [0, 1]


@pytest.mark.parametrize("empty_first", [False, True])
def test_first_ticket_reaches_downstream_before_upstream_finishes(tmp_path, local_ray, monkeypatch, empty_first):
    pq.write_table(pa.table({"value": [0, 1, 2, 3]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 4, 2)
    original = ray.data.from_items
    monkeypatch.setattr(
        ray.data, "from_items", lambda items, **kwargs: original(items, **{**kwargs, "override_num_blocks": 1})
    )
    context = ray.data.DataContext.get_current()
    original_sizes = (context.target_min_block_size, context.target_max_block_size)
    ops = [
        TableMapper(
            WaitForDownstream(str(store.root), empty_first), num_proc=1, num_cpus=1, ray_execution_mode="actor"
        ),
        TableMapper(identity, num_proc=1, num_cpus=1, ray_execution_mode="actor"),
    ]
    result = _process_units(store, units, source, ops)
    assert [v for block in result for v in block["value"].to_pylist()] == ([1, 2, 3] if empty_first else [0, 1, 2, 3])
    assert (context.target_min_block_size, context.target_max_block_size) == original_sizes
    assert store.get_commit(units[0].unit_id, 1).rows == (0 if empty_first else 1)


def test_completed_later_unit_is_not_blocked_by_slow_first_unit(tmp_path, local_ray, monkeypatch):
    pq.write_table(pa.table({"value": [0, 1]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 2, 2)
    original = ray.data.from_items
    monkeypatch.setattr(
        ray.data, "from_items", lambda items, **kwargs: original(items, **{**kwargs, "override_num_blocks": 2})
    )
    ops = [
        TableMapper(SlowFirstUnit(str(store.root)), num_proc=2, num_cpus=0.5, ray_execution_mode="actor"),
        TableMapper(identity, num_proc=1, num_cpus=0.5, ray_execution_mode="actor"),
    ]
    result = _process_units(store, units, source, ops)
    assert [value for table in result for value in table["value"].to_pylist()] == [0, 1]


@pytest.mark.parametrize("crash_point", ["before_write", "during_write", "after_commit"])
def test_actor_restart_replays_only_uncommitted_work(tmp_path, local_ray, monkeypatch, crash_point):
    pq.write_table(pa.table({"value": [0, 1, 2, 3]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 4, 1)
    calls = tmp_path / "calls"
    original = ray.data.from_items
    monkeypatch.setattr(
        ray.data, "from_items", lambda items, **kwargs: original(items, **{**kwargs, "override_num_blocks": 1})
    )
    if crash_point == "after_commit":
        monkeypatch.setattr(ray_unit_runner, "_UnitTicketActor", ExitAfterCommitOnce)
    op = TableMapper(
        RecordCalls(str(calls), crash_point=None if crash_point == "after_commit" else crash_point),
        num_proc=1,
        num_cpus=1,
        ray_execution_mode="actor",
    )
    result = _process_units(store, units, source, [op])
    assert [v for block in result for v in block["value"].to_pylist()] == [0, 1, 2, 3]
    observed = [int(value) for value in calls.read_text().splitlines()]
    assert observed.count(1) == (1 if crash_point == "after_commit" else 2)
    assert all(observed.count(value) == 1 for value in (0, 2, 3))


@pytest.mark.parametrize("mode", ["actor", "task"])
def test_native_and_recovery_share_all_planned_transformations(tmp_path, local_ray, monkeypatch, mode):
    from data_juicer.core.data.ray_batch_plan import RayBatchPlan

    original = RayBatchPlan.map_batches
    builds = []

    def add_shared_column(table):
        return table.append_column("shared_plan", pa.array([True] * table.num_rows))

    def extend_plan(plan, function, **options):
        original(plan, function, **options)
        if "compute" in options:
            builds.append(options["compute"])
            original(plan, add_shared_column, batch_size=2, batch_format="pyarrow")
        return plan

    monkeypatch.setattr(RayBatchPlan, "map_batches", extend_plan)
    table = pa.table({"text": [" A\tB ", " C ", " D "]})
    pq.write_table(table, tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 3, 1, unit_size=2)
    op = WhitespaceNormalizationMapper(num_proc=1, ray_execution_mode=mode)
    native = RayDataset(ray.data.from_arrow(table), auto_op_parallelism=False).process([op]).data.take_all()
    recovered = [row for block in _process_units(store, units, source, [op]) for row in block.to_pylist()]
    assert recovered == native
    assert all(row["shared_plan"] is True for row in recovered)
    assert len(builds) == 2
    assert type(builds[0]) is type(builds[1])
