"""Exercise the public partitioned executor dispatch with recovery units."""

import json
import sqlite3

import pytest
import ray
import yaml
from jsonargparse import Namespace

from data_juicer.config import init_configs
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.core.executor.unit_protocol import ProtocolError
from data_juicer.ops.base_op import Filter, Mapper
from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
from data_juicer.ops.filter.suffix_filter import SuffixFilter
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper
from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager
from data_juicer.utils.constant import Fields


class DuplicateMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        return {key: [value for item in values for value in (item, item)] for key, values in samples.items()}


class VariableMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        keep = [i for i, value in enumerate(samples["id"]) if value % 2]
        output = {key: [values[i] for i in keep] for key, values in samples.items()}
        output["mapped"] = [True] * len(keep)
        return output


class BatchSizeMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        samples["batch_rows"] = [len(samples["id"])] * len(samples["id"])
        return samples


class MaximumBatchFilter(Filter):
    _batched_op = True

    def compute_stats_batched(self, samples):
        for stats in samples[Fields.stats]:
            stats["compute_batch_rows"] = len(samples["id"])
        return samples

    def process_batched(self, samples):
        return [value == max(samples["id"]) for value in samples["id"]]


class OutputFormatMapper(Mapper):
    _batched_op = True

    def __init__(self, output_format, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_format = output_format

    def process_batched(self, samples):
        if self.output_format == "generator":
            return ({key: [values[i]] for key, values in samples.items()} for i in range(len(samples["id"])))
        if self.output_format == "dataframe":
            import pandas as pd

            return pd.DataFrame(samples)
        import numpy as np

        samples["tensor"] = np.array([np.full((2, 2), value) for value in samples["id"]])
        return samples


class FailingSingleMapper(Mapper):
    def process_single(self, sample):
        if sample["id"] == 1:
            raise ValueError("bad sample")
        return sample


@pytest.fixture
def local_ray():
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


def _executor(tmp_path, *, resume=False):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = Namespace(auto_op_parallelism=False)
    executor.ckpt_manager = RayCheckpointManager(
        str(tmp_path / "checkpoints"),
        checkpoint_enabled=True,
        checkpoint_strategy=CheckpointStrategy.EVERY_OP,
    )
    executor.job_id = "unit-integration"
    executor.recovery_mode = "streaming"
    executor.unit_size_cfg = 2
    executor._is_resuming = resume
    return executor


def _source(executor, *, changed=False):
    rows = [
        {"text": " A\tB ", "id": 0},
        {"text": "!!!", "id": 1},
        {"text": " C ", "id": 2},
        {"text": "???", "id": 3},
    ]
    if changed:
        rows[0]["text"] = "different input"
    return RayDataset(ray.data.from_items(rows, override_num_blocks=1), cfg=executor.cfg)


def test_streaming_dispatch_uses_units_for_mapper_filter_and_resumes(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1), TextLengthFilter(min_len=1, max_len=2, num_proc=1)]
    assert executor._should_use_streaming_recovery(_source(executor), ops)
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    assert sorted(row["id"] for row in result.data.take_all()) == [2]
    source_directory = tmp_path / "checkpoints" / "unit_recovery" / "source"
    assert [path.name for path in source_directory.iterdir()] == ["manifest.json"]

    database = tmp_path / "checkpoints" / "unit_recovery" / "state" / "units.sqlite3"
    with sqlite3.connect(database) as db:
        committed_before = db.execute(
            "SELECT COUNT(*) FROM stage_state WHERE committed_attempt_id IS NOT NULL"
        ).fetchone()[0]
        attempts_before = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    assert committed_before == 4

    restarted = _executor(tmp_path, resume=True)
    restored = restarted._process_with_simple_partitioning(_source(restarted), ops)
    assert sorted(row["id"] for row in restored.data.take_all()) == [2]
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == attempts_before


def test_streaming_shared_state_hook_uses_partition_path(tmp_path):
    class SharedFilter(TextLengthFilter):
        def _prepare_for_ray_map_batches(self):
            raise AssertionError("Selection must not initialize shared state")

    executor = _executor(tmp_path)
    op = SharedFilter(num_proc=1)
    assert not executor._should_use_streaming_recovery(None, [op])
    with pytest.raises(ProtocolError, match="Shared Ray state"):
        RayDataset(None, auto_op_parallelism=False).process([op], unit_recovery=Namespace(stage_count=1))


def test_streaming_resume_rejects_changed_input(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1)]
    executor._process_with_simple_partitioning(_source(executor), ops)
    restarted = _executor(tmp_path, resume=True)
    with pytest.raises(ProtocolError, match="Input content or read order changed"):
        restarted._process_with_simple_partitioning(_source(restarted, changed=True), ops)


def test_streaming_resume_rejects_missing_input_manifest(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1)]
    executor._process_with_simple_partitioning(_source(executor), ops)
    manifest = tmp_path / "checkpoints" / "unit_recovery" / "source" / "manifest.json"
    manifest.unlink()
    restarted = _executor(tmp_path, resume=True)
    with pytest.raises(ProtocolError, match="Unit snapshot is missing on resume"):
        restarted._process_with_simple_partitioning(_source(restarted), ops)


@pytest.mark.parametrize("explicit_resume", [False, True])
def test_resume_after_crash_during_snapshot(tmp_path, local_ray, explicit_resume):
    recovery = tmp_path / "checkpoints" / "unit_recovery"
    stale = recovery / ".source.crashed.tmp"
    stale.mkdir(parents=True)
    (stale / "unit-000000000000.parquet").write_bytes(b"partial")
    restarted = _executor(tmp_path, resume=True)
    restarted.cfg._resume_requested = explicit_resume
    ops = [WhitespaceNormalizationMapper(num_proc=1), TextLengthFilter(min_len=1, max_len=2, num_proc=1)]
    result = restarted._process_with_simple_partitioning(_source(restarted), ops)
    assert sorted(row["id"] for row in result.data.take_all()) == [2]
    assert not stale.exists()
    assert (recovery / "source" / "manifest.json").is_file()


def test_streaming_export_keeps_expansion_and_input_unit_order(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [DuplicateMapper(num_proc=1, batch_size=1)]
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    assert [row["id"] for row in result.data.take_all()] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_streaming_all_filtered_completes_without_schema(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [TextLengthFilter(min_len=100, num_proc=1), WhitespaceNormalizationMapper(num_proc=1)]
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    assert result.data.take_all() == []
    database = tmp_path / "checkpoints" / "unit_recovery" / "state" / "units.sqlite3"
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE rows=0 AND files_json='[]'").fetchone()[0] == 4


@pytest.mark.parametrize("use_partition_size", [False, True])
@pytest.mark.parametrize("drop_all", [False, True])
def test_full_executor_exports_streaming_unit_result(tmp_path, local_ray, use_partition_size, drop_all):
    input_path = tmp_path / "input.jsonl"
    with input_path.open("w") as stream:
        for text in [" A\tB ", "!!!", " C "]:
            stream.write(json.dumps({"text": text}) + "\n")
    config_path = tmp_path / "job.yaml"
    export_path = tmp_path / "output.jsonl"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_name": "unit-integration",
                "executor_type": "ray_partitioned",
                "dataset_path": str(input_path),
                "export_path": str(export_path),
                "work_dir": str(tmp_path / "work"),
                "checkpoint_dir": str(tmp_path / "work" / "checkpoints"),
                "auto_op_parallelism": False,
                "strict_preflight": False,
                "checkpoint": {"enabled": True, "strategy": "every_op"},
                "partition": {
                    "recovery_mode": "streaming",
                    **({"size": 2} if use_partition_size else {"unit_size": 2}),
                },
                "process": [
                    {"whitespace_normalization_mapper": {"num_proc": 1, "ray_execution_mode": "actor"}},
                    {
                        "text_length_filter": {
                            "min_len": 100 if drop_all else 1,
                            "max_len": 200 if drop_all else 2,
                            "num_proc": 1,
                            "ray_execution_mode": "actor",
                        }
                    },
                ],
            }
        )
    )
    cfg = init_configs(["--config", str(config_path)])
    executor = PartitionedRayExecutor(cfg)
    executor.run(skip_return=True)
    if drop_all:
        assert export_path.is_file()
        assert export_path.read_bytes() == b""
    else:
        shards = list(export_path.glob("*.json"))
        assert shards
        assert [json.loads(line)["text"] for shard in shards for line in shard.read_text().splitlines()] == ["C"]


def test_empty_mapper_output_does_not_impose_input_schema(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [VariableMapper(batch_size=1, num_proc=1), TextLengthFilter(min_len=1, num_proc=1)]
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    rows = result.data.take_all()
    assert [row["id"] for row in rows] == [1, 3]
    assert all(row["mapped"] is True for row in rows)


def test_empty_units_skip_downstream_stats_schema(tmp_path, local_ray):
    executor = _executor(tmp_path)
    executor.unit_size_cfg = 1
    ops = [TextLengthFilter(min_len=4, num_proc=1), AlphanumericFilter(min_ratio=0.1, num_proc=1)]
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    assert [row["id"] for row in result.data.take_all()] == [0]


def test_operator_batches_span_upstream_files_within_each_unit(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [DuplicateMapper(batch_size=1, num_proc=1), BatchSizeMapper(batch_size=3, num_proc=1)]
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    rows = result.data.take_all()
    assert [row["id"] for row in rows] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [row["batch_rows"] for row in rows] == [3, 3, 3, 1, 3, 3, 3, 1]


def test_single_mapper_keeps_arrow_exception_wrapper(tmp_path, local_ray):
    executor = _executor(tmp_path)
    result = executor._process_with_simple_partitioning(
        _source(executor), [FailingSingleMapper(num_proc=1, skip_op_error=True)]
    )
    assert [row["id"] for row in result.data.take_all()] == [0, 2, 3]


@pytest.mark.parametrize("batch_size", [2, "auto", None])
def test_filter_predicate_batching_matches_native_within_unit(tmp_path, local_ray, batch_size):
    executor = _executor(tmp_path)
    executor.unit_size_cfg = 4
    op = MaximumBatchFilter(batch_size=batch_size, num_proc=1, ray_execution_mode="actor")
    expected = _source(executor).process([op]).data.take_all()
    actual = executor._process_with_simple_partitioning(_source(executor), [op]).data.take_all()
    assert actual == expected
    assert [row["id"] for row in actual] == [3]


def test_stats_and_predicate_batch_sizes_are_independent(monkeypatch):
    import pyarrow as pa

    from data_juicer.core.data.ray_batch_plan import BatchPlanWorker, RayBatchPlan

    plans = []

    def capture(plan, dataset, recovery=None):
        plans.append(plan)
        return dataset

    monkeypatch.setattr(RayBatchPlan, "apply", capture)
    RayDataset(None, auto_op_parallelism=False)._run_single_op(
        MaximumBatchFilter(batch_size=3, num_proc=1, ray_execution_mode="actor"), defer_schema_probe=True
    )
    stage = BatchPlanWorker(plans[0])
    tables = [pa.table({"id": list(range(start, min(start + 2, 1004)))}) for start in range(0, 1004, 2)]
    result = [row for table in stage.process_unit(tables) for row in table.to_pylist()]
    assert [row["id"] for row in result] == [999, 1003]
    assert [row[Fields.stats]["compute_batch_rows"] for row in result] == [3, 2]


@pytest.mark.parametrize("output_format", ["generator", "dataframe", "tensor"])
def test_native_output_formats_survive_unit_checkpoint(tmp_path, local_ray, output_format):
    import numpy as np

    executor = _executor(tmp_path)
    op = OutputFormatMapper(output_format, batch_size=2, num_proc=1, ray_execution_mode="actor")
    expected = _source(executor).process([op]).data.take_all()
    actual = executor._process_with_simple_partitioning(_source(executor), [op]).data.take_all()
    if output_format == "tensor":
        for rows in (actual, expected):
            for row in rows:
                row["tensor"] = np.asarray(row["tensor"]).tolist()
    assert sorted(actual, key=lambda row: row["id"]) == sorted(expected, key=lambda row: row["id"])


def test_forced_batched_non_stats_filter_stats_only_keeps_rows(tmp_path, local_ray):
    executor = _executor(tmp_path)
    op = SuffixFilter(batch_mode=True, batch_size=2, num_proc=1, ray_execution_mode="actor", skip_op_error=False)
    result = _source(executor).process([op], stats_only=True).data.take_all()
    assert sorted(row["id"] for row in result) == [0, 1, 2, 3]
    assert all(row[Fields.stats] == {} for row in result)


@pytest.mark.parametrize("case", ["mapper_filter", "expansion", "suffix"])
def test_streaming_and_native_ray_actors_produce_same_records(tmp_path, local_ray, case):
    executor = _executor(tmp_path)
    kwargs = {"num_proc": 1, "ray_execution_mode": "actor"}
    if case == "mapper_filter":
        ops = [WhitespaceNormalizationMapper(**kwargs), TextLengthFilter(min_len=1, max_len=2, **kwargs)]
    elif case == "expansion":
        ops = [DuplicateMapper(batch_size=1, **kwargs), WhitespaceNormalizationMapper(**kwargs)]
    else:
        ops = [SuffixFilter(suffixes=[".txt"], **kwargs)]

    def source():
        result = _source(executor)
        if case == "suffix":
            result.data = result.data.map(lambda row: {**row, Fields.suffix: ".txt" if row["id"] % 2 else ".pdf"})
        return result

    native = source().process(ops).data.take_all()
    recovered = executor._process_with_simple_partitioning(source(), ops).data.take_all()
    assert sorted(native, key=lambda row: row["id"]) == sorted(recovered, key=lambda row: row["id"])
    if case == "suffix":
        assert [row["id"] for row in recovered] == [1, 3]
        assert all(Fields.stats not in row for row in recovered)


class LoggedStatsTextLengthFilter(TextLengthFilter):
    def __init__(self, log_path, *, failure_marker=None, state_dir=None, **kwargs):
        from pathlib import Path

        super().__init__(**kwargs)
        self.log_path = Path(log_path)
        self.failure_marker = Path(failure_marker) if failure_marker is not None else None
        self.state_dir = Path(state_dir) if state_dir is not None else None

    def _record(self, phase, samples):
        with self.log_path.open("a") as stream:
            stream.write(json.dumps({"phase": phase, "ids": list(samples["id"])}) + "\n")

    def compute_stats_batched(self, samples):
        samples = super().compute_stats_batched(samples)
        self._record("compute", samples)
        return samples

    def process_batched(self, samples):
        self._record("predicate", samples)
        if self.failure_marker is not None and 2 in samples["id"] and not self.failure_marker.exists():
            import pyarrow.parquet as pq

            # Fail inside the real predicate, only after this attempt's stats are on disk.
            with sqlite3.connect(self.state_dir / "units.sqlite3") as db:
                attempt_id = db.execute(
                    "SELECT attempt_id FROM attempts WHERE unit_id=? AND stage=0 ORDER BY generation DESC LIMIT 1",
                    ("000000000001",),
                ).fetchone()[0]
                assert (
                    db.execute(
                        "SELECT committed_attempt_id FROM stage_state WHERE unit_id=? AND stage=0",
                        ("000000000000",),
                    ).fetchone()[0]
                    is not None
                )
            stats_paths = sorted((self.state_dir / "attempts" / attempt_id).glob("stats.parquet"))
            persisted = [row for path in stats_paths for row in pq.read_table(path).to_pylist()]
            assert sorted(row["id"] for row in persisted) == [2, 3]
            assert all(row[Fields.stats]["text_len"] == len(row["text"]) for row in persisted)
            with self.failure_marker.open("x") as stream:
                stream.write(attempt_id)
            raise RuntimeError("injected predicate failure after stats persistence")
        return super().process_batched(samples)


def _read_stats_test_json(path):
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    assert paths
    return [json.loads(line) for shard in paths for line in shard.read_text().splitlines() if line.strip()]


def _committed_stats_attempts(state_dir):
    with sqlite3.connect(state_dir / "units.sqlite3") as db:
        db.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in db.execute(
                "SELECT a.* FROM attempts a JOIN stage_state s ON s.committed_attempt_id=a.attempt_id "
                "ORDER BY a.unit_id, a.stage"
            )
        ]


@pytest.mark.parametrize("ray_execution_mode", ["actor", "task"])
@pytest.mark.parametrize("drop_all", [False, True])
def test_streaming_stats_export_matches_native_before_filtering(tmp_path, local_ray, ray_execution_mode, drop_all):
    import pyarrow.parquet as pq

    executor = _executor(tmp_path)
    native_path = tmp_path / "native-stats"
    recovered_path = tmp_path / "recovered-stats"
    kwargs = {
        "min_len": 100 if drop_all else 4,
        "batch_size": 2,
        "num_proc": 1,
        "ray_execution_mode": ray_execution_mode,
        "skip_op_error": False,
    }
    native = _source(executor).process([TextLengthFilter(stats_export_path=str(native_path), **kwargs)]).data.take_all()
    ops = [TextLengthFilter(stats_export_path=str(recovered_path), **kwargs)]
    assert executor._should_use_streaming_recovery(_source(executor), ops)
    recovered = executor._process_with_simple_partitioning(_source(executor), ops).data.take_all()
    expected_ids = [] if drop_all else [0]
    assert sorted(row["id"] for row in recovered) == expected_ids
    assert sorted(recovered, key=lambda row: row["id"]) == sorted(native, key=lambda row: row["id"])

    native_stats = sorted(_read_stats_test_json(native_path), key=lambda row: row["id"])
    recovered_stats = sorted(_read_stats_test_json(recovered_path), key=lambda row: row["id"])
    assert recovered_stats == native_stats
    assert [row["id"] for row in recovered_stats] == [0, 1, 2, 3]
    assert [row[Fields.stats]["text_len"] for row in recovered_stats] == [5, 3, 3, 3]

    state_dir = tmp_path / "checkpoints" / "unit_recovery" / "state"
    commits = _committed_stats_attempts(state_dir)
    assert len(commits) == 2
    committed_stats = []
    committed_business = []
    for ordinal, commit in enumerate(commits):
        assert commit["unit_id"] == f"{ordinal:012d}"
        files = json.loads(commit["files_json"])
        stats_files = json.loads(commit["stats_files_json"])
        assert stats_files
        assert {entry["path"] for entry in files}.isdisjoint(entry["path"] for entry in stats_files)
        for entry in files + stats_files:
            path = state_dir / entry["path"]
            assert path.parent == state_dir / "attempts" / commit["attempt_id"]
            assert pq.read_metadata(path).num_rows == entry["rows"]
        stats_rows = [row for entry in stats_files for row in pq.read_table(state_dir / entry["path"]).to_pylist()]
        business_rows = [row for entry in files for row in pq.read_table(state_dir / entry["path"]).to_pylist()]
        assert sorted(row["id"] for row in stats_rows) == [ordinal * 2, ordinal * 2 + 1]
        assert sum(entry["rows"] for entry in stats_files) == 2
        assert commit["rows"] == sum(entry["rows"] for entry in files) == len(business_rows)
        assert sorted(row["id"] for row in business_rows) == [
            value for value in expected_ids if ordinal * 2 <= value < ordinal * 2 + 2
        ]
        if not business_rows:
            assert files == []
        committed_stats.extend(stats_rows)
        committed_business.extend(business_rows)
    assert sorted(committed_stats, key=lambda row: row["id"]) == recovered_stats
    assert sorted(committed_business, key=lambda row: row["id"]) == sorted(recovered, key=lambda row: row["id"])


def test_streaming_predicate_failure_replays_only_pending_stats_unit(tmp_path, local_ray, monkeypatch):
    import pyarrow.parquet as pq

    # One ticket block and one actor keep the failure after the first unit's commit.
    from_items = ray.data.from_items
    monkeypatch.setattr(
        ray.data, "from_items", lambda items, **kwargs: from_items(items, **{**kwargs, "override_num_blocks": 1})
    )
    state_dir = tmp_path / "checkpoints" / "unit_recovery" / "state"
    log_path = tmp_path / "filter-calls.jsonl"
    marker = tmp_path / "predicate-failed-once"
    stats_path = tmp_path / "stats"
    kwargs = {
        "log_path": str(log_path),
        "failure_marker": str(marker),
        "state_dir": str(state_dir),
        "stats_export_path": str(stats_path),
        "min_len": 4,
        "batch_size": 2,
        "num_proc": 1,
        "ray_execution_mode": "actor",
        "skip_op_error": False,
    }
    executor = _executor(tmp_path)
    with pytest.raises(ray.exceptions.RayTaskError, match="injected predicate failure after stats persistence"):
        executor._process_with_simple_partitioning(_source(executor), [LoggedStatsTextLengthFilter(**kwargs)])

    expected_calls = [{"phase": phase, "ids": ids} for ids in ([0, 1], [2, 3]) for phase in ("compute", "predicate")]
    assert _read_stats_test_json(log_path) == expected_calls
    first_commits = _committed_stats_attempts(state_dir)
    assert len(first_commits) == 1
    assert first_commits[0]["unit_id"] == "000000000000"
    failed_attempt = marker.read_text()
    with sqlite3.connect(state_dir / "units.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
        assert db.execute(
            "SELECT committed_attempt_id FROM stage_state WHERE unit_id=? AND stage=0", ("000000000001",)
        ).fetchone() == (None,)
        assert db.execute(
            "SELECT files_json, stats_files_json, rows FROM attempts WHERE attempt_id=?", (failed_attempt,)
        ).fetchone() == (None, None, None)
    orphan_stats = sorted((state_dir / "attempts" / failed_attempt).glob("stats.parquet"))
    assert sorted(row["id"] for path in orphan_stats for row in pq.read_table(path).to_pylist()) == [2, 3]
    assert not stats_path.exists()

    restarted = _executor(tmp_path, resume=True)
    restored = restarted._process_with_simple_partitioning(_source(restarted), [LoggedStatsTextLengthFilter(**kwargs)])
    assert [row["id"] for row in restored.data.take_all()] == [0]
    assert _read_stats_test_json(log_path) == expected_calls + expected_calls[-2:]
    commits = _committed_stats_attempts(state_dir)
    assert len(commits) == 2
    assert commits[0] == first_commits[0]
    assert commits[1]["attempt_id"] != failed_attempt
    assert commits[1]["generation"] == 2
    assert marker.read_text() == failed_attempt
    with sqlite3.connect(state_dir / "units.sqlite3") as db:
        assert db.execute("SELECT unit_id, COUNT(*) FROM attempts GROUP BY unit_id ORDER BY unit_id").fetchall() == [
            ("000000000000", 1),
            ("000000000001", 2),
        ]
    exported = sorted(_read_stats_test_json(stats_path), key=lambda row: row["id"])
    assert [row["id"] for row in exported] == [0, 1, 2, 3]
    assert [row[Fields.stats]["text_len"] for row in exported] == [5, 3, 3, 3]
    committed_stats_paths = [
        state_dir / entry["path"] for commit in commits for entry in json.loads(commit["stats_files_json"])
    ]
    assert set(committed_stats_paths).isdisjoint(orphan_stats)
    assert all(path.is_file() for path in orphan_stats)
    assert (
        sorted(
            [row for path in committed_stats_paths for row in pq.read_table(path).to_pylist()],
            key=lambda row: row["id"],
        )
        == exported
    )


def test_streaming_stats_publish_failure_resumes_without_recomputing_or_duplicate_json(
    tmp_path, local_ray, monkeypatch
):
    from data_juicer.core.data.ray_batch_plan import StatsExport

    log_path = tmp_path / "filter-calls.jsonl"
    stats_path = tmp_path / "stats"
    kwargs = {
        "log_path": str(log_path),
        "stats_export_path": str(stats_path),
        "min_len": 4,
        "batch_size": 2,
        "num_proc": 1,
        "ray_execution_mode": "actor",
        "skip_op_error": False,
    }
    stats_path.mkdir()
    unrelated = stats_path / "unrelated.bin"
    unrelated.write_bytes(b"preserve existing destination content")
    publish = StatsExport.publish
    publish_calls = []

    def publish_then_fail_once(export, paths):
        publish_calls.append(tuple(paths))
        publish(export, paths)
        if len(publish_calls) == 1:
            raise RuntimeError("injected failure after stats publication")

    monkeypatch.setattr(StatsExport, "publish", publish_then_fail_once)
    executor = _executor(tmp_path)
    with pytest.raises(RuntimeError, match="injected failure after stats publication"):
        executor._process_with_simple_partitioning(_source(executor), [LoggedStatsTextLengthFilter(**kwargs)])

    state_dir = tmp_path / "checkpoints" / "unit_recovery" / "state"
    commits_before = _committed_stats_attempts(state_dir)
    assert len(commits_before) == 2
    with sqlite3.connect(state_dir / "units.sqlite3") as db:
        attempts_before = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    assert attempts_before == 2
    for phase in ("compute", "predicate"):
        assert sorted(
            value for call in _read_stats_test_json(log_path) if call["phase"] == phase for value in call["ids"]
        ) == [0, 1, 2, 3]
    calls_before = log_path.read_bytes()
    exported_before = sorted(_read_stats_test_json(stats_path), key=lambda row: row["id"])
    assert [row["id"] for row in exported_before] == [0, 1, 2, 3]
    assert [row[Fields.stats]["text_len"] for row in exported_before] == [5, 3, 3, 3]
    assert publish_calls == [
        tuple(
            str(state_dir / entry["path"])
            for commit in commits_before
            for entry in json.loads(commit["stats_files_json"])
        )
    ]

    restarted = _executor(tmp_path, resume=True)
    restored = restarted._process_with_simple_partitioning(_source(restarted), [LoggedStatsTextLengthFilter(**kwargs)])
    assert [row["id"] for row in restored.data.take_all()] == [0]
    assert _committed_stats_attempts(state_dir) == commits_before
    with sqlite3.connect(state_dir / "units.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == attempts_before
    assert log_path.read_bytes() == calls_before
    assert len(publish_calls) == 2
    assert publish_calls[1] == publish_calls[0]
    exported_after = sorted(_read_stats_test_json(stats_path), key=lambda row: row["id"])
    assert exported_after == exported_before
    assert [row["id"] for row in exported_after] == [0, 1, 2, 3]
    assert unrelated.read_bytes() == b"preserve existing destination content"


def test_empty_upstream_stats_export_commits_without_schema(tmp_path, local_ray):
    stats_path = tmp_path / "stats"
    executor = _executor(tmp_path)
    ops = [
        TextLengthFilter(min_len=100, num_proc=1, ray_execution_mode="actor"),
        TextLengthFilter(stats_export_path=str(stats_path), num_proc=1, ray_execution_mode="actor"),
    ]
    assert executor._process_with_simple_partitioning(_source(executor), ops).data.take_all() == []
    state_dir = tmp_path / "checkpoints" / "unit_recovery" / "state"
    commits = _committed_stats_attempts(state_dir)
    assert len(commits) == 4
    assert all(commit["rows"] == 0 and commit["stats_files_json"] == "[]" for commit in commits)
    assert not list(stats_path.glob("*.json"))
    restarted = _executor(tmp_path, resume=True)
    assert restarted._process_with_simple_partitioning(_source(restarted), ops).data.take_all() == []
    assert _committed_stats_attempts(state_dir) == commits


def test_partial_stats_publication_rebuilds_only_owned_files(tmp_path, local_ray, monkeypatch):
    stats_path = tmp_path / "stats"
    stats_path.mkdir()
    unrelated = stats_path / "other-export.json"
    unrelated.write_text('{"id":99}\n')
    log_path = tmp_path / "calls.jsonl"
    kwargs = dict(
        log_path=str(log_path),
        stats_export_path=stats_path.as_uri(),
        min_len=4,
        batch_size=2,
        num_proc=1,
        ray_execution_mode="actor",
        skip_op_error=False,
    )
    write_json = ray.data.Dataset.write_json
    writes = []

    def partial_write_once(data, path, **options):
        writes.append(path)
        if len(writes) == 1:
            partial = data.limit(1)
            partial._set_uuid(data._uuid)
            write_json(partial, path, **options)
            (stats_path / f"{data._uuid}stale.json").write_text('{"id":-1}\n')
            raise RuntimeError("injected partial statistics publication")
        return write_json(data, path, **options)

    monkeypatch.setattr(ray.data.Dataset, "write_json", partial_write_once)
    executor = _executor(tmp_path)
    with pytest.raises(RuntimeError, match="partial statistics publication"):
        executor._process_with_simple_partitioning(_source(executor), [LoggedStatsTextLengthFilter(**kwargs)])
    assert len(_read_stats_test_json(stats_path)) == 3
    calls_before = log_path.read_bytes()
    state_dir = tmp_path / "checkpoints" / "unit_recovery" / "state"
    commits = _committed_stats_attempts(state_dir)
    restarted = _executor(tmp_path, resume=True)
    result = restarted._process_with_simple_partitioning(_source(restarted), [LoggedStatsTextLengthFilter(**kwargs)])
    assert [row["id"] for row in result.data.take_all()] == [0]
    assert sorted(row["id"] for row in _read_stats_test_json(stats_path)) == [0, 1, 2, 3, 99]
    assert unrelated.read_text() == '{"id":99}\n'
    assert log_path.read_bytes() == calls_before
    assert _committed_stats_attempts(state_dir) == commits


def test_multiple_stats_stages_share_directory_without_erasing_each_other(tmp_path, local_ray):
    from collections import Counter

    stats_path = tmp_path / "stats"
    kwargs = dict(stats_export_path=str(stats_path), num_proc=1, ray_execution_mode="actor")
    ops = [TextLengthFilter(min_len=4, **kwargs), AlphanumericFilter(min_ratio=0.1, **kwargs)]
    executor = _executor(tmp_path)
    assert [
        row["id"] for row in executor._process_with_simple_partitioning(_source(executor), ops).data.take_all()
    ] == [0]
    expected = Counter({0: 2, 1: 1, 2: 1, 3: 1})
    assert Counter(row["id"] for row in _read_stats_test_json(stats_path)) == expected
    restarted = _executor(tmp_path, resume=True)
    restarted._process_with_simple_partitioning(_source(restarted), ops)
    assert Counter(row["id"] for row in _read_stats_test_json(stats_path)) == expected
