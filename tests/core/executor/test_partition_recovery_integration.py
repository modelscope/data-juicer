"""End-to-end checks of side-path checkpointing on a local Ray cluster."""

import json
import os
from collections import Counter
from pathlib import Path

import pytest
import ray
import yaml
from jsonargparse import Namespace

from data_juicer.config import init_configs
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor import ray_partition_runner
from data_juicer.core.executor.partition_protocol import COMMIT_DIR, Commit, LocalStore, ProtocolError
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops.base_op import Mapper
from data_juicer.ops.deduplicator.document_deduplicator import DocumentDeduplicator
from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper
from data_juicer.utils.ckpt_utils import RayCheckpointManager
from data_juicer.utils.constant import Fields


class DuplicateMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        return {key: [value for item in values for value in (item, item)] for key, values in samples.items()}


class OddMapper(Mapper):
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


class LoggingMapper(Mapper):
    """Record processed ids; optionally fail once when it sees ``fail_id``."""

    _batched_op = True

    def __init__(self, log_path, fail_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.fail_id = fail_id

    def process_batched(self, samples):
        marker = Path(self.log_path + ".failed")
        if self.fail_id in samples["id"] and not marker.exists():
            marker.write_text("x")
            raise RuntimeError("injected mapper failure")
        with open(self.log_path, "a") as stream:
            stream.write(json.dumps(samples["id"]) + "\n")
        return samples


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


class ExitAfterFirstCommitActor(ray_partition_runner.CheckpointStageActor):
    """Kill the worker after the first committed batch has been yielded."""

    def __call__(self, batch):
        for table in super().__call__(batch):
            yield table
            marker = Path(self.context.store.root, "exited-once")
            try:
                with open(marker, "x"):
                    pass
            except FileExistsError:
                continue
            os._exit(86)


def exit_after_first_commit_task(batch, context, plan):
    for table in ray_partition_runner.run_checkpoint_task(batch, context, plan):
        yield table
        try:
            with open(Path(context.store.root, "exited-once"), "x"):
                pass
        except FileExistsError:
            continue
        os._exit(86)


@pytest.fixture
def local_ray():
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


def _executor(tmp_path, *, resume=False, size=2):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = Namespace(auto_op_parallelism=False)
    executor.ckpt_manager = RayCheckpointManager(str(tmp_path / "checkpoints"), checkpoint_enabled=True)
    executor.job_id = "checkpoint-integration"
    executor.partition_size_cfg = size
    executor._is_resuming = resume
    return executor


def _source(executor, *, changed=False, blocks=1):
    rows = [
        {"text": " A\tB ", "id": 0},
        {"text": "!!!", "id": 1},
        {"text": " C ", "id": 2},
        {"text": "???", "id": 3},
    ]
    if changed:
        rows[0]["text"] = "different input"
    return RayDataset(ray.data.from_items(rows, override_num_blocks=blocks), cfg=executor.cfg)


def _run(executor, ops, source=None):
    dataset, recovery = executor._process_with_partition_recovery(source or _source(executor), ops)
    try:
        rows = dataset.data.take_all()
        recovery.finish()
    finally:
        recovery.close()
    return rows


def _store(tmp_path):
    return LocalStore(str(tmp_path / "checkpoints" / "partition_recovery"))


def _commits(tmp_path):
    store = _store(tmp_path)
    return sorted((Commit.from_bytes(store.get(key)) for key in store.list(COMMIT_DIR)), key=lambda c: c.seq)


def _logged_ids(path):
    if not os.path.exists(path):
        return []
    return [value for line in Path(path).read_text().splitlines() for value in json.loads(line)]


def _by_id(rows):
    return sorted(rows, key=lambda row: row["id"])


@pytest.mark.parametrize("mode", ["actor", "task"])
def test_checkpointed_rows_match_native_and_resume_reuses_commits(tmp_path, local_ray, mode):
    log_path = str(tmp_path / "calls.jsonl")
    kwargs = {"num_proc": 1, "ray_execution_mode": mode}
    ops = [
        LoggingMapper(log_path, batch_size=1, **kwargs),
        WhitespaceNormalizationMapper(**kwargs),
        TextLengthFilter(min_len=1, max_len=2, **kwargs),
    ]
    native = _source(_executor(tmp_path)).process(ops).data.take_all()
    Path(log_path).unlink()

    rows = _run(_executor(tmp_path), ops)
    assert _by_id(rows) == _by_id(native)
    assert [row["id"] for row in rows] == [2]
    commits = _commits(tmp_path)
    assert {commit.stage for commit in commits} == {0, 1, 2}
    assert all(sum(stop - start for _, start, stop in commit.consumed) <= 2 for commit in commits)
    assert sorted(_logged_ids(log_path)) == [0, 1, 2, 3]

    assert _run(_executor(tmp_path, resume=True), ops) == rows
    assert _commits(tmp_path) == commits
    assert sorted(_logged_ids(log_path)) == [0, 1, 2, 3]


def test_failed_run_resumes_only_unconsumed_rows(tmp_path, local_ray):
    log_path = str(tmp_path / "calls.jsonl")
    kwargs = {"num_proc": 1, "ray_execution_mode": "actor"}
    ops = [
        LoggingMapper(log_path, batch_size=1, **kwargs),
        LoggingMapper(log_path + ".second", fail_id=3, batch_size=1, **kwargs),
    ]
    with pytest.raises(Exception, match="injected mapper failure"):
        _run(_executor(tmp_path), ops)
    committed_first = sorted(
        index
        for commit in _commits(tmp_path)
        if commit.stage == 0
        for _, start, stop in commit.consumed
        for index in range(start, stop)
    )
    first_calls = _logged_ids(log_path)
    assert committed_first and sorted(first_calls) == sorted(set(first_calls))

    rows = _run(_executor(tmp_path, resume=True), ops)
    assert sorted(row["id"] for row in rows) == [0, 1, 2, 3]
    resumed_calls = _logged_ids(log_path)[len(first_calls) :]
    assert sorted(resumed_calls) == sorted(set(range(4)) - set(committed_first))
    # The failed commit batch reran as a whole; committed batches did not.
    second = Counter(_logged_ids(log_path + ".second"))
    assert set(second) == {0, 1, 2, 3} and sum(second.values()) <= 4 + 2


@pytest.mark.parametrize("mode", ["actor", "task"])
def test_worker_death_after_yield_replays_the_committed_batch(tmp_path, local_ray, monkeypatch, mode):
    if mode == "actor":
        monkeypatch.setattr(ray_partition_runner, "CheckpointStageActor", ExitAfterFirstCommitActor)
    else:
        monkeypatch.setattr(ray_partition_runner, "run_checkpoint_task", exit_after_first_commit_task)
    kwargs = {"num_proc": 1, "ray_execution_mode": mode}
    log_path = str(tmp_path / "calls.jsonl")
    ops = [
        LoggingMapper(log_path, batch_size=1, **kwargs),
        DuplicateMapper(batch_size=1, **kwargs),
        WhitespaceNormalizationMapper(**kwargs),
    ]
    rows = _run(_executor(tmp_path), ops)
    assert (tmp_path / "checkpoints" / "partition_recovery" / "exited-once").exists()
    assert sorted(row["id"] for row in rows) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert sorted(_logged_ids(log_path)) == [0, 1, 2, 3]


@pytest.mark.parametrize("block_rows,max_rows,expected", [(2, 8, [2, 2, 2, 2]), (8, 3, [3, 3, 2])])
def test_commit_size_does_not_merge_blocks(tmp_path, local_ray, block_rows, max_rows, expected):
    import pyarrow as pa

    from data_juicer.core.data.ray_batch_plan import RayBatchPlan, map_whole_blocks
    from data_juicer.core.executor.partition_protocol import SOURCE_SEQ, with_keys

    store = LocalStore(str(tmp_path))
    coordinator = ray_partition_runner._CommitCoordinator.remote(store, 1, 8)
    context = ray_partition_runner.StageContext(store, 0, coordinator, max_rows)
    plan = RayBatchPlan().map_batches(lambda batch: batch, batch_format="pyarrow")
    table = with_keys(pa.table({"id": list(range(8))}), SOURCE_SEQ)
    source = ray.data.from_arrow([table.slice(start, block_rows) for start in range(0, 8, block_rows)])
    try:
        result = map_whole_blocks(
            source,
            ray_partition_runner.run_checkpoint_task,
            fn_args=(context, plan),
            batch_format="pyarrow",
        )
        assert sorted(row["id"] for row in result.take_all()) == list(range(8))
        commits = [Commit.from_bytes(store.get(key)) for key in store.list(COMMIT_DIR)]
        assert sorted(commit.rows for commit in commits) == sorted(expected)
    finally:
        ray.kill(coordinator)


def test_gpu_whole_block_plan_has_no_bundling_threshold(local_ray):
    from data_juicer.core.data.ray_batch_plan import map_whole_blocks

    result = map_whole_blocks(ray.data.range(4), lambda batch: batch, num_gpus=1)
    assert result._logical_plan.dag.min_rows_per_bundled_input is None


def test_expansion_shrinking_and_batches_within_commits(tmp_path, local_ray):
    ops = [DuplicateMapper(batch_size=1, num_proc=1), BatchSizeMapper(batch_size=3, num_proc=1)]
    rows = _run(_executor(tmp_path, size=4), ops)
    assert sorted(row["id"] for row in rows) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert max(row["batch_rows"] for row in rows) == 3

    shrinking = tmp_path / "shrinking"
    rows = _run(_executor(shrinking), [OddMapper(batch_size=1, num_proc=1), TextLengthFilter(min_len=1, num_proc=1)])
    assert sorted(row["id"] for row in rows) == [1, 3]
    assert all(row["mapped"] is True for row in rows)


def test_all_filtered_run_completes_and_resumes(tmp_path, local_ray):
    ops = [TextLengthFilter(min_len=100, num_proc=1), WhitespaceNormalizationMapper(num_proc=1)]
    assert _run(_executor(tmp_path), ops) == []
    commits = _commits(tmp_path)
    assert {commit.stage for commit in commits} == {0} and all(commit.rows == 0 for commit in commits)
    assert _run(_executor(tmp_path, resume=True), ops) == []
    assert _commits(tmp_path) == commits


@pytest.mark.parametrize("output_format", ["generator", "dataframe", "tensor"])
def test_native_output_formats_survive_checkpoint(tmp_path, local_ray, output_format):
    import numpy as np

    op = OutputFormatMapper(output_format, batch_size=2, num_proc=1, ray_execution_mode="actor")
    expected = _source(_executor(tmp_path)).process([op]).data.take_all()
    actual = _run(_executor(tmp_path), [op])
    if output_format == "tensor":
        for rows in (actual, expected):
            for row in rows:
                row["tensor"] = np.asarray(row["tensor"]).tolist()
    assert _by_id(actual) == _by_id(expected)


def _read_json_rows(path):
    return [json.loads(line) for shard in sorted(Path(path).glob("*.json")) for line in shard.read_text().splitlines()]


@pytest.mark.parametrize("mode", ["actor", "task"])
def test_stats_export_matches_native_and_republishes_on_resume(tmp_path, local_ray, mode):
    native_path, checkpoint_path = tmp_path / "native-stats", tmp_path / "ckpt-stats"
    kwargs = {"min_len": 4, "batch_size": 2, "num_proc": 1, "ray_execution_mode": mode}
    native = _source(_executor(tmp_path)).process([TextLengthFilter(stats_export_path=str(native_path), **kwargs)])
    native_rows = native.data.take_all()
    ops = [TextLengthFilter(stats_export_path=str(checkpoint_path), **kwargs)]
    assert _by_id(_run(_executor(tmp_path), ops)) == _by_id(native_rows)
    exported = _by_id(_read_json_rows(checkpoint_path))
    assert exported == _by_id(_read_json_rows(native_path))
    assert [row[Fields.stats]["text_len"] for row in exported] == [5, 3, 3, 3]

    _run(_executor(tmp_path, resume=True), ops)
    assert _by_id(_read_json_rows(checkpoint_path)) == exported


def test_stats_publish_failure_resumes_without_recompute(tmp_path, local_ray, monkeypatch):
    from data_juicer.core.data.ray_batch_plan import StatsExport

    log_path = str(tmp_path / "calls.jsonl")
    stats_path = tmp_path / "stats"
    stats_path.mkdir()
    (stats_path / "unrelated.bin").write_bytes(b"keep")
    publish = StatsExport.publish
    calls = []

    def publish_then_fail_once(export, data, identity):
        calls.append(tuple(identity))
        publish(export, data, identity)
        if len(calls) == 1:
            raise RuntimeError("injected failure after stats publication")

    monkeypatch.setattr(StatsExport, "publish", publish_then_fail_once)
    ops = [
        LoggingMapper(log_path, batch_size=2, num_proc=1),
        TextLengthFilter(stats_export_path=str(stats_path), min_len=4, num_proc=1),
    ]
    with pytest.raises(RuntimeError, match="injected failure after stats publication"):
        _run(_executor(tmp_path), ops)
    before = _by_id(_read_json_rows(stats_path))
    assert [row["id"] for row in before] == [0, 1, 2, 3]
    calls_before = _logged_ids(log_path)

    assert [row["id"] for row in _run(_executor(tmp_path, resume=True), ops)] == [0]
    assert _logged_ids(log_path) == calls_before
    assert calls[1] == calls[0]
    assert _by_id(_read_json_rows(stats_path)) == before
    assert (stats_path / "unrelated.bin").read_bytes() == b"keep"


def test_multiple_stats_stages_share_directory(tmp_path, local_ray):
    stats_path = tmp_path / "stats"
    kwargs = dict(stats_export_path=str(stats_path), num_proc=1, ray_execution_mode="actor")
    ops = [TextLengthFilter(min_len=4, **kwargs), AlphanumericFilter(min_ratio=0.1, **kwargs)]
    assert [row["id"] for row in _run(_executor(tmp_path), ops)] == [0]
    expected = Counter({0: 2, 1: 1, 2: 1, 3: 1})
    assert Counter(row["id"] for row in _read_json_rows(stats_path)) == expected
    _run(_executor(tmp_path, resume=True), ops)
    assert Counter(row["id"] for row in _read_json_rows(stats_path)) == expected


def test_resume_rejects_changed_input_and_fresh_run_rejects_existing_checkpoint(tmp_path, local_ray):
    ops = [WhitespaceNormalizationMapper(num_proc=1)]
    _run(_executor(tmp_path), ops)
    resumed = _executor(tmp_path, resume=True)
    with pytest.raises(ProtocolError, match="Input content, read order, or operator config changed"):
        _run(resumed, ops, _source(resumed, changed=True))
    with pytest.raises(ProtocolError, match="already exists"):
        _run(_executor(tmp_path), ops)
    other_ops = [WhitespaceNormalizationMapper(num_proc=1), TextLengthFilter(min_len=1, num_proc=1)]
    with pytest.raises(ProtocolError, match="changed since the checkpoint"):
        _run(_executor(tmp_path, resume=True), other_ops)


def test_resume_accepts_a_different_commit_size(tmp_path, local_ray):
    log_path = str(tmp_path / "calls.jsonl")
    ops = [LoggingMapper(log_path, fail_id=2, batch_size=1, num_proc=1)]
    with pytest.raises(Exception, match="injected mapper failure"):
        _run(_executor(tmp_path, size=1), ops)
    rows = _run(_executor(tmp_path, resume=True, size=3), ops)
    assert sorted(row["id"] for row in rows) == [0, 1, 2, 3]
    assert sorted(_logged_ids(log_path)) == [0, 1, 2, 3]


def test_global_operators_end_the_checkpointed_prefix(tmp_path):
    class SharedFilter(TextLengthFilter):
        def _prepare_for_ray_map_batches(self):
            return True

    executor = _executor(tmp_path)
    mapper = WhitespaceNormalizationMapper(num_proc=1)
    assert executor._checkpoint_prefix_length([mapper, DocumentDeduplicator(), mapper]) == 1
    assert executor._checkpoint_prefix_length([mapper, SharedFilter(num_proc=1)]) == 1
    assert executor._checkpoint_prefix_length([mapper, TextLengthFilter(num_proc=1)]) == 2
    executor.ckpt_manager.checkpoint_enabled = False
    assert executor._checkpoint_prefix_length([mapper]) == 0
    with pytest.raises(ProtocolError, match="Shared Ray state"):
        RayDataset(None, auto_op_parallelism=False).process(
            [SharedFilter(num_proc=1)], partition_recovery=Namespace(stage_count=1)
        )


def _write_job(tmp_path, process, *, extra=None):
    input_path = tmp_path / "input.jsonl"
    with input_path.open("w") as stream:
        for text in [" A\tB ", "!!!", " C ", " C "]:
            stream.write(json.dumps({"text": text}) + "\n")
    config = {
        "project_name": "checkpoint-integration",
        "executor_type": "ray_partitioned",
        "dataset_path": str(input_path),
        "export_path": str(tmp_path / "output.jsonl"),
        "work_dir": str(tmp_path / "work"),
        "checkpoint_dir": str(tmp_path / "work" / "checkpoints"),
        "auto_op_parallelism": False,
        "strict_preflight": False,
        "checkpoint": {"enabled": True, "strategy": "every_op"},
        "partition": {"size": 2},
        "process": process,
        **(extra or {}),
    }
    config_path = tmp_path / "job.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _exported_texts(tmp_path):
    export_path = tmp_path / "output.jsonl"
    shards = sorted(export_path.glob("*.json")) if export_path.is_dir() else [export_path]
    return sorted(json.loads(line)["text"] for shard in shards for line in shard.read_text().splitlines())


@pytest.mark.parametrize("drop_all", [False, True])
def test_full_executor_exports_checkpointed_result(tmp_path, local_ray, drop_all):
    config_path = _write_job(
        tmp_path,
        [
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
    )
    PartitionedRayExecutor(init_configs(["--config", str(config_path)])).run(skip_return=True)
    assert _exported_texts(tmp_path) == ([] if drop_all else ["C", "C"])


def test_full_executor_checkpoints_the_prefix_before_a_deduplicator(tmp_path, local_ray):
    config_path = _write_job(
        tmp_path,
        [
            {"whitespace_normalization_mapper": {"num_proc": 1}},
            {"ray_document_deduplicator": {}},
            {"text_length_filter": {"min_len": 1, "max_len": 3, "num_proc": 1}},
        ],
    )
    PartitionedRayExecutor(init_configs(["--config", str(config_path)])).run(skip_return=True)
    assert _exported_texts(tmp_path) == ["!!!", "A B", "C"]
    store = LocalStore(str(tmp_path / "work" / "checkpoints" / "partition_recovery"))
    stages = {Commit.from_bytes(store.get(key)).stage for key in store.list(COMMIT_DIR)}
    assert stages == {0}


def test_unsupported_checkpoint_scheme_fails_before_running(tmp_path, local_ray):
    config_path = _write_job(
        tmp_path,
        [{"whitespace_normalization_mapper": {"num_proc": 1}}],
        extra={"checkpoint_dir": "gs://bucket/checkpoints"},
    )
    with pytest.raises(ValueError, match="not supported"):
        PartitionedRayExecutor(init_configs(["--config", str(config_path)]))


def test_removed_checkpoint_strategies_are_rejected(tmp_path):
    config_path = _write_job(tmp_path, [{"whitespace_normalization_mapper": {}}])
    with pytest.raises(BaseException):
        init_configs(["--config", str(config_path), "--checkpoint.strategy", "every_n_ops"])


def test_parquet_round_trip_keeps_nullable_schema_changes(tmp_path, local_ray):
    class NullableMapper(Mapper):
        _batched_op = True

        def process_batched(self, samples):
            samples["extra"] = [str(value) if value == 0 else None for value in samples["id"]]
            return samples

    op = NullableMapper(batch_size=1, num_proc=1)
    rows = _run(_executor(tmp_path, size=4), [op])
    assert _by_id(rows)[0]["extra"] == "0"
    assert _run(_executor(tmp_path, resume=True, size=4), [op]) == rows
