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
    from data_juicer.core.executor.operator_unit_adapter import operator_unit_stage

    class SharedFilter(TextLengthFilter):
        def _prepare_for_ray_map_batches(self):
            raise AssertionError("Selection must not initialize shared state")

    executor = _executor(tmp_path)
    op = SharedFilter(num_proc=1)
    assert not executor._should_use_streaming_recovery(None, [op])
    with pytest.raises(ProtocolError, match="shared Ray state"):
        operator_unit_stage(op)


def test_streaming_resume_rejects_changed_input(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1)]
    executor._process_with_simple_partitioning(_source(executor), ops)
    restarted = _executor(tmp_path, resume=True)
    with pytest.raises(ProtocolError, match="Input content or read order changed"):
        restarted._process_with_simple_partitioning(_source(restarted, changed=True), ops)


def test_streaming_resume_rejects_missing_snapshot_file(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1)]
    executor._process_with_simple_partitioning(_source(executor), ops)
    snapshot_file = tmp_path / "checkpoints" / "unit_recovery" / "source" / "unit-000000000000.parquet"
    snapshot_file.unlink()
    restarted = _executor(tmp_path, resume=True)
    with pytest.raises(ProtocolError, match="snapshot file is missing or modified"):
        restarted._process_with_simple_partitioning(_source(restarted), ops)


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


def test_filter_predicate_batching_matches_native_within_unit(tmp_path, local_ray):
    executor = _executor(tmp_path)
    executor.unit_size_cfg = 4
    op = MaximumBatchFilter(batch_size=2, num_proc=1, ray_execution_mode="actor")
    expected = _source(executor).process([op]).data.take_all()
    actual = executor._process_with_simple_partitioning(_source(executor), [op]).data.take_all()
    assert actual == expected
    assert [row["id"] for row in actual] == [3]


def test_stats_and_predicate_batch_sizes_are_independent():
    import pyarrow as pa

    from data_juicer.core.executor.operator_unit_adapter import operator_unit_stage

    stage = operator_unit_stage(MaximumBatchFilter(batch_size=3))
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
