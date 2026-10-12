"""Crash and output-contract tests for the local recovery-unit protocol."""

import json
import multiprocessing
import os
import sqlite3
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_juicer.core.executor import unit_protocol
from data_juicer.core.executor.unit_protocol import SCHEMA_VERSION, LocalUnitStore, PreparedOutput, ProtocolError
from data_juicer.core.executor.unit_runner import collect_final_units, execute_unit_attempt, run_local_units


def test_old_checkpoint_database_is_rejected_before_running(tmp_path):
    with sqlite3.connect(tmp_path / "units.sqlite3") as db:
        db.execute("CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, files_json TEXT, rows INTEGER)")
    with pytest.raises(ProtocolError, match="incompatible protocol version; expected version 5"):
        LocalUnitStore(str(tmp_path), "job-1")


@pytest.mark.parametrize(
    "version, schema_column", [(1, False), (2, True), (2, False), (3, True), (3, False), (4, True), (4, False)]
)
def test_incompatible_checkpoint_is_rejected_without_migration(tmp_path, version, schema_column):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    store.commit(attempt, store.write_output(attempt, []))
    with sqlite3.connect(store.db_path) as db:
        assert SCHEMA_VERSION == db.execute("SELECT schema_version FROM runs").fetchone()[0] == 5
        columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
        assert "schema_hex" not in columns
        assert "stats_files_json" in columns
        db.execute("UPDATE runs SET schema_version=?", (version,))
        if schema_column:
            db.execute("ALTER TABLE attempts ADD COLUMN schema_hex TEXT")
    with pytest.raises(ProtocolError, match="incompatible protocol version; expected version 5"):
        LocalUnitStore(str(tmp_path), "job-1")
    with pytest.raises(ProtocolError, match="incompatible protocol version; expected version 5"):
        LocalUnitStore(str(tmp_path), "another-job")
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT schema_version FROM runs").fetchone()[0] == version
        assert ("schema_hex" in {row[1] for row in db.execute("PRAGMA table_info(attempts)")}) == schema_column


@pytest.mark.parametrize("version", [3, 4])
def test_v3_attempts_schema_is_rejected_without_migration(tmp_path, version):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    store.commit(attempt, store.write_output(attempt, []))
    with sqlite3.connect(store.db_path) as db:
        db.execute("ALTER TABLE attempts DROP COLUMN stats_files_json")
        db.execute("UPDATE runs SET schema_version=?", (version,))
        before = tuple(db.iterdump())
    # A v3 layout is incompatible even if its version marker claims v4.
    for run_id in ("job-1", "another-job"):
        with pytest.raises(ProtocolError, match="incompatible protocol version; expected version 5"):
            LocalUnitStore(str(tmp_path), run_id)
    with sqlite3.connect(store.db_path) as db:
        assert tuple(db.iterdump()) == before


def _store(tmp_path, *, rows=5, size=2, stages=2):
    store = LocalUnitStore(str(tmp_path), "job-1")
    units = store.plan(
        input_fingerprint="ordered-input-hash",
        config_fingerprint="operator-chain-hash",
        input_rows=rows,
        unit_size=size,
        stage_count=stages,
    )
    return store, units


def test_plan_is_stable_and_changed_provenance_fails_closed(tmp_path):
    store, units = _store(tmp_path)
    assert [(u.unit_id, u.start, u.stop) for u in units] == [
        ("000000000000", 0, 2),
        ("000000000001", 2, 4),
        ("000000000002", 4, 5),
    ]
    assert _store(tmp_path)[1] == units
    with pytest.raises(ProtocolError, match="differs"):
        store.plan(
            input_fingerprint="changed",
            config_fingerprint="operator-chain-hash",
            input_rows=5,
            unit_size=2,
            stage_count=2,
        )
    with pytest.raises(ProtocolError, match="differs"):
        store.plan(
            input_fingerprint="ordered-input-hash",
            config_fingerprint="changed",
            input_rows=5,
            unit_size=2,
            stage_count=2,
        )


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("empty_tables", [[], [pa.table({})], [pa.table({"value": []}), pa.table({"other": []})]])
def test_zero_output_and_many_outputs_are_complete_commits(tmp_path, empty_tables, stats):
    store, (unit,) = _store(tmp_path, rows=2, size=2)
    first = store.begin(unit.unit_id, 0)
    first_output = store.write_output(first, empty_tables, stats=stats)
    assert first_output == PreparedOutput(())
    assert list((store.root / "attempts" / first.attempt_id).iterdir()) == []
    zero_commit = store.commit(first, first_output)
    assert zero_commit.rows == 0
    assert store.pending(0) == ()
    store, _ = _store(tmp_path, rows=2, size=2)
    assert store.get_commit(unit.unit_id, 0) == zero_commit

    second = store.begin(unit.unit_id, 1)
    assert second.input_commit_id == first.attempt_id
    # The unit emits more rows than it received; both batches land in one file.
    files = store.write_output(
        second,
        [
            pa.table({"copy": [0, 1]}),
            pa.table({"copy": [2, 3, 4]}),
        ],
    )
    assert store.commit(second, files).rows == 5
    reopened, _ = _store(tmp_path, rows=2, size=2)
    restored = reopened.get_commit(unit.unit_id, 1)
    assert restored is not None and restored.rows == 5
    assert len(restored.files) == 1
    assert restored.stats_files == ()
    assert reopened.pending(1) == ()


@pytest.mark.parametrize("empty_output", [False, True])
@pytest.mark.parametrize("different_stats", [False, True])
def test_business_and_stats_outputs_share_one_commit(tmp_path, empty_output, different_stats):
    store, (unit,) = _store(tmp_path, rows=3, size=3)
    attempt = store.begin(unit.unit_id, 0)
    last_stats = pa.table({"other_stat": [True]}) if different_stats else pa.table({"score": [0.5]})
    stats = store.write_output(
        attempt,
        [pa.table({"score": [0.1, 0.9]}), pa.table({"score": []}), last_stats],
        stats=True,
    )
    tables = [] if empty_output else [pa.table({"value": [1]}), pa.table({"value": [2]})]
    business = store.write_output(attempt, tables)
    assert stats.files == ()
    assert business.stats_files == ()
    expected_stats = [f"attempts/{attempt.attempt_id}/stats.parquet"]
    if different_stats:
        expected_stats.append(f"attempts/{attempt.attempt_id}/stats-00000001.parquet")
    assert [item.path for item in stats.stats_files] == expected_stats
    expected_parts = [] if empty_output else [f"attempts/{attempt.attempt_id}/part.parquet"]
    assert [item.path for item in business.files] == expected_parts
    output = PreparedOutput(business.files, stats.stats_files)
    assert store.get_commit(unit.unit_id, 0) is None
    assert store.pending(0) == (unit,)
    assert store.pending(1) == ()
    committed = store.commit(attempt, output)
    assert committed.files == business.files
    assert committed.stats_files == stats.stats_files
    assert committed.rows == (0 if empty_output else 2)
    assert sum(item.rows for item in committed.stats_files) == 3
    with sqlite3.connect(store.db_path) as db:
        row = db.execute("SELECT files_json, stats_files_json, rows FROM attempts").fetchone()
        assert json.loads(row[0]) == [item.__dict__ for item in business.files]
        assert json.loads(row[1]) == [item.__dict__ for item in stats.stats_files]
        assert row[2] == committed.rows
    reopened, units = _store(tmp_path, rows=3, size=3)
    assert reopened.get_commit(unit.unit_id, 0) == committed
    assert reopened.pending(0) == ()
    assert reopened.pending(1) == (unit,)
    assert reopened.begin(unit.unit_id, 1).input_commit_id == attempt.attempt_id
    blocks = collect_final_units(reopened, units, 0)
    assert len(blocks) == len(tables)
    assert all(block.equals(table) for block, table in zip(blocks, tables))


def test_business_and_stats_manifests_roll_back_together(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    business = store.write_output(attempt, [pa.table({"value": [1]})])
    stats = store.write_output(attempt, [pa.table({"score": [0.9]})], stats=True)
    output = PreparedOutput(business.files, stats.stats_files)
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "CREATE TRIGGER fail_commit BEFORE UPDATE OF committed_attempt_id ON stage_state "
            "BEGIN SELECT RAISE(ABORT, 'injected commit failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected commit failure"):
        store.commit(attempt, output)
    assert store.get_commit(unit.unit_id, 0) is None
    assert store.pending(0) == (unit,)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT files_json, stats_files_json, rows FROM attempts").fetchone() == (None, None, None)
        db.execute("DROP TRIGGER fail_commit")
    committed = store.commit(attempt, output)
    assert store.get_commit(unit.unit_id, 0) == committed
    assert committed.files and committed.stats_files


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize(
    "tables",
    [
        [pa.table({"value": [1]}), pa.table({"value": ["two"]}), pa.table({"other": [True]})],
        [pa.table({"value": [None]}), pa.table({"value": ["two"]}), pa.table({"value": [None]})],
        [pa.table({"stats": [{"a": 1}]}), pa.table({"stats": [{"b": True}]})],
    ],
)
def test_one_unit_can_commit_files_with_different_schemas(tmp_path, tables, stats):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [tables[0], pa.table({"empty": []}), *tables[1:]], stats=stats)
    committed = store.commit(attempt, output)
    assert committed.rows == (0 if stats else len(tables))
    files = committed.stats_files if stats else committed.files
    assert len(files) == len(tables)
    reopened, units = _store(tmp_path, rows=1, size=1, stages=1)
    assert reopened.get_commit(unit.unit_id, 0) == committed
    blocks = [pq.read_table(reopened.root / item.path) for item in files]
    assert all(block.equals(table) for block, table in zip(blocks, tables))


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("field_metadata", [False, True])
def test_batch_metadata_changes_are_preserved(tmp_path, stats, field_metadata):
    store, (unit,) = _store(tmp_path, rows=2, size=2, stages=1)
    tables = []
    for value in (1, 2):
        metadata = {b"tag": str(value).encode(), b"data_juicer.batch_rows": b"user-value"}
        field = pa.field("value", pa.int64(), metadata=metadata if field_metadata else None)
        schema = pa.schema([field], metadata=None if field_metadata else metadata)
        tables.append(pa.Table.from_pydict({"value": [value]}, schema=schema))
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, tables, stats=stats)
    committed = store.commit(attempt, output)
    files = committed.stats_files if stats else committed.files
    assert len(files) == 2
    restored = list(store.read_output(files))
    assert len(restored) == len(tables)
    assert all(actual.equals(expected, check_metadata=True) for actual, expected in zip(restored, tables))
    assert all(
        pq.read_table(store.root / item.path).equals(table, check_metadata=True) for item, table in zip(files, tables)
    )


@pytest.mark.parametrize("stats", [False, True])
def test_unregistered_extension_batches_can_be_committed(tmp_path, stats):
    class CustomType(pa.ExtensionType):
        def __init__(self):
            super().__init__(pa.int64(), "data_juicer.test_unregistered")

        def __arrow_ext_serialize__(self):
            return b"custom"

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    store, (unit,) = _store(tmp_path, rows=4, size=4, stages=1)
    tables = [
        pa.table({"value": pa.ExtensionArray.from_storage(CustomType(), pa.array(values))})
        for values in ([1, 2], [3, 4])
    ]
    expected = []
    for index, table in enumerate(tables):
        path = tmp_path / f"native-{index}.parquet"
        pq.write_table(table, path)
        expected.append(pq.read_table(path))
    attempt = store.begin(unit.unit_id, 0)
    committed = store.commit(attempt, store.write_output(attempt, tables, stats=stats))
    files = committed.stats_files if stats else committed.files
    assert len(files) == 1
    assert store.get_commit(unit.unit_id, 0) == committed
    restored = list(store.read_output(files))
    assert len(restored) == len(expected)
    assert all(actual.equals(want, check_metadata=True) for actual, want in zip(restored, expected))


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("rows", [2**20 + 1, 64 * 2**20 + 1])
def test_large_producer_batch_keeps_its_boundary_on_readback(tmp_path, stats, rows):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    table = pa.table({"value": pa.repeat(pa.scalar(7, type=pa.int8()), rows)})
    last = table.slice(0, 1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [table, last], stats=stats)
    committed = store.commit(attempt, output)
    files = committed.stats_files if stats else committed.files
    assert len(files) == 1
    restored = store.read_output(files)
    assert next(restored).equals(table, check_metadata=True)
    assert next(restored).equals(last, check_metadata=True)
    assert list(restored) == []


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("row_group_bytes", [None, 1])
def test_same_schema_batches_share_row_groups_and_keep_boundaries(tmp_path, monkeypatch, stats, row_group_bytes):
    if row_group_bytes is not None:
        monkeypatch.setattr(unit_protocol, "_ROW_GROUP_BYTES", row_group_bytes)
    store, (unit,) = _store(tmp_path, rows=4, size=4, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    batches = [pa.table({"value": list(range(start, start + 2))}) for start in range(0, 8, 2)]
    output = store.write_output(attempt, iter(batches), stats=stats)
    files = output.stats_files if stats else output.files
    assert len(files) == 1
    with pq.ParquetFile(store.root / files[0].path) as parquet:
        assert parquet.metadata.num_row_groups == (1 if row_group_bytes is None else 4)
        assert parquet.read()["value"].to_pylist() == list(range(8))
    committed = store.commit(attempt, output)
    assert store.get_commit(unit.unit_id, 0) == committed
    reads = []
    read_group = pq.ParquetFile.read_row_group

    def tracked_read(parquet, index, **kwargs):
        reads.append(index)
        return read_group(parquet, index, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", tracked_read)
    restored = store.read_output(files)
    assert reads == []
    assert next(restored).equals(batches[0], check_metadata=True)
    assert reads == [0]
    assert all(actual.equals(expected, check_metadata=True) for actual, expected in zip(list(restored), batches[1:]))
    assert reads == ([0] if row_group_bytes is None else [0, 1, 2, 3])


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("schema_change", [False, True])
def test_generator_failure_never_commits_partial_output(tmp_path, stats, schema_change):
    store, (unit,) = _store(tmp_path, rows=2, size=2, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    tables = [pa.table({"value": [1]}), pa.table({"value": ["two"] if schema_change else [2]})]

    def interrupted():
        yield from tables
        raise RuntimeError("interrupted output generation")

    with pytest.raises(RuntimeError, match="interrupted output generation"):
        store.write_output(attempt, interrupted(), stats=stats)
    directory = store.root / "attempts" / attempt.attempt_id
    assert not list(directory.glob("*.tmp"))
    assert len(list(directory.glob("*.parquet"))) == (1 if schema_change else 0)
    reopened, _ = _store(tmp_path, rows=2, size=2, stages=1)
    assert reopened.get_commit(unit.unit_id, 0) is None
    assert reopened.pending(0) == (unit,)
    retry = reopened.begin(unit.unit_id, 0)
    output = reopened.write_output(retry, tables, stats=stats)
    committed = reopened.commit(retry, output)
    files = committed.stats_files if stats else committed.files
    rows = [row for item in files for row in pq.read_table(store.root / item.path).to_pylist()]
    assert rows == [row for table in tables for row in table.to_pylist()]
    assert reopened.pending(0) == ()


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("failure", ["write_table", "close", "fsync", "link"])
def test_output_io_failure_removes_temporary_file_and_can_retry(tmp_path, monkeypatch, stats, failure):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    target = pq.ParquetWriter if failure in ("write_table", "close") else os
    original = getattr(target, failure)

    def fail(*args, **kwargs):
        if failure in ("write_table", "close"):
            original(*args, **kwargs)
        raise OSError("injected output IO failure")

    with monkeypatch.context() as patch:
        patch.setattr(target, failure, fail)
        with pytest.raises(OSError, match="injected output IO failure"):
            store.write_output(attempt, [pa.table({"value": [1]})], stats=stats)
    assert list((store.root / "attempts" / attempt.attempt_id).iterdir()) == []
    assert store.get_commit(unit.unit_id, 0) is None
    retry = store.begin(unit.unit_id, 0)
    committed = store.commit(retry, store.write_output(retry, [pa.table({"value": [1]})], stats=stats))
    assert store.get_commit(unit.unit_id, 0) == committed


def _exit_during_output(directory, stats, crash_point):
    store = LocalUnitStore(str(directory), "job-1")
    attempt = store.begin("000000000000", 0)

    def batches():
        yield pa.table({"value": [0, 1]})
        if crash_point == "row_group":
            os._exit(86)
        yield pa.table({"value": [2, 3]})

    output = store.write_output(attempt, batches(), stats=stats)
    if crash_point == "after_commit":
        store.commit(attempt, output)
    os._exit(86)


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("crash_point", ["row_group", "before_commit", "after_commit"])
def test_process_exit_at_output_boundaries_recovers_complete_unit(tmp_path, stats, crash_point):
    store, (unit,) = _store(tmp_path, rows=4, size=4, stages=1)
    child = multiprocessing.get_context("spawn").Process(
        target=_exit_during_output, args=(tmp_path, stats, crash_point)
    )
    child.start()
    try:
        child.join(timeout=60)
        assert child.exitcode == 86
    finally:
        if child.is_alive():
            child.kill()
            child.join()
        child.close()
    reopened, _ = _store(tmp_path, rows=4, size=4, stages=1)
    committed = reopened.get_commit(unit.unit_id, 0)
    if crash_point == "after_commit":
        assert committed is not None
        assert reopened.pending(0) == ()
        with pytest.raises(ProtocolError, match="already committed"):
            reopened.begin(unit.unit_id, 0)
    else:
        assert committed is None
        assert reopened.pending(0) == (unit,)
        if crash_point == "row_group":
            assert list(store.root.rglob("*.tmp"))
            assert not list(store.root.rglob("*.parquet"))
        retry = reopened.begin(unit.unit_id, 0)
        assert retry.generation == 2
        output = reopened.write_output(retry, [pa.table({"value": list(range(4))})], stats=stats)
        committed = reopened.commit(retry, output)
    files = committed.stats_files if stats else committed.files
    assert len(files) == 1
    assert pq.read_table(store.root / files[0].path)["value"].to_pylist() == list(range(4))
    assert reopened.pending(0) == ()


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_uncommitted_output_is_reprocessed_and_late_attempt_is_fenced(tmp_path, empty, stats):
    store, (unit,) = _store(tmp_path, rows=2, size=2, stages=1)
    old = store.begin(unit.unit_id, 0)
    old_files = store.write_output(old, [] if empty else [pa.table({"value": ["old"]})], stats=stats)
    # Simulate a crash after durable output, before the metadata commit.
    restarted, _ = _store(tmp_path, rows=2, size=2, stages=1)
    assert restarted.get_commit(unit.unit_id, 0) is None
    assert restarted.pending(0) == (unit,)
    current = restarted.begin(unit.unit_id, 0)
    assert current.generation == old.generation + 1
    current_files = restarted.write_output(current, [] if empty else [pa.table({"value": ["new"]})], stats=stats)
    with pytest.raises(ProtocolError, match="stale"):
        restarted.commit(old, old_files)
    accepted = restarted.commit(current, current_files)
    assert restarted.get_commit(unit.unit_id, 0) == accepted
    with pytest.raises(ProtocolError, match="already committed"):
        restarted.begin(unit.unit_id, 0)
    with pytest.raises(ProtocolError, match="already committed"):
        restarted.commit(old, old_files)


def test_downstream_requires_upstream_and_can_commit_empty_result(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1)
    assert store.pending(1) == ()
    with pytest.raises(ProtocolError, match="Upstream"):
        store.begin(unit.unit_id, 1)
    first = store.begin(unit.unit_id, 0)
    store.commit(first, store.write_output(first, [pa.table({"value": [1, 2, 3]})]))
    assert store.pending(1) == (unit,)
    second = store.begin(unit.unit_id, 1)
    # A Filter can drop every output of the previous stage.
    store.commit(second, store.write_output(second, []))
    assert store.get_commit(unit.unit_id, 1).rows == 0
    assert store.pending(1) == ()


@pytest.mark.parametrize("stats", [False, True])
def test_empty_output_is_fenced_when_upstream_commit_changes(tmp_path, stats):
    store, (unit,) = _store(tmp_path, rows=1, size=1)
    first = store.begin(unit.unit_id, 0)
    store.commit(first, store.write_output(first, []))
    second = store.begin(unit.unit_id, 1)
    output = store.write_output(second, [pa.table({"score": [0.1]})] if stats else [], stats=stats)
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE stage_state SET committed_attempt_id='replaced' WHERE stage=0")
    with pytest.raises(ProtocolError, match="Upstream commit changed"):
        store.commit(second, output)
    assert store.get_commit(unit.unit_id, 1) is None


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_missing_or_modified_output_cannot_be_committed(tmp_path, damage, stats):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [pa.table({"value": [1]})], stats=stats)
    path = tmp_path / (output.stats_files if stats else output.files)[0].path
    if stats:
        business = store.write_output(attempt, [pa.table({"value": [1]})])
        output = replace(output, files=business.files)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"damaged")
    with pytest.raises(ProtocolError, match="Missing or modified"):
        store.commit(attempt, output)
    assert store.get_commit(unit.unit_id, 0) is None
    assert store.pending(0) == (unit,)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT files_json, stats_files_json, rows FROM attempts").fetchone() == (None, None, None)


@pytest.mark.parametrize("stats", [False, True])
def test_attempt_output_files_cannot_be_overwritten(tmp_path, stats):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [pa.table({"value": [1]})], stats=stats)
    with pytest.raises(FileExistsError):
        store.write_output(attempt, [pa.table({"value": [2]})], stats=stats)
    committed = store.commit(attempt, output)
    assert committed.rows == (0 if stats else 1)
    assert store.get_commit(unit.unit_id, 0) == committed


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("invalid", ["rows", "duplicate", "zero"])
def test_invalid_file_manifest_cannot_be_committed(tmp_path, invalid, stats):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [pa.table({"value": [1]})], stats=stats)
    item = (output.stats_files if stats else output.files)[0]
    files = (item, item) if invalid == "duplicate" else (replace(item, rows=0 if invalid == "zero" else 2),)
    invalid_output = PreparedOutput((), files) if stats else PreparedOutput(files)
    with pytest.raises(ProtocolError, match="Invalid attempt output file|row count mismatch"):
        store.commit(attempt, invalid_output)
    assert store.get_commit(unit.unit_id, 0) is None


@pytest.mark.parametrize("stats", [False, True])
def test_forged_attempt_cannot_commit_another_units_output(tmp_path, stats):
    store, units = _store(tmp_path, rows=3, size=2, stages=1)
    first = store.begin(units[0].unit_id, 0)
    second = store.begin(units[1].unit_id, 0)
    files = store.write_output(first, [pa.table({"value": [1]})], stats=stats)
    with pytest.raises(ProtocolError, match="Invalid attempt output file"):
        store.commit(second, files)


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_committed_file_damage_fails_closed_on_recovery(tmp_path, damage, stats):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [pa.table({"value": [1]})], stats=stats)
    path = tmp_path / (output.stats_files if stats else output.files)[0].path
    if stats:
        business = store.write_output(attempt, [pa.table({"value": [1]})])
        output = replace(output, files=business.files)
    store.commit(attempt, output)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"damaged")
    reopened, _ = _store(tmp_path, rows=1, size=1, stages=1)
    with pytest.raises(ProtocolError, match="Committed output file"):
        reopened.get_commit(unit.unit_id, 0)


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("damage", ["total_rows", "file_rows", "duplicate", "foreign_file", "zero"])
def test_committed_manifest_damage_fails_closed_on_recovery(tmp_path, damage, stats):
    store, units = _store(tmp_path, rows=2, size=1, stages=1)
    first, second = (store.begin(unit.unit_id, 0) for unit in units)
    output = store.write_output(first, [pa.table({"value": [1]})], stats=stats)
    store.commit(first, output)
    files = [(output.stats_files if stats else output.files)[0].__dict__.copy()]
    rows = 1
    if damage == "total_rows":
        rows = 2
    elif damage == "file_rows":
        files[0]["rows"] = rows = 2
    elif damage == "duplicate":
        files *= 2
        rows = 2
    elif damage == "foreign_file":
        foreign = store.write_output(second, [pa.table({"value": [1]})], stats=stats)
        files = [(foreign.stats_files if stats else foreign.files)[0].__dict__]
    else:
        files[0]["rows"] = rows = 0
    if stats and damage != "total_rows":
        rows = 0  # Stats must be validated independently of the business row total.
    column = "stats_files_json" if stats else "files_json"
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            f"UPDATE attempts SET {column}=?, rows=? WHERE attempt_id=?",
            (json.dumps(files), rows, first.attempt_id),
        )
    reopened, _ = _store(tmp_path, rows=2, size=1, stages=1)
    with pytest.raises(ProtocolError, match="Committed output"):
        reopened.get_commit(units[0].unit_id, 0)


@pytest.mark.parametrize("on_recovery", [False, True])
def test_duplicate_paths_across_business_and_stats_are_rejected(tmp_path, on_recovery):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [pa.table({"value": [1]})])
    # A different spelling of the same path must not bypass cross-list checks.
    duplicate = replace(output.files[0], path=output.files[0].path.replace("/part", "/./part"))
    if on_recovery:
        store.commit(attempt, output)
        with sqlite3.connect(store.db_path) as db:
            db.execute("UPDATE attempts SET stats_files_json=?", (json.dumps([duplicate.__dict__]),))
        reopened, _ = _store(tmp_path, rows=1, size=1, stages=1)
        with pytest.raises(ProtocolError, match="Committed output file"):
            reopened.get_commit(unit.unit_id, 0)
    else:
        with pytest.raises(ProtocolError, match="Invalid attempt output file"):
            store.commit(attempt, replace(output, stats_files=(duplicate,)))
        assert store.get_commit(unit.unit_id, 0) is None
        assert store.pending(0) == (unit,)


@pytest.mark.parametrize("read_rows", [0, 1, 3])
def test_runner_rejects_incorrect_snapshot_read_before_commit(tmp_path, read_rows):
    store, units = _store(tmp_path, rows=2, size=2, stages=1)

    def incorrect_read(start, stop):
        yield pa.table({"value": list(range(read_rows))})

    def identity(table):
        assert table.num_rows > 0
        yield table

    with pytest.raises(ProtocolError, match="input row count"):
        run_local_units(store, units, incorrect_read, [identity])
    assert store.get_commit(units[0].unit_id, 0) is None


def test_runner_validates_upstream_reference_and_input_rows(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1)
    first = store.begin(unit.unit_id, 0)
    upstream = store.commit(first, store.write_output(first, [pa.table({"value": [1]})]))
    second = store.begin(unit.unit_id, 1)

    def source(start, stop):
        pytest.fail("Downstream must read committed files")

    def drop_all(table):
        return ()

    with pytest.raises(ProtocolError, match="Upstream stage has not committed"):
        execute_unit_attempt(store, second, unit, source, drop_all)
    with pytest.raises(ProtocolError, match="does not reference the upstream commit"):
        execute_unit_attempt(store, replace(second, input_commit_id="wrong"), unit, source, drop_all, upstream)
    with pytest.raises(ProtocolError, match="input row count"):
        execute_unit_attempt(store, second, unit, source, drop_all, replace(upstream, rows=2))
    assert store.get_commit(unit.unit_id, 1) is None


@pytest.mark.parametrize("invalid_source", [False, True])
def test_runner_rejects_non_table_input_or_output(tmp_path, invalid_source):
    store, units = _store(tmp_path, rows=1, size=1, stages=1)

    def source(start, stop):
        yield None if invalid_source else pa.table({"value": [1]})

    def operator(table):
        yield None

    with pytest.raises(TypeError, match="PyArrow tables"):
        run_local_units(store, units, source, [operator])
    assert store.get_commit(units[0].unit_id, 0) is None


def test_runner_restarts_from_committed_units_with_expansion_and_drop_all(tmp_path):
    store, units = _store(tmp_path, rows=4, size=2)
    reads = []
    expands = []
    filtered = []
    fail_once = {"value": True}

    def source(start, stop):
        reads.append((start, stop))
        yield pa.table({"value": list(range(start, stop))})

    def expand(table):
        expands.append(table.column("value")[0].as_py())
        yield table
        yield table

    def keep_first_unit(table):
        first = table.column("value")[0].as_py()
        filtered.append(first)
        if first == 2 and fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("worker died")
        kept = table.filter(pa.array([value < 2 for value in table["value"].to_pylist()]))
        if kept.num_rows:
            yield kept

    with pytest.raises(RuntimeError, match="worker died"):
        run_local_units(store, units, source, [expand, keep_first_unit])
    assert len(reads) == len(expands) == 2
    assert store.get_commit(units[0].unit_id, 1).rows == 4
    assert store.get_commit(units[1].unit_id, 1) is None

    reopened, same_units = _store(tmp_path, rows=4, size=2)
    blocks = run_local_units(reopened, same_units, source, [expand, keep_first_unit])
    assert len(reads) == len(expands) == 2  # Committed Mapper units were not repeated.
    assert filtered == [0, 0, 2, 2, 2]  # Only the unfinished Filter unit repeats.
    assert [value for block in blocks for value in block["value"].to_pylist()] == [0, 1, 0, 1]
    assert reopened.get_commit(units[1].unit_id, 1).rows == 0


@pytest.mark.parametrize("emit_empty_table", [False, True])
def test_runner_recovers_empty_output_without_schema_or_operator_calls(tmp_path, emit_empty_table):
    store, units = _store(tmp_path, rows=2, size=1, stages=3)
    calls = []

    def source(start, stop):
        yield pa.table({})
        yield pa.table({"value": list(range(start, stop))})
        yield pa.table({"other": []})

    def drop_all(table):
        assert table.num_rows > 0
        calls.append(table["value"][0].as_py())
        return (pa.table({"dropped": []}),) if emit_empty_table else ()

    for unit in units:
        attempt = store.begin(unit.unit_id, 0)
        output = execute_unit_attempt(store, attempt, unit, source, drop_all)
        assert output == PreparedOutput(())
        store.commit(attempt, output)
    assert calls == [0, 1]

    def must_not_run(*args):
        pytest.fail("Committed or empty units must not call the source or operator")

    reopened, same_units = _store(tmp_path, rows=2, size=1, stages=3)
    stages = [must_not_run] * 3
    assert run_local_units(reopened, same_units, must_not_run, stages) == ()
    for stage in range(3):
        assert reopened.pending(stage) == ()
        for unit in units:
            committed = reopened.get_commit(unit.unit_id, stage)
            assert committed.rows == 0
            assert committed.files == ()
    assert list(store.root.rglob("*.parquet")) == []
    recovered, same_units = _store(tmp_path, rows=2, size=1, stages=3)
    assert run_local_units(recovered, same_units, must_not_run, stages) == ()


def test_runner_keeps_different_unit_schemas_separate(tmp_path):
    store, units = _store(tmp_path, rows=2, size=1, stages=1)
    tables = [pa.table({"value": [1]}), pa.table({"value": ["two"]})]

    def source(start, stop):
        yield tables[start]

    def identity(table):
        yield table

    blocks = run_local_units(store, units, source, [identity])
    assert len(blocks) == len(tables)
    assert all(block.equals(table) for block, table in zip(blocks, tables))


def test_runner_empty_plan_does_not_invoke_source_or_operator(tmp_path):
    store, units = _store(tmp_path, rows=0, stages=2)

    def must_not_run(*args):
        pytest.fail("An empty plan has no input to execute")

    assert units == ()
    assert run_local_units(store, units, must_not_run, [must_not_run] * 2) == ()
