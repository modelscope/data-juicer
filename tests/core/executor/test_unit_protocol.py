"""Crash and output-contract tests for the local recovery-unit protocol."""

import json
import sqlite3
from dataclasses import replace

import pyarrow as pa
import pytest

from data_juicer.core.executor.unit_protocol import SCHEMA_VERSION, LocalUnitStore, PreparedOutput, ProtocolError
from data_juicer.core.executor.unit_runner import collect_final_units, execute_unit_attempt, run_local_units


def test_old_checkpoint_database_is_rejected_before_running(tmp_path):
    with sqlite3.connect(tmp_path / "units.sqlite3") as db:
        db.execute("CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, files_json TEXT, rows INTEGER)")
    with pytest.raises(ProtocolError, match="incompatible protocol version"):
        LocalUnitStore(str(tmp_path), "job-1")


@pytest.mark.parametrize("version, schema_column", [(1, False), (2, True), (2, False), (3, True)])
def test_incompatible_checkpoint_is_rejected_without_migration(tmp_path, version, schema_column):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    store.commit(attempt, store.write_output(attempt, []))
    with sqlite3.connect(store.db_path) as db:
        assert SCHEMA_VERSION == db.execute("SELECT schema_version FROM runs").fetchone()[0] == 3
        assert "schema_hex" not in {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
        db.execute("UPDATE runs SET schema_version=?", (version,))
        if schema_column:
            db.execute("ALTER TABLE attempts ADD COLUMN schema_hex TEXT")
    with pytest.raises(ProtocolError, match="incompatible protocol version"):
        LocalUnitStore(str(tmp_path), "job-1")
    with pytest.raises(ProtocolError, match="incompatible protocol version"):
        LocalUnitStore(str(tmp_path), "another-job")
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT schema_version FROM runs").fetchone()[0] == version
        assert ("schema_hex" in {row[1] for row in db.execute("PRAGMA table_info(attempts)")}) == schema_column


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


@pytest.mark.parametrize("empty_tables", [[], [pa.table({})], [pa.table({"value": []}), pa.table({"other": []})]])
def test_zero_output_and_many_outputs_are_complete_commits(tmp_path, empty_tables):
    store, (unit,) = _store(tmp_path, rows=2, size=2)
    first = store.begin(unit.unit_id, 0)
    first_output = store.write_output(first, empty_tables)
    assert first_output == PreparedOutput(())
    assert list((store.root / "attempts" / first.attempt_id).iterdir()) == []
    zero_commit = store.commit(first, first_output)
    assert zero_commit.rows == 0
    assert store.pending(0) == ()
    store, _ = _store(tmp_path, rows=2, size=2)
    assert store.get_commit(unit.unit_id, 0) == zero_commit

    second = store.begin(unit.unit_id, 1)
    assert second.input_commit_id == first.attempt_id
    # The unit emits more rows than it received, across two physical files.
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
    assert len(restored.files) == 2
    assert reopened.pending(1) == ()


def test_one_unit_can_commit_files_with_different_schemas(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    tables = [pa.table({"value": [1]}), pa.table({"value": ["two"]}), pa.table({"other": [True]})]
    output = store.write_output(attempt, [tables[0], pa.table({"empty": []}), *tables[1:]])
    committed = store.commit(attempt, output)
    assert committed.rows == 3
    assert len(committed.files) == 3
    reopened, units = _store(tmp_path, rows=1, size=1, stages=1)
    assert reopened.get_commit(unit.unit_id, 0) == committed
    blocks = collect_final_units(reopened, units, 0)
    assert len(blocks) == len(tables)
    assert all(block.equals(table) for block, table in zip(blocks, tables))


@pytest.mark.parametrize("empty", [False, True])
def test_uncommitted_output_is_reprocessed_and_late_attempt_is_fenced(tmp_path, empty):
    store, (unit,) = _store(tmp_path, rows=2, size=2, stages=1)
    old = store.begin(unit.unit_id, 0)
    old_files = store.write_output(old, [] if empty else [pa.table({"value": ["old"]})])
    # Simulate a crash after durable output, before the metadata commit.
    restarted, _ = _store(tmp_path, rows=2, size=2, stages=1)
    assert restarted.get_commit(unit.unit_id, 0) is None
    assert restarted.pending(0) == (unit,)
    current = restarted.begin(unit.unit_id, 0)
    assert current.generation == old.generation + 1
    current_files = restarted.write_output(current, [] if empty else [pa.table({"value": ["new"]})])
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


def test_empty_output_is_fenced_when_upstream_commit_changes(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1)
    first = store.begin(unit.unit_id, 0)
    store.commit(first, store.write_output(first, []))
    second = store.begin(unit.unit_id, 1)
    output = store.write_output(second, [])
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE stage_state SET committed_attempt_id='replaced' WHERE stage=0")
    with pytest.raises(ProtocolError, match="Upstream commit changed"):
        store.commit(second, output)
    assert store.get_commit(unit.unit_id, 1) is None


@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_missing_or_modified_output_cannot_be_committed(tmp_path, damage):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    files = store.write_output(attempt, [pa.table({"value": [1]})])
    path = tmp_path / files.files[0].path
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"damaged")
    with pytest.raises(ProtocolError, match="Missing or modified"):
        store.commit(attempt, files)
    assert store.get_commit(unit.unit_id, 0) is None


def test_attempt_output_files_cannot_be_overwritten(tmp_path):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [pa.table({"value": [1]})])
    with pytest.raises(FileExistsError):
        store.write_output(attempt, [pa.table({"value": [2]})])
    assert store.commit(attempt, output).rows == 1


@pytest.mark.parametrize("invalid", ["rows", "duplicate", "zero"])
def test_invalid_file_manifest_cannot_be_committed(tmp_path, invalid):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    output = store.write_output(attempt, [pa.table({"value": [1]})])
    item = output.files[0]
    files = (item, item) if invalid == "duplicate" else (replace(item, rows=0 if invalid == "zero" else 2),)
    with pytest.raises(ProtocolError, match="Invalid attempt output file|row count mismatch"):
        store.commit(attempt, PreparedOutput(files))
    assert store.get_commit(unit.unit_id, 0) is None


def test_forged_attempt_cannot_commit_another_units_output(tmp_path):
    store, units = _store(tmp_path, rows=3, size=2, stages=1)
    first = store.begin(units[0].unit_id, 0)
    second = store.begin(units[1].unit_id, 0)
    files = store.write_output(first, [pa.table({"value": [1]})])
    with pytest.raises(ProtocolError, match="Invalid attempt output file"):
        store.commit(second, files)


@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_committed_file_damage_fails_closed_on_recovery(tmp_path, damage):
    store, (unit,) = _store(tmp_path, rows=1, size=1, stages=1)
    attempt = store.begin(unit.unit_id, 0)
    files = store.write_output(attempt, [pa.table({"value": [1]})])
    store.commit(attempt, files)
    path = tmp_path / files.files[0].path
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"damaged")
    reopened, _ = _store(tmp_path, rows=1, size=1, stages=1)
    with pytest.raises(ProtocolError, match="Committed output file"):
        reopened.get_commit(unit.unit_id, 0)


@pytest.mark.parametrize("damage", ["total_rows", "file_rows", "duplicate", "foreign_file", "zero"])
def test_committed_manifest_damage_fails_closed_on_recovery(tmp_path, damage):
    store, units = _store(tmp_path, rows=2, size=1, stages=1)
    first, second = (store.begin(unit.unit_id, 0) for unit in units)
    output = store.write_output(first, [pa.table({"value": [1]})])
    store.commit(first, output)
    files = [output.files[0].__dict__.copy()]
    rows = 1
    if damage == "total_rows":
        rows = 2
    elif damage == "file_rows":
        files[0]["rows"] = rows = 2
    elif damage == "duplicate":
        files *= 2
        rows = 2
    elif damage == "foreign_file":
        foreign = store.write_output(second, [pa.table({"value": [1]})])
        files = [foreign.files[0].__dict__]
    else:
        files[0]["rows"] = rows = 0
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "UPDATE attempts SET files_json=?, rows=? WHERE attempt_id=?",
            (json.dumps(files), rows, first.attempt_id),
        )
    reopened, _ = _store(tmp_path, rows=2, size=1, stages=1)
    with pytest.raises(ProtocolError, match="Committed output"):
        reopened.get_commit(units[0].unit_id, 0)


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
