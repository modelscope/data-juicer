"""Ray-free checks of checkpoint storage and the lineage commit log."""

import os

import pyarrow as pa
import pytest

from data_juicer.core.executor.partition_protocol import (
    COMMIT_DIR,
    SEQ_COLUMN,
    SOURCE_SEQ,
    Commit,
    CommitLog,
    IntervalSet,
    LocalStore,
    ProtocolError,
    S3Store,
    commit_key,
    key_ranges,
    open_store,
    read_tables,
    select_ranges,
    with_keys,
    write_tables,
)


def _persist(store, commit):
    assert store.put_if_absent(commit_key(commit.stage, commit.seq), commit.to_bytes())


def _prepare(store, log, stage, consumed, rows):
    files = write_tables(store, [pa.table({"v": list(range(rows))})]) if rows else ()
    commit, new = log.prepare(stage, consumed, files)
    if new:
        _persist(store, commit)
    return commit, new


def test_key_ranges_compress_unsorted_keys():
    assert key_ranges([3, 3, -1, 3, -1], [2, 0, 5, 1, 4]) == ((-1, 4, 6), (3, 0, 3))
    assert key_ranges([], []) == ()
    with pytest.raises(ProtocolError, match="same upstream row twice"):
        key_ranges([1, 1], [0, 0])


def test_interval_set_merges_and_reports_gaps():
    ranges = IntervalSet()
    ranges.add(4, 6)
    ranges.add(0, 2)
    ranges.add(2, 4)
    assert (ranges.starts, ranges.stops) == ([0], [6])
    ranges.add(8, 9)
    assert ranges.overlaps(5, 7) and ranges.overlaps(7, 9) and not ranges.overlaps(6, 8)
    assert ranges.gaps(10) == [(6, 8), (9, 10)]


def test_local_store_creates_only_once_and_hides_temporaries(tmp_path):
    store = LocalStore(str(tmp_path / "ckpt"))
    assert store.put_if_absent("commits/a.json", b"first")
    assert not store.put_if_absent("commits/a.json", b"second")
    assert store.get("commits/a.json") == b"first"
    (tmp_path / "ckpt" / "commits" / ".a.json.x.tmp").write_bytes(b"partial")
    assert store.list("commits") == ["commits/a.json"]
    assert store.list("missing") == []
    assert store.get("missing.json") is None


def test_open_store_accepts_posix_and_s3_only(tmp_path):
    assert isinstance(open_store(str(tmp_path)), LocalStore)
    assert open_store(f"file://{tmp_path}").root == str(tmp_path)
    s3 = open_store("s3://bucket/prefix/run")
    assert isinstance(s3, S3Store) and (s3.bucket, s3.prefix) == ("bucket", "prefix/run")
    for location in ("gs://bucket/x", "hdfs://cluster/x", "oss://bucket/x"):
        with pytest.raises(ValueError, match="not supported"):
            open_store(location)


def test_files_round_trip_and_detect_tampering(tmp_path):
    store = LocalStore(str(tmp_path))
    tables = [pa.table({"a": [1, 2]}), pa.table({"a": [3]}), pa.table({"b": ["x"]}), pa.table({"a": []})]
    files = write_tables(store, tables)
    assert [item.rows for item in files] == [3, 1]
    assert [table.to_pylist() for table in read_tables(store, files)] == [[{"a": 1}, {"a": 2}, {"a": 3}], [{"b": "x"}]]
    with open(os.path.join(store.root, files[0].key), "ab") as stream:
        stream.write(b"x")
    with pytest.raises(ProtocolError, match="missing or modified"):
        list(read_tables(store, files))


def test_keys_select_requested_rows():
    table = with_keys(pa.table({"v": list("abcde")}), 7)
    assert table[SEQ_COLUMN].to_pylist() == [7] * 5
    assert select_ranges(table, [(0, 1), (3, 5)])["v"].to_pylist() == ["a", "d", "e"]


def test_exact_duplicate_returns_winner_and_partial_overlap_fails(tmp_path):
    store = LocalStore(str(tmp_path))
    log = CommitLog(store, stage_count=2, source_rows=10)
    first, new = _prepare(store, log, 0, [(SOURCE_SEQ, 0, 4)], 3)
    assert new and first.seq == 0
    retry, new = _prepare(store, log, 0, [(SOURCE_SEQ, 0, 4)], 2)
    assert not new and retry == first
    with pytest.raises(ProtocolError, match="already consumed"):
        log.prepare(0, [(SOURCE_SEQ, 3, 6)], ())
    with pytest.raises(ProtocolError, match="unknown upstream"):
        log.prepare(1, [(99, 0, 1)], ())
    with pytest.raises(ProtocolError, match="malformed or unknown"):
        log.prepare(1, [(first.seq, 0, 4)], ())
    with pytest.raises(ProtocolError, match="first stage"):
        log.prepare(0, [(first.seq, 0, 1)], ())


def test_reloaded_log_plans_pending_rows_and_verifies_delivery(tmp_path):
    store = LocalStore(str(tmp_path))
    log = CommitLog(store, stage_count=2, source_rows=10)
    first, _ = _prepare(store, log, 0, [(SOURCE_SEQ, 0, 4)], 3)
    _prepare(store, log, 0, [(SOURCE_SEQ, 4, 7)], 0)
    downstream, _ = _prepare(store, log, 1, [(first.seq, 0, 2)], 5)
    # An attempt that never wrote its manifest leaves no trace in the reloaded log.
    log.prepare(0, [(SOURCE_SEQ, 7, 10)], ())

    reloaded = CommitLog(store, stage_count=2, source_rows=10)
    plan = reloaded.plan()
    assert plan["source"] == [(7, 10)]
    assert [(commit.seq, gaps) for commit, gaps in plan["stages"][1]] == [(first.seq, [(2, 3)])]
    assert plan["final"] == [downstream]
    assert reloaded.next_seq == 3
    with pytest.raises(ProtocolError, match="never consumed by stage 0"):
        reloaded.verify()

    rest, _ = _prepare(store, reloaded, 0, [(SOURCE_SEQ, 7, 10)], 1)
    tail, _ = _prepare(store, reloaded, 1, [(first.seq, 2, 3), (rest.seq, 0, 1)], 1)
    with pytest.raises(ProtocolError, match="never delivered downstream"):
        reloaded.verify()
    reloaded.deliver([(downstream.seq, 0, 5)])
    reloaded.deliver([(downstream.seq, 0, 5)])
    with pytest.raises(ProtocolError, match="already consumed"):
        reloaded.deliver([(downstream.seq, 4, 5), (tail.seq, 0, 1)])
    reloaded.deliver([(tail.seq, 0, 1)])
    assert reloaded.verify() == {0: (), 1: ()}


def test_log_rejects_foreign_or_duplicate_manifests(tmp_path):
    store = LocalStore(str(tmp_path))
    commit = Commit(0, 0, ((SOURCE_SEQ, 0, 2),), ())
    store.put(f"{COMMIT_DIR}/0000-0000000000000005.json", commit.to_bytes())
    with pytest.raises(ProtocolError, match="does not belong"):
        CommitLog(store, stage_count=1, source_rows=2)

    other = LocalStore(str(tmp_path / "other"))
    _persist(other, commit)
    _persist(other, Commit(0, 1, ((SOURCE_SEQ, 0, 2),), ()))
    with pytest.raises(ProtocolError, match="duplicates"):
        CommitLog(other, stage_count=1, source_rows=2)


def test_stats_files_are_returned_by_stage(tmp_path):
    store = LocalStore(str(tmp_path))
    log = CommitLog(store, stage_count=1, source_rows=2)
    stats = write_tables(store, [pa.table({"s": [1, 2]})], kind="stats")
    commit, _ = log.prepare(0, [(SOURCE_SEQ, 0, 2)], (), stats)
    _persist(store, commit)
    assert CommitLog(store, stage_count=1, source_rows=2).verify() == {0: stats}
