import json
import os

import pytest
import ray

from data_juicer.core.executor.unit_protocol import ProtocolError
from data_juicer.core.executor.unit_snapshot import SNAPSHOT_VERSION, open_or_create_snapshot


@pytest.fixture(scope="module")
def local_ray():
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


def _rows(count):
    return [{"id": index, "text": f"row-{index}"} for index in range(count)]


def _dataset(rows, blocks):
    return ray.data.from_items(rows, override_num_blocks=blocks)


@pytest.mark.parametrize("rows, blocks, unit_size", [(23, 4, 5), (23, 1, 5), (20, 7, 4), (5, 5, 8), (1, 1, 3)])
def test_manifest_identifies_input_without_persisting_rows(tmp_path, local_ray, rows, blocks, unit_size):
    directory = tmp_path / "source"
    manifest, source = open_or_create_snapshot(_dataset(_rows(rows), blocks), str(directory), unit_size, resuming=False)
    assert manifest["version"] == SNAPSHOT_VERSION
    assert manifest["rows"] == rows
    assert manifest["unit_size"] == unit_size
    assert "files" not in manifest
    assert os.listdir(directory) == ["manifest.json"]
    assert (directory / "manifest.json").stat().st_size < 10_000
    assert not list(tmp_path.glob(".*.tmp"))
    assert source.take_all() == _rows(rows)

    reference, _ = open_or_create_snapshot(
        _dataset(_rows(rows), 1), str(tmp_path / "reference"), unit_size, resuming=False
    )
    assert manifest["fingerprint"] == reference["fingerprint"]
    resumed, restored = open_or_create_snapshot(
        _dataset(_rows(rows), blocks + 2), str(directory), unit_size, resuming=True
    )
    assert resumed == manifest
    assert restored.take_all() == _rows(rows)


@pytest.mark.parametrize("change", ["order", "content", "rows"])
def test_resume_rejects_changed_input(tmp_path, local_ray, change):
    directory = str(tmp_path / "source")
    open_or_create_snapshot(_dataset(_rows(12), 3), directory, 5, resuming=False)
    rows = _rows(12)
    if change == "order":
        rows[3], rows[9] = rows[9], rows[3]
    elif change == "content":
        rows[7]["text"] = "changed"
    else:
        rows = rows[:11]
    with pytest.raises(ProtocolError, match="Input content or read order changed"):
        open_or_create_snapshot(_dataset(rows, 3), directory, 5, resuming=True)


def test_rejects_reserved_fingerprint_column(tmp_path, local_ray):
    rows = [{"id": 0, "__dj_unit_offset__": "user data"}]
    with pytest.raises(ProtocolError, match="reserved recovery column"):
        open_or_create_snapshot(_dataset(rows, 1), str(tmp_path / "source"), 1, resuming=False)


def test_resume_rejects_modified_manifest(tmp_path, local_ray):
    directory = tmp_path / "source"
    open_or_create_snapshot(_dataset(_rows(12), 3), str(directory), 5, resuming=False)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ProtocolError, match="Input content or read order changed"):
        open_or_create_snapshot(_dataset(_rows(12), 3), str(directory), 5, resuming=True)
