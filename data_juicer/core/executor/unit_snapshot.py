"""Durable input identity for recovery units."""

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import ray

from data_juicer.core.executor.ray_executor_partitioned import (
    _PARTITION_CONTENT_HASH_ALGORITHM,
    _combine_partition_hash_partials,
    _hash_partition_batch,
)
from data_juicer.core.executor.unit_protocol import ProtocolError, _fsync_dir

SNAPSHOT_VERSION = 3
_OFFSET_COLUMN = "__dj_unit_offset__"


def _scan_batch(batch):
    offsets = batch[_OFFSET_COLUMN]
    table = batch.drop_columns([_OFFSET_COLUMN])
    rows = table.num_rows
    first = offsets[0].as_py() if rows else 0
    if rows and offsets[rows - 1].as_py() - first != rows - 1:
        raise ProtocolError("Input block row offsets are not contiguous")
    partial = _hash_partition_batch(table).to_pylist()[0]
    schema_hash = hashlib.sha256(table.schema.serialize().to_pybytes()).digest()
    return pa.table(
        {
            "start": [first],
            "row_count": [rows],
            "sequence_hash": [partial["sequence_hash"]],
            "schema_hash": [schema_hash],
        }
    )


def _fingerprint(schema_bytes, partials):
    content_hash, _ = _combine_partition_hash_partials(partials)
    payload = schema_bytes + f"{_PARTITION_CONTENT_HASH_ALGORITHM}:{content_hash}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _scan(dataset):
    """Materialize and fingerprint the ordered input on Ray workers."""
    data = dataset.materialize()
    total_rows = data.count()
    schema = data.schema()
    schema = getattr(schema, "base_schema", schema)
    if not isinstance(schema, pa.Schema):
        raise ProtocolError("Input needs a concrete Arrow schema")
    if _OFFSET_COLUMN in schema.names:
        raise ProtocolError(f"Input contains reserved recovery column {_OFFSET_COLUMN}")
    schema_bytes = schema.serialize().to_pybytes()
    if total_rows == 0:
        return _fingerprint(schema_bytes, []), 0, schema_bytes, data
    offsets = ray.data.range(total_rows).rename_columns({"id": _OFFSET_COLUMN})
    partials = data.zip(offsets).map_batches(_scan_batch, batch_format="pyarrow", batch_size=None).take_all()
    partials = sorted((item for item in partials if item["row_count"]), key=lambda item: item["start"])
    expected_start = 0
    for partial in partials:
        if partial["start"] != expected_start:
            raise ProtocolError("Input blocks do not cover the input rows exactly once")
        expected_start += partial["row_count"]
    if expected_start != total_rows:
        raise ProtocolError("Input blocks do not cover the input rows exactly once")
    schema_hash = hashlib.sha256(schema_bytes).digest()
    if {partial["schema_hash"] for partial in partials} != {schema_hash}:
        raise ProtocolError("Input schema changed between batches")
    return _fingerprint(schema_bytes, partials), total_rows, schema_bytes, data


def open_or_create_snapshot(dataset, directory: str, unit_size: int, *, resuming: bool):
    """Persist the input identity and return the verified materialized Dataset."""
    if unit_size < 1:
        raise ValueError("unit_size must be positive")
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if resuming:
        if not manifest_path.is_file():
            raise ProtocolError("Unit snapshot is missing on resume")
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        if manifest.get("version") != SNAPSHOT_VERSION or manifest.get("unit_size") != unit_size:
            raise ProtocolError("Unit snapshot plan changed")
        fingerprint, rows, schema, data = _scan(dataset)
        if (fingerprint, rows, schema.hex()) != (
            manifest.get("fingerprint"),
            manifest.get("rows"),
            manifest.get("schema_hex"),
        ):
            raise ProtocolError("Input content or read order changed since the unit snapshot")
        return manifest, data

    if root.exists():
        raise ProtocolError("Unit snapshot already exists; resume or use a new job ID")
    root.parent.mkdir(parents=True, exist_ok=True)
    for stale in root.parent.glob(f".{root.name}.*.tmp"):
        shutil.rmtree(stale, ignore_errors=True)
    tmp = root.parent / f".{root.name}.{uuid.uuid4().hex}.tmp"
    tmp.mkdir()
    try:
        fingerprint, rows, schema, data = _scan(dataset)
        manifest = {
            "version": SNAPSHOT_VERSION,
            "fingerprint": fingerprint,
            "rows": rows,
            "unit_size": unit_size,
            "schema_hex": schema.hex(),
        }
        with (tmp / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_dir(tmp)
        os.replace(tmp, root)
        _fsync_dir(root.parent)
        return manifest, data
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
