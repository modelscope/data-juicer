"""Shared validation and catalog loading for existing-table writers."""

import ntpath
from copy import deepcopy
from urllib.parse import urlparse


def validate_iceberg_artifact(export_path, shard_size=0, encrypt=False):
    """Validate the local artifact anchor and unsupported file options."""
    if not isinstance(export_path, str) or not export_path.strip():
        raise ValueError("Iceberg export_path must be a non-empty local artifact path, not a table URI")
    drive, _ = ntpath.splitdrive(export_path)
    if urlparse(export_path).scheme and not (len(drive) == 2 and drive[1] == ":"):
        raise ValueError("Iceberg export_path must be a non-empty local artifact path, not a table URI")
    if shard_size > 0:
        raise ValueError("Iceberg export does not support export_shard_size > 0")
    if encrypt:
        raise ValueError("Iceberg export does not support local-file encryption")


def _validate_catalog_name(catalog_kwargs):
    """Require an explicit catalog name for writes."""
    if not isinstance(catalog_kwargs, dict):
        raise ValueError("Iceberg export requires catalog_kwargs as a dict")
    name = catalog_kwargs.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Iceberg catalog name must be an explicit non-empty string")


def iceberg_export_config(export_path, shard_size, encrypt, kwargs):
    """Validate write configuration without resolving or logging credentials."""
    validate_iceberg_artifact(export_path, shard_size, encrypt)
    if set(kwargs) - {"table_identifier", "catalog_kwargs", "mode"}:
        raise ValueError("Iceberg export only accepts table_identifier, catalog_kwargs and mode")
    identifier = kwargs.get("table_identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Iceberg export requires a non-empty table_identifier")
    catalog_kwargs = kwargs.get("catalog_kwargs")
    _validate_catalog_name(catalog_kwargs)
    mode = kwargs.get("mode", "overwrite")
    if mode not in ("append", "overwrite"):
        raise ValueError("Iceberg export mode must be append or overwrite")
    try:
        import pyiceberg.catalog  # noqa: F401
    except ImportError as e:
        raise RuntimeError("Iceberg writing requires py-data-juicer[iceberg]") from e
    return {"table_identifier": identifier, "catalog_kwargs": deepcopy(catalog_kwargs), "mode": mode}


def load_iceberg_table(table_identifier, catalog_kwargs):
    """Load an existing table without substituting a file writer."""
    _validate_catalog_name(catalog_kwargs)
    from pyiceberg.catalog import load_catalog
    from pyiceberg.exceptions import NoSuchTableError

    properties = deepcopy(catalog_kwargs)
    name = properties.pop("name")
    try:
        return load_catalog(name, **properties).load_table(table_identifier)
    except NoSuchTableError as e:
        raise ValueError(f"Iceberg target table does not exist: {table_identifier}") from e


def validate_iceberg_schema(schema, table):
    """Use PyIceberg's writer compatibility check before any table mutation."""
    from pyiceberg.io.pyarrow import _check_pyarrow_schema_compatible

    _check_pyarrow_schema_compatible(table.schema(), schema, format_version=table.metadata.format_version)


def align_iceberg_schema(schema, table):
    """Restore top-level required flags that Arrow conversions drop."""
    import pyarrow as pa

    required = {field.name for field in table.schema().fields if field.required}
    fields = [field.with_nullable(False) if field.name in required else field for field in schema]
    return pa.schema(fields, metadata=schema.metadata)


def align_iceberg_nullability(arrow, table):
    """Mark required columns non-nullable after checking they contain no nulls."""
    schema = align_iceberg_schema(arrow.schema, table)
    if schema.equals(arrow.schema):
        return arrow
    for field in arrow.schema:
        if field.nullable and not schema.field(field.name).nullable and arrow.column(field.name).null_count:
            raise ValueError(f"Iceberg required column {field.name} contains null values")
    return arrow.cast(schema)


def iceberg_datasink(table_identifier, catalog_kwargs, mode):
    """Retain Ray's atomic data commit while forbidding schema commits."""
    from ray.data._internal.datasource.iceberg_datasink import IcebergDatasink
    from ray.data._internal.savemode import SaveMode

    _validate_catalog_name(catalog_kwargs)
    if mode not in ("append", "overwrite"):
        raise ValueError("Iceberg export mode must be append or overwrite")
    required_methods = ("write", "on_write_start", "on_write_complete", "_reload_table", "_update_schema")
    if any(not callable(getattr(IcebergDatasink, name, None)) for name in required_methods):
        raise RuntimeError("This Ray IcebergDatasink API is unsupported; use a Ray 2.52-compatible version")

    class FixedSchemaIcebergDatasink(IcebergDatasink):
        """Validate fixed schemas and isolate native Parquet file names."""

        def write(self, blocks, ctx):
            import uuid

            from ray.data.block import BlockAccessor

            # Native write resets PyIceberg's file counter for every block.
            # A fresh invocation UUID also isolates retries of the same task.
            attempt_uuid = uuid.uuid4()
            shared_uuid = self._write_uuid

            def unique_blocks():
                for block_index, block in enumerate(blocks):
                    self._write_uuid = uuid.uuid5(attempt_uuid, str(block_index))
                    yield align_iceberg_nullability(BlockAccessor.for_block(block).to_arrow(), self._table)

            try:
                return super().write(unique_blocks(), ctx)
            finally:
                # Keep the driver's shared UUID and native commit lifecycle intact.
                self._write_uuid = shared_uuid

        def _update_schema(self, incoming_schema):
            self._reload_table()
            validate_iceberg_schema(incoming_schema, self._table)
            update = self._table.update_schema().union_by_name(incoming_schema)
            if update._apply() != self._table.schema():
                raise ValueError("Iceberg export does not support automatic schema evolution")

    sink = FixedSchemaIcebergDatasink(
        table_identifier=table_identifier,
        catalog_kwargs=deepcopy(catalog_kwargs),
        mode=SaveMode.APPEND if mode == "append" else SaveMode.OVERWRITE,
    )
    if any(not hasattr(sink, name) for name in ("_table", "_write_uuid")):
        raise RuntimeError("This Ray IcebergDatasink state API is unsupported; use a Ray 2.52-compatible version")
    return sink
