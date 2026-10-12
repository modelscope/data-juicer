# Streaming recovery units

`partition.recovery_mode: streaming` in `ray_partitioned` enables durable recovery units for Mapper/Filter chains when checkpointing is enabled. A recovery unit is a fixed range of ordered input rows, not a Ray block or an operator output block. Mappers may expand a unit and Filters may remove every row without changing the unit's identity.

## Input identity and unit boundaries

The executor materializes the input as Ray data, computes an order-sensitive fingerprint on Ray workers, and persists only a small `manifest.json` containing the fingerprint, row count, Arrow schema, and unit size. It does not copy the input into the checkpoint directory. On resume, it reads and fingerprints the source again before reusing committed work. Changed content, order, schema, operator configuration, or an explicitly configured unit size fails closed.

The input is assigned dense offsets and strictly repartitioned into exact unit-sized Ray blocks in the object store. The last unit may be shorter. `partition.unit_size` accepts a positive integer; `auto` targets about 30 seconds of measured single-actor work, clamped to 64–2048 rows, and falls back to 512 rows. If `partition.size` is explicitly set, it supplies the automatic unit size.

## Ticket pipeline

Each operator is one stage in the native `RayDataset.process` plan. Stage 0 consumes one exact input unit. Later stages consume one-row tickets containing the unit range, completed stage, and committed attempt ID. A worker validates the ticket, reads only the committed upstream files, runs the native operator batch plan, and emits the next ticket after committing its result. Operator actors stay alive across units, and units may advance independently between stages.

Each `(unit, stage)` has at most one committed attempt. Workers write immutable attempt-owned Parquet files, fsync them, and ask a coordinator to commit the result in SQLite. A generation fence rejects late workers from older attempts. An all-filtered unit commits successfully with no output file. Parquet files coalesce producer batches into larger row groups while recording their original row boundaries so downstream stages recover the same batching semantics.

Filter statistics are written as attempt-owned files and published only after final ticket validation. Operators with shared cross-unit state are rejected because they require a separate replay contract.

## Resume behavior

At startup, the executor reconstructs tickets through the latest stage completed by every unit. A claimed `(unit, stage)` that is already committed skips computation and forwards the committed ticket. An unfinished attempt receives a new generation and is recomputed. Final collection verifies exactly one final ticket for every unit, then reads committed final-stage files in input-unit order.

A crash before the input manifest and SQLite plan are published leaves no reusable work, so restart removes the incomplete temporary manifest and rebuilds it. A checkpoint created by the retired tee-sink protocol is rejected rather than mixed with recovery-unit state.

## Storage and deployment constraints

The checkpoint contains:

- `source/manifest.json`: input identity only;
- `state/units.sqlite3`: plan, attempts, generations, and committed winners;
- `state/attempts/<attempt_id>/*.parquet`: committed or orphaned attempt output and statistics.

The current protocol requires every Ray worker to see the same POSIX checkpoint directory because it relies on local SQLite transactions, hard links, and fsync. A multi-node deployment therefore needs a shared POSIX filesystem. Supporting object storage would require a conditional-commit metadata service and a different file publication protocol.

The principal runtime costs are full input fingerprinting, per-stage durable output, SQLite coordination, and reading committed upstream files between stages. Cross-unit state and external side effects remain outside this retry contract unless the operator provides its own idempotent or transactional recovery mechanism.
