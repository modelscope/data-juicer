# Checkpoint commit protocol

This document describes how `ray_partitioned` checkpoints the leading
Mapper/Filter operators of a recipe (`checkpoint.strategy: every_op`). For
configuration and usage see [PartitionAndCheckpoint.md](PartitionAndCheckpoint.md).

## Data path

All operators run in one Ray Data plan, and rows move between them through
the Ray object store exactly as without checkpoints. Checkpointing is a side
path: a worker that finishes an outer batch of `partition.size` rows writes its
output to storage, commits it, and only then yields the batch downstream. A
downstream commit therefore never references a row whose producing batch is
not committed.

## Lineage keys

Every row carries two internal columns while it is inside the plan:

- `__dj_ckpt_seq__`: the commit that produced the row, or `-1` for an input row;
- `__dj_ckpt_idx__`: the row's position within that commit (the input offset for
  input rows).

Before an operator runs, the worker records the batch's keys as compact
ranges `(seq, start, stop)` and drops the columns, so operators never see them.
After the commit, the output is tagged with the new commit's sequence number and
local positions. The columns are removed before the export.

## Commits

A commit is a small JSON manifest at `commits/{stage:04d}-{seq:016d}.json`
recording:

- the operator stage and its sequence number;
- the consumed key ranges;
- the produced Parquet files under `data/`, each with row count and SHA-256;
- the Filter statistics files, if the stage exports statistics.

Commits are not planned ahead. Their number and boundaries follow the batches
that Ray Data actually forms, so a Mapper may expand rows and a Filter may drop
all of them; an empty output still commits its consumed ranges.

A Ray actor pinned to the driver node coordinates commits. It handles metadata
only; data files are written by the workers. Under a lock it checks the consumed
ranges against the stage's committed ranges:

- no overlap: a new sequence number is assigned and the manifest is written
  with a create-if-absent operation;
- an identical consumed set: the existing commit is returned, and the worker
  yields that commit's files instead of its own. This covers Ray task retries
  and lineage reconstruction, because consumers may already hold the first
  result;
- any other overlap: the run fails, since the same row would be counted twice.

If a manifest write fails, the coordinator stops accepting commits and the run
must be resumed.

## Storage

`LocalStore` writes to a temporary file, fsyncs it, publishes with `os.link`
(create if absent) or `os.replace`, and fsyncs the directory. `S3Store` uses
`boto3` `put_object` with `IfNoneMatch='*'`, which AWS S3 and S3-compatible
services such as MinIO implement as a conditional create. Other URI schemes are
rejected because they do not provide that guarantee here.

## Run identity

`run.json` records the protocol version, an order-sensitive fingerprint of the
input computed on Ray workers, the row count, the Arrow schema, a hash of the
checkpointed operators' configuration, and the stage count. The input is not
copied. A new run fails if commits already exist; a resume fails if the input,
its order or the operator configuration changed, or if commits exist without
`run.json`. The commit size is not part of the identity.

## Resume

Loading every manifest gives, per stage, the committed consumed ranges. The
resumed plan is built stage by stage:

- the input of stage 0 is the input rows that stage 0 has not consumed;
- the input of stage `k` is the stream from stage `k - 1` plus the committed
  stage `k - 1` rows that stage `k` has not consumed, read from storage on the
  workers;
- the output is the stream from the last stage plus all committed last-stage
  rows.

The final map records every delivered key. After the export, the coordinator
verifies that every stage consumed all rows produced by its predecessor and
that every final row was delivered. Statistics are published only after that
check.

## Export

The export streams with the operators, so an interrupted run usually leaves part
of it behind, and a resumed run writes the complete output again. Each run
registers an export attempt, `exports/{n:08d}.json`, and names its files
`dj{n:04d}-{id}_*`:

1. before writing, files of attempts older than the previous one are removed,
   so repeated failures keep at most one earlier partial export;
2. the attempt writes the complete output and the delivery check above runs;
3. only then are the files of every earlier attempt removed.

A failed resume therefore never leaves less exported output than it found.
Files in the export directory that no attempt of the job wrote are kept, with a
warning. Formats that Ray does not write as named files (Lance) cannot replace
a partial export; resume stops until it is removed.

## Scope

Only the leading Mapper/Filter operators are checkpointed. A Deduplicator,
`Pipeline`, shared-state Filter, or `converge_after` operator needs all of its
input at once; it and every later operator run without checkpoints on the
merged output. External side effects of operators are outside this contract.
