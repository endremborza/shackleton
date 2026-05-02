# shackleton

[![pypi](https://img.shields.io/pypi/v/shackleton.svg)](https://pypi.org/project/shackleton/)

Persistent tabular store backed by Parquet or Arrow IPC, built on [polars](https://pola.rs).

- **Partitioned writes** — by column value or xxhash bucket
- **Sorted merge** — maintain sorted order on extend via `id_col`
- **Lazy reads** — returns `pl.LazyFrame` so polars can push down predicates
- **Schema evolution** — diagonal concat handles mismatched columns transparently
- **Thread-safe** — per-directory stdlib locks; polars I/O releases the GIL

Single dependency: `polars`.

---

## Install

```
pip install shackleton
```

---

## Quick start

```python
from pathlib import Path
import polars as pl
from shackleton import TableRepo

repo = TableRepo(Path("/tmp/mydata"))

# Write
repo.extend(pl.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]}))
repo.extend(pl.DataFrame({"id": [4, 5], "value": [40.0, 50.0]}))

# Read
df = repo.get_full_df()         # pl.DataFrame — all rows
lf = repo.get_full_lf()         # pl.LazyFrame — push down filters yourself

# Overwrite everything
repo.replace_all(pl.DataFrame({"id": [99], "value": [0.0]}))

# Delete all files
repo.purge()
```

---

## Column partitioning

Data is split into hive-style directories (`col=val/data.parquet`).

```python
repo = TableRepo(Path("/tmp/events"), partition_cols=["region", "env"])

repo.extend(pl.DataFrame({
    "region": ["us", "us", "eu"],
    "env":    ["prod", "dev", "prod"],
    "count":  [100, 50, 80],
}))

# Read only us/prod — no full scan
df = repo.get_partition_df({"region": "us", "env": "prod"})

# Wildcard: all eu partitions
df_eu = repo.get_partition_df({"region": "eu"})
```

---

## Hash partitioning

When partition values have unbounded cardinality, hash into fixed buckets instead.
Uses polars' native xxhash — vectorised, GIL-free.

```python
from shackleton import HashPartitioner

hp = HashPartitioner(col="user_id", num_groups=128)
repo = TableRepo(Path("/tmp/users"), partition_cols=hp)

repo.extend(pl.DataFrame({
    "user_id": ["alice", "bob", "carol", "dave"],
    "score":   [0.9, 0.7, 0.8, 0.6],
}))

df = repo.get_full_df()
# The hash key column is stripped — only original columns are stored
```

---

## Deduplication with `dedup_cols`

When `dedup_cols` is set, each `extend` drops duplicate rows on those columns
after merging with existing data — only the first occurrence is kept. Combine
with `id_col` for sorted-merge dedup; use alone for unordered dedup.
Not supported with `max_records > 0` (raises at construction time).

```python
repo = TableRepo(Path("/tmp/events"), dedup_cols=["event_id"])

repo.extend(pl.DataFrame({"event_id": [1, 2, 3], "val": [10, 20, 30]}))
# Re-ingest with overlapping IDs — existing rows are kept, new ones appended
repo.extend(pl.DataFrame({"event_id": [2, 4], "val": [99, 40]}))

repo.get_full_df().shape  # (4, 2) — event_id 2 kept once
```

---

## Sorted merge with `id_col`

When `id_col` is set, each extend keeps the file sorted by that column
(when `max_records == 0`) and `replace_records` does an upsert: matching IDs
are updated, new IDs appended.

```python
repo = TableRepo(Path("/tmp/prices"), id_col="ticker")

repo.extend(pl.DataFrame({
    "ticker": ["AAPL", "GOOG", "MSFT"],
    "price":  [180.0, 140.0, 380.0],
}))

# Update GOOG, add TSLA
repo.replace_records(pl.DataFrame({
    "ticker": ["GOOG", "TSLA"],
    "price":  [145.0, 220.0],
}))

repo.get_full_df().sort("ticker")
# ┌────────┬───────┐
# │ ticker ┆ price │
# ╞════════╪═══════╡
# │ AAPL   ┆ 180.0 │
# │ GOOG   ┆ 145.0 │
# │ MSFT   ┆ 380.0 │
# │ TSLA   ┆ 220.0 │
# └────────┴───────┘
```

---

## File size limits with `max_records`

Cap each file at N rows. Useful when downstream tools (Spark, DuckDB) benefit
from many smaller files, or when partial reads matter.

```python
repo = TableRepo(Path("/tmp/logs"), max_records=100_000)

for batch in my_log_stream():
    repo.extend(batch)
# Creates file-00000000000000000000.parquet, file-00000000000000000001.parquet, …
```

---

## Streaming ingestion with batch writers

For row-at-a-time or small-DataFrame pipelines: accumulate in memory, flush in
batches.

```python
# Dict record writer — for APIs / scrapers
with repo.get_extending_record_writer(batch_size=50_000) as writer:
    for event in event_stream():
        writer.add_to_batch({"ts": event.ts, "value": event.value})
# Final partial batch is flushed on context exit

# DataFrame writer — for chunked ETL
with repo.get_extending_df_writer(batch_size=500_000) as writer:
    for chunk_df in chunked_source():
        writer.add_to_batch(chunk_df)

# Replacing variants — upsert on each flush
with repo.get_replacing_record_writer(batch_size=10_000) as writer:
    for record in updates():
        writer.add_to_batch(record)
```

---

## Schema evolution

Columns added in later writes fill missing values with `null` in earlier files.
No migration step needed.

```python
repo = TableRepo(Path("/tmp/evolving"))
repo.extend(pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
repo.extend(pl.DataFrame({"a": [5], "c": [6]}))        # new column c

df = repo.get_full_df()
# shape: (3, 3) — columns a, b, c — nulls where data didn't exist
```

---

## Partition-level overwrite

`replace_partition` atomically clears and rewrites only the partitions covered
by the DataFrame. `purge_partition` deletes matching files without rewriting.

```python
repo = TableRepo(Path("/tmp/events"), partition_cols=["date"])
repo.extend(pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
    "val":  [1, 2, 3],
}))

# Replace just the 2024-01-01 partition with corrected data
repo.replace_partition(pl.DataFrame({
    "date": ["2024-01-01"],
    "val":  [99],
}))

# Or wipe it without rewriting
repo.purge_partition({"date": "2024-01-01"})
```

---

## Compression

Control the codec and, for Parquet, the compression level.

```python
repo = TableRepo(Path("/tmp/data"), compression="zstd", compression_level=3)
repo_ipc = TableRepo(Path("/tmp/fast"), ipc=True, compression="lz4")
```

---

## Arrow IPC format

Slightly faster reads for local workloads that don't need Parquet compatibility.

```python
repo = TableRepo(Path("/tmp/fast"), ipc=True)   # writes .arrow files
repo.extend(df)
df = repo.get_full_df()
```

---

## Lazy reads and predicate pushdown

`get_full_lf()` and `get_partition_lf()` return `pl.LazyFrame`. Polars will
push predicates and column selections into the scan.

```python
result = (
    repo.get_full_lf()
    .filter(pl.col("region") == "us")
    .select(["ticker", "price"])
    .sort("price", descending=True)
    .limit(10)
    .collect()
)
```

---

## Parallel partition processing with `map_partitions`

Apply a function to each partition's collected DataFrame in parallel. All files
within a partition are combined before your function is called, so it always
receives one complete DataFrame per partition.

```python
repo = TableRepo(Path("/tmp/events"), partition_cols=["region"])
repo.extend(pl.DataFrame({
    "region": ["us", "us", "eu", "eu", "ap"],
    "revenue": [100.0, 200.0, 150.0, 50.0, 300.0],
}))

# Sequential (workers=None, default)
totals = repo.map_partitions(lambda df: df["revenue"].sum())
# e.g. [300.0, 200.0, 300.0]  — one value per partition

# Parallel with auto-detected thread count
totals = repo.map_partitions(lambda df: df["revenue"].sum(), workers=0)

# Parallel with fixed thread pool
totals = repo.map_partitions(lambda df: df["revenue"].sum(), workers=4)
```

The `workers` parameter controls parallelism:

| `workers` | Behaviour |
|---|---|
| `None` | Sequential — no thread pool |
| `0` | Thread pool sized to `os.cpu_count()` |
| `> 0` | Thread pool of exactly that size |

Because polars I/O releases the GIL, threads overlap on I/O with no contention.

The function can return any type; results preserve partition order.

```python
# Compute per-partition summary stats
def summarise(df: pl.DataFrame) -> dict:
    return {"n": len(df), "mean": df["revenue"].mean()}

stats = repo.map_partitions(summarise, workers=0)
# [{"n": 2, "mean": 150.0}, {"n": 2, "mean": 100.0}, {"n": 1, "mean": 300.0}]

# Collect into a single DataFrame
import polars as pl
result = pl.DataFrame(repo.map_partitions(summarise, workers=0))
```

---

## Performance

Benchmarked on a single core, NVMe SSD, 1 million rows (4 columns: i64, f64, str, str).
Run `uv run bench.py` to reproduce.

| Scenario | Throughput |
|---|---|
| extend — single file | ~33M rows/s |
| extend — sorted merge (`id_col`) | ~49M rows/s |
| extend — 26 column partitions | ~11M rows/s |
| extend — 64 hash partitions | ~7M rows/s |
| extend — `max_records=10k` | ~7M rows/s |
| extend — IPC format | ~23M rows/s |
| read full (collect) | ~131M rows/s |
| read full (lazy + filter) | ~170M rows/s |
| read single partition (1 of 26) | ~18M rows/s |
| read full — IPC format | ~102M rows/s |
| replace_records (10k upserts) | ~107k rows/s |

`replace_records` does a full read-merge-rewrite, so throughput scales with
existing file size, not batch size. Use partitioned repos to bound the merge scope.
