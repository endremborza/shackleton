# Possible extensions

Design constraint: shackleton stays lean — no new mandatory dependencies.
Extensions are listed roughly by value/complexity ratio.

---

## 1. `map_partitions(fn)` — per-partition transform

Apply a function to each partition's collected `DataFrame` and collect results.
Parquetranger had this as `map_partitions` with optional `parallel_map`.

```python
# sketch
def map_partitions(self, fn: Callable[[pl.DataFrame], T]) -> list[T]:
    return [fn(lf.collect()) for lf in self.lfs]
```

Useful for per-partition aggregations, model scoring, or rewrite-in-place
transforms. Parallel variant could use `concurrent.futures.ThreadPoolExecutor`
(polars releases the GIL).

**Complexity:** low. **Value:** high for analytical pipelines.

---

## 2. `drop_partition(values)` — delete a single partition

Delete files belonging to one partition without purging the entire store.

```python
repo.drop_partition({"region": "us", "env": "dev"})
```

Implemented as a targeted glob + `Path.unlink`. No schema tracking needed.

**Complexity:** very low. **Value:** medium (ETL reruns, data corrections).

---

## 3. File compaction — `compact(target_rows)`

Merge small files within a partition into fewer larger ones.

```python
repo.compact(target_rows=500_000)
```

Read all files in a partition, concat, rewrite in target-sized chunks.
Critical when many small extends cause file count to grow unboundedly with
`max_records` set.

**Complexity:** low. **Value:** high for long-running append pipelines.

---

## 4. Process-level file locking

Current locks are thread-local (`threading.Lock`). Multi-process writers
(e.g. `multiprocessing`, separate worker processes) would race.

Replace or augment with `fcntl.flock` (POSIX) or `filelock` (cross-platform).
`filelock` is tiny, mature, and cross-platform — the one dependency worth adding.

**Complexity:** low. **Value:** high if workers run in separate processes.

---

## 5. Lightweight partition index

A small sidecar file (e.g. `_index.arrow`) tracking row counts and min/max
per partition without reading data files. Enables cheap `n_rows` and
partition pruning without file I/O.

```python
repo.n_rows              # fast: reads index only
repo.partition_stats()   # min/max of id_col per partition
```

**Complexity:** medium (must be kept consistent on every write). **Value:** medium.

---

## 6. `FixedRecordWriter` — project to known columns

Parquetranger had `FixedRecordWriter`: each dict is projected to a fixed column
set before batching. Keys not in `cols` are dropped; missing keys produce `null`.

```python
with repo.get_extending_record_writer(cols=["ts", "value", "tag"]) as w:
    for raw in messy_stream():
        w.add_to_batch(raw)  # extra keys silently dropped
```

Avoids schema explosion from inconsistent upstream sources.

**Complexity:** trivial. **Value:** medium for schema-strict pipelines.

---

## 7. Cloud storage backend

Polars `scan_parquet` / `scan_ipc` accept `s3://`, `gs://`, `az://` paths
natively (via `object_store` feature). `TableRepo` could accept string paths
and pass them through, with the constraint that writes use polars' `sink_parquet`
(streaming) or `write_parquet` on a collected frame.

Main friction: `Path` must become `str | Path`; directory listing needs
`fsspec` or cloud-SDK calls instead of `Path.glob`.

**Complexity:** medium. **Value:** high for cloud-native pipelines.

---

## 8. Time-range partition helpers

Convenience wrappers for the common case of partitioning by date:

```python
repo = TableRepo(root, partition_cols=["year", "month", "day"])
repo.extend_with_date(df, date_col="ts")   # auto-extracts year/month/day
repo.get_date_range_lf("2024-01-01", "2024-03-31")
```

Pure sugar over existing `partition_cols` — no core changes needed.

**Complexity:** low. **Value:** high for time-series workloads.

---

## 9. Async write support

`async def aextend(self, df)` wrapping the synchronous write in
`asyncio.get_event_loop().run_in_executor`. No new dependencies; polars
releases the GIL so thread-pool execution is safe.

**Complexity:** low. **Value:** medium for async pipelines (FastAPI ingestors, etc.).

---

## Non-goals (intentionally excluded)

- **Pandas/PyArrow support** — polars only; parquetranger covers the pandas path.
- **`env_parents` / `env_ctx`** — environment switching added complexity for
  little gain; solved at the call-site by constructing repos with different paths.
- **Parquet metadata embedding** — `extra_metadata` pickle storage in parquet
  file headers is fragile; use a sidecar file if metadata is needed (see §5).
- **`atqo` / parallel_map dependency** — stdlib `ThreadPoolExecutor` is sufficient
  for GIL-free polars I/O.
