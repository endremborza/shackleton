# Performance

## Running Benchmarks

```bash
uv run bench.py
uv run bench.py --rows <n>
```

Rows default to `DEFAULT_ROWS` in `bench.py`; columns are an int id, a float and two strings.

## Metrics

All measured in rows/second.

| Operation | Baseline (`DEFAULT_ROWS`, polars 2.0) |
| --- | --: |
| Single-file extend | ~50M rows/s |
| Extend into stored rows (id_col merge) | ~17M rows/s |
| Partitioned write (`N_CATEGORIES` partitions) | ~9.4M rows/s |
| Hash-partitioned write (`N_BUCKETS` buckets) | ~5.5M rows/s |
| Extend with `max_records=MAX_RECORDS` | ~11M rows/s |
| IPC extend | ~30M rows/s |
| Full read (collect) | ~127M rows/s |
| Lazy read + filter | ~158M rows/s |
| Single partition read | ~19M rows/s |
| IPC full read | ~122M rows/s |
| Replace records (`N_UPSERTS` upserts) | ~248k rows/s |

## Known Bottlenecks

- `extend` into a stored file and `replace_records` read-merge-rewrite the partition, so their cost scales with the stored partition, not the batch: an append-heavy store partitions so that each partition stays small or is written once
- With `max_records` unset, a merge with stored data streams through polars' sink; with `max_records > 0`, `replace_records` and `compact` collect the whole partition in memory
- Hash-partitioned writes scale sub-linearly with bucket count due to per-partition file overhead
- Partition reads are limited by parquet file open latency when partitions are small

## What to Watch

- Polars version upgrades can significantly affect parquet I/O throughput
- Row group size interacts with partition count — too many small files hurts read performance
- The GIL-releasing path through polars I/O makes thread-pool scaling important for parallel reads
