# Performance

## Running Benchmarks

```bash
uv run bench.py
uv run bench.py --rows 5000000
```

Default: 1M rows, 5 columns (int, float, str, date, bool).

## Metrics

All measured in rows/second.

| Operation | Baseline (1M rows) |
|-----------|-------------------|
| Single-file extend | ~33M rows/s |
| Extend with sorted merge (id_col) | ~49M rows/s |
| Partitioned write (26 partitions) | ~11M rows/s |
| Hash-partitioned write (64 buckets) | ~7M rows/s |
| Full read (collect) | ~131M rows/s |
| Lazy read + filter | ~170M rows/s |
| Single partition read | ~18M rows/s |
| Replace records (10k upserts) | ~107k rows/s |

## Known Bottlenecks

- `replace_records` requires a full merge — O(n) in total rows, not O(k) in replacement size
- Hash-partitioned writes scale sub-linearly with bucket count due to per-partition file overhead
- Partition reads are limited by parquet file open latency when partitions are small

## What to Watch

- Polars version upgrades can significantly affect parquet I/O throughput
- Row group size interacts with partition count — too many small files hurts read performance
- The GIL-releasing path through polars I/O makes thread-pool scaling important for parallel reads
