# shackleton

Polars-based persistent tabular store backed by Parquet or Arrow IPC files.

## `DfBatchWriter`
Accumulates DataFrames and flushes to a ``TableRepo`` in batches.

    Record count is tracked by row count across accumulated DataFrames.

    Args:
        trepo: Target repository.
        batch_size: Flush when total accumulated rows reach this threshold.
        writer_fn: ``TableRepo`` method to call on flush. Defaults to ``extend``.

### `add_to_batch(self, df: polars.dataframe.frame.DataFrame) -> None`

### `close(self) -> None`

## `HashPartitioner`
Partition rows by hashing a column into `num_groups` zero-padded buckets.

    Uses polars' native xxhash for vectorised, GIL-free partitioning.
    The computed column (``key``) is stripped before writing to files.

### `add_to_df(self, df: polars.dataframe.frame.DataFrame) -> polars.dataframe.frame.DataFrame`

## `RecordWriter`
Accumulates dict records and flushes to a ``TableRepo`` in batches.

    Use as a context manager to guarantee the final partial batch is flushed.

    Args:
        trepo: Target repository.
        batch_size: Flush when this many records have been accumulated.
        writer_fn: ``TableRepo`` method to call on flush. Defaults to ``extend``.

### `add_to_batch(self, record: dict) -> None`

### `close(self) -> None`

## `TableRepo`
Persistent tabular store backed by Parquet or Arrow IPC files.

    Writes are partitioned by column(s) or hash bucket, with optional per-file
    record limits. Thread-safe: concurrent extends are serialised per directory
    via stdlib locks (polars I/O releases the GIL).

    Args:
        root_path: Root directory for all data files.
        id_col: Column used as a unique row identifier. Required for
            ``replace_records``; enables sorted merge on extend.
        partition_cols: Column name(s) to partition by, or a
            ``HashPartitioner`` for hash-based bucketing. Partition values
            are encoded as hive-style directories (``col=val/``).
        max_records: Maximum rows per file. ``0`` means one file per
            partition (unlimited rows). When ``> 0``, files are named
            ``file-<zero-padded-index><ext>``.
        ipc: Use Arrow IPC format instead of Parquet.
        compression: Compression codec passed to the writer
            (e.g. ``"zstd"``, ``"gzip"``). ``None`` uses the writer default.

### `compact(self, target_rows: int = 500000) -> None`
Merge small files within each partition into ``target_rows``-sized files.

        Partitions with a single file are skipped. Useful after many small
        extends when ``max_records`` has caused file count to grow unboundedly.

### `extend(self, df: polars.dataframe.frame.DataFrame) -> None`
Append rows to the store.

### `get_extending_df_writer(self, batch_size: int = 1000000) -> 'DfBatchWriter'`
Writer that batches DataFrames and flushes via ``extend``.

### `get_extending_record_writer(self, batch_size: int = 1000000) -> 'RecordWriter'`
Writer that batches dicts and flushes via ``extend``.

### `get_full_df(self) -> polars.dataframe.frame.DataFrame`
Collect all data into a single DataFrame.

### `get_full_lf(self) -> polars.lazyframe.frame.LazyFrame`
Lazy frame over all data in the store.

### `get_partition_df(self, partitions: dict[str, typing.Any]) -> polars.dataframe.frame.DataFrame`
Collect partitions matching ``partitions`` dict into a DataFrame.

### `get_partition_lf(self, partitions: dict[str, typing.Any]) -> polars.lazyframe.frame.LazyFrame`
Lazy frame over partitions matching ``partitions`` dict.

        Keys absent from ``partitions`` are treated as wildcards.
        For ``HashPartitioner`` repos, use ``repo.partition_cols.key``
        as the dict key with the zero-padded bucket string as the value.

### `get_replacing_df_writer(self, batch_size: int = 1000000) -> 'DfBatchWriter'`
Writer that batches DataFrames and flushes via ``replace_records``.

### `get_replacing_record_writer(self, batch_size: int = 1000000) -> 'RecordWriter'`
Writer that batches dicts and flushes via ``replace_records``.

### `map_partitions(self, fn: Callable[[polars.dataframe.frame.DataFrame], ~T], *, workers: int | None = None) -> list[~T]`
Apply ``fn`` to each partition's collected DataFrame; return results.

        All files within a partition directory are combined before ``fn`` is
        called, so ``fn`` always receives a single DataFrame per partition.

        Args:
            fn: Function applied to each partition's collected DataFrame.
            workers: Thread-pool size. ``None`` = sequential. ``0`` uses
                ``os.cpu_count()`` threads; positive int fixes the pool size.

### `purge(self) -> None`
Delete all data files.

### `replace_all(self, df: polars.dataframe.frame.DataFrame) -> None`
Purge and rewrite entirely with ``df``.

### `replace_records(self, df: polars.dataframe.frame.DataFrame) -> None`
Update rows with matching ``id_col``; append the rest.

        Requires ``id_col`` to be set. For multi-file partitions
        (``max_records > 0``) performs a full read-merge-rewrite.

## `TableShack`
Persistent tabular store backed by Parquet or Arrow IPC files.

    Writes are partitioned by column(s) or hash bucket, with optional per-file
    record limits. Thread-safe: concurrent extends are serialised per directory
    via stdlib locks (polars I/O releases the GIL).

    Args:
        root_path: Root directory for all data files.
        id_col: Column used as a unique row identifier. Required for
            ``replace_records``; enables sorted merge on extend.
        partition_cols: Column name(s) to partition by, or a
            ``HashPartitioner`` for hash-based bucketing. Partition values
            are encoded as hive-style directories (``col=val/``).
        max_records: Maximum rows per file. ``0`` means one file per
            partition (unlimited rows). When ``> 0``, files are named
            ``file-<zero-padded-index><ext>``.
        ipc: Use Arrow IPC format instead of Parquet.
        compression: Compression codec passed to the writer
            (e.g. ``"zstd"``, ``"gzip"``). ``None`` uses the writer default.

### `compact(self, target_rows: int = 500000) -> None`
Merge small files within each partition into ``target_rows``-sized files.

        Partitions with a single file are skipped. Useful after many small
        extends when ``max_records`` has caused file count to grow unboundedly.

### `extend(self, df: polars.dataframe.frame.DataFrame) -> None`
Append rows to the store.

### `get_extending_df_writer(self, batch_size: int = 1000000) -> 'DfBatchWriter'`
Writer that batches DataFrames and flushes via ``extend``.

### `get_extending_record_writer(self, batch_size: int = 1000000) -> 'RecordWriter'`
Writer that batches dicts and flushes via ``extend``.

### `get_full_df(self) -> polars.dataframe.frame.DataFrame`
Collect all data into a single DataFrame.

### `get_full_lf(self) -> polars.lazyframe.frame.LazyFrame`
Lazy frame over all data in the store.

### `get_partition_df(self, partitions: dict[str, typing.Any]) -> polars.dataframe.frame.DataFrame`
Collect partitions matching ``partitions`` dict into a DataFrame.

### `get_partition_lf(self, partitions: dict[str, typing.Any]) -> polars.lazyframe.frame.LazyFrame`
Lazy frame over partitions matching ``partitions`` dict.

        Keys absent from ``partitions`` are treated as wildcards.
        For ``HashPartitioner`` repos, use ``repo.partition_cols.key``
        as the dict key with the zero-padded bucket string as the value.

### `get_replacing_df_writer(self, batch_size: int = 1000000) -> 'DfBatchWriter'`
Writer that batches DataFrames and flushes via ``replace_records``.

### `get_replacing_record_writer(self, batch_size: int = 1000000) -> 'RecordWriter'`
Writer that batches dicts and flushes via ``replace_records``.

### `map_partitions(self, fn: Callable[[polars.dataframe.frame.DataFrame], ~T], *, workers: int | None = None) -> list[~T]`
Apply ``fn`` to each partition's collected DataFrame; return results.

        All files within a partition directory are combined before ``fn`` is
        called, so ``fn`` always receives a single DataFrame per partition.

        Args:
            fn: Function applied to each partition's collected DataFrame.
            workers: Thread-pool size. ``None`` = sequential. ``0`` uses
                ``os.cpu_count()`` threads; positive int fixes the pool size.

### `purge(self) -> None`
Delete all data files.

### `replace_all(self, df: polars.dataframe.frame.DataFrame) -> None`
Purge and rewrite entirely with ``df``.

### `replace_records(self, df: polars.dataframe.frame.DataFrame) -> None`
Update rows with matching ``id_col``; append the rest.

        Requires ``id_col`` to be set. For multi-file partitions
        (``max_records > 0``) performs a full read-merge-rewrite.
