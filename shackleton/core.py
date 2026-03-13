import contextlib
import fcntl
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import polars as pl

T = TypeVar("T")

FILE_NAME = "data"

_PATH_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    with _LOCKS_LOCK:
        if key not in _PATH_LOCKS:
            _PATH_LOCKS[key] = threading.Lock()
        return _PATH_LOCKS[key]


@contextlib.contextmanager
def _dir_lock(path_str: str):
    """Combined thread + process lock for a directory.

    Acquires an in-process threading.Lock first, then an fcntl.LOCK_EX on a
    ``.lock`` sidecar file so that concurrent writers in separate processes are
    also serialised.
    """
    with _get_lock(path_str):
        lock_path = Path(path_str) / ".lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield


@dataclass
class HashPartitioner:
    """Partition rows by hashing a column into `num_groups` zero-padded buckets.

    Uses polars' native xxhash for vectorised, GIL-free partitioning.
    The computed column (``key``) is stripped before writing to files.
    """

    col: str
    num_groups: int = 128

    @property
    def key(self) -> str:
        return f"__pqr-hash-{self.col}-{self.num_groups}__"

    def add_to_df(self, df: pl.DataFrame) -> pl.DataFrame:
        width = len(str(self.num_groups - 1))
        return df.with_columns(
            (pl.col(self.col).cast(pl.String).hash() % self.num_groups)
            .cast(pl.String)
            .str.zfill(width)
            .alias(self.key)
        )


@dataclass
class TableRepo:
    """Persistent tabular store backed by Parquet or Arrow IPC files.

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
    """

    root_path: Path
    id_col: str | None = None
    partition_cols: list[str] | HashPartitioner = field(default_factory=list)
    max_records: int = 0
    ipc: bool = False
    compression: str | None = None

    @cached_property
    def extension(self) -> str:
        return ".arrow" if self.ipc else ".parquet"

    # --- write ---

    def extend(self, df: pl.DataFrame) -> None:
        """Append rows to the store."""
        self._for_each_partition(df, self._extend_dir)

    def replace_records(self, df: pl.DataFrame) -> None:
        """Update rows with matching ``id_col``; append the rest.

        Requires ``id_col`` to be set. For multi-file partitions
        (``max_records > 0``) performs a full read-merge-rewrite.
        """
        assert self.id_col is not None, "replace_records requires id_col"
        self._for_each_partition(df, self._replace_dir)

    def replace_all(self, df: pl.DataFrame) -> None:
        """Purge and rewrite entirely with ``df``."""
        self.purge()
        self.extend(df)

    def purge(self) -> None:
        """Delete all data files."""
        for p in self.paths:
            p.unlink()

    # --- read ---

    def get_full_lf(self) -> pl.LazyFrame:
        """Lazy frame over all data in the store."""
        parts = list(self.paths)
        if not parts:
            return pl.LazyFrame()
        return pl.concat(map(self._lazy_read, parts), how="diagonal")

    def get_full_df(self) -> pl.DataFrame:
        """Collect all data into a single DataFrame."""
        return self.get_full_lf().collect()

    def get_partition_lf(self, partitions: dict[str, Any]) -> pl.LazyFrame:
        """Lazy frame over partitions matching ``partitions`` dict.

        Keys absent from ``partitions`` are treated as wildcards.
        For ``HashPartitioner`` repos, use ``repo.partition_cols.key``
        as the dict key with the zero-padded bucket string as the value.
        """
        paths = list(self._partition_paths(partitions))
        if not paths:
            return pl.LazyFrame()
        return pl.concat(map(self._lazy_read, paths), how="diagonal")

    def get_partition_df(self, partitions: dict[str, Any]) -> pl.DataFrame:
        """Collect partitions matching ``partitions`` dict into a DataFrame."""
        return self.get_partition_lf(partitions).collect()

    # --- iteration ---

    @property
    def paths(self) -> Iterable[Path]:
        """All data file paths in the store."""
        return self.root_path.glob("**/*" + self.extension)

    @property
    def n_files(self) -> int:
        """Number of data files currently stored."""
        return len(list(self.paths))

    @property
    def lfs(self) -> Iterable[pl.LazyFrame]:
        """Lazy frame per data file."""
        return map(self._lazy_read, self.paths)

    @property
    def dfs(self) -> Iterable[pl.DataFrame]:
        """Collected DataFrame per data file."""
        return map(pl.LazyFrame.collect, self.lfs)

    # --- batch writers ---

    def get_extending_record_writer(
        self, batch_size: int = 1_000_000
    ) -> "RecordWriter":
        """Writer that batches dicts and flushes via ``extend``."""
        return RecordWriter(self, batch_size)

    def get_replacing_record_writer(
        self, batch_size: int = 1_000_000
    ) -> "RecordWriter":
        """Writer that batches dicts and flushes via ``replace_records``."""
        return RecordWriter(self, batch_size, TableRepo.replace_records)

    def get_extending_df_writer(self, batch_size: int = 1_000_000) -> "DfBatchWriter":
        """Writer that batches DataFrames and flushes via ``extend``."""
        return DfBatchWriter(self, batch_size)

    def get_replacing_df_writer(self, batch_size: int = 1_000_000) -> "DfBatchWriter":
        """Writer that batches DataFrames and flushes via ``replace_records``."""
        return DfBatchWriter(self, batch_size, TableRepo.replace_records)

    # --- partition-level operations ---

    def map_partitions(
        self,
        fn: Callable[[pl.DataFrame], T],
        *,
        workers: int | None = None,
    ) -> list[T]:
        """Apply ``fn`` to each partition's collected DataFrame; return results.

        All files within a partition directory are combined before ``fn`` is
        called, so ``fn`` always receives a single DataFrame per partition.

        Args:
            fn: Function applied to each partition's collected DataFrame.
            workers: Thread-pool size. ``None`` = sequential. ``0`` uses
                ``os.cpu_count()`` threads; positive int fixes the pool size.
        """
        dirs = self._partition_dirs
        if not dirs:
            return []

        def _apply(d: Path) -> T:
            files = sorted(d.glob("*" + self.extension))
            lf = pl.concat([self._lazy_read(f) for f in files], how="diagonal")
            return fn(lf.collect())

        if workers is None:
            return [_apply(d) for d in dirs]

        n = workers if workers > 0 else None
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(_apply, dirs))

    def compact(self, target_rows: int = 500_000) -> None:
        """Merge small files within each partition into ``target_rows``-sized files.

        Partitions with a single file are skipped. Useful after many small
        extends when ``max_records`` has caused file count to grow unboundedly.
        """
        for d in self._partition_dirs:
            with _dir_lock(str(d)):
                files = sorted(d.glob("*" + self.extension))
                if len(files) <= 1:
                    continue
                df = pl.concat(
                    [self._lazy_read(f) for f in files], how="diagonal"
                ).collect()
                for f in files:
                    f.unlink()
                for idx, offset in enumerate(range(0, max(len(df), 1), target_rows)):
                    self._write(
                        df.slice(offset, target_rows),
                        d / f"file-{idx:020d}{self.extension}",
                    )

    # --- internals ---

    def _lazy_read(self, path: Path) -> pl.LazyFrame:
        lf = (
            pl.scan_ipc(path)
            if self.ipc
            else pl.scan_parquet(path, hive_partitioning=False)
        )
        return lf.set_sorted(self.id_col) if self.id_col else lf

    def _write(self, df: pl.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_fn = pl.DataFrame.write_ipc if self.ipc else pl.DataFrame.write_parquet
        if self.compression:
            write_fn = partial(write_fn, compression=self.compression)
        write_fn(df, path)

    def _merge_extend(self, new: pl.LazyFrame, old: pl.LazyFrame) -> pl.LazyFrame:
        if self.id_col:
            return old.merge_sorted(new.sort(self.id_col), key=self.id_col)
        return pl.concat([old, new], how="diagonal")

    def _merge_replace(self, new: pl.LazyFrame, old: pl.LazyFrame) -> pl.LazyFrame:
        return (
            old.merge_sorted(new.sort(self.id_col), key=self.id_col)
            .unique(subset=self.id_col, keep="last")
            .sort(self.id_col)
        )

    def _for_each_partition(self, df: pl.DataFrame, handler: Callable) -> None:
        keys = self._part_keys
        if not keys:
            handler(df, self.root_path)
            return

        is_hash = isinstance(self.partition_cols, HashPartitioner)
        if is_hash:
            df = self.partition_cols.add_to_df(df)

        for group_vals, group_df in df.group_by(keys):
            group_vals = group_vals if isinstance(group_vals, tuple) else (group_vals,)
            hive_parts = [f"{k}={v}" for k, v in zip(keys, group_vals)]
            part_dir = Path(self.root_path, *hive_parts)
            if is_hash:
                group_df = group_df.drop(keys)
            handler(group_df, part_dir)

    def _extend_dir(self, df: pl.DataFrame, root: Path) -> None:
        with _dir_lock(str(root)):
            if self.max_records > 0:
                existing = sorted(root.glob("*" + self.extension))
                self._append_parts(df, root, existing)
            else:
                path = self._single_path(root)
                if not path.exists():
                    out = df.sort(self.id_col) if self.id_col else df
                else:
                    out = self._merge_extend(df.lazy(), self._lazy_read(path)).collect()
                self._write(out, path)

    def _replace_dir(self, df: pl.DataFrame, root: Path) -> None:
        with _dir_lock(str(root)):
            existing = sorted(root.glob("*" + self.extension))
            if not existing:
                self._write(df.sort(self.id_col), self._single_path(root))
                return
            if self.max_records > 0:
                old = pl.concat(
                    [self._lazy_read(p) for p in existing], how="diagonal"
                ).collect()
                out = self._merge_replace(df.lazy(), old.lazy()).collect()
                for p in existing:
                    p.unlink()
                self._append_parts(out, root, [])
            else:
                path = existing[0]
                out = self._merge_replace(df.lazy(), self._lazy_read(path)).collect()
                self._write(out, path)

    def _append_parts(self, df: pl.DataFrame, root: Path, existing: list[Path]) -> None:
        if existing:
            last = existing[-1]
            old = self._lazy_read(last).collect()
            spare = self.max_records - len(old)
            if spare > 0:
                merged = pl.concat([old, df.slice(0, spare)], how="diagonal")
                self._write(merged, last)
                df = df.slice(spare)
        while len(df) > 0:
            new_path = root / f"file-{len(existing):020d}{self.extension}"
            self._write(df.slice(0, self.max_records), new_path)
            df = df.slice(self.max_records)
            existing.append(new_path)

    def _partition_paths(self, partitions: dict[str, Any]) -> Iterable[Path]:
        keys = self._part_keys
        if not keys:
            return []
        g = "/".join(f"{c}={partitions.get(c, '*')}" for c in keys)
        return self.root_path.glob(f"{g}/*{self.extension}")

    def _single_path(self, root: Path) -> Path:
        return (root / FILE_NAME).with_suffix(self.extension)

    @property
    def _partition_dirs(self) -> list[Path]:
        seen: set[Path] = set()
        result = []
        for p in self.paths:
            d = p.parent
            if d not in seen:
                seen.add(d)
                result.append(d)
        return result

    @cached_property
    def _part_keys(self) -> list[str]:
        if not self.partition_cols:
            return []
        if isinstance(self.partition_cols, HashPartitioner):
            return [self.partition_cols.key]
        return list(self.partition_cols)


@dataclass
class RecordWriter:
    """Accumulates dict records and flushes to a ``TableRepo`` in batches.

    Use as a context manager to guarantee the final partial batch is flushed.

    Args:
        trepo: Target repository.
        batch_size: Flush when this many records have been accumulated.
        writer_fn: ``TableRepo`` method to call on flush. Defaults to ``extend``.
    """

    trepo: TableRepo
    batch_size: int = 1_000_000
    writer_fn: Callable | None = None

    def __post_init__(self):
        if self.writer_fn is None:
            self.writer_fn = TableRepo.extend
        self._batch: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def add_to_batch(self, record: dict) -> None:
        self._batch.append(record)
        if len(self._batch) >= self.batch_size:
            self._flush()

    def close(self) -> None:
        if self._batch:
            self._flush()

    def _flush(self) -> None:
        self.writer_fn(self.trepo, pl.DataFrame(self._batch))
        self._batch = []


@dataclass
class DfBatchWriter:
    """Accumulates DataFrames and flushes to a ``TableRepo`` in batches.

    Record count is tracked by row count across accumulated DataFrames.

    Args:
        trepo: Target repository.
        batch_size: Flush when total accumulated rows reach this threshold.
        writer_fn: ``TableRepo`` method to call on flush. Defaults to ``extend``.
    """

    trepo: TableRepo
    batch_size: int = 1_000_000
    writer_fn: Callable | None = None

    def __post_init__(self):
        if self.writer_fn is None:
            self.writer_fn = TableRepo.extend
        self._batch: list[pl.DataFrame] = []
        self._count: int = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def add_to_batch(self, df: pl.DataFrame) -> None:
        self._batch.append(df)
        self._count += len(df)
        if self._count >= self.batch_size:
            self._flush()

    def close(self) -> None:
        if self._batch:
            self._flush()

    def _flush(self) -> None:
        self.writer_fn(self.trepo, pl.concat(self._batch, how="diagonal"))
        self._batch = []
        self._count = 0
