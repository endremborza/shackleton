from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Callable, Iterable

import polars as pl

FILE_NAME = "data"


@dataclass
class TableShack:
    root_path: Path
    id_col: str | None = None
    partition_cols: list[str] = field(default_factory=list)
    ipc: bool = False
    compression: str | None = None

    @cached_property
    def extension(self):
        return ".arrow" if self.ipc else ".parquet"

    def extend(self, df: pl.DataFrame):
        return self._write_meta(df, self._extend)

    def replace_records(self, df: pl.DataFrame):
        return self._write_meta(df, self._replace_records)

    def lazy_read(self, path: Path) -> pl.LazyFrame:
        if self.ipc:
            o = pl.scan_ipc(path)
        else:
            o = pl.scan_parquet(path, hive_partitioning=False)
        if self.id_col:
            return o.set_sorted(self.id_col)
        return o

    def replace_all(self, df: pl.DataFrame):
        """purges everything and writes df instead"""
        self.purge()
        self.extend(df)

    def purge(self):
        """purges everything"""
        for p in self.paths:
            p.unlink()

    def get_full_lf(self) -> pl.LazyFrame:
        return pl.concat(map(self.lazy_read, self.paths))

    def get_full_df(self) -> pl.DataFrame:
        return self.get_full_lf().collect()

    def get_partition_paths(self, partitions: dict[str, Any]) -> Iterable[Path]:
        g = "/".join(f"{c}={partitions.get(c, '*')}" for c in self.partition_cols)
        return self.root_path.glob(f"{g}/*{self.extension}")

    def get_partition_lf(self, partitions: dict[str, Any]) -> pl.LazyFrame:
        return pl.concat(map(self.lazy_read, self.get_partition_paths(partitions)))

    def get_partition_df(self, partitions: dict[str, Any]) -> pl.DataFrame:
        return self.get_partition_lf(partitions).collect()

    @property
    def paths(self) -> Iterable[Path]:
        return self.root_path.glob("**/*" + self.extension)

    @property
    def lfs(self) -> Iterable[pl.LazyFrame]:
        return map(self.lazy_read, self.paths)

    @property
    def dfs(self) -> Iterable[pl.DataFrame]:
        return map(pl.LazyFrame.collect, self.lfs)

    def _extend(self, df: pl.LazyFrame, old_df: pl.LazyFrame) -> pl.LazyFrame:
        if self.id_col:
            return old_df.merge_sorted(df.sort(self.id_col), key=self.id_col)
        else:
            return pl.concat([old_df, df])

    def _replace_records(self, df: pl.LazyFrame, old_df: pl.LazyFrame) -> pl.LazyFrame:
        assert self.id_col is not None, "can only replace id'd records"
        return old_df.merge_sorted(df, key=self.id_col).unique(
            subset=self.id_col, keep="last"
        )

    def _write_meta(
        self,
        df: pl.DataFrame,
        fun: Callable[[pl.LazyFrame, pl.LazyFrame], pl.LazyFrame],
        subdirs: tuple[str, ...] = (),
        partitioned=False,
    ):
        if self.partition_cols and not partitioned:
            return self._gb_handle(df, fun)

        true_path = self._get_df_path(subdirs)
        true_path.parent.mkdir(exist_ok=True, parents=True)

        if not true_path.exists():
            out = df.sort(self.id_col) if self.id_col else df
        else:
            out = fun(df.lazy(), self.lazy_read(true_path)).collect()

        fun = pl.DataFrame.write_ipc if self.ipc else pl.DataFrame.write_parquet
        self._add_compression(fun)(out, true_path)

    def _gb_handle(self, df: pl.DataFrame, fun):
        for gid, gdf in df.group_by(self.partition_cols):
            hive_names = [f"{k}={v}" for k, v in zip(self.partition_cols, gid)]
            self._write_meta(gdf.drop(self.partition_cols), fun, hive_names, True)

    def _add_compression(self, fun):
        if self.compression:
            return partial(fun, compression=self.compression)
        return fun

    def _get_df_path(self, subdirs: tuple = ()):
        return Path(self.root_path, *subdirs, FILE_NAME).with_suffix(self.extension)
