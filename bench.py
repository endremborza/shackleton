"""Shackleton performance benchmarks.

Usage:
    uv run bench.py
    uv run bench.py --rows 500_000
"""

import argparse
import tempfile
import time
from pathlib import Path

import polars as pl

from shackleton import HashPartitioner, TableRepo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.arange(0, n, eager=True),
            "value": pl.Series([float(i) * 1.1 for i in range(n)]),
            "category": pl.Series([chr(65 + i % 26) for i in range(n)]),
            "tag": pl.Series([f"tag-{i % 100}" for i in range(n)]),
        }
    )


def bench(label: str, fn, *, unit: str = "rows/s", scale: int = 1) -> float:
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    rate = scale / elapsed
    print(f"  {label:<45} {rate:>12,.0f} {unit}  ({elapsed:.3f}s)")
    return rate


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def bench_extend_single(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "extend_single")
    bench("extend (single file)", lambda: repo.extend(df), scale=len(df))


def bench_extend_with_id(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "extend_id", id_col="id")
    bench("extend + sorted merge (id_col)", lambda: repo.extend(df), scale=len(df))


def bench_extend_partitioned(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "extend_part", partition_cols=["category"])
    bench("extend (26 column partitions)", lambda: repo.extend(df), scale=len(df))


def bench_extend_hash(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "extend_hash", partition_cols=HashPartitioner("tag", 64))
    bench("extend (64 hash partitions)", lambda: repo.extend(df), scale=len(df))


def bench_extend_max_records(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "extend_max", max_records=10_000)
    bench("extend (max_records=10k files)", lambda: repo.extend(df), scale=len(df))


def bench_read_full(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "read_full")
    repo.extend(df)
    bench("read full (collect)", lambda: repo.get_full_df(), scale=len(df))


def bench_read_full_lazy(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "read_lazy")
    repo.extend(df)
    bench(
        "read full (lazy filter + collect)",
        lambda: repo.get_full_lf().filter(pl.col("value") > 0).collect(),
        scale=len(df),
    )


def bench_read_partition(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "read_part", partition_cols=["category"])
    repo.extend(df)
    bench(
        "get_partition_df (1 of 26 partitions)",
        lambda: repo.get_partition_df({"category": "A"}),
        scale=len(df) // 26,
    )


def bench_replace_records(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "replace_rec", id_col="id")
    repo.extend(df)
    updates = df.sample(n=min(10_000, len(df) // 10), seed=42).with_columns(
        pl.col("value") * 2
    )
    bench("replace_records (10k updates)", lambda: repo.replace_records(updates), scale=len(updates))


def bench_ipc_extend(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "ipc_extend", ipc=True)
    bench("extend IPC/Arrow format", lambda: repo.extend(df), scale=len(df))


def bench_ipc_read(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "ipc_read", ipc=True)
    repo.extend(df)
    bench("read full IPC/Arrow format", lambda: repo.get_full_df(), scale=len(df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    args = parser.parse_args()

    n = args.rows
    print(f"\nShackleton benchmark — {n:,} rows\n")

    df = make_df(n)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        print("--- Write ---")
        bench_extend_single(df, root)
        bench_extend_with_id(df, root)
        bench_extend_partitioned(df, root)
        bench_extend_hash(df, root)
        bench_extend_max_records(df, root)
        bench_ipc_extend(df, root)

        print("\n--- Read ---")
        bench_read_full(df, root)
        bench_read_full_lazy(df, root)
        bench_read_partition(df, root)
        bench_ipc_read(df, root)

        print("\n--- Update ---")
        bench_replace_records(df, root)

    print()


if __name__ == "__main__":
    main()
