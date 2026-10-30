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

DEFAULT_ROWS = 1_000_000
N_CATEGORIES = 26
N_BUCKETS = 64
MAX_RECORDS = 10_000
N_UPSERTS = 10_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.arange(0, n, eager=True),
            "value": pl.Series([float(i) * 1.1 for i in range(n)]),
            "category": pl.Series([chr(65 + i % N_CATEGORIES) for i in range(n)]),
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
    repo.extend(df.with_columns(pl.col("id") * 2))
    interleaved = df.with_columns(pl.col("id") * 2 + 1)
    bench(
        "extend into stored rows (id_col merge)",
        lambda: repo.extend(interleaved),
        scale=len(df),
    )


def bench_extend_partitioned(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "extend_part", partition_cols=["category"])
    bench(
        f"extend ({N_CATEGORIES} column partitions)",
        lambda: repo.extend(df),
        scale=len(df),
    )


def bench_extend_hash(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(
        root / "extend_hash", partition_cols=HashPartitioner("tag", N_BUCKETS)
    )
    bench(
        f"extend ({N_BUCKETS} hash partitions)",
        lambda: repo.extend(df),
        scale=len(df),
    )


def bench_extend_max_records(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "extend_max", max_records=MAX_RECORDS)
    bench(
        f"extend (max_records={MAX_RECORDS:,})",
        lambda: repo.extend(df),
        scale=len(df),
    )


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
        f"get_partition_df (1 of {N_CATEGORIES} partitions)",
        lambda: repo.get_partition_df({"category": "A"}),
        scale=len(df) // N_CATEGORIES,
    )


def bench_replace_records(df: pl.DataFrame, root: Path) -> None:
    repo = TableRepo(root / "replace_rec", id_col="id")
    repo.extend(df)
    updates = df.sample(n=min(N_UPSERTS, len(df) // 10), seed=42).with_columns(
        pl.col("value") * 2
    )
    bench(
        f"replace_records ({len(updates):,} updates)",
        lambda: repo.replace_records(updates),
        scale=len(updates),
    )


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
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
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
