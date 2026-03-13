import random
import threading
from base64 import b16encode

import polars as pl
import pytest

from shackleton import HashPartitioner, TableRepo

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_df(n: int, seed: int = 0, with_id: bool = False) -> pl.DataFrame:
    rng = random.Random(seed)
    data: dict = {
        "A": [rng.random() for _ in range(n)],
        "B": [rng.randint(0, 9) for _ in range(n)],
        "C": [rng.choice(list("xyz")) for _ in range(n)],
    }
    if with_id:
        data["id"] = list(range(n))
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Basic extend / replace_all / purge
# ---------------------------------------------------------------------------


def test_extend_single_file(tmp_path):
    repo = TableRepo(tmp_path / "data")
    df1 = make_df(10, seed=1)
    df2 = make_df(5, seed=2)
    repo.extend(df1)
    repo.extend(df2)
    assert repo.get_full_df().shape[0] == 15
    assert repo.n_files == 1


def test_replace_all(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(make_df(20))
    df = make_df(7, seed=99)
    repo.replace_all(df)
    result = repo.get_full_df()
    assert result.shape[0] == 7


def test_purge(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(make_df(10))
    repo.purge()
    assert repo.n_files == 0
    assert repo.get_full_df().shape[0] == 0


def test_dfs_lfs_paths(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["B"])
    repo.extend(make_df(50))
    assert all(df.shape[0] > 0 for df in repo.dfs)
    assert all(lf.collect().shape[0] > 0 for lf in repo.lfs)
    assert len(list(repo.paths)) == repo.n_files


# ---------------------------------------------------------------------------
# max_records splitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["n_rows", "max_records", "expected_files"],
    [
        (9, 0, 1),
        (9, 10, 1),
        (9, 5, 2),
        (9, 3, 3),
    ],
)
def test_max_records_file_count(tmp_path, n_rows, max_records, expected_files):
    repo = TableRepo(tmp_path / "data", max_records=max_records)
    repo.extend(make_df(n_rows, seed=1))
    assert repo.n_files == expected_files
    assert repo.get_full_df().shape[0] == n_rows


def test_max_records_multiple_extends(tmp_path):
    repo = TableRepo(tmp_path / "data", max_records=5)
    for seed in range(4):
        repo.extend(make_df(4, seed=seed))
    assert repo.get_full_df().shape[0] == 16
    # 16 rows at 5/file → 4 files (5, 5, 5, 1)
    assert repo.n_files == 4


def test_max_records_fills_last_file(tmp_path):
    repo = TableRepo(tmp_path / "data", max_records=10)
    repo.extend(make_df(7))
    repo.extend(make_df(6))
    # 13 rows → file-0: 10, file-1: 3
    assert repo.n_files == 2
    assert repo.get_full_df().shape[0] == 13


# ---------------------------------------------------------------------------
# partition_cols
# ---------------------------------------------------------------------------


def test_partition_single_col(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["B"])
    df = make_df(100)
    repo.extend(df)
    full = repo.get_full_df()
    assert full.shape[0] == 100
    assert set(full["B"].to_list()).issubset(set(range(10)))


def test_partition_multi_col(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["B", "C"])
    df = make_df(200)
    repo.extend(df)
    assert repo.get_full_df().shape[0] == 200


def test_get_partition(tmp_path):
    rng = random.Random(42)
    byte_set = [b16encode(rng.randbytes(4)).decode() for _ in range(5)]
    repo = TableRepo(tmp_path / "data", partition_cols=["B", "C"])
    df = pl.DataFrame(
        {
            "A": [rng.random() for _ in range(2000)],
            "B": [rng.randint(0, 20) for _ in range(2000)],
            "C": [rng.choice(byte_set) for _ in range(2000)],
        }
    )
    repo.extend(df)
    part = repo.get_partition_df({"B": 4})
    assert part.shape[0] == df.filter(pl.col("B") == 4).shape[0]

    part2 = repo.get_partition_df({"C": byte_set[0]})
    assert part2.shape[0] == df.filter(pl.col("C") == byte_set[0]).shape[0]


def test_partition_extend_twice(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    df = make_df(50)
    repo.extend(df)
    repo.extend(df)
    assert repo.get_full_df().shape[0] == 100


def test_partition_replace_all(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(30))
    df2 = make_df(10, seed=99)
    repo.replace_all(df2)
    assert repo.get_full_df().shape[0] == 10


def test_partition_max_records(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"], max_records=3)
    repo.extend(make_df(30))
    assert repo.get_full_df().shape[0] == 30
    # Each partition ("x", "y", "z") should have multiple files
    assert repo.n_files > 3


# ---------------------------------------------------------------------------
# id_col and replace_records
# ---------------------------------------------------------------------------


def test_replace_records_basic(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    repo.extend(df)

    updates = pl.DataFrame({"id": [2, 4], "val": [200, 400]})
    repo.replace_records(updates)

    result = repo.get_full_df().sort("id")
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["val"].to_list() == [10, 200, 30, 400]


def test_replace_records_no_id_raises(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with pytest.raises(AssertionError):
        repo.replace_records(pl.DataFrame({"A": [1]}))


def test_replace_records_dedup(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    df = pl.DataFrame({"id": [1, 2, 3], "val": [1, 2, 3]})
    repo.extend(df)
    # Replace with duplicate ids in the input — last wins
    dups = pl.DataFrame({"id": [2, 2], "val": [20, 21]})
    repo.replace_records(dups)
    result = repo.get_full_df().sort("id")
    assert result.filter(pl.col("id") == 2)["val"][0] in (20, 21)
    assert result.shape[0] == 3


def test_replace_records_with_max_records(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id", max_records=2)
    df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    repo.extend(df)
    assert repo.n_files == 2  # 2 + 1

    updates = pl.DataFrame({"id": [2, 3, 4], "val": [200, 300, 400]})
    repo.replace_records(updates)

    result = repo.get_full_df().sort("id")
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["val"].to_list() == [10, 200, 300, 400]
    assert repo.n_files == 2  # 4 rows at max_records=2


def test_replace_records_empty_repo(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    df = pl.DataFrame({"id": [1, 2], "val": [1, 2]})
    repo.replace_records(df)
    assert repo.get_full_df().shape[0] == 2


# ---------------------------------------------------------------------------
# IPC format
# ---------------------------------------------------------------------------


def test_ipc_format(tmp_path):
    repo = TableRepo(tmp_path / "data", ipc=True)
    df = make_df(20)
    repo.extend(df)
    assert repo.n_files == 1
    assert list(repo.paths)[0].suffix == ".arrow"
    assert repo.get_full_df().shape[0] == 20


def test_ipc_with_partitions(tmp_path):
    repo = TableRepo(tmp_path / "data", ipc=True, partition_cols=["C"])
    repo.extend(make_df(30))
    assert repo.get_full_df().shape[0] == 30


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def test_compression(tmp_path):
    repo = TableRepo(tmp_path / "data", compression="zstd")
    repo.extend(make_df(50))
    assert repo.get_full_df().shape[0] == 50


# ---------------------------------------------------------------------------
# HashPartitioner
# ---------------------------------------------------------------------------


def test_hash_partitioner_distributes(tmp_path):
    hp = HashPartitioner("B", num_groups=4)
    repo = TableRepo(tmp_path / "data", partition_cols=hp)
    df = make_df(200)
    repo.extend(df)

    total = repo.get_full_df().shape[0]
    assert total == 200
    # Buckets: 1–4 files (collisions allowed); at least 1
    assert 1 <= repo.n_files <= 4


def test_hash_partitioner_key_stripped(tmp_path):
    hp = HashPartitioner("B", num_groups=4)
    repo = TableRepo(tmp_path / "data", partition_cols=hp)
    repo.extend(make_df(20))
    df = repo.get_full_df()
    assert hp.key not in df.columns


def test_hash_partitioner_get_partition(tmp_path):
    hp = HashPartitioner("B", num_groups=8)
    repo = TableRepo(tmp_path / "data", partition_cols=hp)
    df = make_df(100)
    repo.extend(df)

    # All rows with B=3 land in the same bucket
    b3_bucket = str(int(pl.Series([3]).cast(pl.String).hash()[0]) % 8).zfill(1)
    part = repo.get_partition_df({hp.key: b3_bucket})
    b3_count = df.filter(pl.col("B") == 3).shape[0]
    # Partition contains all B=3 rows (and possibly other B values in same bucket)
    assert part.filter(pl.col("B") == 3).shape[0] == b3_count


def test_hash_partitioner_extend_twice(tmp_path):
    hp = HashPartitioner("C", num_groups=3)
    repo = TableRepo(tmp_path / "data", partition_cols=hp)
    df = make_df(30)
    repo.extend(df)
    repo.extend(df)
    assert repo.get_full_df().shape[0] == 60


# ---------------------------------------------------------------------------
# RecordWriter
# ---------------------------------------------------------------------------


def test_record_writer_extend(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with repo.get_extending_record_writer(7) as writer:
        for i in range(30):
            writer.add_to_batch({"i": i, "val": i * 2})
    assert repo.get_full_df().shape[0] == 30


def test_record_writer_replace(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="i")
    df = pl.DataFrame({"i": list(range(10)), "val": list(range(10))})
    repo.extend(df)

    with repo.get_replacing_record_writer(3) as writer:
        for i in range(5):
            writer.add_to_batch({"i": i, "val": i * 100})

    result = repo.get_full_df().sort("i")
    assert result.shape[0] == 10
    assert result.filter(pl.col("i") == 0)["val"][0] == 0


def test_record_writer_context_manager_flushes(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with repo.get_extending_record_writer(100) as writer:
        for i in range(5):  # below batch_size
            writer.add_to_batch({"i": i})
    # context exit should have flushed
    assert repo.get_full_df().shape[0] == 5


# ---------------------------------------------------------------------------
# DfBatchWriter
# ---------------------------------------------------------------------------


def test_df_writer_extend(tmp_path):
    repo = TableRepo(tmp_path / "data")
    dfs = [make_df(10, seed=i) for i in range(5)]
    with repo.get_extending_df_writer(25) as writer:
        for df in dfs:
            writer.add_to_batch(df)
    assert repo.get_full_df().shape[0] == 50


def test_df_writer_replaces(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    base = pl.DataFrame({"id": list(range(20)), "val": list(range(20))})
    repo.extend(base)

    updates = [
        pl.DataFrame({"id": [0, 1, 2], "val": [100, 101, 102]}),
        pl.DataFrame({"id": [3, 4, 5], "val": [103, 104, 105]}),
    ]
    with repo.get_replacing_df_writer(10) as writer:
        for df in updates:
            writer.add_to_batch(df)

    result = repo.get_full_df().sort("id")
    assert result.shape[0] == 20
    assert result.filter(pl.col("id") == 0)["val"][0] == 100


def test_df_writer_context_flushes(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with repo.get_extending_df_writer(1000) as writer:
        writer.add_to_batch(make_df(3))
    assert repo.get_full_df().shape[0] == 3


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_extend(tmp_path):
    repo = TableRepo(tmp_path / "data")
    n_threads = 8
    rows_per_thread = 100

    def worker(seed):
        repo.extend(make_df(rows_per_thread, seed=seed))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert repo.get_full_df().shape[0] == n_threads * rows_per_thread


def test_concurrent_extend_partitioned(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    n_threads = 6
    rows_per_thread = 50

    def worker(seed):
        repo.extend(make_df(rows_per_thread, seed=seed))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert repo.get_full_df().shape[0] == n_threads * rows_per_thread


# ---------------------------------------------------------------------------
# Schema evolution (diagonal concat)
# ---------------------------------------------------------------------------


def test_schema_mismatch_extend(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(pl.DataFrame({"A": [1, 2], "B": [3, 4]}))
    # Add column C that didn't exist before
    repo.extend(pl.DataFrame({"A": [5], "C": [6]}))
    full = repo.get_full_df()
    assert full.shape[0] == 3
    # B and C both present (nulls for missing values)
    assert "B" in full.columns
    assert "C" in full.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_extend(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(pl.DataFrame({"A": pl.Series([], dtype=pl.Int64)}))
    # Empty extend should not create a file (nothing to write in _append_parts)
    # or create an empty file - behaviour: _extend_dir writes an empty file
    # The important thing is get_full_df works
    assert repo.get_full_df().shape[0] == 0


# ---------------------------------------------------------------------------
# Process-level locking (.lock sidecar)
# ---------------------------------------------------------------------------


def test_lock_file_created(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(make_df(10))
    assert (tmp_path / "data" / ".lock").exists()


def test_lock_file_created_partitioned(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(30))
    # Each partition dir should have a .lock file
    lock_files = list((tmp_path / "data").glob("**/.lock"))
    assert len(lock_files) == 3  # x, y, z partitions


# ---------------------------------------------------------------------------
# map_partitions
# ---------------------------------------------------------------------------


def test_map_partitions_basic(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(90))
    counts = repo.map_partitions(lambda df: len(df))
    assert sum(counts) == 90
    assert len(counts) == 3  # x, y, z


def test_map_partitions_parallel(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(90))
    counts_seq = sorted(repo.map_partitions(lambda df: len(df)))
    counts_par = sorted(repo.map_partitions(lambda df: len(df), workers=2))
    assert counts_seq == counts_par


def test_map_partitions_empty(tmp_path):
    repo = TableRepo(tmp_path / "data")
    assert repo.map_partitions(lambda df: len(df)) == []


def test_map_partitions_no_partition_combines_files(tmp_path):
    repo = TableRepo(tmp_path / "data", max_records=5)
    repo.extend(make_df(15))
    assert repo.n_files == 3
    # map_partitions should combine all 3 files into one call
    result = repo.map_partitions(lambda df: df.shape[0])
    assert result == [15]


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------


def test_compact_reduces_files(tmp_path):
    repo = TableRepo(tmp_path / "data", max_records=10)
    for _ in range(5):
        repo.extend(make_df(10))
    assert repo.n_files == 5
    repo.compact(target_rows=25)
    assert repo.n_files == 2  # 50 rows → ceil(50/25) = 2
    assert repo.get_full_df().shape[0] == 50


def test_compact_single_file_noop(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(make_df(20))
    repo.compact(target_rows=10)
    assert repo.n_files == 1
    assert repo.get_full_df().shape[0] == 20


def test_compact_partitioned(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"], max_records=5)
    repo.extend(make_df(90))
    n_before = repo.n_files
    repo.compact(target_rows=50)
    assert repo.get_full_df().shape[0] == 90
    assert repo.n_files < n_before


def test_compact_ipc(tmp_path):
    repo = TableRepo(tmp_path / "data", ipc=True, max_records=10)
    repo.extend(make_df(40))
    repo.compact(target_rows=20)
    assert repo.n_files == 2
    assert repo.get_full_df().shape[0] == 40


# ---------------------------------------------------------------------------


def test_large_extend_performance(tmp_path):
    """Smoke test: 40 batches of 10k rows should complete without error."""
    repo = TableRepo(tmp_path / "data")
    rng = random.Random(7)
    for _ in range(40):
        df = pl.DataFrame(
            {
                "A": [rng.random() for _ in range(10_000)],
                "B": [rng.randint(0, 1000) for _ in range(10_000)],
            }
        )
        repo.extend(df)
    assert repo.get_full_df().shape[0] == 400_000
