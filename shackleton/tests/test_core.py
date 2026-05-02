import random
import threading
from base64 import b16encode

import polars as pl
import pytest

from shackleton import HashPartitioner, TableRepo


def make_df(n: int, seed: int = 0) -> pl.DataFrame:
    rng = random.Random(seed)
    return pl.DataFrame(
        {
            "A": [rng.random() for _ in range(n)],
            "B": [rng.randint(0, 9) for _ in range(n)],
            "C": [rng.choice(list("xyz")) for _ in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# Basic extend / replace_all / purge
# ---------------------------------------------------------------------------


def test_extend_single_file(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(make_df(10, seed=1))
    repo.extend(make_df(5, seed=2))
    assert repo.get_full_df().shape[0] == 15
    assert repo.n_files == 1


def test_replace_all(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(make_df(20))
    repo.replace_all(make_df(7, seed=99))
    assert repo.get_full_df().shape[0] == 7


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
    assert repo.n_files == 4  # 16 rows at 5/file → (5,5,5,1)


def test_max_records_fills_last_file(tmp_path):
    repo = TableRepo(tmp_path / "data", max_records=10)
    repo.extend(make_df(7))
    repo.extend(make_df(6))
    assert repo.n_files == 2
    assert repo.get_full_df().shape[0] == 13


# ---------------------------------------------------------------------------
# partition_cols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("partition_cols", [["B"], ["B", "C"]])
def test_partition_extend(tmp_path, partition_cols):
    repo = TableRepo(tmp_path / "data", partition_cols=partition_cols)
    repo.extend(make_df(100))
    assert repo.get_full_df().shape[0] == 100


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
    assert (
        repo.get_partition_df({"B": 4}).shape[0] == df.filter(pl.col("B") == 4).shape[0]
    )
    assert (
        repo.get_partition_df({"C": byte_set[0]}).shape[0]
        == df.filter(pl.col("C") == byte_set[0]).shape[0]
    )


def test_partition_extend_twice(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    df = make_df(50)
    repo.extend(df)
    repo.extend(df)
    assert repo.get_full_df().shape[0] == 100


def test_partition_replace_all(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(30))
    repo.replace_all(make_df(10, seed=99))
    assert repo.get_full_df().shape[0] == 10


def test_partition_max_records(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"], max_records=3)
    repo.extend(make_df(30))
    assert repo.get_full_df().shape[0] == 30
    assert repo.n_files > 3


# ---------------------------------------------------------------------------
# id_col and replace_records
# ---------------------------------------------------------------------------


def test_replace_records_insert_and_update(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    repo.extend(pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]}))
    repo.replace_records(pl.DataFrame({"id": [2, 4], "val": [200, 400]}))
    result = repo.get_full_df().sort("id")
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["val"].to_list() == [10, 200, 30, 400]


def test_replace_records_no_id_raises(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with pytest.raises(AssertionError):
        repo.replace_records(pl.DataFrame({"A": [1]}))


def test_replace_records_dedup(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    repo.extend(pl.DataFrame({"id": [1, 2, 3], "val": [1, 2, 3]}))
    repo.replace_records(pl.DataFrame({"id": [2, 2], "val": [20, 21]}))
    result = repo.get_full_df().sort("id")
    assert result.shape[0] == 3
    assert result.filter(pl.col("id") == 2)["val"][0] in (20, 21)


def test_replace_records_with_max_records(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id", max_records=2)
    repo.extend(pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]}))
    assert repo.n_files == 2  # 2+1
    repo.replace_records(pl.DataFrame({"id": [2, 3, 4], "val": [200, 300, 400]}))
    result = repo.get_full_df().sort("id")
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["val"].to_list() == [10, 200, 300, 400]
    assert repo.n_files == 2  # 4 rows at max_records=2


def test_replace_records_empty_repo(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    repo.replace_records(pl.DataFrame({"id": [1, 2], "val": [1, 2]}))
    assert repo.get_full_df().shape[0] == 2


def test_id_col_sorted_merge_on_extend(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    repo.extend(pl.DataFrame({"id": [1, 3], "val": [10, 30]}))
    repo.extend(pl.DataFrame({"id": [2, 4], "val": [20, 40]}))
    result = repo.get_full_df()
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["val"].to_list() == [10, 20, 30, 40]


# ---------------------------------------------------------------------------
# dedup_cols
# ---------------------------------------------------------------------------


def test_dedup_cols_max_records_raises(tmp_path):
    with pytest.raises(ValueError, match="dedup_cols"):
        TableRepo(tmp_path / "data", dedup_cols=["id"], max_records=100)


def test_dedup_cols_initial_write(tmp_path):
    repo = TableRepo(tmp_path / "data", dedup_cols=["id"])
    repo.extend(pl.DataFrame({"id": [1, 1, 2], "val": [10, 99, 20]}))
    result = repo.get_full_df()
    assert result.shape[0] == 2
    assert set(result["id"].to_list()) == {1, 2}


def test_dedup_cols_extend(tmp_path):
    repo = TableRepo(tmp_path / "data", dedup_cols=["id"])
    repo.extend(pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]}))
    repo.extend(pl.DataFrame({"id": [2, 4], "val": [99, 40]}))
    result = repo.get_full_df().sort("id")
    assert result.shape[0] == 4
    assert set(result["id"].to_list()) == {1, 2, 3, 4}
    # id=4 is new, so its value is unambiguous
    assert result.filter(pl.col("id") == 4)["val"][0] == 40


def test_dedup_cols_with_id_col_old_wins(tmp_path):
    # id_col + dedup_cols: merge_sorted places old rows first, keep="first" retains them
    repo = TableRepo(tmp_path / "data", id_col="id", dedup_cols=["id"])
    repo.extend(pl.DataFrame({"id": [1, 2], "val": [10, 20]}))
    repo.extend(pl.DataFrame({"id": [2, 3], "val": [99, 30]}))
    result = repo.get_full_df().sort("id")
    assert result.shape[0] == 3
    assert result["id"].to_list() == [1, 2, 3]
    assert result.filter(pl.col("id") == 2)["val"][0] == 20  # old value kept
    assert result.filter(pl.col("id") == 3)["val"][0] == 30  # new row inserted


# ---------------------------------------------------------------------------
# IPC / Parquet format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ipc", [False, True])
def test_format_basic(tmp_path, ipc):
    repo = TableRepo(tmp_path / "data", ipc=ipc)
    repo.extend(make_df(20))
    assert repo.n_files == 1
    assert list(repo.paths)[0].suffix == (".arrow" if ipc else ".parquet")
    assert repo.get_full_df().shape[0] == 20


@pytest.mark.parametrize("ipc", [False, True])
def test_format_with_partitions(tmp_path, ipc):
    repo = TableRepo(tmp_path / "data", ipc=ipc, partition_cols=["C"])
    repo.extend(make_df(30))
    assert repo.get_full_df().shape[0] == 30


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"compression": "zstd"},
        {"compression": "zstd", "compression_level": 1},
    ],
)
def test_compression(tmp_path, kw):
    repo = TableRepo(tmp_path / "data", **kw)
    repo.extend(make_df(50))
    assert repo.get_full_df().shape[0] == 50


# ---------------------------------------------------------------------------
# HashPartitioner
# ---------------------------------------------------------------------------


def test_hash_partitioner_distributes(tmp_path):
    hp = HashPartitioner("B", num_groups=4)
    repo = TableRepo(tmp_path / "data", partition_cols=hp)
    repo.extend(make_df(200))
    assert repo.get_full_df().shape[0] == 200
    assert 1 <= repo.n_files <= 4


def test_hash_partitioner_key_stripped(tmp_path):
    hp = HashPartitioner("B", num_groups=4)
    repo = TableRepo(tmp_path / "data", partition_cols=hp)
    repo.extend(make_df(20))
    assert hp.key not in repo.get_full_df().columns


def test_hash_partitioner_get_partition(tmp_path):
    hp = HashPartitioner("B", num_groups=8)
    repo = TableRepo(tmp_path / "data", partition_cols=hp)
    df = make_df(100)
    repo.extend(df)
    b3_bucket = str(int(pl.Series([3]).cast(pl.String).hash()[0]) % 8).zfill(1)
    part = repo.get_partition_df({hp.key: b3_bucket})
    assert (
        part.filter(pl.col("B") == 3).shape[0] == df.filter(pl.col("B") == 3).shape[0]
    )


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
    result = repo.get_full_df().sort("i")
    assert result.shape[0] == 30
    assert result["val"].to_list() == [i * 2 for i in range(30)]


def test_record_writer_replace(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="i")
    repo.extend(pl.DataFrame({"i": list(range(10)), "val": list(range(10))}))
    with repo.get_replacing_record_writer(3) as writer:
        for i in range(5):
            writer.add_to_batch({"i": i, "val": i * 100})
    result = repo.get_full_df().sort("i")
    assert result.shape[0] == 10
    assert result.filter(pl.col("i") < 5)["val"].to_list() == [
        i * 100 for i in range(5)
    ]
    assert result.filter(pl.col("i") >= 5)["val"].to_list() == list(range(5, 10))


def test_record_writer_context_manager_flushes(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with repo.get_extending_record_writer(100) as writer:
        for i in range(5):
            writer.add_to_batch({"i": i})
    assert repo.get_full_df().shape[0] == 5


# ---------------------------------------------------------------------------
# DfBatchWriter
# ---------------------------------------------------------------------------


def test_df_writer_extend(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with repo.get_extending_df_writer(25) as writer:
        for i in range(5):
            writer.add_to_batch(make_df(10, seed=i))
    assert repo.get_full_df().shape[0] == 50


def test_df_writer_replaces(tmp_path):
    repo = TableRepo(tmp_path / "data", id_col="id")
    repo.extend(pl.DataFrame({"id": list(range(20)), "val": list(range(20))}))
    with repo.get_replacing_df_writer(10) as writer:
        writer.add_to_batch(pl.DataFrame({"id": [0, 1, 2], "val": [100, 101, 102]}))
        writer.add_to_batch(pl.DataFrame({"id": [3, 4, 5], "val": [103, 104, 105]}))
    result = repo.get_full_df().sort("id")
    assert result.shape[0] == 20
    assert result.filter(pl.col("id") <= 5)["val"].to_list() == list(range(100, 106))
    assert result.filter(pl.col("id") > 5)["val"].to_list() == list(range(6, 20))


def test_df_writer_context_flushes(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with repo.get_extending_df_writer(1000) as writer:
        writer.add_to_batch(make_df(3))
    assert repo.get_full_df().shape[0] == 3


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "partition_cols,n_threads,rows_per_thread",
    [
        ([], 8, 100),
        (["C"], 6, 50),
    ],
)
def test_concurrent_extend(tmp_path, partition_cols, n_threads, rows_per_thread):
    repo = TableRepo(tmp_path / "data", partition_cols=partition_cols)

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
    repo.extend(pl.DataFrame({"A": [5], "C": [6]}))
    full = repo.get_full_df()
    assert full.shape[0] == 3
    assert "B" in full.columns
    assert "C" in full.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_extend(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(pl.DataFrame({"A": pl.Series([], dtype=pl.Int64)}))
    assert repo.get_full_df().shape[0] == 0


# ---------------------------------------------------------------------------
# Process-level locking (.lock sidecar)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "partition_cols,n,expected_locks",
    [
        ([], 10, 1),
        (["C"], 30, 3),  # x, y, z partitions
    ],
)
def test_lock_file_created(tmp_path, partition_cols, n, expected_locks):
    repo = TableRepo(tmp_path / "data", partition_cols=partition_cols)
    repo.extend(make_df(n))
    assert len(list((tmp_path / "data").glob("**/.lock"))) == expected_locks


# ---------------------------------------------------------------------------
# map_partitions
# ---------------------------------------------------------------------------


def test_map_partitions_basic(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(90))
    counts = repo.map_partitions(lambda df: len(df))
    assert sum(counts) == 90
    assert len(counts) == 3


def test_map_partitions_parallel(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(90))
    assert sorted(repo.map_partitions(lambda df: len(df))) == sorted(
        repo.map_partitions(lambda df: len(df), workers=2)
    )


def test_map_partitions_empty(tmp_path):
    repo = TableRepo(tmp_path / "data")
    assert repo.map_partitions(lambda df: len(df)) == []


def test_map_partitions_no_partition_combines_files(tmp_path):
    repo = TableRepo(tmp_path / "data", max_records=5)
    repo.extend(make_df(15))
    assert repo.n_files == 3
    assert repo.map_partitions(lambda df: df.shape[0]) == [15]


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
# purge_partition / replace_partition
# ---------------------------------------------------------------------------


def test_purge_partition(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(90))
    x_count = repo.get_partition_df({"C": "x"}).shape[0]
    repo.purge_partition({"C": "x"})
    result = repo.get_full_df()
    assert result.filter(pl.col("C") == "x").shape[0] == 0
    assert result.shape[0] == 90 - x_count


def test_purge_partition_no_match(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(30))
    n = repo.get_full_df().shape[0]
    repo.purge_partition({"C": "nonexistent"})
    assert repo.get_full_df().shape[0] == n


def test_replace_partition_basic(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(90))
    repo.replace_partition(pl.DataFrame({"A": [9.0], "B": [5], "C": ["x"]}))
    x_rows = repo.get_full_df().filter(pl.col("C") == "x")
    assert x_rows.shape[0] == 1
    assert x_rows["A"][0] == 9.0


def test_replace_partition_no_partition_raises(tmp_path):
    repo = TableRepo(tmp_path / "data")
    with pytest.raises(ValueError, match="partition_cols"):
        repo.replace_partition(pl.DataFrame({"A": [1]}))


def test_replace_partition_with_max_records(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"], max_records=5)
    repo.extend(make_df(90))
    repo.replace_partition(
        pl.DataFrame({"A": [1.0, 2.0], "B": [1, 2], "C": ["x", "x"]})
    )
    assert repo.get_full_df().filter(pl.col("C") == "x").shape[0] == 2


# ---------------------------------------------------------------------------
# get_partition_lf edge cases
# ---------------------------------------------------------------------------


def test_get_partition_lf_no_match(tmp_path):
    repo = TableRepo(tmp_path / "data", partition_cols=["C"])
    repo.extend(make_df(30))
    assert repo.get_partition_lf({"C": "nonexistent"}).collect().shape[0] == 0


def test_get_partition_lf_no_partition_cols(tmp_path):
    repo = TableRepo(tmp_path / "data")
    repo.extend(make_df(10))
    assert repo.get_partition_lf({}).collect().shape[0] == 0


# ---------------------------------------------------------------------------
# Performance smoke test
# ---------------------------------------------------------------------------


def test_large_extend_performance(tmp_path):
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
