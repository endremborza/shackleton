import random
from base64 import b16encode

import polars as pl
import pytest

from shackleton import TableShack


@pytest.mark.parametrize(["id_col", "comp"], [("A", None), (None, "gzip")])
def test_import(tmp_path, id_col, comp):
    shack = TableShack(
        tmp_path / "data", id_col=id_col, partition_cols=["B", "C"], compression=comp
    )
    n = 2_000
    rng = random.Random(426)
    byte_set = [b16encode(rng.randbytes(4)) for _ in range(5)]
    df = pl.DataFrame(
        {
            "A": [rng.random() for _ in range(n)],
            "B": [rng.randint(0, 20) for _ in range(n)],
            "C": [rng.choice(byte_set) for _ in range(n)],
        }
    )
    shack.extend(df)
    assert shack.get_full_df().shape[0] == n

    assert len(list(shack.get_partition_paths({"B": 4}))) == 5
    assert len(list(shack.get_partition_paths({"C": byte_set[0]}))) == 21
    assert shack.get_partition_df({"B": 4}).shape[0] == 97
    shack.extend(df)
    assert shack.get_full_df().shape[0] == df.shape[0] * 2

    shack.replace_all(df)
    assert shack.get_full_df().shape[0] == df.shape[0]
    if id_col:
        shack.replace_records(df)
        assert shack.get_full_df().shape[0] == df.shape[0]

    some_dfs = False
    for _df in shack.dfs:
        assert _df.shape[0] > 0
        some_dfs = True
    assert some_dfs

    n2 = 5678
    df2 = pl.DataFrame(
        {
            "A": [rng.random() for _ in range(n2)],
            "B": [rng.randint(0, 19) for _ in range(n2)],
            "C": [rng.choice(byte_set) for _ in range(n2)],
        }
    )

    shack.extend(df2)
    assert shack.get_full_df().shape[0] == (n + n2)
