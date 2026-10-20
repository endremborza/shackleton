"""Polars-based persistent tabular store backed by Parquet or Arrow IPC files."""

from .core import (  # noqa: F401
    DfBatchWriter,
    HashDriftError,
    HashPartitioner,
    RecordWriter,
    TableRepo,
)

TableShack = TableRepo  # backward-compat alias

__version__ = "2.0.0"
