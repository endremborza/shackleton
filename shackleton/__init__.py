"""Polars-based persistent tabular store backed by Parquet or Arrow IPC files."""

from .core import DfBatchWriter, HashPartitioner, RecordWriter, TableRepo  # noqa: F401

TableShack = TableRepo  # backward-compat alias

__version__ = "1.2.0"
