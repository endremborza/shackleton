## v2.0.1

- Fixed composite-key merges to be order-exact: stored rows now reliably come before incoming ones on equal keys, where polars' streaming engine could previously reorder ties.
- Fixed `replace_records` to work against a partition with no stored data yet.
- Writes are now atomic: data is staged to a temp file and moved into place with `os.replace`, so a crash mid-merge can no longer leave a partially-written file.
- Fixed partitions to be flagged sorted on read only when they are a single file, since appended parts are unsorted and were previously mis-flagged.

---

## v2.0.0

- **Breaking:** requires polars ≥2; existing `HashPartitioner` stores built under older polars versions will raise `HashDriftError` on their next bucket-addressed operation until migrated.
- Detects hash bucket drift before writes, `get_partition_*` and `purge_partition`, and adds `TableRepo.rehash()` to rebuild a store under the installed polars' hashing; full reads are unaffected and skip the check.
- `id_col` now accepts a list of columns for composite-key merges and dedup.
- Merges are now order-exact: on equal keys stored rows precede incoming ones, so `extend` with `dedup_cols` keeps the stored row and `replace_records` keeps the batch's last row instead of an arbitrary one.
- Writes are now staged to a temp file and atomically renamed into place, and merges with existing data stream instead of collecting fully in memory.
- `paths` are now returned sorted, so reads are deterministic.

---

## v1.2.0

deduped

---

## v1.1.0

realign, simplify
