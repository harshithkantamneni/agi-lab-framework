"""Backfill the four single-valued metadata columns on `memories` FILE-LEVEL.

Re-derives {program_id, role, phase, deliverable_type} for every existing row by
grouping rows by source_path, building each file's whole-file text (all its
chunks in id order), calling tools.retrieval.metadata.derive_metadata ONCE per
file, and applying that single result to ALL of that file's rows.

This is a metadata-only UPDATE. It NEVER inserts, deletes, or changes ids/
chunk_text, so the separate sqlite-vec table (keyed by id) is untouched.

--dry-run is the DEFAULT and writes nothing. --apply is required to write.
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import OrderedDict, defaultdict

from tools.retrieval.metadata import derive_metadata

DEFAULT_DB = "tools/lab_memory.db"

_META_COLS = ("program_id", "role", "phase", "deliverable_type")


def _group_rows_by_file(conn: sqlite3.Connection):
    """Return OrderedDict[source_path] -> list[(id, chunk_text)] in id order.

    Reads id, source_path, chunk_text only. Rows are ordered by id so that the
    whole-file text reconstruction is deterministic and matches ingestion order.
    """
    by_file: "OrderedDict[str, list]" = OrderedDict()
    cur = conn.execute(
        "SELECT id, source_path, chunk_text FROM memories ORDER BY id"
    )
    for row_id, source_path, chunk_text in cur:
        by_file.setdefault(source_path, []).append((row_id, chunk_text))
    return by_file


def _derive_per_file(by_file) -> dict:
    """Return {source_path: derived_metadata_dict}, one derive call per file."""
    derived = {}
    for source_path, rows in by_file.items():
        whole_text = " ".join(chunk_text for _id, chunk_text in rows)
        derived[source_path] = derive_metadata(source_path, whole_text)
    return derived


def _existing_program_tags(conn: sqlite3.Connection) -> dict:
    """Return {id: program_id} for rows whose program_id is currently non-empty."""
    return {
        row_id: prog
        for row_id, prog in conn.execute(
            "SELECT id, program_id FROM memories WHERE program_id != ''"
        )
    }


def _coverage_from_conn(conn: sqlite3.Connection) -> dict:
    """Compute coverage stats directly from the committed table state."""
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    files = conn.execute(
        "SELECT COUNT(DISTINCT source_path) FROM memories"
    ).fetchone()[0]
    counts = {}
    for col in _META_COLS:
        counts[col] = conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {col} != ''"
        ).fetchone()[0]
    any_tag = conn.execute(
        "SELECT COUNT(*) FROM memories "
        "WHERE program_id != '' OR role != '' OR phase != ''"
    ).fetchone()[0]
    data_memories_prog = conn.execute(
        "SELECT COUNT(*) FROM memories "
        "WHERE program_id != '' AND source_path LIKE 'data/memories/%'"
    ).fetchone()[0]
    return {
        "total": total,
        "files": files,
        "counts": counts,
        "any_tag": any_tag,
        "data_memories_prog": data_memories_prog,
    }


def _coverage_from_derived(by_file, derived) -> dict:
    """Compute the coverage that WOULD result from applying `derived`."""
    total = sum(len(rows) for rows in by_file.values())
    counts = {c: 0 for c in _META_COLS}
    any_tag = 0
    data_memories_prog = 0
    for source_path, rows in by_file.items():
        meta = derived[source_path]
        n = len(rows)
        for col in _META_COLS:
            if meta[col]:
                counts[col] += n
        if meta["program_id"] or meta["role"] or meta["phase"]:
            any_tag += n
        if meta["program_id"] and source_path.startswith("data/memories/"):
            data_memories_prog += n
    return {
        "total": total,
        "files": len(by_file),
        "counts": counts,
        "any_tag": any_tag,
        "data_memories_prog": data_memories_prog,
    }


def _reproduced_program_tags(by_file, derived, existing: dict) -> tuple:
    """Return (reproduced, total_existing): how many existing program_id tags the
    derivation reproduces (same program_id) on the same rows."""
    total_existing = len(existing)
    reproduced = 0
    for source_path, rows in by_file.items():
        new_prog = derived[source_path]["program_id"]
        for row_id, _chunk in rows:
            old = existing.get(row_id)
            if old is not None and new_prog == old:
                reproduced += 1
    return reproduced, total_existing


def _pct(n: int, total: int) -> float:
    return (100.0 * n / total) if total else 0.0


def _print_coverage(cov: dict, *, label: str, reproduced=None) -> None:
    total = cov["total"]
    print(f"--- {label} coverage ---")
    print(f"total rows : {total}")
    print(f"files      : {cov['files']}")
    for col in _META_COLS:
        n = cov["counts"][col]
        print(f"{col:16s}: {n:6d} ({_pct(n, total):5.1f}%)")
    print(
        f"{'ANY-tag':16s}: {cov['any_tag']:6d} "
        f"({_pct(cov['any_tag'], total):5.1f}%)  "
        f"[program OR role OR phase]"
    )
    print(f"data/memories program-tagged rows: {cov['data_memories_prog']}")
    if reproduced is not None:
        rep, tot = reproduced
        print(f"existing program_id reproduced  : {rep}/{tot}")


def apply_backfill(db_path: str, apply: bool) -> dict:
    """Run the backfill. When apply=False (default) writes nothing.

    Returns a stats dict (also printed). Metadata-only UPDATEs in one
    transaction; never INSERT/DELETE, never touches id/chunk_text/vectors.
    """
    conn = sqlite3.connect(db_path)
    try:
        by_file = _group_rows_by_file(conn)
        derived = _derive_per_file(by_file)
        existing = _existing_program_tags(conn)
        reproduced = _reproduced_program_tags(by_file, derived, existing)

        if apply:
            cur = conn.cursor()
            cur.execute("BEGIN")
            try:
                for source_path, rows in by_file.items():
                    meta = derived[source_path]
                    payload = (
                        meta["program_id"],
                        meta["role"],
                        meta["phase"],
                        meta["deliverable_type"],
                    )
                    cur.executemany(
                        "UPDATE memories SET program_id=?, role=?, phase=?, "
                        "deliverable_type=? WHERE id=?",
                        [payload + (row_id,) for row_id, _chunk in rows],
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            # Report from the committed state.
            cov = _coverage_from_conn(conn)
            print("MODE: APPLY (changes committed)")
            _print_coverage(cov, label="committed", reproduced=reproduced)
        else:
            cov = _coverage_from_derived(by_file, derived)
            print("MODE: DRY-RUN (no changes written; pass --apply to write)")
            _print_coverage(cov, label="dry-run (derived)", reproduced=reproduced)
        return {"coverage": cov, "reproduced": reproduced, "applied": apply}
    finally:
        conn.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill memories metadata columns (file-level, dry-run by default)."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="path to lab_memory.db")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write the UPDATEs (default is dry-run, writes nothing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="explicit dry-run (the default; writes nothing)",
    )
    args = parser.parse_args(argv)
    apply_backfill(args.db, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
