"""Phase-level training digest aggregator — reads all per-cell digests for a
given phase and emits a cross-cell summary at
data/digests/training/<phase>_summary.{md,json}.

Per spec: docs/superpowers/specs/2026-05-05-training-digest-observer.md § 4.

Explicitly NOT in aggregate: ANOVA tables, F statistics, p-values,
hypothesis tests, decision-rule firing, mechanism explanations, benchmark
scores. Those are P9 statistical_reviewer / P10 mechanism_extractor /
P11 measurement_theorist's domains. Test
test_aggregator_does_not_replicate_anova (Task 15) enforces this.

CLI:
    python3 tools/training_digest_aggregate.py <phase> [--regen]
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# Ensure the repo root (parent of tools/) is on sys.path so direct invocation
# (`python tools/training_digest_aggregate.py <phase>`) can resolve the
# sibling `tools.training_digest` import. Without this, only pytest's
# auto-pathing and callers that already set sys.path would work.
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_ON_PATH = str(_TOOLS_DIR.parent)
if _REPO_ROOT_ON_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_ON_PATH)

REPO = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = "1.0"


def discover_cells(phase: str, repo_root: Path | None = None) -> list[str]:
    """Read data/checkpoints/<phase>/run_index.json and return the list of
    cell IDs (top-level keys) in lexicographic order. Empty if missing or
    malformed."""
    root = repo_root if repo_root is not None else REPO
    ri_path = root / "data" / "checkpoints" / phase / "run_index.json"
    if not ri_path.exists():
        return []
    try:
        ri = json.loads(ri_path.read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(ri, dict):
        return []
    return sorted(ri.keys())


def load_cell_digests(cells: list[str], phase: str,
                      repo_root: Path | None = None) -> list[dict]:
    """Load per-cell digest dicts from data/digests/training/<cell_id>.json.

    For cells whose JSON sidecar doesn't exist, calls
    tools.training_digest.generate_digest() to produce it. Cells that can't
    be generated (e.g., missing stdout.log) are silently skipped — the
    aggregator runs over whatever cells succeed.

    Returns a list of digest dicts in the same order as the input cells
    (minus any skipped ones).
    """
    root = repo_root if repo_root is not None else REPO
    digest_dir = root / "data" / "digests" / "training"

    # Lazy import to avoid circular dependency at module load
    from tools.training_digest import generate_digest

    out: list[dict] = []
    for cell_id in cells:
        sidecar = digest_dir / f"{cell_id}.json"
        if sidecar.exists():
            try:
                out.append(json.loads(sidecar.read_text()))
                continue
            except json.JSONDecodeError:
                pass  # fall through to regen

        # Generate if missing or unparseable
        digest = generate_digest(cell_id, phase, repo_root=root)
        if digest is not None:
            out.append(digest)

    return out


def build_metrics_table(cell_digests: list[dict]) -> list[dict]:
    """Build a cell-by-cell comparison table (list of row dicts).

    Columns: cell_id, model, lr, seed, best_ppl, recent_avg_lm, wall_h,
    eta_overshoot_pct, rc, atexit_clean, shape_class, nan_class.

    Sorted by cell_id for deterministic output.
    """
    rows: list[dict] = []
    for d in cell_digests:
        cfg = d.get("config", {}) or {}
        outcome = d.get("outcome", {}) or {}
        lt = d.get("loss_trajectory", {}) or {}
        nan = d.get("nan_incidents", {}) or {}

        rows.append({
            "cell_id": d.get("cell_id"),
            "model": cfg.get("model"),
            "lr": cfg.get("lr"),
            "seed": cfg.get("seed"),
            "best_ppl": outcome.get("best_ppl"),
            "recent_avg_lm": outcome.get("recent_avg_lm"),
            "wall_h": cfg.get("actual_hours"),
            "eta_overshoot_pct": cfg.get("eta_overshoot_pct"),
            "rc": outcome.get("rc"),
            "atexit_clean": outcome.get("atexit_clean"),
            "shape_class": lt.get("shape_class"),
            "nan_class": nan.get("classification"),
        })

    return sorted(rows, key=lambda r: r.get("cell_id") or "")


# Default outlier threshold: |deviation / median| > 20%
_DEFAULT_OUTLIER_THRESHOLD = 0.20


def detect_outliers(metrics_table: list[dict],
                    threshold: float = _DEFAULT_OUTLIER_THRESHOLD) -> list[dict]:
    """Flag cells whose best_ppl deviates from their cell-mate median.

    Cell-mates share (model, lr) and differ only in seed. Groups with
    fewer than 2 valid cells (best_ppl != None) are skipped.

    Returns list of outlier descriptors:
        {cell_id, group_key, group_median_best_ppl, cell_best_ppl,
         deviation_pct (signed), direction ("high" or "low")}

    Sorted by cell_id for determinism.
    """
    # Group rows by (model, lr); skip rows with None best_ppl
    groups: dict[tuple, list[dict]] = {}
    for row in metrics_table:
        if row.get("best_ppl") is None:
            continue
        key = (row.get("model"), row.get("lr"))
        groups.setdefault(key, []).append(row)

    outliers: list[dict] = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        ppls = [m["best_ppl"] for m in members]
        median_ppl = statistics.median(ppls)
        if median_ppl <= 0:
            continue
        for m in members:
            deviation = (m["best_ppl"] - median_ppl) / median_ppl
            if abs(deviation) > threshold:
                outliers.append({
                    "cell_id": m["cell_id"],
                    "group_key": key,
                    "group_median_best_ppl": median_ppl,
                    "cell_best_ppl": m["best_ppl"],
                    "deviation_pct": round(deviation * 100, 2),
                    "direction": "high" if deviation > 0 else "low",
                })

    return sorted(outliers, key=lambda o: o["cell_id"] or "")


# Anomaly level priority for sorting (lower = higher priority)
_LEVEL_PRIORITY = {"FATAL": 0, "ERROR": 1, "WARNING": 2}


def build_anomaly_catalogue(cell_digests: list[dict]) -> list[dict]:
    """Flatten and dedupe per-cell anomalies into a phase-level catalogue.

    Inputs: list of digest dicts (each with `cell_id` and optional `anomalies`).

    Output: list of catalogue entries, each:
        {level, text, cells: [list of cell_ids in which this anomaly appeared]}

    Deduped by (level, text) — the same anomaly across multiple cells
    becomes one entry with multiple cell_ids. Sorted by level priority
    (FATAL → ERROR → WARNING) then alphabetic by text.
    """
    # (level, text) -> set of cell_ids
    grouped: dict[tuple[str, str], set[str]] = {}
    for d in cell_digests:
        cell_id = d.get("cell_id")
        for a in (d.get("anomalies") or []):
            key = (a.get("level", ""), a.get("text", ""))
            grouped.setdefault(key, set()).add(cell_id)

    catalogue = [
        {"level": level, "text": text, "cells": sorted(cells)}
        for (level, text), cells in grouped.items()
    ]

    catalogue.sort(key=lambda e: (
        _LEVEL_PRIORITY.get(e["level"], 99),
        e["text"],
    ))

    return catalogue


from datetime import datetime, timezone


def _format_phase_label(phase: str) -> str:
    """phase3_factorial -> 'Phase 3 Factorial'."""
    parts = phase.replace("_", " ").split()
    return " ".join(p.capitalize() if p[0].isalpha() else p for p in parts)


def render_summary_markdown(phase: str, metrics_table: list[dict],
                             outliers: list[dict],
                             anomaly_catalogue: list[dict]) -> str:
    """Render the phase summary as markdown narrative.

    Sections: title, cell count + total wall, cell-by-cell table, outliers,
    anomaly catalogue, forward-carry candidates, footer.
    """
    label = _format_phase_label(phase)
    cell_count = len(metrics_table)
    cell_ids = [r.get("cell_id") for r in metrics_table]
    total_wall_h = sum((r.get("wall_h") or 0) for r in metrics_table)

    lines: list[str] = []
    lines.append(f"# {label} ({phase}) — Training Summary")
    lines.append("")
    lines.append(f"**Cells:** {cell_count} ({', '.join(str(c) for c in cell_ids)})")
    lines.append(f"**Total wall:** {total_wall_h:.2f}h")
    lines.append("")

    # Cell-by-cell table
    lines.append("## Cell-by-cell metrics")
    lines.append("")
    lines.append("| cell | model | lr | seed | best_ppl | recent_avg_lm | wall_h | shape | nan |")
    lines.append("|------|-------|----|------|----------|---------------|--------|-------|-----|")
    for r in metrics_table:
        lines.append(
            f"| {r.get('cell_id', '?')} | {r.get('model', '?')} | "
            f"{r.get('lr', '?')} | {r.get('seed', '?')} | "
            f"{r.get('best_ppl', '?')} | {r.get('recent_avg_lm', '?')} | "
            f"{r.get('wall_h', '?')} | {r.get('shape_class', '?')} | "
            f"{r.get('nan_class', '?')} |"
        )
    lines.append("")

    # Outliers
    lines.append("## Outliers vs cell-mates")
    lines.append("")
    if not outliers:
        lines.append("None.")
    else:
        for o in outliers:
            sign = "+" if o["deviation_pct"] >= 0 else ""
            group_str = f"{o['group_key'][0]}, lr={o['group_key'][1]}"
            lines.append(
                f"- **{o['cell_id']}**: best_ppl {o['cell_best_ppl']:.3f} vs "
                f"{group_str} cell-mates median {o['group_median_best_ppl']:.3f} "
                f"({sign}{o['deviation_pct']}%). Direction: {o['direction']}."
            )
    lines.append("")

    # Anomaly catalogue
    lines.append("## Anomaly catalogue")
    lines.append("")
    if not anomaly_catalogue:
        lines.append("None.")
    else:
        for a in anomaly_catalogue:
            cells_str = ", ".join(a["cells"])
            lines.append(f"- **{a['level']}** ({cells_str}): {a['text']}")
    lines.append("")

    # Forward-carry candidates — derived from outliers + FATAL/ERROR anomalies
    fwd_candidates: list[str] = []
    for o in outliers:
        fwd_candidates.append(
            f"{o['cell_id']} outlier ({o['direction']} best_ppl) — investigate at P10 mechanism extraction."
        )
    for a in anomaly_catalogue:
        if a["level"] in ("FATAL", "ERROR"):
            cells_str = ", ".join(a["cells"])
            fwd_candidates.append(f"{a['level']} in {cells_str}: {a['text']}")
    lines.append("## Forward-carry candidates")
    lines.append("")
    if not fwd_candidates:
        lines.append("None.")
    else:
        for c in fwd_candidates:
            lines.append(f"- {c}")
    lines.append("")

    # Footer
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    lines.append("---")
    lines.append(
        f"*Generated by tools/training_digest_aggregate.py from "
        f"data/digests/training/*.json on {today}.*"
    )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-cell training digests into a phase summary."
    )
    parser.add_argument("phase", nargs="?", help="Phase name (e.g., phase3_factorial)")
    parser.add_argument("--regen", action="store_true", help="Force regen even if summary is fresh.")
    args = parser.parse_args(argv)

    if not args.phase:
        parser.print_usage(sys.stderr)
        return 2

    root = REPO

    cells = discover_cells(args.phase, repo_root=root)
    if not cells:
        print(f"error: no cells found for phase {args.phase!r}", file=sys.stderr)
        return 1

    digests = load_cell_digests(cells, args.phase, repo_root=root)
    if not digests:
        print(f"error: no digests could be loaded/generated for phase {args.phase!r}",
              file=sys.stderr)
        return 1

    table = build_metrics_table(digests)
    outliers = detect_outliers(table)
    catalogue = build_anomaly_catalogue(digests)

    md = render_summary_markdown(args.phase, table, outliers, catalogue)
    summary_json = {
        "schema_version": SCHEMA_VERSION,
        "phase": args.phase,
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "cell_count": len(table),
        "metrics_table": table,
        "outliers": outliers,
        "anomaly_catalogue": catalogue,
    }

    digest_dir = root / "data" / "digests" / "training"
    digest_dir.mkdir(parents=True, exist_ok=True)
    (digest_dir / f"{args.phase}_summary.md").write_text(md)
    (digest_dir / f"{args.phase}_summary.json").write_text(json.dumps(summary_json, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
