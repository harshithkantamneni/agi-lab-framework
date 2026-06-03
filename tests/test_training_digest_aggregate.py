"""Validates tools/training_digest_aggregate.py — phase-level rollup."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _setup_aggregate_repo(tmp_path):
    """Build a sandbox repo with phase3_factorial run_index containing 4 cells (A42, B42, C42, D42).

    Each cell gets a copy of the A42 stdout fixture (so generate_digest can run for all 4).
    The cells differ in their cmd field (model + lr) so config extraction differentiates them.
    """
    import json
    ckpt_dir = tmp_path / "data/checkpoints/phase3_factorial"
    ckpt_dir.mkdir(parents=True)

    fixture_log_text = (REPO / "tests/fixtures/training_digest/phase3_A42_sample.log").read_text()
    fixture_entry = json.loads(
        (REPO / "tests/fixtures/training_digest/phase3_A42_run_index_entry.json").read_text()
    )

    run_index = {}
    for cell_id, model, lr in [("A42", "dense50m", 0.001), ("B42", "dense50m", 0.002),
                                ("C42", "moe50m_8exp", 0.001), ("D42", "moe50m_8exp", 0.002)]:
        # Copy stdout.log to data/runs/phase3_<cell_id>/stdout.log
        runs_dir = tmp_path / f"data/runs/phase3_{cell_id}"
        runs_dir.mkdir(parents=True)
        (runs_dir / "stdout.log").write_text(fixture_log_text)

        entry = dict(fixture_entry)
        entry["cell"] = cell_id[0]
        entry["seed"] = int(cell_id[1:])
        entry["run_id"] = f"phase3_{cell_id}"
        cmd = (entry["cmd"]
               .replace("dense50m", model)
               .replace("--lr 0.001", f"--lr {lr}")
               .replace("--weight-seed 42", f"--weight-seed {cell_id[1:]}"))
        entry["cmd"] = cmd
        run_index[cell_id] = entry

    (ckpt_dir / "run_index.json").write_text(json.dumps(run_index, indent=2))
    return tmp_path


def test_module_imports():
    import tools.training_digest_aggregate as agg
    assert hasattr(agg, "main")
    assert hasattr(agg, "REPO")
    assert hasattr(agg, "SCHEMA_VERSION")
    assert agg.SCHEMA_VERSION == "1.0"


def test_discover_cells_returns_all_cells_for_phase(tmp_path):
    import tools.training_digest_aggregate as agg
    repo = _setup_aggregate_repo(tmp_path)
    cells = agg.discover_cells("phase3_factorial", repo_root=repo)
    assert isinstance(cells, list)
    assert sorted(cells) == ["A42", "B42", "C42", "D42"]


def test_discover_cells_returns_empty_for_missing_phase(tmp_path):
    import tools.training_digest_aggregate as agg
    cells = agg.discover_cells("nonexistent_phase", repo_root=tmp_path)
    assert cells == []


def test_discover_cells_ignores_non_dict_entries(tmp_path):
    """run_index.json should be a dict[cell_id -> entry]; ignore weird shapes."""
    import json
    import tools.training_digest_aggregate as agg
    ckpt_dir = tmp_path / "data/checkpoints/weird_phase"
    ckpt_dir.mkdir(parents=True)
    # run_index is a list, not a dict — should produce empty cell list
    (ckpt_dir / "run_index.json").write_text(json.dumps(["A42", "B42"]))
    cells = agg.discover_cells("weird_phase", repo_root=tmp_path)
    assert cells == []


def test_main_no_args_prints_usage_and_returns_nonzero():
    import tools.training_digest_aggregate as agg
    rc = agg.main(argv=[])
    assert rc != 0


def test_load_cell_digests_auto_generates_missing(tmp_path, monkeypatch):
    """When per-cell JSON sidecars don't exist yet, loader generates them via generate_digest."""
    import tools.training_digest_aggregate as agg
    import tools.training_digest as td
    repo = _setup_aggregate_repo(tmp_path)
    monkeypatch.setattr(td, "REPO", repo)
    monkeypatch.setattr(agg, "REPO", repo)

    cells = ["A42", "B42", "C42", "D42"]
    digests = agg.load_cell_digests(cells, "phase3_factorial", repo_root=repo)

    assert isinstance(digests, list)
    assert len(digests) == 4
    cell_ids = {d["cell_id"] for d in digests}
    assert cell_ids == set(cells)

    # Sidecars now exist on disk
    for c in cells:
        assert (repo / f"data/digests/training/{c}.json").exists()


def test_load_cell_digests_uses_existing_sidecar(tmp_path, monkeypatch):
    """If a sidecar is fresh, loader reads it instead of regenerating."""
    import tools.training_digest_aggregate as agg
    import tools.training_digest as td
    repo = _setup_aggregate_repo(tmp_path)
    monkeypatch.setattr(td, "REPO", repo)
    monkeypatch.setattr(agg, "REPO", repo)

    # First load — generates sidecars
    agg.load_cell_digests(["A42"], "phase3_factorial", repo_root=repo)
    sidecar = repo / "data/digests/training/A42.json"
    first_mtime = sidecar.stat().st_mtime

    import time
    time.sleep(0.05)

    # Second load — should reuse sidecar (no regen)
    agg.load_cell_digests(["A42"], "phase3_factorial", repo_root=repo)
    second_mtime = sidecar.stat().st_mtime
    assert first_mtime == second_mtime, "expected sidecar reused, but mtime changed"


def test_load_cell_digests_skips_unknown_cell(tmp_path, monkeypatch):
    """Cells without stdout.log should be skipped (not crash the aggregator)."""
    import tools.training_digest_aggregate as agg
    import tools.training_digest as td
    repo = _setup_aggregate_repo(tmp_path)
    monkeypatch.setattr(td, "REPO", repo)
    monkeypatch.setattr(agg, "REPO", repo)

    digests = agg.load_cell_digests(["A42", "GHOST"], "phase3_factorial", repo_root=repo)
    cell_ids = {d["cell_id"] for d in digests}
    assert "A42" in cell_ids
    assert "GHOST" not in cell_ids


def test_build_metrics_table_canonical(tmp_path, monkeypatch):
    """Build a comparison table over 4 cells; assert key columns are present and sane."""
    import tools.training_digest_aggregate as agg
    import tools.training_digest as td
    repo = _setup_aggregate_repo(tmp_path)
    monkeypatch.setattr(td, "REPO", repo)
    monkeypatch.setattr(agg, "REPO", repo)

    digests = agg.load_cell_digests(["A42", "B42", "C42", "D42"], "phase3_factorial", repo_root=repo)
    table = agg.build_metrics_table(digests)

    assert isinstance(table, list)
    assert len(table) == 4
    # Each row is a dict with key columns
    expected_cols = {"cell_id", "model", "lr", "seed", "best_ppl", "recent_avg_lm",
                     "wall_h", "eta_overshoot_pct", "rc", "atexit_clean",
                     "shape_class", "nan_class"}
    for row in table:
        missing = expected_cols - set(row)
        assert not missing, f"missing columns: {missing} in row {row}"

    # Models reflect the cmd-derived config
    by_cell = {r["cell_id"]: r for r in table}
    assert by_cell["A42"]["model"] == "dense50m"
    assert by_cell["A42"]["lr"] == 0.001
    assert by_cell["B42"]["lr"] == 0.002
    assert by_cell["C42"]["model"] == "moe50m_8exp"
    assert by_cell["D42"]["lr"] == 0.002


def test_build_metrics_table_sorts_by_cell_id():
    """Table should be sorted by cell_id for deterministic output."""
    import tools.training_digest_aggregate as agg
    digests = [
        {"cell_id": "C42", "config": {"model": "moe", "lr": 0.001, "seed": 42},
         "outcome": {"rc": 0, "atexit_clean": True, "best_ppl": 3.0, "recent_avg_lm": 5.0},
         "loss_trajectory": {"shape_class": "monotone_decreasing"},
         "nan_incidents": {"classification": "false_positive"}},
        {"cell_id": "A42", "config": {"model": "dense", "lr": 0.001, "seed": 42},
         "outcome": {"rc": 0, "atexit_clean": True, "best_ppl": 2.1, "recent_avg_lm": 5.5},
         "loss_trajectory": {"shape_class": "monotone_decreasing"},
         "nan_incidents": {"classification": "false_positive"}},
        {"cell_id": "B42", "config": {"model": "dense", "lr": 0.002, "seed": 42},
         "outcome": {"rc": 0, "atexit_clean": True, "best_ppl": 2.0, "recent_avg_lm": 5.3},
         "loss_trajectory": {"shape_class": "monotone_decreasing"},
         "nan_incidents": {"classification": "false_positive"}},
    ]
    table = agg.build_metrics_table(digests)
    assert [r["cell_id"] for r in table] == ["A42", "B42", "C42"]


def test_detect_outliers_flags_high_outlier():
    """C42 best_ppl 3.18 vs C-cell-mates median 2.43 — should be flagged."""
    import tools.training_digest_aggregate as agg
    table = [
        {"cell_id": "C42", "model": "moe50m_8exp", "lr": 0.001, "seed": 42, "best_ppl": 3.18},
        {"cell_id": "C43", "model": "moe50m_8exp", "lr": 0.001, "seed": 43, "best_ppl": 2.61},
        {"cell_id": "C44", "model": "moe50m_8exp", "lr": 0.001, "seed": 44, "best_ppl": 2.43},
    ]
    outliers = agg.detect_outliers(table)
    assert len(outliers) == 1
    o = outliers[0]
    assert o["cell_id"] == "C42"
    assert o["direction"] == "high"
    assert o["deviation_pct"] > 20.0
    assert o["group_key"] == ("moe50m_8exp", 0.001)
    # group_median_best_ppl should be 2.61 (median of 3.18, 2.61, 2.43)
    assert abs(o["group_median_best_ppl"] - 2.61) < 1e-3


def test_detect_outliers_flags_low_outlier():
    """A cell that's much better than its cell-mates should also be flagged."""
    import tools.training_digest_aggregate as agg
    table = [
        {"cell_id": "X1", "model": "M", "lr": 0.001, "seed": 42, "best_ppl": 5.0},
        {"cell_id": "X2", "model": "M", "lr": 0.001, "seed": 43, "best_ppl": 5.2},
        {"cell_id": "X3", "model": "M", "lr": 0.001, "seed": 44, "best_ppl": 3.0},  # 40% below median
    ]
    outliers = agg.detect_outliers(table)
    assert len(outliers) == 1
    assert outliers[0]["cell_id"] == "X3"
    assert outliers[0]["direction"] == "low"


def test_detect_outliers_no_outlier_in_tight_group():
    """All cells within 5% of median — no outliers."""
    import tools.training_digest_aggregate as agg
    table = [
        {"cell_id": "A42", "model": "dense", "lr": 0.001, "seed": 42, "best_ppl": 2.12},
        {"cell_id": "A43", "model": "dense", "lr": 0.001, "seed": 43, "best_ppl": 2.14},
        {"cell_id": "A44", "model": "dense", "lr": 0.001, "seed": 44, "best_ppl": 2.17},
    ]
    outliers = agg.detect_outliers(table)
    assert outliers == []


def test_detect_outliers_skips_singleton_groups():
    """A cell-mate group with <2 cells can't establish a reference; skip without crash."""
    import tools.training_digest_aggregate as agg
    table = [
        {"cell_id": "Z42", "model": "alone", "lr": 0.001, "seed": 42, "best_ppl": 10.0},
        {"cell_id": "A42", "model": "dense", "lr": 0.001, "seed": 42, "best_ppl": 2.12},
        {"cell_id": "A43", "model": "dense", "lr": 0.001, "seed": 43, "best_ppl": 2.14},
        {"cell_id": "A44", "model": "dense", "lr": 0.001, "seed": 44, "best_ppl": 2.17},
    ]
    outliers = agg.detect_outliers(table)
    # Z42 is alone in its group — should NOT appear as an outlier
    assert "Z42" not in [o["cell_id"] for o in outliers]


def test_detect_outliers_handles_none_best_ppl():
    """Cells with best_ppl=None should be skipped, not crash."""
    import tools.training_digest_aggregate as agg
    table = [
        {"cell_id": "A42", "model": "dense", "lr": 0.001, "seed": 42, "best_ppl": 2.12},
        {"cell_id": "A43", "model": "dense", "lr": 0.001, "seed": 43, "best_ppl": None},
        {"cell_id": "A44", "model": "dense", "lr": 0.001, "seed": 44, "best_ppl": 2.17},
    ]
    outliers = agg.detect_outliers(table)
    # No outlier among A42 (2.12) and A44 (2.17) — and A43 (None) is skipped
    assert outliers == []


def test_detect_outliers_custom_threshold():
    """Caller can pass a stricter or looser threshold."""
    import tools.training_digest_aggregate as agg
    table = [
        {"cell_id": "A42", "model": "dense", "lr": 0.001, "seed": 42, "best_ppl": 2.12},
        {"cell_id": "A43", "model": "dense", "lr": 0.001, "seed": 43, "best_ppl": 2.14},
        {"cell_id": "A44", "model": "dense", "lr": 0.001, "seed": 44, "best_ppl": 2.50},  # ~17% high
    ]
    # Default threshold 0.20 → no outlier
    assert agg.detect_outliers(table) == []
    # Stricter 0.10 → A44 flagged
    strict = agg.detect_outliers(table, threshold=0.10)
    assert len(strict) == 1
    assert strict[0]["cell_id"] == "A44"


def test_build_anomaly_catalogue_dedups_by_level_and_text():
    """Same anomaly appearing in multiple cells produces one catalogue entry."""
    import tools.training_digest_aggregate as agg
    digests = [
        {"cell_id": "A42", "anomalies": [
            {"line_no": 100, "level": "FATAL", "text": "core dumped"},
            {"line_no": 200, "level": "WARNING", "text": "low memory"},
        ]},
        {"cell_id": "A43", "anomalies": [
            {"line_no": 105, "level": "FATAL", "text": "core dumped"},  # same as A42
            {"line_no": 300, "level": "ERROR", "text": "checkpoint corrupted"},
        ]},
        {"cell_id": "A44", "anomalies": [
            {"line_no": 250, "level": "WARNING", "text": "low memory"},  # same as A42
        ]},
    ]
    cat = agg.build_anomaly_catalogue(digests)
    assert len(cat) == 3  # core dumped, checkpoint corrupted, low memory

    # Find each entry
    by_text = {e["text"]: e for e in cat}
    assert sorted(by_text["core dumped"]["cells"]) == ["A42", "A43"]
    assert by_text["checkpoint corrupted"]["cells"] == ["A43"]
    assert sorted(by_text["low memory"]["cells"]) == ["A42", "A44"]


def test_build_anomaly_catalogue_sorted_by_level():
    """FATAL first, then ERROR, then WARNING."""
    import tools.training_digest_aggregate as agg
    digests = [
        {"cell_id": "X", "anomalies": [
            {"line_no": 1, "level": "WARNING", "text": "w"},
            {"line_no": 2, "level": "FATAL", "text": "f"},
            {"line_no": 3, "level": "ERROR", "text": "e"},
        ]},
    ]
    cat = agg.build_anomaly_catalogue(digests)
    assert [e["level"] for e in cat] == ["FATAL", "ERROR", "WARNING"]


def test_build_anomaly_catalogue_empty_when_no_anomalies():
    import tools.training_digest_aggregate as agg
    digests = [
        {"cell_id": "A42", "anomalies": []},
        {"cell_id": "B42", "anomalies": []},
    ]
    assert agg.build_anomaly_catalogue(digests) == []


def test_build_anomaly_catalogue_handles_missing_anomalies_key():
    """A digest with no `anomalies` key (e.g., partial digest) should not crash."""
    import tools.training_digest_aggregate as agg
    digests = [
        {"cell_id": "A42"},  # missing anomalies
        {"cell_id": "B42", "anomalies": [
            {"line_no": 1, "level": "FATAL", "text": "boom"},
        ]},
    ]
    cat = agg.build_anomaly_catalogue(digests)
    assert len(cat) == 1
    assert cat[0]["cells"] == ["B42"]


def test_render_summary_markdown_canonical(tmp_path, monkeypatch):
    """Real 4-cell phase produces a markdown with all sections."""
    import tools.training_digest_aggregate as agg
    import tools.training_digest as td
    repo = _setup_aggregate_repo(tmp_path)
    monkeypatch.setattr(td, "REPO", repo)
    monkeypatch.setattr(agg, "REPO", repo)

    cells = ["A42", "B42", "C42", "D42"]
    digests = agg.load_cell_digests(cells, "phase3_factorial", repo_root=repo)
    table = agg.build_metrics_table(digests)
    outliers = agg.detect_outliers(table)
    catalogue = agg.build_anomaly_catalogue(digests)

    md = agg.render_summary_markdown(
        phase="phase3_factorial",
        metrics_table=table,
        outliers=outliers,
        anomaly_catalogue=catalogue,
    )

    # Title with phase
    assert "phase3_factorial" in md.lower() or "Phase 3" in md

    # Cell count
    assert "4" in md  # cells

    # Cell-by-cell table — all 4 cells appear
    for cell in cells:
        assert cell in md

    # Headers from the metrics table
    assert "best_ppl" in md or "Best PPL" in md
    assert "shape" in md.lower() or "shape_class" in md

    # Outliers section
    assert "Outliers" in md or "outliers" in md.lower()

    # Anomaly catalogue section
    assert "Anomal" in md or "anomal" in md.lower()

    # Footer cite
    assert "tools/training_digest_aggregate.py" in md or "training_digest_aggregate" in md


def test_main_writes_md_and_json_files(tmp_path, monkeypatch):
    import tools.training_digest_aggregate as agg
    import tools.training_digest as td
    repo = _setup_aggregate_repo(tmp_path)
    monkeypatch.setattr(td, "REPO", repo)
    monkeypatch.setattr(agg, "REPO", repo)

    rc = agg.main(argv=["phase3_factorial"])
    assert rc == 0

    md_path = repo / "data/digests/training/phase3_factorial_summary.md"
    json_path = repo / "data/digests/training/phase3_factorial_summary.json"
    assert md_path.exists(), "expected phase3_factorial_summary.md"
    assert json_path.exists(), "expected phase3_factorial_summary.json"

    md = md_path.read_text()
    assert "A42" in md and "B42" in md and "C42" in md and "D42" in md


def test_main_returns_nonzero_for_unknown_phase(tmp_path, monkeypatch):
    import tools.training_digest_aggregate as agg
    monkeypatch.setattr(agg, "REPO", tmp_path)
    rc = agg.main(argv=["nonexistent_phase"])
    assert rc != 0


def test_summary_json_has_schema_version_and_no_anova(tmp_path, monkeypatch):
    """The JSON sidecar has schema_version and passes the ANOVA-exclusion walk."""
    import tools.training_digest_aggregate as agg
    import tools.training_digest as td
    import json
    repo = _setup_aggregate_repo(tmp_path)
    monkeypatch.setattr(td, "REPO", repo)
    monkeypatch.setattr(agg, "REPO", repo)

    rc = agg.main(argv=["phase3_factorial"])
    assert rc == 0

    json_path = repo / "data/digests/training/phase3_factorial_summary.json"
    summary = json.loads(json_path.read_text())
    assert summary["schema_version"] == "1.0"
    assert summary["phase"] == "phase3_factorial"

    # ANOVA-exclusion walk
    BANNED = {"anova_F", "f_statistic", "p_value", "bonferroni_alpha",
              "effect_size", "degrees_of_freedom", "mean_square",
              "sum_of_squares", "confidence_interval", "null_hypothesis",
              "alternative_hypothesis"}
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                assert k not in BANNED, f"banned ANOVA field {k!r} in summary JSON"
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(summary)


def test_aggregator_output_does_not_contain_anova_fields(tmp_path, monkeypatch):
    """Recursively walk the aggregator's full output and assert no banned ANOVA keys appear."""
    import tools.training_digest_aggregate as agg
    import tools.training_digest as td
    repo = _setup_aggregate_repo(tmp_path)
    monkeypatch.setattr(td, "REPO", repo)
    monkeypatch.setattr(agg, "REPO", repo)

    cells = ["A42", "B42", "C42", "D42"]
    digests = agg.load_cell_digests(cells, "phase3_factorial", repo_root=repo)
    table = agg.build_metrics_table(digests)
    outliers = agg.detect_outliers(table)
    catalogue = agg.build_anomaly_catalogue(digests)

    # The composite output that would go in the aggregate JSON
    composite = {
        "metrics_table": table,
        "outliers": outliers,
        "anomaly_catalogue": catalogue,
    }

    BANNED = {
        "anova_F", "f_statistic", "p_value", "bonferroni_alpha",
        "effect_size", "degrees_of_freedom", "mean_square",
        "sum_of_squares", "confidence_interval", "null_hypothesis",
        "alternative_hypothesis",
    }

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                assert k not in BANNED, \
                    f"banned ANOVA field {k!r} found in aggregator output at {path}"
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(composite, "<root>")
