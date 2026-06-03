import json
import sqlite3
import pytest
from pathlib import Path


@pytest.fixture
def fake_repo(tmp_path):
    """A minimal repo-shaped dir with one of each data source."""
    root = tmp_path / "repo"
    (root / "data/eval").mkdir(parents=True)
    (root / "data/infra").mkdir(parents=True)
    (root / "data/memories").mkdir(parents=True)
    (root / "programs/prog_a").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)

    # experiments.db
    db = root / "data/experiments.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE experiments (id INTEGER PRIMARY KEY, created_at TEXT,
        updated_at TEXT, stream TEXT, type TEXT, title TEXT, description TEXT,
        status TEXT, result TEXT, metrics TEXT, tags TEXT, parent_id INTEGER,
        commit_hash TEXT, config_json TEXT, random_seed INTEGER, data_version TEXT)""")
    con.execute("INSERT INTO experiments (created_at, stream, type, status, title) VALUES "
                "('2026-04-13 03:00:00','math','hypothesis','proposed','t1')")
    con.execute("INSERT INTO experiments (created_at, stream, type, status, title) VALUES "
                "('2026-04-20 03:00:00','arch','training','complete','t2')")
    con.commit(); con.close()

    # cost_rollup.json
    (root / "data/infra/cost_rollup.json").write_text(json.dumps({
        "schema_version": "1.0", "generated_at": "2026-05-30T05:14:54Z",
        "week": "2026-W21", "total_cost_usd": 42.5, "total_input_tokens": 1000,
        "total_output_tokens": 2000, "session_count": 17, "by_model": {}}))

    # scorecard.md
    (root / "data/eval/scorecard.md").write_text(
        "# Scorecard\n## Latest Scores\n"
        "### MMLU\n- **Overall accuracy**: 0.2895 (66/228)\n"
        "### HellaSwag\n- **Overall accuracy**: 0.2300 (23/100)\n")

    # calibration telemetry (jsonl) — matches the REAL lab schema: each record has
    # `confidence` (float) + `outcome` (truthy=correct, false=wrong, null=unresolved).
    (root / "data/infra/calibration_telemetry.jsonl").write_text(
        '{"confidence": 0.9, "outcome": true}\n'
        '{"confidence": 0.9, "outcome": false}\n'
        '{"confidence": 0.5, "outcome": true}\n')

    # memories log.md (D-N decision headers)
    (root / "data/memories/log.md").write_text(
        "# Log\n\n## D-10 (2026-04-13) — first decision.\nbody\n\n"
        "## D-11 (2026-04-20) — second decision; ORG_ADAPTATION flagged.\nbody\n")

    # evaluator report
    (root / "data/evaluator_report.md").write_text("## Overall: PASS_WITH_FLAGS\nnotes\n")

    # a closure memo (closed program)
    (root / "programs/prog_a/closure_memo.md").write_text(
        "# Closure\npublic_type: envelope\nVerdict O4 binding.\n")
    return root
