from __future__ import annotations
import json
import re
import sqlite3
import subprocess
from pathlib import Path

_DEC_RE = re.compile(r"^#{2,3}\s+(D-\d{1,4})\b[^\n]*?—\s*(.*)$", re.MULTILINE)
_SCORE_RE = re.compile(
    r"^###\s+(?P<name>[\w\-]+)\s*\n-\s+\*\*Overall accuracy\*\*:\s+"
    r"(?P<acc>[0-9.]+)\s*\((?P<correct>\d+)/(?P<total>\d+)\)", re.MULTILINE)


class Sources:
    """Read-only access to the lab's raw data. repo_root makes it testable."""

    def __init__(self, repo_root: Path):
        self.root = Path(repo_root)

    # ---- structured ----
    def experiments(self) -> list[dict]:
        db = self.root / "data/experiments.db"
        if not db.exists():
            return []
        con = None
        try:
            con = sqlite3.connect(db)
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute("SELECT * FROM experiments")]
        except sqlite3.Error:
            return []
        finally:
            if con is not None:
                con.close()

    def cost_rollup(self) -> dict | None:
        return self._read_json(self.root / "data/infra/cost_rollup.json")

    def calibration_telemetry(self) -> list[dict]:
        p = self.root / "data/infra/calibration_telemetry.jsonl"
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # ---- markdown ----
    def scorecard_scores(self) -> dict:
        p = self.root / "data/eval/scorecard.md"
        if not p.exists():
            return {}
        text = p.read_text(encoding="utf-8", errors="replace")
        scores = {}
        for m in _SCORE_RE.finditer(text):
            scores[m["name"]] = {"accuracy": float(m["acc"]),
                                 "correct": int(m["correct"]), "total": int(m["total"])}
        return scores

    def decision_headers(self) -> list[dict]:
        p = self.root / "data/memories/log.md"
        out = []
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            for m in _DEC_RE.finditer(text):
                out.append({"id": m.group(1), "headline": m.group(2).strip()})
        return out

    def evaluator_overall(self) -> str | None:
        p = self.root / "data/evaluator_report.md"
        if not p.exists():
            return None
        m = re.search(r"^##\s+Overall:\s+(\w+)", p.read_text(errors="replace"), re.MULTILINE)
        return m.group(1) if m else None

    def closure_memos(self) -> list[dict]:
        out = []
        for memo in sorted(self.root.glob("programs/*/closure_memo.md")):
            out.append({"program": memo.parent.name,
                        "text": memo.read_text(encoding="utf-8", errors="replace")})
        return out

    def papers(self) -> list[Path]:
        return sorted(self.root.glob("programs/*/paper_draft*.md"))

    # ---- git ----
    def git_log(self, since: str | None = None) -> list[dict]:
        cmd = ["git", "-C", str(self.root), "log", "--pretty=%H|%ct|%s"]
        if since:
            cmd += [f"--since={since}"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return []
        commits = []
        for line in out.stdout.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({"sha": parts[0], "ts": int(parts[1]), "subject": parts[2]})
        return commits

    # ---- helpers ----
    @staticmethod
    def _read_json(p: Path):
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return None
