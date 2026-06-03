from __future__ import annotations
import re
from collections import Counter
from tools.lab_assessment.types import DimensionResult
from tools.lab_assessment.sources import Sources

_TYPE_KEYWORDS = [
    ("envelope", ["envelope", "null", "below the floor", "below the measurement"]),
    ("methodology", ["methodology", "governance contribution"]),
    ("reproducibility", ["reproducibility report", "replication"]),
    ("positive", ["positive-result", "positive result"]),
]


def classify_output_type(memo_text: str) -> str:
    m = re.search(r"public_type:\s*(\w+)", memo_text)
    if m:
        return m.group(1).strip()
    low = memo_text.lower()
    for label, kws in _TYPE_KEYWORDS:
        if any(k in low for k in kws):
            return label
    return "unclassified"


def compute(src: Sources) -> DimensionResult:
    memos = src.closure_memos()
    papers = src.papers()
    decisions = src.decision_headers()
    metrics, caveats = {}, []

    metrics["programs_closed"] = len(memos)
    metrics["papers"] = len(papers)
    metrics["decisions_total"] = len(decisions)
    mix = Counter(classify_output_type(m["text"]) for m in memos)
    metrics["output_type_mix"] = dict(mix)

    if not memos:
        level, rationale = "N/A", "no closed programs found"
    elif len(memos) >= 2 and papers:
        level = "Solid"
        rationale = (f"{len(memos)} closed programs producing {len(papers)} paper drafts across "
                     f"{len(mix)} output types — honest mix of positive/null/methodology work")
    else:
        level, rationale = "Developing", f"{len(memos)} closed program(s)"
    caveats.append("Novelty / 'is the science good' is a qualitative judgment, not computed here.")

    return DimensionResult(
        dimension="research", metrics=metrics, verdict_level=level,
        verdict_rationale=rationale,
        relative_to="Real-Mission output goals (publishable small-scale research, four output types)",
        caveats=caveats)
