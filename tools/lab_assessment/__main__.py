from __future__ import annotations
import argparse
import subprocess
import time
from pathlib import Path

from tools.lab_assessment.sources import Sources
from tools.lab_assessment import research, benchmarks, ops, governance, rag
from tools.lab_assessment.report import render_markdown, render_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compute the lab assessment report.")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--rag-quality", action="store_true",
                    help="also run the heavy known-item retrieval eval (loads models)")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    src = Sources(repo)
    results = [research.compute(src), benchmarks.compute(src), ops.compute(src),
               governance.compute(src), rag.compute(src)]

    if args.rag_quality:
        q = rag.compute_retrieval_quality(repo)
        for r in results:
            if r.dimension == "rag":
                r.metrics.update(q)

    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        sha = "unknown"
    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "lab_assessment.md").write_text(render_markdown(results, generated, sha))
    (out / "lab_assessment.json").write_text(render_json(results, generated, sha))
    print(f"wrote {out/'lab_assessment.md'} and lab_assessment.json")
    for r in results:
        print(f"  {r.dimension:12} {r.verdict_level}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
