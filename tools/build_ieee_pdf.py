#!/usr/bin/env python3
"""tools/build_ieee_pdf.py — convert lab paper markdown to IEEE-conference PDF.

Input: a PI-approved markdown paper draft (e.g.,
       programs/methodology_structural_anti_forgery/paper_draft_v2.md).
Output: same directory, same basename + `_ieee.pdf` suffix.

Pipeline:
    paper.md  -- pandoc + ieee_conference.tex template -->  paper_ieee.tex
    paper_ieee.tex  -- xelatex (×2 for cross-refs) -->  paper_ieee.pdf

The pandoc template `tools/ieee_template/ieee_conference.tex` controls IEEE
formatting: 2-column, IEEEtran conference class, author block convention from
Alt-D paper Appendix E.2 (collective authorship + submitter-of-record).

Usage:
    python3 tools/build_ieee_pdf.py <paper.md>
    python3 tools/build_ieee_pdf.py <paper.md> --abstract-from-section "Abstract"
    python3 tools/build_ieee_pdf.py <paper.md> --keep-tex
    python3 tools/build_ieee_pdf.py <paper.md> --strip-frontmatter

Notes:
    - Markdown YAML frontmatter (--- ... ---) is honored: title/author/abstract/
      keywords map to IEEE template variables.
    - If no YAML frontmatter, falls back: first H1 = title; first paragraph
      after "## Abstract" = abstract; etc.
    - Default to first non-trivial paragraph as abstract if none found.
    - PI-approved drafts have a header section with status / date / authors /
      target venues etc. that should be stripped from the PDF body. Pass
      --strip-frontmatter to drop everything above the first "## " heading
      that's not a YAML block. Use cautiously.

Returns 0 on success, non-zero on any pandoc/xelatex error.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "tools" / "ieee_template" / "ieee_conference.tex"


def extract_or_synthesize_metadata(md_text: str) -> tuple[dict, str]:
    """Return (metadata_dict, body_text). Honors YAML frontmatter if present;
    otherwise infers title/abstract from the markdown structure."""
    metadata: dict = {}
    body = md_text

    # YAML frontmatter
    yaml_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", md_text, re.DOTALL)
    if yaml_match:
        yaml_text = yaml_match.group(1)
        body = yaml_match.group(2)
        for line in yaml_text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                metadata[k.strip()] = v.strip().strip('"').strip("'")

    # Title: last `# ` heading before the first `## ` heading (i.e., the
    # title that immediately precedes the body). This handles drafts that
    # put a `# Change Log` or other frontmatter H1 above the real title;
    # picking the LAST H1 before the body skips those wrappers.
    # Strip code fences first so a `# comment` inside a shell block doesn't
    # match as a heading.
    if "title" not in metadata:
        prose = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
        first_h2 = re.search(r"^##\s+", prose, re.MULTILINE)
        head_region = prose[:first_h2.start()] if first_h2 else prose
        h1_matches = re.findall(r"^#\s+(?!#)(.+?)$", head_region, re.MULTILINE)
        if h1_matches:
            metadata["title"] = h1_matches[-1].strip()

    # Abstract + Keywords: extract from body into IEEE template metadata,
    # then strip from body so pandoc doesn't render them twice. Layout assumed:
    #
    #     ## Abstract
    #     <one-paragraph abstract>
    #
    #     **Keywords:** kw1, kw2, ...
    #
    #     ---
    #
    #     ## 1. Introduction
    #
    # We capture the entire ##Abstract-through-next-## span and remove it,
    # then parse abstract (first para) and keywords (line starting with
    # **Keywords:**) out of that span individually.
    if "abstract" not in metadata:
        am = re.search(
            r"^(##\s*(?:\d+\.\s*)?Abstract\s*\n+(.+?))(?=\n##\s|\Z)",
            body, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if am:
            span = am.group(2).strip()
            # Abstract = text up to the first **Keywords:** marker or first
            # horizontal rule, whichever comes first.
            # Match **Keywords:**, *Keywords:*, _Keywords:_ — markdown bold OR italic
            stop = re.search(
                r"\n\s*(?:\*\*|\*|_)Keywords?:?(?:\*\*|\*|_)|\n\s*---\s*$",
                span, re.MULTILINE)
            abstract_text = (span[:stop.start()] if stop else span).strip()
            abstract_text = re.sub(r"\s+", " ", abstract_text)
            metadata["abstract"] = abstract_text

            if "keywords" not in metadata:
                km = re.search(
                    r"(?:\*\*|\*|_)Keywords?:?(?:\*\*|\*|_)\s*(.+?)(?=\n\s*---|\Z)",
                    span, re.DOTALL | re.IGNORECASE)
                if km:
                    kw_text = re.sub(r"\s+", " ", km.group(1)).strip().rstrip(".")
                    metadata["keywords"] = kw_text

            body = body.replace(am.group(1), "", 1)
            # Trim leftover horizontal rule that often follows the abstract block
            body = re.sub(r"^\s*---\s*\n", "", body, count=1, flags=re.MULTILINE)

    return metadata, body


def strip_lab_frontmatter(body: str) -> str:
    """Drop the lab-internal status block at the top of paper drafts.

    Lab convention: Title is `# X` followed by a metadata block listing
    Date / Author / Status / Implements / Target venues / etc., then the
    real content starts at the first `## ` heading.

    For IEEE submission this metadata is internal; the title goes in the
    template's title variable; everything else gets dropped.
    """
    lines = body.splitlines(keepends=True)
    out = []
    in_intro_block = True
    seen_first_h2 = False

    for line in lines:
        if seen_first_h2:
            out.append(line)
            continue
        if line.startswith("## "):
            seen_first_h2 = True
            out.append(line)
            in_intro_block = False
            continue
        if in_intro_block:
            # Drop everything in the pre-h2 block (title and metadata)
            continue
        out.append(line)
    return "".join(out)


def strip_yaml_frontmatter(text: str) -> str:
    """Remove --- ... --- block from the very top if present."""
    m = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.DOTALL)
    if m:
        return text[m.end():]
    return text


def _rewrite_longtables_to_tablestar(tex_path: Path) -> None:
    """Convert pandoc-generated longtable blocks to IEEE-friendly table*+tabular.

    Pandoc emits, for any markdown table:

        \\begin{longtable}[]{@{}lll@{}}
        \\toprule\\noalign{}
        Header A & Header B & Header C \\\\
        \\midrule\\noalign{}\\endhead
        \\bottomrule\\noalign{}\\endlastfoot
        cell & cell & cell \\\\
        ...
        \\end{longtable}

    For wider/aligned tables pandoc emits a MULTI-LINE column spec:

        \\begin{longtable}[]{@{}
          >{\\raggedright\\arraybackslash}p{(\\linewidth - ...)}
          >{\\raggedright\\arraybackslash}p{(\\linewidth - ...)}
          ...
          @{}}

    longtable cannot exist inside IEEE 2-column layout. table*+tabular spans
    both columns and is the canonical IEEE wide-table form.

    Implementation: when we hit `\\begin{longtable}`, walk forward by
    brace-balance to capture the full column spec across however many lines
    pandoc split it across. Then emit the equivalent table*+tabular pair.
    """
    text = tex_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out_lines: list[str] = []
    inside = False
    after_endhead = False
    skip_until_brace_balance = 0  # >0 = consuming a multi-line colspec
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not inside and stripped.startswith("\\begin{longtable}"):
            # Walk forward across lines balancing braces to capture the
            # complete column spec (which may span multiple lines for
            # raggedright/raggedleft / explicit-width tables).
            buf = stripped
            j = i
            # Find the start of the column-spec block: the brace after the
            # optional `[...]` placement modifier.
            # \begin{longtable}[]{...}  OR  \begin{longtable}{...}
            head_m = re.match(
                r"\\begin\{longtable\}(?:\[[^\]]*\])?\{",
                buf,
            )
            if head_m:
                # Tally braces from after the opening `{` of the colspec.
                depth = 1
                # Slice off everything up to and including the opening brace
                rest = buf[head_m.end():]
                colspec_chunks: list[str] = []
                while True:
                    for ch in rest:
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                break
                    if depth == 0:
                        # Found closing brace within this chunk — capture
                        # everything up to that brace.
                        end_idx = rest.index("}") if "}" in rest else len(rest)
                        # But brace counting could close at a NESTED level
                        # earlier; for the cases pandoc emits there is no
                        # nesting in column specs, so a simple index works.
                        # Walk again to find the actual closing position.
                        d = 1
                        end_idx = -1
                        for k, ch in enumerate(rest):
                            if ch == "{":
                                d += 1
                            elif ch == "}":
                                d -= 1
                                if d == 0:
                                    end_idx = k
                                    break
                        if end_idx >= 0:
                            colspec_chunks.append(rest[:end_idx])
                        break
                    # Need more lines to balance braces.
                    j += 1
                    if j >= len(lines):
                        break
                    colspec_chunks.append(rest)
                    rest = lines[j]
                colspec = "".join(colspec_chunks).replace("\n", " ").strip()
                # Reduce internal whitespace for cleanliness.
                colspec = re.sub(r"\s+", " ", colspec)
                if not colspec:
                    colspec = "@{}l@{}"
            else:
                colspec = "@{}l@{}"
                j = i

            out_lines.append("\\begin{table*}[ht]\n")
            out_lines.append("\\centering\n")
            out_lines.append(f"\\begin{{tabular}}{{{colspec}}}\n")
            inside = True
            after_endhead = False
            i = j + 1
            continue

        if stripped.startswith("\\end{longtable}"):
            out_lines.append("\\end{tabular}\n")
            out_lines.append("\\end{table*}\n")
            inside = False
            after_endhead = False
            i += 1
            continue

        if inside:
            # Strip longtable-specific row-class markers
            if any(t in stripped for t in (
                    "\\endhead", "\\endlastfoot", "\\endfoot", "\\endfirsthead")):
                if "\\endhead" in stripped:
                    after_endhead = True
                i += 1
                continue
            # \noalign{} is longtable-specific; strip but keep the rest
            line = line.replace("\\noalign{}", "")
        out_lines.append(line)
        i += 1
    tex_path.write_text("".join(out_lines), encoding="utf-8")


def _wrap_section_titles_bold(body: str) -> str:
    """Wrap \\section*{...} title content in \\textbf{...} with balanced
    braces. The IEEE template's \\bfseries format hook is observed not to
    win against later font-state changes in some compile paths, so we
    apply bold directly to the title text via \\textbf which is local-
    scoped and survives.
    """
    out = []
    i = 0
    cmds = ("\\section*", "\\subsection*", "\\subsubsection*")
    while i < len(body):
        # Find next sectioning command
        next_pos = -1
        next_cmd = None
        for cmd in cmds:
            p = body.find(cmd + "{", i)
            if p != -1 and (next_pos == -1 or p < next_pos):
                next_pos = p
                next_cmd = cmd
        if next_pos == -1:
            out.append(body[i:])
            break
        # Append everything before the command
        out.append(body[i:next_pos])
        # Position of the opening brace
        open_brace = next_pos + len(next_cmd)
        # Walk to find matching closing brace
        depth = 1
        j = open_brace + 1
        while j < len(body) and depth > 0:
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            j += 1
        # body[open_brace+1 : j-1] is the title content (between { and })
        title_body = body[open_brace + 1:j - 1]
        out.append(f"{next_cmd}{{\\textbf{{{title_body}}}}}")
        i = j
    return "".join(out)


def _unnumber_sectioning_commands(tex_path: Path) -> None:
    """Convert \\section{} -> \\section*{\\textbf{...}} (and sub-variants)
    in the body.

    Operates only on text after \\begin{document} to avoid mangling the
    template's section-format \\def\\section{...} redefinitions in the
    preamble.

    The markdown source carries manual numbers in heading text (e.g.,
    '5. Structural Remediation') that 44+ body cross-references target.
    IEEEtran's auto-numbered \\section{} would prefix a Roman numeral,
    duplicating the count. The starred (unnumbered) form keeps the manual
    numbers standalone. We then wrap each starred title in \\textbf{} so
    headings render bold (the template's \\bfseries format hook does not
    reliably override IEEEtran's font state in all compile paths).
    """
    text = tex_path.read_text(encoding="utf-8")
    marker = "\\begin{document}"
    if marker in text:
        preamble, _, body = text.partition(marker)
        body = re.sub(r"\\section(?!\*)\{", r"\\section*{", body)
        body = re.sub(r"\\subsection(?!\*)\{", r"\\subsection*{", body)
        body = re.sub(r"\\subsubsection(?!\*)\{", r"\\subsubsection*{", body)
        body = _wrap_section_titles_bold(body)
        text = preamble + marker + body
    tex_path.write_text(text, encoding="utf-8")


def _rewrite_figures_to_figurestar(tex_path: Path) -> None:
    """Convert `figure` -> `figure*` so figures span both IEEE columns.

    The lab's paper figures are designed at 7-inch (full IEEE text width)
    and look cramped when rendered into a single ~3.4-inch column.
    `figure*` is the standard IEEE wide-figure form. This is a global
    rewrite — all figures in lab papers are full-width designs.
    """
    text = tex_path.read_text(encoding="utf-8")
    text = text.replace("\\begin{figure}", "\\begin{figure*}")
    text = text.replace("\\end{figure}", "\\end{figure*}")
    tex_path.write_text(text, encoding="utf-8")


def _rewrite_verbatim_to_fvextra(tex_path: Path) -> None:
    """Switch pandoc's plain `verbatim` env to fvextra's `Verbatim`.

    Pandoc with --no-highlight emits `\\begin{verbatim}...\\end{verbatim}` for
    every fenced code block. LaTeX's built-in `verbatim` env does NOT wrap
    long lines, so regex patterns, file paths, and other unbreakable tokens
    overrun the IEEE 2-column layout (~3.4 inches per column at typewriter
    font ≈ 35-40 chars max). fvextra's `Verbatim` env honors the global
    \\fvset{breaklines=true,breakanywhere=true} declared in the template,
    which is what we want.

    This is purely a name-swap — the body content is preserved verbatim.
    """
    text = tex_path.read_text(encoding="utf-8")
    text = text.replace("\\begin{verbatim}", "\\begin{Verbatim}")
    text = text.replace("\\end{verbatim}", "\\end{Verbatim}")
    tex_path.write_text(text, encoding="utf-8")


def run(cmd: list[str], cwd: Path) -> None:
    """Run a subprocess, surfacing output on failure."""
    # capture as bytes (pdflatex emits non-UTF8 bytes for some font names)
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True)
    if result.returncode != 0:
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        sys.stderr.write(f"\nCommand failed: {' '.join(cmd)}\n")
        sys.stderr.write(f"--- stdout ---\n{stdout}\n")
        sys.stderr.write(f"--- stderr ---\n{stderr}\n")
        raise SystemExit(result.returncode)


def build(md_path: Path, *, keep_tex: bool, strip_frontmatter: bool) -> Path:
    md_text = md_path.read_text(encoding="utf-8")

    # Extract metadata FIRST (while title # heading is still present), THEN
    # strip the lab frontmatter block. Order matters — strip removes the title.
    metadata, body = extract_or_synthesize_metadata(md_text)

    if strip_frontmatter:
        body = strip_yaml_frontmatter(body)
        body = strip_lab_frontmatter(body)

    out_dir = md_path.parent.resolve()
    base = md_path.stem
    if base.endswith("_v2"):
        base = base[:-3] + "_ieee_v2"
    elif base.endswith("_v1"):
        base = base[:-3] + "_ieee_v1"
    else:
        base = base + "_ieee"
    tex_path = out_dir / f"{base}.tex"
    pdf_path = out_dir / f"{base}.pdf"

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # Pandoc reads from a temp file (so we can pass the cleaned body)
        clean_md = tmpdir / "clean.md"
        clean_md.write_text(body, encoding="utf-8")

        # Build pandoc command. --shift-heading-level-by=-1 maps the
        # markdown heading levels onto IEEEtran's section ladder:
        #   ##   -> \section{}      (I., II., III., ...)
        #   ###  -> \subsection{}   (A., B., C., ...)
        #   #### -> \subsubsection{}(1), 2), 3), ...)
        # without it, pandoc's default treats ## as \subsection{} (A.,B.,C.)
        # which makes top-level sections look like subsections in IEEE.
        cmd = [
            "pandoc",
            str(clean_md),
            "--from", "markdown+yaml_metadata_block+raw_tex+grid_tables+pipe_tables",
            "--to", "latex",
            "--standalone",
            "--template", str(TEMPLATE_PATH),
            "--shift-heading-level-by=-1",
            "--output", str(tex_path),
            "--no-highlight",
        ]
        for k, v in metadata.items():
            if v:
                cmd.extend(["--metadata", f"{k}={v}"])

        run(cmd, cwd=out_dir)

        # Post-process: pandoc emits longtable for any markdown table, but
        # longtable fundamentally cannot live inside IEEE's 2-column layout
        # (longtable spans pages, IEEE 2-col is column-bound). Rewrite to
        # table*+tabular which spans BOTH columns of an IEEE conf paper —
        # the conventional IEEE solution for any wide table.
        _rewrite_longtables_to_tablestar(tex_path)
        _rewrite_figures_to_figurestar(tex_path)
        _rewrite_verbatim_to_fvextra(tex_path)
        _unnumber_sectioning_commands(tex_path)

        # Compile LaTeX (twice for cross-refs / TOC, though IEEE conf has no TOC).
        # Use pdflatex (not xelatex): IEEEtran.cls + xelatex has broken font
        # encoding — \scshape, \bfseries, \textbf all silently fall back to
        # the regular roman face. pdflatex with IEEEtran works correctly
        # (small caps in section heads, bold via \textbf, etc.).
        for _ in range(2):
            run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory", str(out_dir.resolve()),
                    str(tex_path.resolve()),
                ],
                cwd=out_dir,
            )

    # Cleanup auxiliary files (.aux .log .out)
    for ext in (".aux", ".log", ".out", ".toc"):
        aux = out_dir / f"{base}{ext}"
        aux.unlink(missing_ok=True)

    if not keep_tex:
        tex_path.unlink(missing_ok=True)

    if not pdf_path.exists():
        sys.stderr.write(f"ERROR: PDF not produced at {pdf_path}\n")
        raise SystemExit(2)

    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Build IEEE-conference PDF from a markdown paper.")
    parser.add_argument("md_path", type=Path, help="Path to the markdown paper draft")
    parser.add_argument("--keep-tex", action="store_true",
                        help="Keep the intermediate LaTeX file alongside the PDF")
    parser.add_argument("--strip-frontmatter", action="store_true",
                        help="Drop the lab-internal status block at the top "
                             "of the markdown (date/author/venues table). On by default.")
    parser.add_argument("--no-strip-frontmatter", dest="strip_frontmatter",
                        action="store_false")
    parser.set_defaults(strip_frontmatter=True)
    args = parser.parse_args()

    if not args.md_path.exists():
        sys.stderr.write(f"ERROR: paper not found: {args.md_path}\n")
        return 1
    if not TEMPLATE_PATH.exists():
        sys.stderr.write(f"ERROR: IEEE template missing: {TEMPLATE_PATH}\n")
        return 1
    if not shutil.which("pandoc"):
        sys.stderr.write("ERROR: pandoc not in PATH. brew install pandoc.\n")
        return 1
    if not shutil.which("pdflatex"):
        sys.stderr.write("ERROR: pdflatex not in PATH. Install MacTeX.\n")
        return 1

    pdf_path = build(args.md_path, keep_tex=args.keep_tex,
                     strip_frontmatter=args.strip_frontmatter)
    print(f"OK: {pdf_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
