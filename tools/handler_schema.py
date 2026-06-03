"""Parse YAML schema blocks from work_queue_handlers.md.

Each handler section starts with a header of the form ``## `<type>``` and may
optionally contain a YAML code block with the keys
``expected_deliverable_pattern`` (str | null) and ``next_action_template``
(dict | null). load_schema() returns a dict mapping type -> schema-dict.
Handlers without a YAML block are absent from the result.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def load_schema(path: Path) -> dict[str, dict[str, Any]]:
    """Parse YAML schema blocks from a handlers.md file.

    Args:
        path: Path to a work_queue_handlers.md-style file.

    Returns:
        Dict mapping handler type name -> parsed schema dict.
        Handlers with no YAML block are absent from the result.
    """
    text = Path(path).read_text()
    schemas: dict[str, dict[str, Any]] = {}
    # Split on `## \`<type>\`` headers (optional trailing text like "(urgent)"
    # or "(low priority)" after the closing backtick is ignored).
    # sections[0] is preamble, then alternating (type, body) pairs.
    sections = re.split(r"\n## `([^`]+)`[^\n]*\n", text)
    for i in range(1, len(sections), 2):
        type_name = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        # Clamp body to before any `---` separator so the YAML regex
        # cannot bleed past a section boundary.
        section_end = body.find("\n---\n")
        if section_end != -1:
            body = body[:section_end]
        m = re.search(r"```yaml\n(.*?)\n```", body, re.DOTALL)
        if not m:
            continue
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to parse handler schema. "
                "Install it with: pip install pyyaml"
            )
        block = yaml.safe_load(m.group(1)) or {}
        schemas[type_name] = block
    return schemas


def render_template(template: Any, payload: dict[str, Any]) -> Any:
    """Recursively render a template by substituting {var} placeholders.

    Supports str (substitute placeholders), dict (recurse on values), list
    (recurse on items), and pass-through for anything else.

    Derived vars: ``to_phase_num`` is auto-derived from ``to_phase`` by
    extracting its first numeric run (so ``"P10"`` -> ``"10"``).

    Missing vars are left as ``{var}`` in the output (not replaced with empty
    string), so the caller can detect unrendered placeholders.
    """
    derived = dict(payload)
    if "to_phase" in payload:
        m = re.search(r"\d+", str(payload["to_phase"]))
        if m:
            derived["to_phase_num"] = m.group(0)

    if isinstance(template, str):
        out = template
        for k, v in derived.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(template, dict):
        return {k: render_template(v, payload) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(x, payload) for x in template]
    return template
