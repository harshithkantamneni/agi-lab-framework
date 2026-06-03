from __future__ import annotations
from dataclasses import dataclass, field

VERDICT_LEVELS = ["Strong", "Solid", "Developing", "Weak", "N/A"]


@dataclass
class DimensionResult:
    dimension: str
    metrics: dict                 # name -> value | "unavailable"
    verdict_level: str            # one of VERDICT_LEVELS
    verdict_rationale: str
    relative_to: str
    caveats: list = field(default_factory=list)

    def __post_init__(self):
        if self.verdict_level not in VERDICT_LEVELS:
            raise ValueError(f"verdict_level {self.verdict_level!r} not in {VERDICT_LEVELS}")

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "metrics": self.metrics,
            "verdict": {"level": self.verdict_level, "rationale": self.verdict_rationale,
                        "relative_to": self.relative_to},
            "caveats": list(self.caveats),
        }
