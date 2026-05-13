"""Universal + topic-conditional default citations.

Derived from val data analysis: `Art. 100 Abs. 1 BGG` appears in 9/10 val gold sets.
Topic defaults are conservative — only include high-frequency, high-confidence anchors.
"""
from __future__ import annotations

UNIVERSAL_DEFAULTS: list[str] = [
    "Art. 100 Abs. 1 BGG",
]

TOPIC_DEFAULTS: dict[str, list[str]] = {
    "detention_stpo": [
        "Art. 221 StPO", "Art. 226 StPO", "Art. 227 StPO",
        "Art. 228 StPO", "Art. 229 StPO",
        "Art. 5 Abs. 3 BV", "Art. 31 BV", "Art. 10 Abs. 2 BV",
    ],
    "family_zgb": [
        "Art. 133 ZGB", "Art. 176 ZGB", "Art. 285 ZGB",
    ],
    "werkvertrag_or": [
        "Art. 363 OR", "Art. 368 OR", "Art. 371 OR",
    ],
    "erbrecht_zgb": [
        "Art. 457 ZGB", "Art. 458 ZGB", "Art. 522 ZGB",
    ],
    "strafrecht": [
        "Art. 29 Abs. 2 BV", "Art. 32 BV",
    ],
    "bger_appeal": [
        "Art. 82 BGG", "Art. 113 BGG",
    ],
}


def get_defaults(topic: str | None) -> list[str]:
    base = list(UNIVERSAL_DEFAULTS)
    if topic and topic in TOPIC_DEFAULTS:
        base.extend(TOPIC_DEFAULTS[topic])
    return base
