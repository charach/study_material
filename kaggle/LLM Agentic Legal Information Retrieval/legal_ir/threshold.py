"""Dynamic emit-count calibration based on score-gap elbow + topic prior."""
from __future__ import annotations

import numpy as np


TOPIC_PRIOR: dict[str, int] = {
    "detention_stpo": 45,
    "iv_ivg":         30,
    "erbrecht":       12,
    "family_zgb":     14,
    "werkvertrag":    20,
    "sachenrecht":    20,
    "strafrecht":     30,
    "auftragsrecht":  25,
    "default":        20,
}


def _count_questions(query: str) -> int:
    return query.count("?")


def _word_count(query: str) -> int:
    return len(query.split())


def _elbow(scores: list[float], lo: int = 5, hi: int = 50) -> int:
    if len(scores) < lo + 2:
        return max(1, len(scores))
    arr = np.asarray(scores[: hi + 1], dtype=float)
    gaps = arr[:-1] - arr[1:]
    window = gaps[lo:hi] if hi <= len(gaps) else gaps[lo:]
    if window.size == 0:
        return lo
    return int(np.argmax(window) + lo)


def calibrate_emit_count(
    rerank_scores: list[float],
    query: str,
    topic: str | None = None,
) -> int:
    sorted_scores = sorted(rerank_scores, reverse=True)
    elbow = _elbow(sorted_scores)
    prior = TOPIC_PRIOR.get(topic or "default", TOPIC_PRIOR["default"])
    multi_issue_bonus = _count_questions(query) * 5
    length_bonus = (_word_count(query) // 100) * 2
    n = int(0.5 * prior + 0.4 * elbow + 0.1 * (prior + multi_issue_bonus + length_bonus))
    return max(8, min(50, n))
