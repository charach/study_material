"""Macro F1 (per-query F1 averaged). Exact string match, order-invariant."""
from __future__ import annotations
from typing import Mapping


def per_query_f1(gold: list[str] | set[str], pred: list[str] | set[str]) -> float:
    g, p = set(gold), set(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(g)
    return 2 * precision * recall / (precision + recall)


def macro_f1(
    gold_dict: Mapping[str, list[str]],
    pred_dict: Mapping[str, list[str]],
) -> float:
    """Average per-query F1 over the union of query ids in gold/pred."""
    qids = set(gold_dict) | set(pred_dict)
    if not qids:
        return 0.0
    total = 0.0
    for qid in qids:
        total += per_query_f1(gold_dict.get(qid, []), pred_dict.get(qid, []))
    return total / len(qids)


def split_citations(s: str | None) -> list[str]:
    if not s or not isinstance(s, str) or not s.strip():
        return []
    return [c.strip() for c in s.split(";") if c.strip()]


def load_gold(df) -> dict[str, list[str]]:
    return {row["query_id"]: split_citations(row["gold_citations"]) for _, row in df.iterrows()}
