"""Step 2 verification: legal_ir.normalize.resolve_against_corpus on train gold.

Expected: 97%+ resolved.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from legal_ir.config import TRAIN_CSV, VAL_CSV
from legal_ir.normalize import build_corpus_index, resolve_against_corpus
from legal_ir.eval import split_citations, macro_f1
from legal_ir.pipeline import extract_seed_citations
from legal_ir.defaults import get_defaults

print("[1] Imports OK. Building corpus index ...")
idx = build_corpus_index()
print(f"    corpus citations: {len(idx.corpus_cits):,}")
print(f"    parents with children: {len(idx.parent_to_children):,}")

print("\n[2] Resolving TRAIN gold ...")
train = pd.read_csv(TRAIN_CSV)
strategy = Counter()
total = 0; resolved = 0
sample_no_match: list[str] = []
for _, r in train.iterrows():
    for c in split_citations(r["gold_citations"]):
        total += 1
        out, tag = resolve_against_corpus(c)
        strategy[tag] += 1
        if out:
            resolved += 1
        elif len(sample_no_match) < 8:
            sample_no_match.append(c)
print(f"    total={total}  resolved={resolved}  rate={100*resolved/total:.2f}%")
for k, v in strategy.most_common():
    print(f"      {k:<22} {v:>5} ({100*v/total:.2f}%)")
print(f"    sample no-match: {sample_no_match}")
assert resolved / total >= 0.97, "expected ≥97% resolved on train"

print("\n[3] Resolving VAL gold ...")
val = pd.read_csv(VAL_CSV)
vtotal = 0; vresolved = 0
for _, r in val.iterrows():
    for c in split_citations(r["gold_citations"]):
        vtotal += 1
        out, _ = resolve_against_corpus(c)
        if out:
            vresolved += 1
print(f"    total={vtotal}  resolved={vresolved}  rate={100*vresolved/vtotal:.2f}%")

print("\n[4] Pipeline smoke test (regex+defaults only, no models) on val ...")
gold_dict: dict[str, list[str]] = {}
pred_dict: dict[str, list[str]] = {}
for _, r in val.iterrows():
    gold_dict[r["query_id"]] = split_citations(r["gold_citations"])
    seeds = extract_seed_citations(r["query"])
    defaults = []
    for d in get_defaults(None):
        resolved_d, _ = resolve_against_corpus(d)
        defaults.extend(resolved_d)
    pred = list(dict.fromkeys(seeds + defaults))
    pred_dict[r["query_id"]] = pred
    print(f"    {r['query_id']}: pred={len(pred)} (seeds={len(seeds)}, defaults={len(defaults)})")
f1 = macro_f1(gold_dict, pred_dict)
print(f"\n    Macro F1 (regex+defaults only) = {f1:.4f}")

print("\nDONE.")
