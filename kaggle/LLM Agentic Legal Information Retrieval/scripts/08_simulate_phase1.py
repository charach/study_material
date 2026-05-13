"""Run Phase 1 locally end-to-end using legal_ir/ modules.

Outputs:
  build/phase1_submission.csv  — to upload to Kaggle's Submit Predictions
  prints VAL Macro F1
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from legal_ir.config import VAL_CSV, TEST_CSV
from legal_ir.normalize import build_corpus_index, MULTILANG_ABBR
from legal_ir.eval import macro_f1, per_query_f1, split_citations
from legal_ir.pipeline import extract_seed_citations
from legal_ir.defaults import get_defaults
from legal_ir.normalize import resolve_against_corpus, granularity_filter, dedup_preserve_order

print(f"multilang abbr entries: {len(MULTILANG_ABBR)}")
idx = build_corpus_index()
print(f"corpus size: {len(idx.corpus_cits):,}, parents: {len(idx.parent_to_children):,}")

def predict(query: str) -> list[str]:
    seeds = extract_seed_citations(query)
    defaults = []
    for d in get_defaults(None):
        r, _ = resolve_against_corpus(d)
        defaults.extend(r)
    return granularity_filter(dedup_preserve_order(seeds + defaults))

print("\n=== VAL evaluation ===")
val = pd.read_csv(VAL_CSV)
gold_d: dict[str, list[str]] = {r["query_id"]: split_citations(r["gold_citations"]) for _, r in val.iterrows()}
pred_d: dict[str, list[str]] = {r["query_id"]: predict(r["query"]) for _, r in val.iterrows()}
score = macro_f1(gold_d, pred_d)
print(f"VAL Macro F1 = {score:.4f}")
for qid in sorted(pred_d):
    g, p = gold_d[qid], pred_d[qid]
    print(f"  {qid}: pred={len(p):>2} gold={len(g):>2} F1={per_query_f1(g, p):.3f}")

print("\n=== TEST submission ===")
test = pd.read_csv(TEST_CSV)
rows = []
for _, r in test.iterrows():
    cits = predict(r["query"])
    rows.append({"query_id": r["query_id"], "predicted_citations": ";".join(cits)})
sub = pd.DataFrame(rows)
out = ROOT / "build" / "phase1_submission.csv"
out.parent.mkdir(exist_ok=True)
sub.to_csv(out, index=False)
print(f"Wrote {out}  rows={len(sub)}")
print(sub.head().to_string(index=False))
counts = sub["predicted_citations"].str.count(";").add(1)
print(f"\nemit count: min={counts.min()} mean={counts.mean():.1f} max={counts.max()}")
