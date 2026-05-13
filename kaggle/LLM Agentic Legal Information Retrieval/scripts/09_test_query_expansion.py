"""Test alias expansion: ensure citation-context restriction kills false positives."""
from __future__ import annotations
import sys, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from legal_ir.config import VAL_CSV, TEST_CSV
from legal_ir.normalize import MULTILANG_ABBR

CITATION_ALIAS_RE = re.compile(
    r'\b(?:Art(?:icle)?\.?|al\.?)\s*\d+[a-z]?'
    r'(?:\s*(?:Abs\.|al\.|para\.)\s*\d+)?'
    r'(?:\s*(?:lit\.|let\.)\s*[a-z])?'
    r'\s+([A-Z][A-Za-zÄÖÜäöüß0-9\.\-]+)'
)

def expand(q):
    found = set()
    for m in CITATION_ALIAS_RE.finditer(q):
        code = m.group(1)
        de = MULTILANG_ABBR.get(code)
        if de and de != code:
            found.add((code, de))
    return found

for name, path in [("VAL", VAL_CSV), ("TEST", TEST_CSV)]:
    df = pd.read_csv(path)
    counts = []
    print(f"\n=== {name} ===")
    for _, r in df.iterrows():
        ex = expand(r["query"])
        counts.append(len(ex))
        if ex:
            print(f"  {r['query_id']}: {[f'{a}->{b}' for a,b in ex]}")
    n_with_match = sum(1 for c in counts if c > 0)
    print(f"queries with ≥1 expansion: {n_with_match}/{len(counts)}")
    print(f"total expansion events: {sum(counts)}")
