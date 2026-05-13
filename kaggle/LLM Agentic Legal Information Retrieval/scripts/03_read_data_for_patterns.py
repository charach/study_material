"""Read train/val/test queries + gold to derive concrete patterns.

Dumps a focused 'insights.txt' that we'll read directly.
"""
from __future__ import annotations
import re
import random
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda_output"

random.seed(7)

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "val.csv")
test = pd.read_csv(DATA / "test.csv")
laws = pd.read_csv(DATA / "laws_de.csv")

def split_cits(s):
    if not isinstance(s, str) or not s.strip(): return []
    return [c.strip() for c in s.split(";") if c.strip()]

def classify(cit: str) -> str:
    if cit.startswith("Art."): return "article"
    if cit.startswith("BGE"): return "bge"
    if re.match(r"^\d+[A-Za-z]_\d+/\d+", cit): return "bger"
    return "other"

lines: list[str] = []
def out(s=""): lines.append(s)

# 1) Dump 5 SHORT train queries (median ~ 145 words) and 5 LONG train queries
train["wc"] = train["query"].fillna("").str.split().str.len()
short_tr = train.nsmallest(5, "wc")
long_tr = train.nlargest(5, "wc").head(5)

out("=" * 100)
out("PART 1: TRAIN QUERIES — short examples (likely typical LEXam style)")
out("=" * 100)
for _, r in short_tr.iterrows():
    out(f"\n--- {r['query_id']} ({r['wc']} words) ---")
    out(f"QUERY: {r['query'][:1500]}")
    out(f"GOLD ({len(split_cits(r['gold_citations']))} cites): {r['gold_citations']}")

out("\n" + "=" * 100)
out("PART 2: TRAIN QUERIES — LONG examples (max length)")
out("=" * 100)
for _, r in long_tr.iterrows():
    out(f"\n--- {r['query_id']} ({r['wc']} words) ---")
    out(f"QUERY (first 1500 chars): {r['query'][:1500]}")
    out(f"GOLD ({len(split_cits(r['gold_citations']))} cites): {r['gold_citations']}")

# 2) ALL 10 val queries — read them fully
out("\n" + "=" * 100)
out("PART 3: ALL 10 VAL QUERIES (full text) — KEY for distribution matching")
out("=" * 100)
for _, r in val.iterrows():
    cits = split_cits(r["gold_citations"])
    art = [c for c in cits if classify(c) == "article"]
    bge = [c for c in cits if classify(c) == "bge"]
    bger = [c for c in cits if classify(c) == "bger"]
    out(f"\n--- {r['query_id']} (gold={len(cits)}: {len(art)} art / {len(bge)} bge / {len(bger)} bger) ---")
    out(f"QUERY: {r['query']}")
    out(f"GOLD ARTICLES: {art}")
    out(f"GOLD BGE:      {bge}")
    out(f"GOLD BGER:     {bger}")

# 3) Random 10 test queries (no gold)
out("\n" + "=" * 100)
out("PART 4: 10 TEST QUERIES (no gold) — compare to val style")
out("=" * 100)
test_sample = test.sample(10, random_state=11)
for _, r in test_sample.iterrows():
    out(f"\n--- {r['query_id']} ---")
    out(f"QUERY: {r['query']}")

# 4) Lexical overlap test: do val gold citations literally appear (as abbreviations or numbers) in the query?
out("\n" + "=" * 100)
out("PART 5: LEXICAL OVERLAP — do gold citations' tokens appear in queries?")
out("=" * 100)
RE_ART_TOKEN = re.compile(r"Art\.\s*(\d+[a-z]?)\s*(?:Abs\.\s*(\d+))?\s*(?:lit\.\s*([a-z]))?\s*([A-Za-zÄÖÜäöüß]+)")
for _, r in val.iterrows():
    cits = split_cits(r["gold_citations"])
    q = r["query"]
    matches = []
    for c in cits:
        # Heuristic: does the law code abbreviation appear verbatim in query?
        m = re.search(r"\b([A-Z][A-Za-zÄÖÜäöüß]{1,8})$", c)
        abbr = m.group(1) if m else ""
        # Also check article number
        art_num_m = re.match(r"Art\.\s*(\d+[a-z]?)", c)
        art_num = art_num_m.group(1) if art_num_m else ""
        abbr_hit = abbr and (abbr.lower() in q.lower())
        num_hit = art_num and (f"art. {art_num}" in q.lower() or f"art {art_num}" in q.lower())
        if abbr_hit or num_hit:
            matches.append((c, "abbr+num" if (abbr_hit and num_hit) else ("abbr" if abbr_hit else "num")))
    out(f"\n{r['query_id']}: gold={len(cits)}, mentions={len(matches)}")
    for c, why in matches:
        out(f"   [{why}] {c}")

# 5) Check val gold citations against corpus citation strings (exact match rate)
out("\n" + "=" * 100)
out("PART 6: VAL GOLD vs CORPUS — exact match coverage")
out("=" * 100)
laws_cit_set = set(laws["citation"].astype(str))
# Quick sample-check on court_considerations (we don't load full 2.4GB here)
import csv
court_cit_set: set[str] = set()
with open(DATA / "court_considerations.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # header
    for i, row in enumerate(reader):
        if row:
            court_cit_set.add(row[0])
        if i % 500_000 == 0:
            print(f"   scanning court_considerations citation column: {i:,}", flush=True)
print(f"   court_considerations unique citations: {len(court_cit_set):,}")

all_corpus_cits = laws_cit_set | court_cit_set
out(f"laws_de unique citations: {len(laws_cit_set):,}")
out(f"court_considerations unique citations: {len(court_cit_set):,}")
out(f"combined corpus citations: {len(all_corpus_cits):,}")

# Per-val-row exact-match
val_hits = []
for _, r in val.iterrows():
    cits = split_cits(r["gold_citations"])
    hits = [c for c in cits if c in all_corpus_cits]
    misses = [c for c in cits if c not in all_corpus_cits]
    val_hits.append((r["query_id"], len(cits), len(hits), misses))
    out(f"\n{r['query_id']}: {len(hits)}/{len(cits)} gold in corpus")
    if misses:
        out(f"   MISSING from corpus:")
        for m in misses[:20]:
            out(f"     - {m}")
        if len(misses) > 20:
            out(f"     ... and {len(misses)-20} more")

# 6) Train gold coverage (entire training set)
out("\n" + "=" * 100)
out("PART 7: TRAIN GOLD vs CORPUS — coverage")
out("=" * 100)
all_train_cits = []
for s in train["gold_citations"]:
    all_train_cits.extend(split_cits(s))
all_train_set = set(all_train_cits)
hit = sum(1 for c in all_train_set if c in all_corpus_cits)
out(f"train unique gold cites: {len(all_train_set):,}")
out(f"  in corpus (exact): {hit:,} ({100*hit/max(1,len(all_train_set)):.2f}%)")

by_type = Counter(classify(c) for c in all_train_set)
type_hit = Counter()
for c in all_train_set:
    if c in all_corpus_cits:
        type_hit[classify(c)] += 1
for t, n in by_type.most_common():
    h = type_hit[t]
    out(f"  type={t}: {h}/{n} = {100*h/max(1,n):.2f}% in corpus")

# 7) Inspect a SAMPLE of train misses to understand the gap
out("\n" + "=" * 100)
out("PART 8: SAMPLE of train gold MISSES from corpus (to find normalization rules)")
out("=" * 100)
misses_train = [c for c in all_train_set if c not in all_corpus_cits]
out(f"total train misses: {len(misses_train):,}")
out("sample 40 misses + what's the closest match in corpus?")
laws_by_prefix = {}
# Quick lookup: per abbreviation, what citations exist?
def article_abbr(cit: str) -> str:
    parts = cit.strip().split()
    return parts[-1] if parts else ""
abbr_to_cits = {}
for c in laws_cit_set:
    ab = article_abbr(c)
    abbr_to_cits.setdefault(ab, []).append(c)

sample_misses = random.sample(misses_train, min(40, len(misses_train)))
for m in sample_misses:
    ab = article_abbr(m)
    art_m = re.match(r"Art\.\s*(\d+[a-z]?)", m)
    art_num = art_m.group(1) if art_m else None
    cand = abbr_to_cits.get(ab, [])
    if art_num:
        same_article = [x for x in cand if re.match(rf"Art\.\s*{art_num}\b", x)]
        same_article = same_article[:5]
    else:
        same_article = cand[:3]
    out(f"  miss: {m!r}")
    out(f"     same Article in corpus: {same_article}")

# 8) Train query examples (10 random) — look at structure
out("\n" + "=" * 100)
out("PART 9: 10 RANDOM TRAIN QUERIES (full text) — actual LEXam structure")
out("=" * 100)
sample_tr = train.sample(10, random_state=22)
for _, r in sample_tr.iterrows():
    cits = split_cits(r["gold_citations"])
    out(f"\n--- {r['query_id']} ({r['wc']} words, gold={len(cits)}) ---")
    out(f"QUERY: {r['query'][:2000]}")
    out(f"GOLD: {r['gold_citations']}")

with open(OUT / "insights.txt", "w") as f:
    f.write("\n".join(lines))
print(f"\nWrote {OUT / 'insights.txt'} ({len(lines)} lines)")
