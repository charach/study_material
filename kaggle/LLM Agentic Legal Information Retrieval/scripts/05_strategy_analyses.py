"""Analyses backing the strategy section:
 1) Val gold retrievability: regex / lexical / semantic-only — how much of gold can each layer recover?
 2) Topic Canon Jaccard: cluster val queries by gold overlap.
 3) court_considerations base citation structure (BGE without E., BGer without E.).
 4) Train Mode A vs Mode B split.
"""
from __future__ import annotations
import re
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda_output"

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "val.csv")
test = pd.read_csv(DATA / "test.csv")
laws = pd.read_csv(DATA / "laws_de.csv")

def split_cits(s):
    if not isinstance(s, str) or not s.strip(): return []
    return [c.strip() for c in s.split(";") if c.strip()]

def classify(c: str) -> str:
    if c.startswith("Art."): return "article"
    if c.startswith("BGE"): return "bge"
    if re.match(r"^\d+[A-Za-z]_\d+/\d+", c): return "bger"
    return "other"

# Multilang abbreviation map (FR/IT → DE)
MULTILANG_ABBR = {
    "LAI": "IVG", "LAA": "UVG", "LAVS": "AHVG", "LAMal": "KVG",
    "LPP": "BVG", "LPGA": "ATSG", "LACI": "AVIG",
    "CO": "OR", "CC": "ZGB", "CP": "StGB", "CPP": "StPO", "CPC": "ZPO",
    "LTF": "BGG", "LDIP": "IPRG", "LIFD": "DBG", "LP": "SchKG",
    "LPM": "MSchG", "LCD": "UWG", "LDA": "URG", "LCart": "KG",
    "LB": "BankG", "LAT": "RPG", "LPE": "USG", "LRTV": "RTVG",
    "LCR": "SVG", "LStup": "BetmG", "Cst.": "BV", "Cst": "BV",
}

# Regex citation extractors
RE_ART = re.compile(
    r"\bArt(?:icle)?\.?\s*(\d+[a-z]?(?:bis|ter|quater)?)"
    r"(?:\s*(?:Abs\.|al\.|para\.)\s*(\d+))?"
    r"(?:\s*(?:lit\.|let\.|para\.)\s*([a-z]))?"
    r"\s+([A-Z][A-Za-zÄÖÜäöüß0-9\.\-]+)"
)
RE_BGE = re.compile(r"\bBGE\s+(\d+)\s+([IVX]+)\s+(\d+)(?:\s+E\.?\s*([\d\.]+))?")
RE_BGER = re.compile(r"\b(\d+[A-Za-z]_\d+/\d+)(?:\s+E\.?\s*([\d\.]+))?")

# ---------- 1) Val gold retrievability ----------
print("\n=== 1) VAL gold retrievability: regex-extractable vs not ===")
def regex_extract_citations(q: str) -> set[str]:
    out = set()
    for m in RE_ART.finditer(q):
        art, abs_, lit, code = m.groups()
        # Map FR/IT abbreviations to DE primary
        de_code = MULTILANG_ABBR.get(code, code)
        # Build several candidate strings (we don't know exact form):
        for use_code in {code, de_code}:
            if abs_ and lit:
                out.add(f"Art. {art} Abs. {abs_} lit. {lit} {use_code}")
                out.add(f"Art. {art} Abs. {abs_} {use_code}")  # parent
                out.add(f"Art. {art} {use_code}")  # grandparent
            elif abs_:
                out.add(f"Art. {art} Abs. {abs_} {use_code}")
                out.add(f"Art. {art} {use_code}")
            else:
                out.add(f"Art. {art} {use_code}")
    for m in RE_BGE.finditer(q):
        vol, book, page, e = m.groups()
        base = f"BGE {vol} {book} {page}"
        if e:
            out.add(f"{base} E. {e}")
        out.add(base)
    for m in RE_BGER.finditer(q):
        case, e = m.groups()
        if e:
            out.add(f"{case} E. {e}")
        out.add(case)
    return out

# Build corpus citation set for exact-membership check
laws_cits = set(laws["citation"].astype(str))
court_cits: set[str] = set()
with open(DATA / "court_considerations.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f); next(reader)
    for row in reader:
        if row: court_cits.add(row[0])
corpus_cits = laws_cits | court_cits

# Also: corpus text for laws (concatenated) — used for lexical retrieval feasibility
laws_text_index = {cit: text for cit, text in zip(laws["citation"], laws["text"].fillna(""))}

regex_hit_total = 0
regex_hit_in_gold = 0
gold_total = 0
per_q_breakdown = []
for _, r in val.iterrows():
    q = r["query"]
    gold = set(split_cits(r["gold_citations"]))
    extracted = regex_extract_citations(q)
    # Intersect with corpus to be "valid"
    extracted_valid = extracted & corpus_cits
    hits = extracted_valid & gold
    regex_hit_total += len(extracted_valid)
    regex_hit_in_gold += len(hits)
    gold_total += len(gold)
    per_q_breakdown.append({
        "qid": r["query_id"],
        "gold": len(gold),
        "regex_extracted": len(extracted),
        "regex_extracted_valid": len(extracted_valid),
        "regex_correct": len(hits),
        "regex_precision": round(len(hits) / max(1, len(extracted_valid)), 3),
        "regex_recall_of_gold": round(len(hits) / max(1, len(gold)), 3),
    })

print(f"Val total gold: {gold_total}")
print(f"Regex-extracted candidates total: {regex_hit_total}  (corpus-valid only)")
print(f"Regex correct (in gold): {regex_hit_in_gold} = {100*regex_hit_in_gold/gold_total:.2f}% recall of gold")
for x in per_q_breakdown:
    print(f"  {x['qid']}: gold={x['gold']:>2}  regex_correct={x['regex_correct']:>2}  precision={x['regex_precision']}  recall={x['regex_recall_of_gold']}")

# ---------- 2) Topic Canon Jaccard between val queries ----------
print("\n=== 2) Topic Canon: Jaccard between val queries (by gold citation overlap) ===")
val_gold_sets = {r["query_id"]: set(split_cits(r["gold_citations"])) for _, r in val.iterrows()}
qids = list(val_gold_sets.keys())
jaccard = {}
for i, a in enumerate(qids):
    for b in qids[i+1:]:
        A, B = val_gold_sets[a], val_gold_sets[b]
        inter = A & B
        union = A | B
        j = len(inter) / max(1, len(union))
        if len(inter) > 0:
            jaccard[(a, b)] = {
                "jaccard": round(j, 3),
                "intersection": len(inter),
                "a_size": len(A), "b_size": len(B),
                "shared_examples": list(inter)[:8],
            }
# Top overlapping pairs
sorted_pairs = sorted(jaccard.items(), key=lambda kv: -kv[1]["intersection"])
print(f"Val pairs with non-empty intersection: {len(jaccard)}")
for (a, b), d in sorted_pairs[:10]:
    print(f"  {a} ↔ {b}: J={d['jaccard']}  |∩|={d['intersection']}  shared sample={d['shared_examples'][:5]}")

# ---------- 3) court_considerations base citation structure ----------
print("\n=== 3) court_considerations base citation structure ===")
RE_BGE_BASE = re.compile(r"^BGE\s+\d+\s+[IVX]+\s+\d+")
RE_BGER_BASE = re.compile(r"^\d+[A-Za-z]_\d+/\d+")
base_to_e = defaultdict(int)
total_rows = 0
unparsed = 0
type_counter = Counter()
for c in court_cits:
    total_rows += 1
    m = RE_BGE_BASE.match(c)
    if m:
        base_to_e[m.group(0)] += 1
        type_counter["bge"] += 1
        continue
    m = RE_BGER_BASE.match(c)
    if m:
        base_to_e[m.group(0)] += 1
        type_counter["bger"] += 1
        continue
    unparsed += 1
print(f"total court citations: {total_rows:,}")
print(f"  BGE-style: {type_counter['bge']:,}")
print(f"  BGer-style: {type_counter['bger']:,}")
print(f"  unparsed: {unparsed:,}")
print(f"unique base citations (judgments): {len(base_to_e):,}")
e_counts = sorted(base_to_e.values())
print(f"E. per judgment — min={e_counts[0]}, p50={e_counts[len(e_counts)//2]}, "
      f"p90={e_counts[int(0.9*len(e_counts))]}, max={e_counts[-1]}")
print(f"avg E. per judgment: {sum(e_counts)/len(e_counts):.2f}")

# ---------- 4) Train Mode A vs Mode B split ----------
print("\n=== 4) Train Mode A (short Q) vs Mode B (BGE dump) ===")
mode_a = 0
mode_b = 0
mode_b_examples = []
for _, r in train.iterrows():
    q = r["query"]
    is_bge_dump = (
        ("Urteilskopf" in q) or
        re.search(r"\bBGE\s+\d+\s+[IVX]+\s+\d+", q[:300]) is not None or
        ("Regeste" in q[:500] and "Sachverhalt" in q)
    )
    if is_bge_dump:
        mode_b += 1
        if len(mode_b_examples) < 3:
            mode_b_examples.append(r["query_id"])
    else:
        mode_a += 1
print(f"Mode A (short Q): {mode_a}")
print(f"Mode B (BGE dump): {mode_b}")
print(f"Mode B sample IDs: {mode_b_examples}")

# Save
result = {
    "regex_extraction": {
        "val_total_gold": gold_total,
        "val_regex_correct": regex_hit_in_gold,
        "val_regex_recall_pct": round(100*regex_hit_in_gold/gold_total, 2),
        "per_query": per_q_breakdown,
    },
    "topic_canon_jaccard": [
        {"pair": [a, b], **d} for (a, b), d in sorted_pairs
    ],
    "court_corpus": {
        "total_rows": total_rows,
        "bge_rows": type_counter["bge"],
        "bger_rows": type_counter["bger"],
        "unique_judgments": len(base_to_e),
        "avg_e_per_judgment": sum(e_counts) / len(e_counts),
    },
    "train_mode_split": {"mode_a_short": mode_a, "mode_b_bge_dump": mode_b},
}
with open(OUT / "strategy_analyses.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\nSaved {OUT / 'strategy_analyses.json'}")
