"""Task A — Citation Normalization & Mapping.

Implements normalize_and_resolve(citation, corpus_set, corpus_children_index) and measures
how much the train/val matching rate improves.

Rules applied:
1. Whitespace / dot normalization.
2. Multilingual abbreviation mapping (FR/IT → DE): LAI→IVG, LPM→MSchG, CO→OR, etc.
3. Strip non-essential modifiers (E., S., ff., etc.) for case-law shape only.
4. Parent ↔ child resolution:
   - If gold='Art. 187 OR' not in corpus, but corpus has Abs.-children: snap to a representative child OR expand to all children.
   - If gold='Art. 477 Abs. 2 ZGB' not in corpus, but parent 'Art. 477 ZGB' exists: snap to parent.
5. Preserve law-code variants (lit. = Bst., Ziff. = Nr., bis/ter ordinals).
"""
from __future__ import annotations
import re
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda_output"

# ---------- 1) Build the corpus citation set + parent→children index ----------
import pandas as pd
laws = pd.read_csv(DATA / "laws_de.csv")
court_cits: set[str] = set()
with open(DATA / "court_considerations.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            court_cits.add(row[0])
laws_cits: set[str] = set(laws["citation"].astype(str))
corpus_cits: set[str] = laws_cits | court_cits
print(f"laws cits: {len(laws_cits):,} | court cits: {len(court_cits):,} | combined: {len(corpus_cits):,}")

# ---------- 2) Citation parsing ----------
ART_PATTERN = re.compile(
    r"^Art\.\s*(?P<art>\d+[a-z]?(?:bis|ter|quater)?)"
    r"(?:\s*Abs\.\s*(?P<abs>\d+[a-z]?(?:bis|ter)?))?"
    r"(?:\s*lit\.\s*(?P<lit>[a-zA-Z]+))?"
    r"(?:\s*Ziff\.\s*(?P<ziff>\d+))?"
    r"\s+(?P<code>[A-Za-zÄÖÜäöüß0-9\.\-]+)$"
)
BGE_PATTERN = re.compile(r"^BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.?\s*[\d\.a-z]+)?$")
BGER_PATTERN = re.compile(r"^\d+[A-Za-z]_\d+/\d+(?:\s+E\.?\s*[\d\.a-z]+)?$")


def parse_article(cit: str) -> Optional[dict]:
    m = ART_PATTERN.match(cit.strip())
    if not m:
        return None
    return m.groupdict()


# Build parent → children index for laws_de.csv
parent_to_children: dict[str, list[str]] = defaultdict(list)
child_to_parent: dict[str, str] = {}
all_law_parsed: dict[str, dict] = {}
for c in laws_cits:
    p = parse_article(c)
    if not p:
        continue
    all_law_parsed[c] = p
    # parent key: "Art. {art} {code}" — drop Abs/lit/Ziff
    parent = f"Art. {p['art']} {p['code']}"
    if c != parent:
        parent_to_children[parent].append(c)
        child_to_parent[c] = parent

print(f"parsed law articles: {len(all_law_parsed):,}")
print(f"unique parents with ≥1 child: {len(parent_to_children):,}")

# ---------- 3) Multilingual abbreviation map (FR/IT → DE primary) ----------
MULTILANG_ABBR = {
    # Code-level abbreviations seen in queries
    "LAI": "IVG", "LAA": "UVG", "LAVS": "AHVG", "LAMal": "KVG",
    "LPP": "BVG", "LPGA": "ATSG", "LACI": "AVIG",
    "CO": "OR", "CC": "ZGB",
    "CP": "StGB", "CPP": "StPO",
    "CPC": "ZPO",
    "LTF": "BGG",
    "LDIP": "IPRG",
    "LIFD": "DBG",
    "LP": "SchKG",
    "LPM": "MSchG",
    "LCD": "UWG",
    "LDA": "URG",
    "LCart": "KG",
    "LB": "BankG",
    "LDA": "URG",
    "LAT": "RPG",
    "LPE": "USG",
    "LRTV": "RTVG",
    "LCR": "SVG",
    "LStup": "BetmG",
    "Cst.": "BV", "Cst": "BV",
    "Costituzione": "BV",
    # Italian
    "CCS": "ZGB", "CPS": "StGB", "CPP-I": "StPO",
}


def expand_multilang(cit: str) -> list[str]:
    """Return [original] + any variants with FR/IT code swapped to DE primary."""
    variants = {cit}
    m = parse_article(cit)
    if m:
        code = m["code"]
        if code in MULTILANG_ABBR:
            new = MULTILANG_ABBR[code]
            variants.add(cit[: cit.rfind(code)] + new)
    return list(variants)


# ---------- 4) Normalization & resolution ----------
def normalize_whitespace(cit: str) -> str:
    """Conservative normalization: collapse spaces, fix only 'Art.\\d' (no space).
    Do NOT touch internal dots in 'E. 3.2' etc."""
    s = re.sub(r"\s+", " ", cit.strip())
    s = re.sub(r"\bArt\.(?=\d)", "Art. ", s)
    s = re.sub(r"\bAbs\.(?=\d)", "Abs. ", s)
    s = re.sub(r"\blit\.(?=[a-zA-Z])", "lit. ", s)
    s = re.sub(r"\bZiff\.(?=\d)", "Ziff. ", s)
    return s


def resolve_against_corpus(cit: str) -> tuple[list[str], str]:
    """Try to resolve `cit` to one or more corpus citations.
    Returns (resolved_list, strategy_tag).
    """
    c = normalize_whitespace(cit)

    # Strategy 1: exact match
    if c in corpus_cits:
        return [c], "exact"

    # Strategy 2: multilingual abbreviation
    for v in expand_multilang(c):
        if v != c and v in corpus_cits:
            return [v], "multilang_abbr"

    # Strategy 3: parent → expand to all children (if c is a parent with no exact)
    if c in parent_to_children:
        return parent_to_children[c], "parent_to_children"

    # Strategy 4: child → snap to parent (if c is a missing child but parent exists)
    p = parse_article(c)
    if p:
        parent = f"Art. {p['art']} {p['code']}"
        if parent in corpus_cits and parent != c:
            return [parent], "child_to_parent"
        # Also try multilang on parent
        for v in expand_multilang(parent):
            if v in corpus_cits and v != c:
                return [v], "parent_multilang"

    # Strategy 5: drop "lit." or "Ziff." modifier and try
    if p and (p["lit"] or p["ziff"]):
        # Try without lit
        stripped = f"Art. {p['art']}"
        if p["abs"]:
            stripped += f" Abs. {p['abs']}"
        stripped += f" {p['code']}"
        if stripped in corpus_cits:
            return [stripped], "drop_lit_ziff"

    return [], "no_match"


# ---------- 5) Measure on train + val gold ----------
def split_cits(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [c.strip() for c in s.split(";") if c.strip()]


def measure(name: str, df):
    strategy_counter = Counter()
    resolved_count = 0
    total = 0
    per_query_resolved: list[tuple[int, int]] = []
    sample_no_match: list[str] = []
    for _, r in df.iterrows():
        cits = split_cits(r["gold_citations"])
        q_resolved = 0
        for c in cits:
            total += 1
            resolved, strategy = resolve_against_corpus(c)
            strategy_counter[strategy] += 1
            if resolved:
                resolved_count += 1
                q_resolved += 1
            elif strategy == "no_match" and len(sample_no_match) < 20:
                sample_no_match.append(c)
        per_query_resolved.append((len(cits), q_resolved))
    print(f"\n=== {name} ===")
    print(f"total gold citations: {total}")
    for s, n in strategy_counter.most_common():
        print(f"  {s:<22} {n:>5}  ({100*n/total:.2f}%)")
    print(f"  RESOLVED: {resolved_count}/{total} = {100*resolved_count/total:.2f}%")
    print(f"  sample no-match: {sample_no_match[:10]}")
    return {
        "total": total,
        "resolved": resolved_count,
        "resolved_pct": 100*resolved_count/total,
        "strategy_breakdown": dict(strategy_counter),
        "sample_no_match": sample_no_match,
    }


train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "val.csv")
train_res = measure("TRAIN", train)
val_res = measure("VAL", val)

# Save
report = {"train": train_res, "val": val_res, "corpus_sizes": {
    "laws": len(laws_cits), "court": len(court_cits), "combined": len(corpus_cits),
}}
with open(OUT / "normalization_results.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nSaved {OUT / 'normalization_results.json'}")
