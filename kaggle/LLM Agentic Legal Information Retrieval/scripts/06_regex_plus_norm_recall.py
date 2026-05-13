"""Measure regex extraction + normalization (parent-expand) combined recall on val.

Also inspects the 303K 'unparsed' court_considerations citations.
"""
from __future__ import annotations
import re
import json
import csv
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda_output"

val = pd.read_csv(DATA / "val.csv")
laws = pd.read_csv(DATA / "laws_de.csv")

laws_cits = set(laws["citation"].astype(str))
court_cits: set[str] = set()
unparsed_samples: list[str] = []
RE_BGE_BASE = re.compile(r"^BGE\s+\d+\s+[IVX]+\s+\d+")
RE_BGER_BASE = re.compile(r"^\d+[A-Za-z]_\d+/\d+")
with open(DATA / "court_considerations.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f); next(reader)
    for row in reader:
        if not row: continue
        c = row[0]
        court_cits.add(c)
        if not RE_BGE_BASE.match(c) and not RE_BGER_BASE.match(c):
            if len(unparsed_samples) < 50:
                unparsed_samples.append(c)
corpus_cits = laws_cits | court_cits

print(f"Sample of 'unparsed' court citations (first 50):")
for u in unparsed_samples:
    print(f"  {u!r}")

# parent → children index
ART_PATTERN = re.compile(
    r"^Art\.\s*(?P<art>\d+[a-z]?(?:bis|ter|quater)?)"
    r"(?:\s*Abs\.\s*(?P<abs>\d+[a-z]?(?:bis|ter)?))?"
    r"(?:\s*lit\.\s*(?P<lit>[a-zA-Z]+))?"
    r"(?:\s*Ziff\.\s*(?P<ziff>\d+))?"
    r"\s+(?P<code>[A-Za-zÄÖÜäöüß0-9\.\-]+)$"
)
parent_to_children: dict[str, list[str]] = defaultdict(list)
for c in laws_cits:
    m = ART_PATTERN.match(c)
    if not m: continue
    p = m.groupdict()
    parent = f"Art. {p['art']} {p['code']}"
    if c != parent:
        parent_to_children[parent].append(c)

MULTILANG_ABBR = {
    "LAI": "IVG", "LAA": "UVG", "LAVS": "AHVG", "LAMal": "KVG",
    "LPP": "BVG", "LPGA": "ATSG", "LACI": "AVIG",
    "CO": "OR", "CC": "ZGB", "CP": "StGB", "CPP": "StPO", "CPC": "ZPO",
    "LTF": "BGG", "LDIP": "IPRG", "LIFD": "DBG", "LP": "SchKG",
    "LPM": "MSchG", "LCD": "UWG", "LDA": "URG", "LCart": "KG",
    "LB": "BankG", "LAT": "RPG", "LPE": "USG", "LCR": "SVG",
    "LStup": "BetmG", "Cst.": "BV", "Cst": "BV",
}

RE_ART_Q = re.compile(
    r"\bArt(?:icle)?\.?\s*(\d+[a-z]?(?:bis|ter|quater)?)"
    r"(?:\s*(?:Abs\.|al\.|para\.)\s*(\d+))?"
    r"(?:\s*(?:lit\.|let\.|para\.)\s*([a-z]))?"
    r"\s+([A-Z][A-Za-zÄÖÜäöüß0-9\.\-]+)"
)
RE_BGE_Q = re.compile(r"\bBGE\s+(\d+)\s+([IVX]+)\s+(\d+)(?:\s+E\.?\s*([\d\.]+))?")
RE_BGER_Q = re.compile(r"\b(\d+[A-Za-z]_\d+/\d+)(?:\s+E\.?\s*([\d\.]+))?")


def resolve(cit: str) -> list[str]:
    """Try several strategies to find corpus matches."""
    if cit in corpus_cits:
        return [cit]
    if cit in parent_to_children:
        return parent_to_children[cit]
    m = ART_PATTERN.match(cit)
    if m:
        p = m.groupdict()
        parent = f"Art. {p['art']} {p['code']}"
        if parent in corpus_cits and parent != cit:
            return [parent]
    return []


def regex_extract_and_resolve(q: str) -> set[str]:
    out = set()
    for m in RE_ART_Q.finditer(q):
        art, abs_, lit, code = m.groups()
        # Try original + multilang-mapped code
        for c in {code, MULTILANG_ABBR.get(code, code)}:
            candidates = [
                f"Art. {art} Abs. {abs_} lit. {lit} {c}" if (abs_ and lit) else None,
                f"Art. {art} Abs. {abs_} {c}" if abs_ else None,
                f"Art. {art} {c}",
            ]
            for cand in candidates:
                if not cand: continue
                for resolved in resolve(cand):
                    out.add(resolved)
    for m in RE_BGE_Q.finditer(q):
        vol, book, page, e = m.groups()
        base = f"BGE {vol} {book} {page}"
        if e:
            cand = f"{base} E. {e}"
            if cand in corpus_cits: out.add(cand)
        # also try base
        if base in corpus_cits: out.add(base)
    for m in RE_BGER_Q.finditer(q):
        case, e = m.groups()
        if e:
            cand = f"{case} E. {e}"
            if cand in corpus_cits: out.add(cand)
        if case in corpus_cits: out.add(case)
    return out


def split_cits(s):
    if not isinstance(s, str) or not s.strip(): return []
    return [c.strip() for c in s.split(";") if c.strip()]


print("\n=== Regex + Normalization combined recall on val ===")
total_gold = 0
total_correct = 0
total_emitted = 0
rows = []
for _, r in val.iterrows():
    q = r["query"]
    gold = set(split_cits(r["gold_citations"]))
    extracted = regex_extract_and_resolve(q)
    correct = extracted & gold
    total_gold += len(gold)
    total_correct += len(correct)
    total_emitted += len(extracted)
    p = len(correct) / max(1, len(extracted))
    rec = len(correct) / max(1, len(gold))
    f1 = 2*p*rec / max(1e-9, p+rec)
    rows.append({"qid": r["query_id"], "gold": len(gold), "emitted": len(extracted), "correct": len(correct),
                 "p": round(p, 3), "r": round(rec, 3), "f1": round(f1, 3)})
    print(f"  {r['query_id']}: gold={len(gold):>2} emitted={len(extracted):>2} correct={len(correct):>2} P={p:.3f} R={rec:.3f} F1={f1:.3f}")
overall_p = total_correct / max(1, total_emitted)
overall_r = total_correct / max(1, total_gold)
overall_f1 = 2*overall_p*overall_r/max(1e-9, overall_p+overall_r)
# Macro F1 (per-query average)
macro_f1 = sum(x["f1"] for x in rows) / len(rows)
print(f"\nOVERALL: emitted={total_emitted} correct={total_correct} gold={total_gold}")
print(f"  Micro P={overall_p:.3f} R={overall_r:.3f} F1={overall_f1:.3f}")
print(f"  MACRO F1 (competition metric): {macro_f1:.3f}")

with open(OUT / "regex_norm_val_recall.json", "w") as f:
    json.dump({"rows": rows, "macro_f1": macro_f1, "micro_f1": overall_f1,
               "unparsed_sample": unparsed_samples}, f, indent=2, ensure_ascii=False)
print(f"Saved {OUT / 'regex_norm_val_recall.json'}")
