"""Inspect laws_de.csv to figure out (numeric law code) -> (abbreviation like ZGB/OR/StGB) mapping.

Hypothesis: laws_de.csv citation column uses numeric SR (Systematische Rechtssammlung) codes,
while gold_citations in train use German abbreviations (ZGB, OR, ...).
The `title` column likely contains the full law name and abbreviation, which lets us auto-build the map.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda_output"
OUT.mkdir(exist_ok=True)

laws = pd.read_csv(DATA / "laws_de.csv")
print(f"laws_de.csv rows: {len(laws):,}")
print(f"columns: {list(laws.columns)}")

# 1) Sample titles & extract trailing token from citation (the "code" suffix)
print("\n--- Title head (5 distinct) ---")
print(laws["title"].drop_duplicates().head(5).tolist())

# Citation pattern observed: "Art. 1 112", "Art. 3 Abs. 1 112"
# The trailing token after spaces is the code.
def trailing_code(cit: str) -> str | None:
    if not isinstance(cit, str):
        return None
    parts = cit.strip().split()
    if not parts:
        return None
    return parts[-1]

laws["code"] = laws["citation"].apply(trailing_code)
unique_codes = laws["code"].dropna().unique()
print(f"\nunique trailing codes in laws_de.csv: {len(unique_codes)}")
print(f"sample codes: {sorted(unique_codes)[:20]}")

# 2) For each code, look at distinct titles and see if abbreviation appears in title
# Typical Swiss law title format: "Schweizerisches Zivilgesetzbuch (ZGB)" or
# "Bundesgesetz ... (XYZ)" where (XYZ) holds the abbreviation.
RE_PAREN_ABBR = re.compile(r"\(([A-Za-zÄÖÜäöüß0-9]{2,10})\)")

def extract_abbr_candidates(title: str) -> list[str]:
    if not isinstance(title, str):
        return []
    # All "(XYZ)" tokens
    return RE_PAREN_ABBR.findall(title)


# Build code -> abbreviation map
code_to_abbrs: dict[str, Counter] = defaultdict(Counter)
code_to_titles: dict[str, Counter] = defaultdict(Counter)
for code, title in zip(laws["code"], laws["title"]):
    if not code:
        continue
    code_to_titles[code][title] += 1
    for ab in extract_abbr_candidates(title):
        code_to_abbrs[code][ab] += 1

# Pick most frequent non-trivial abbreviation per code
code_map: dict[str, str] = {}
code_map_alts: dict[str, list[str]] = {}
for code, ctr in code_to_abbrs.items():
    if not ctr:
        continue
    ranked = ctr.most_common()
    # filter: skip if abbreviation is pure digits or 1 char
    ranked = [(a, c) for a, c in ranked if not a.isdigit() and len(a) >= 2]
    if not ranked:
        continue
    code_map[code] = ranked[0][0]
    code_map_alts[code] = [a for a, _ in ranked[:5]]

print(f"\ncodes with at least one abbreviation extracted: {len(code_map)} / {len(unique_codes)}")
print("sample mappings:")
for c in list(code_map.keys())[:15]:
    print(f"  {c:>6}  -> {code_map[c]:<10}  (alts: {code_map_alts[c]})")
    # also one title for context
    t = code_to_titles[c].most_common(1)[0][0]
    print(f"           title sample: {t[:120]}")

# 3) Reverse map: abbreviation -> code(s). Multiple codes may share an abbreviation? Inspect.
abbr_to_codes: dict[str, list[str]] = defaultdict(list)
for code, ab in code_map.items():
    abbr_to_codes[ab].append(code)
collisions = {a: cs for a, cs in abbr_to_codes.items() if len(cs) > 1}
print(f"\nabbreviation collisions (>1 code shares abbr): {len(collisions)}")
for a, cs in list(collisions.items())[:10]:
    print(f"  {a}: {cs}")

# 4) Now cross-reference with the train gold citations to validate
train = pd.read_csv(DATA / "train.csv")
RE_ART = re.compile(r"^\s*Art\.\s")

def split_cits(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [c.strip() for c in s.split(";") if c.strip()]

gold_articles = []
for s in train["gold_citations"]:
    for c in split_cits(s):
        if RE_ART.match(c):
            gold_articles.append(c)

print(f"\ntrain gold articles total: {len(gold_articles)}")
# What abbreviations show up in gold?
def article_abbr(cit: str) -> str:
    parts = cit.strip().split()
    return parts[-1] if parts else ""

gold_abbrs = Counter(article_abbr(c) for c in gold_articles)
print("top 20 abbreviations in train gold:")
for ab, n in gold_abbrs.most_common(20):
    have = "✓" if ab in abbr_to_codes else "✗"
    code = abbr_to_codes.get(ab, [])
    print(f"  {have} {ab:<10} {n:>4}  code(s)={code}")

# 5) Save outputs
with open(OUT / "law_code_to_abbr.json", "w") as f:
    json.dump({"primary": code_map, "alternatives": code_map_alts}, f, indent=2, ensure_ascii=False)

# Save reverse
abbr_map_serializable = {a: cs for a, cs in abbr_to_codes.items()}
with open(OUT / "abbr_to_law_code.json", "w") as f:
    json.dump(abbr_map_serializable, f, indent=2, ensure_ascii=False)

# Save gold abbr coverage
coverage = {
    "total_gold_articles": len(gold_articles),
    "unique_gold_abbrs": len(gold_abbrs),
    "gold_abbrs_with_code": sum(1 for ab in gold_abbrs if ab in abbr_to_codes),
    "gold_articles_with_known_abbr": sum(n for ab, n in gold_abbrs.items() if ab in abbr_to_codes),
    "top_unknown_abbrs": [(ab, n) for ab, n in gold_abbrs.most_common(50) if ab not in abbr_to_codes],
}
with open(OUT / "gold_abbr_coverage.json", "w") as f:
    json.dump(coverage, f, indent=2, ensure_ascii=False)

print(f"\nGOLD COVERAGE: {coverage['gold_articles_with_known_abbr']}/{coverage['total_gold_articles']} "
      f"= {100*coverage['gold_articles_with_known_abbr']/coverage['total_gold_articles']:.2f}% "
      f"of gold articles have a known abbreviation in our auto-built map")
print(f"top unknown abbreviations (may need manual mapping):")
for ab, n in coverage["top_unknown_abbrs"][:15]:
    print(f"  {ab:<10} {n:>4}")
