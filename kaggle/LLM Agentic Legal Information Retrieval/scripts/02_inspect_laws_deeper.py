"""Find SR codes for top federal-law abbreviations by name-matching in titles."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda_output"

laws = pd.read_csv(DATA / "laws_de.csv")

def trailing_code(cit):
    if not isinstance(cit, str): return None
    parts = cit.strip().split()
    return parts[-1] if parts else None
laws["code"] = laws["citation"].apply(trailing_code)

# For each top abbreviation, name signatures to look for in title text
# Source: standard Swiss SR (Systematische Rechtssammlung) names.
abbr_to_name_signatures = {
    "ZGB":   ["Zivilgesetzbuch"],
    "OR":    ["Obligationenrecht", "Bundesgesetz betreffend die Ergänzung des Schweizerischen Zivilgesetzbuches"],
    "StGB":  ["Strafgesetzbuch"],
    "BV":    ["Bundesverfassung der Schweizerischen Eidgenossenschaft"],
    "ZPO":   ["Zivilprozessordnung"],
    "StPO":  ["Strafprozessordnung"],
    "JStG":  ["Jugendstrafgesetz"],
    "JStPO": ["Jugendstrafprozessordnung"],
    "BGG":   ["Bundesgerichtsgesetz", "Bundesgesetz über das Bundesgericht"],
    "URG":   ["Urheberrechtsgesetz", "Bundesgesetz über das Urheberrecht"],
    "DSG":   ["Datenschutzgesetz"],
    "AIG":   ["Ausländer- und Integrationsgesetz", "Bundesgesetz über die Ausländerinnen und Ausländer"],
    "RPG":   ["Raumplanungsgesetz"],
    "FIDLEG":["Finanzdienstleistungsgesetz"],
    "VwVG":  ["Verwaltungsverfahrensgesetz"],
    "AsylG": ["Asylgesetz"],
    "SVG":   ["Strassenverkehrsgesetz"],
    "VStG":  ["Verrechnungssteuergesetz", "Bundesgesetz über die Verrechnungssteuer"],
    "FINMAG":["Finanzmarktaufsichtsgesetz"],
    "UWG":   ["Bundesgesetz gegen den unlauteren Wettbewerb"],
    "USG":   ["Umweltschutzgesetz"],
    "ATSG":  ["Bundesgesetz über den Allgemeinen Teil des Sozialversicherungsrechts"],
    "BankG": ["Bankengesetz"],
    "UVG":   ["Unfallversicherungsgesetz"],
    "IRSG":  ["Rechtshilfegesetz", "Bundesgesetz über internationale Rechtshilfe in Strafsachen"],
    "IPRG":  ["Bundesgesetz über das Internationale Privatrecht"],
    "DBG":   ["Bundesgesetz über die direkte Bundessteuer"],
    "SchKG": ["Bundesgesetz über Schuldbetreibung und Konkurs"],
    "GBV":   ["Grundbuchverordnung"],
    "UVPV":  ["Verordnung über die Umweltverträglichkeitsprüfung"],
    "BVG":   ["Bundesgesetz über die berufliche Alters-, Hinterlassenen- und Invalidenvorsorge"],
    "IVG":   ["Invalidenversicherungsgesetz", "Bundesgesetz über die Invalidenversicherung"],
    "AHVG":  ["Bundesgesetz über die Alters- und Hinterlassenenversicherung"],
    "KAG":   ["Kollektivanlagengesetz", "Bundesgesetz über die kollektiven Kapitalanlagen"],
    "LugÜ":  ["Lugano", "Übereinkommen über die gerichtliche Zuständigkeit"],
    "MWSTG": ["Mehrwertsteuergesetz"],
    "MStG":  ["Militärstrafgesetz"],
    "MStP":  ["Militärstrafprozess"],
    "BÜPF":  ["Überwachung des Post- und Fernmeldeverkehrs"],
    "AVIG":  ["Arbeitslosenversicherungsgesetz"],
    "ArG":   ["Arbeitsgesetz"],
    "GwG":   ["Geldwäschereigesetz"],
    "FusG":  ["Fusionsgesetz"],
    "PartG": ["Partnerschaftsgesetz"],
    "FMG":   ["Fernmeldegesetz"],
    "BetmG": ["Betäubungsmittelgesetz"],
    "WaG":   ["Waldgesetz"],
    "GSchG": ["Gewässerschutzgesetz"],
}

found: dict[str, list[tuple[str, int]]] = {}
for abbr, sigs in abbr_to_name_signatures.items():
    code_counter = Counter()
    for sig in sigs:
        mask = laws["title"].fillna("").str.contains(sig, regex=False, na=False)
        if mask.any():
            codes = laws.loc[mask, "code"].dropna().tolist()
            code_counter.update(codes)
    found[abbr] = code_counter.most_common(5)

print("=== ABBR -> top SR codes by title signature ===")
for abbr, lst in found.items():
    if lst:
        print(f"  {abbr:<8} -> {lst}")
    else:
        print(f"  {abbr:<8} -> (NO MATCH)")

# Pick best (most rows) single code per abbreviation
abbr_to_code = {abbr: lst[0][0] for abbr, lst in found.items() if lst}

# Cross-check: how many gold articles can we now map?
train = pd.read_csv(DATA / "train.csv")
import re
RE_ART = re.compile(r"^\s*Art\.\s")
def split_cits(s):
    if not isinstance(s, str) or not s.strip(): return []
    return [c.strip() for c in s.split(";") if c.strip()]
def article_abbr(cit):
    parts = cit.strip().split()
    return parts[-1] if parts else ""

gold_articles = []
for s in train["gold_citations"]:
    for c in split_cits(s):
        if RE_ART.match(c):
            gold_articles.append(c)

gold_abbrs = Counter(article_abbr(c) for c in gold_articles)
covered = sum(n for ab, n in gold_abbrs.items() if ab in abbr_to_code)
print(f"\nGOLD ARTICLE COVERAGE BY KNOWN ABBR: "
      f"{covered}/{len(gold_articles)} = {100*covered/len(gold_articles):.2f}%")

uncovered = [(ab, n) for ab, n in gold_abbrs.most_common() if ab not in abbr_to_code]
print(f"\nTop UNCOVERED gold abbreviations:")
for ab, n in uncovered[:30]:
    print(f"  {ab:<12} {n:>4}")

# Save
with open(OUT / "federal_abbr_to_sr_code.json", "w") as f:
    json.dump({
        "abbr_to_code": abbr_to_code,
        "all_matches_per_abbr": {a: l for a, l in found.items()},
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved abbr->code map ({len(abbr_to_code)} entries) to eda_output/federal_abbr_to_sr_code.json")
