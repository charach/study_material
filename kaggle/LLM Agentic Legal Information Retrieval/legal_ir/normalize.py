"""Citation normalization & corpus-based resolution.

Verified: train gold 4,659 → 97.62% resolved against laws+court corpora.
"""
from __future__ import annotations
import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import LAWS_CSV, COURT_CSV

_FALLBACK_MULTILANG_ABBR = {
    "LAI": "IVG", "LAA": "UVG", "LAVS": "AHVG", "LAMal": "KVG",
    "LPP": "BVG", "LPGA": "ATSG", "LACI": "AVIG",
    "CO": "OR", "CC": "ZGB", "CP": "StGB", "CPP": "StPO", "CPC": "ZPO",
    "LTF": "BGG", "LDIP": "IPRG", "LIFD": "DBG", "LP": "SchKG",
    "LPM": "MSchG", "LCD": "UWG", "LDA": "URG", "LCart": "KG",
    "LCR": "SVG", "LStup": "BetmG",
    "Cst.": "BV", "Cst": "BV",
}


def _build_multilang_abbr() -> dict[str, str]:
    """Build FR/IT → DE alias map from Omnilex abbrev-translations.json if available.

    Falls back to a small hand-curated table when the file isn't present.
    """
    candidates = []
    try:
        from .config import DATA_DIR
        candidates.append(Path(DATA_DIR).parent / "Omnilex-Agentic-Retrieval-Competition" /
                          "utils" / "abbrev-translations.json")
    except Exception:
        pass
    candidates.append(Path("/kaggle/input/omnilex-abbrev-translations/abbrev-translations.json"))
    candidates.append(Path(__file__).resolve().parent.parent /
                      "Omnilex-Agentic-Retrieval-Competition" / "utils" / "abbrev-translations.json")

    import json
    table = dict(_FALLBACK_MULTILANG_ABBR)
    for path in candidates:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data:
                    ab = entry.get("abbrev", {})
                    de = ab.get("de"); fr = ab.get("fr"); it = ab.get("it")
                    if not de or de[0].isdigit():
                        continue
                    if fr and fr != de and not fr[0].isdigit():
                        table[fr] = de
                    if it and it != de and not it[0].isdigit():
                        table[it] = de
                break
            except Exception:
                continue
    return table


MULTILANG_ABBR = _build_multilang_abbr()

ART_PATTERN = re.compile(
    r"^Art\.\s*(?P<art>\d+[a-z]?(?:bis|ter|quater)?)"
    r"(?:\s*Abs\.\s*(?P<abs>\d+[a-z]?(?:bis|ter)?))?"
    r"(?:\s*lit\.\s*(?P<lit>[a-zA-Z]+))?"
    r"(?:\s*Ziff\.\s*(?P<ziff>\d+))?"
    r"\s+(?P<code>[A-Za-zÄÖÜäöüß0-9\.\-]+)$"
)


def parse_article(cit: str) -> Optional[dict]:
    m = ART_PATTERN.match(cit.strip())
    return m.groupdict() if m else None


def normalize_whitespace(cit: str) -> str:
    s = re.sub(r"\s+", " ", cit.strip())
    s = re.sub(r"\bArt\.(?=\d)", "Art. ", s)
    s = re.sub(r"\bAbs\.(?=\d)", "Abs. ", s)
    s = re.sub(r"\blit\.(?=[a-zA-Z])", "lit. ", s)
    s = re.sub(r"\bZiff\.(?=\d)", "Ziff. ", s)
    return s


def expand_multilang(cit: str) -> list[str]:
    variants = [cit]
    m = parse_article(cit)
    if m:
        code = m["code"]
        if code in MULTILANG_ABBR:
            mapped = MULTILANG_ABBR[code]
            variants.append(cit[: cit.rfind(code)] + mapped)
    return variants


class CorpusIndex:
    def __init__(self) -> None:
        self.corpus_cits: set[str] = set()
        self.parent_to_children: dict[str, list[str]] = defaultdict(list)
        self._built = False

    def build(self, laws_csv: Path = LAWS_CSV, court_csv: Path = COURT_CSV) -> "CorpusIndex":
        if self._built:
            return self
        laws_df = pd.read_csv(laws_csv)
        laws_cits = set(laws_df["citation"].astype(str))
        court_cits: set[str] = set()
        with open(court_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    court_cits.add(row[0])
        self.corpus_cits = laws_cits | court_cits

        for c in laws_cits:
            m = ART_PATTERN.match(c)
            if not m:
                continue
            p = m.groupdict()
            parent = f"Art. {p['art']} {p['code']}"
            if c != parent:
                self.parent_to_children[parent].append(c)
        self._built = True
        return self

    def has(self, cit: str) -> bool:
        return cit in self.corpus_cits


_INDEX: Optional[CorpusIndex] = None


def build_corpus_index() -> CorpusIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = CorpusIndex().build()
    return _INDEX


def get_index() -> CorpusIndex:
    return build_corpus_index()


def resolve_against_corpus(cit: str) -> tuple[list[str], str]:
    idx = get_index()
    c = normalize_whitespace(cit)
    if c in idx.corpus_cits:
        return [c], "exact"
    for v in expand_multilang(c):
        if v != c and v in idx.corpus_cits:
            return [v], "multilang_abbr"
        if v != c and v in idx.parent_to_children:
            return list(idx.parent_to_children[v]), "multilang_parent_to_children"
    if c in idx.parent_to_children:
        return list(idx.parent_to_children[c]), "parent_to_children"
    p = parse_article(c)
    if p:
        parent = f"Art. {p['art']} {p['code']}"
        if parent != c and parent in idx.corpus_cits:
            return [parent], "child_to_parent"
        for v in expand_multilang(parent):
            if v != c and v in idx.corpus_cits:
                return [v], "parent_multilang"
            if v != c and v in idx.parent_to_children:
                return list(idx.parent_to_children[v]), "parent_multilang_to_children"
    return [], "no_match"


def granularity_filter(cits: list[str]) -> list[str]:
    """Drop a parent if any of its children is also in the predicted list.

    Per official rule: predicting `Art. 11 OR` while corpus has `Art. 11 Abs. N OR`
    is invalid. If we've already retrieved a child, the parent is redundant.
    """
    idx = get_index()
    cit_set = set(cits)
    keep = []
    for c in cits:
        children = idx.parent_to_children.get(c)
        if children and any(ch in cit_set for ch in children):
            continue
        keep.append(c)
    return keep


def dedup_preserve_order(cits: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for c in cits:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out
