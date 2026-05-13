"""EDA for Kaggle LLM Agentic Legal Information Retrieval.

Outputs JSON stats + sample rows under ./eda_output/.
Run: python eda.py
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "eda_output"
OUT.mkdir(exist_ok=True)

# ---- Citation type classifier ----
# Patterns observed:
#   Article-style: "Art. 111 ZGB", "Art. 221 Abs. 1 lit. b StPO", "Art. 16 ATSG"
#   BGE:           "BGE 137 IV 122 E. 6.2"  (published Federal Supreme Court decisions)
#   BGer (case#):  "1B_210/2023 E. 4.1", "8C_510/2020 E. 2.4"  (unpublished decisions)
RE_ART = re.compile(r"^\s*Art\.\s")
RE_BGE = re.compile(r"^\s*BGE\s")
RE_BGER = re.compile(r"^\s*\d+[A-Za-z]_\d+/\d+")


def classify(cit: str) -> str:
    if RE_ART.match(cit):
        return "article"
    if RE_BGE.match(cit):
        return "bge"
    if RE_BGER.match(cit):
        return "bger"
    return "other"


def split_cits(s: str) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [c.strip() for c in s.split(";") if c.strip()]


def split_law_code(cit: str) -> str | None:
    """Extract trailing law code abbreviation from an article citation."""
    if not RE_ART.match(cit):
        return None
    parts = cit.strip().split()
    return parts[-1] if parts else None


def lang_guess(text: str) -> str:
    """Crude DE vs EN heuristic via stopword counts."""
    if not isinstance(text, str):
        return "unknown"
    t = text.lower()
    de_hits = sum(t.count(w) for w in (" der ", " die ", " und ", " ist ", " nicht ", " auf ", " sich "))
    en_hits = sum(t.count(w) for w in (" the ", " and ", " is ", " of ", " that ", " for ", " with "))
    fr_hits = sum(t.count(w) for w in (" le ", " la ", " les ", " est ", " et ", " que ", " pour "))
    it_hits = sum(t.count(w) for w in (" il ", " la ", " che ", " di ", " per ", " sono ", " sia "))
    scores = {"de": de_hits, "en": en_hits, "fr": fr_hits, "it": it_hits}
    return max(scores, key=scores.get) if any(scores.values()) else "unknown"


# ---- Load queries ----
print("[1/5] Loading queries ...")
train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "val.csv")
test = pd.read_csv(DATA / "test.csv")
samp = pd.read_csv(DATA / "sample_submission.csv")

print(f"  train: {len(train):,} | val: {len(val):,} | test: {len(test):,} | sample_sub: {len(samp):,}")

# ---- Query-side stats ----
print("[2/5] Query-side stats ...")
stats = {}
for name, df in [("train", train), ("val", val), ("test", test)]:
    qlen = df["query"].fillna("").str.len()
    qwords = df["query"].fillna("").str.split().str.len()
    langs = df["query"].fillna("").apply(lang_guess).value_counts().to_dict()
    s = {
        "rows": int(len(df)),
        "query_char_len": {
            "mean": float(qlen.mean()),
            "median": float(qlen.median()),
            "p90": float(qlen.quantile(0.9)),
            "min": int(qlen.min()),
            "max": int(qlen.max()),
        },
        "query_word_len": {
            "mean": float(qwords.mean()),
            "median": float(qwords.median()),
            "p90": float(qwords.quantile(0.9)),
        },
        "lang_guess": langs,
        "id_format_sample": df["query_id"].iloc[0],
    }
    stats[name] = s

# ---- Gold citation stats (train + val) ----
print("[3/5] Citation distributions ...")
cit_stats = {}
all_train_cits: list[str] = []
all_val_cits: list[str] = []
for name, df, sink in [("train", train, all_train_cits), ("val", val, all_val_cits)]:
    cit_lists = df["gold_citations"].apply(split_cits)
    counts = cit_lists.apply(len)
    types_per_query = cit_lists.apply(lambda lst: Counter(classify(c) for c in lst))
    type_totals = Counter()
    for c in types_per_query:
        type_totals.update(c)
    for lst in cit_lists:
        sink.extend(lst)
    cit_stats[name] = {
        "queries_with_gold": int((counts > 0).sum()),
        "avg_citations_per_query": float(counts.mean()),
        "median_citations_per_query": float(counts.median()),
        "p90_citations_per_query": float(counts.quantile(0.9)),
        "p99_citations_per_query": float(counts.quantile(0.99)),
        "max_citations_per_query": int(counts.max()),
        "min_citations_per_query": int(counts.min()),
        "total_citation_mentions": int(counts.sum()),
        "type_totals": dict(type_totals),
        "type_share_pct": {k: round(100 * v / max(1, counts.sum()), 2) for k, v in type_totals.items()},
    }

# Article law-code distribution (train)
law_codes = Counter()
for c in all_train_cits:
    code = split_law_code(c)
    if code:
        law_codes[code] += 1
cit_stats["train"]["top_law_codes"] = law_codes.most_common(30)

# Unique citation cardinality
cit_stats["train"]["unique_citations"] = len(set(all_train_cits))
cit_stats["val"]["unique_citations"] = len(set(all_val_cits))

# How many val gold citations also seen in train?
train_set = set(all_train_cits)
val_set = set(all_val_cits)
cit_stats["val"]["overlap_with_train_unique"] = len(val_set & train_set)
cit_stats["val"]["coverage_pct_in_train"] = round(100 * len(val_set & train_set) / max(1, len(val_set)), 2)

# ---- Corpus stats ----
print("[4/5] Corpus stats ...")
# laws_de.csv is small enough to load fully
laws = pd.read_csv(DATA / "laws_de.csv")
laws_stats = {
    "rows": int(len(laws)),
    "columns": list(laws.columns),
    "unique_citations": int(laws["citation"].nunique()),
    "sample_citations": laws["citation"].head(10).tolist(),
    "text_char_len_mean": float(laws["text"].fillna("").str.len().mean()),
    "text_char_len_p90": float(laws["text"].fillna("").str.len().quantile(0.9)),
    "title_nunique": int(laws["title"].nunique()) if "title" in laws.columns else None,
}

# Coverage: which fraction of train article citations exist verbatim in laws_de?
law_cit_set = set(laws["citation"].astype(str))
train_articles = [c for c in all_train_cits if classify(c) == "article"]
exact_in_laws = sum(1 for c in train_articles if c in law_cit_set)
laws_stats["train_article_exact_match_in_corpus_pct"] = round(
    100 * exact_in_laws / max(1, len(train_articles)), 2
)
laws_stats["train_article_exact_match_count"] = exact_in_laws
laws_stats["train_article_total"] = len(train_articles)

# court_considerations.csv is 2.4GB -> chunked stats only
print("  scanning court_considerations.csv in chunks (this takes ~1 min) ...")
cc_rows = 0
cc_cits_seen: set[str] = set()
cc_text_chars = 0
cc_first_samples: list[dict] = []
for chunk in pd.read_csv(DATA / "court_considerations.csv", chunksize=200_000):
    cc_rows += len(chunk)
    cc_cits_seen.update(chunk["citation"].astype(str).head(50_000))  # capping memory
    cc_text_chars += int(chunk["text"].fillna("").str.len().sum())
    if not cc_first_samples:
        cc_first_samples = chunk.head(3).to_dict(orient="records")

court_stats = {
    "rows": cc_rows,
    "approx_unique_citations_seen": len(cc_cits_seen),
    "total_text_chars": cc_text_chars,
    "avg_text_chars": cc_text_chars / max(1, cc_rows),
    "sample_rows": [
        {"citation": r["citation"], "text_preview": (r["text"] or "")[:200]} for r in cc_first_samples
    ],
}

# Coverage of BGE/BGer citations in train against court_considerations (approx)
# Note: cc_cits_seen is a capped sample; this is a lower-bound coverage estimate.
train_case_cits = [c for c in all_train_cits if classify(c) in ("bge", "bger")]
court_stats["train_case_citations_total"] = len(train_case_cits)
court_stats["note"] = (
    "approx_unique_citations_seen is capped at first 50k per chunk to save memory; "
    "use for sanity only, not exact coverage."
)

# ---- Final dump ----
print("[5/5] Writing eda_output/ ...")
report = {
    "data_files": {
        "train.csv": int(len(train)),
        "val.csv": int(len(val)),
        "test.csv": int(len(test)),
        "sample_submission.csv": int(len(samp)),
        "laws_de.csv": int(len(laws)),
        "court_considerations.csv": cc_rows,
    },
    "submission_format": {
        "columns": list(samp.columns),
        "sample_rows": samp.to_dict(orient="records"),
        "separator_in_predicted_citations": ";",
        "test_id_format_first": test["query_id"].iloc[0],
        "sample_id_format_first": samp["query_id"].iloc[0],
        "id_format_mismatch_warning": (
            test["query_id"].iloc[0] != samp["query_id"].iloc[0]
        ),
    },
    "queries": stats,
    "citations": cit_stats,
    "laws_corpus": laws_stats,
    "court_corpus": court_stats,
}

with open(OUT / "eda_stats.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Also dump a few sample rows for the report
samples = {
    "train_sample": train.head(3).to_dict(orient="records"),
    "val_sample": val.head(3).to_dict(orient="records"),
    "test_sample": test.head(3).to_dict(orient="records"),
    "laws_sample": laws.head(5).to_dict(orient="records"),
}
with open(OUT / "samples.json", "w") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False, default=str)

print("DONE. See eda_output/eda_stats.json")
