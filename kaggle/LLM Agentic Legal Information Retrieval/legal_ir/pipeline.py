"""End-to-end query → predicted_citations pipeline.

Each stage is fail-soft: missing models / indexes degrade gracefully so the
pipeline still emits *something* even on a stripped-down environment.
"""
from __future__ import annotations
import re
from typing import Optional

from .defaults import get_defaults
from .normalize import (
    resolve_against_corpus,
    granularity_filter,
    dedup_preserve_order,
    get_index,
)
from .threshold import calibrate_emit_count

RE_ART_Q = re.compile(
    r"\bArt(?:icle)?\.?\s*(\d+[a-z]?(?:bis|ter|quater)?)"
    r"(?:\s*(?:Abs\.|al\.|para\.)\s*(\d+))?"
    r"(?:\s*(?:lit\.|let\.)\s*([a-z]))?"
    r"\s+([A-Z][A-Za-zÄÖÜäöüß0-9\.\-]+)"
)
RE_BGE_Q = re.compile(r"\bBGE\s+(\d+)\s+([IVX]+)\s+(\d+)(?:\s+E\.?\s*([\d\.]+))?")
RE_BGER_Q = re.compile(r"\b(\d+[A-Za-z]_\d+/\d+)(?:\s+E\.?\s*([\d\.]+))?")


def extract_seed_citations(query: str) -> list[str]:
    """Regex-extract citation mentions from the query, then resolve to corpus form."""
    seeds: list[str] = []
    for m in RE_ART_Q.finditer(query):
        art, abs_, lit, code = m.groups()
        cand_parts = [f"Art. {art}"]
        if abs_: cand_parts.append(f"Abs. {abs_}")
        if lit: cand_parts.append(f"lit. {lit}")
        cand_parts.append(code)
        cand = " ".join(cand_parts)
        resolved, _ = resolve_against_corpus(cand)
        seeds.extend(resolved)
    for m in RE_BGE_Q.finditer(query):
        vol, book, page, e = m.groups()
        base = f"BGE {vol} {book} {page}"
        if e:
            resolved, _ = resolve_against_corpus(f"{base} E. {e}")
            seeds.extend(resolved)
        resolved, _ = resolve_against_corpus(base)
        seeds.extend(resolved)
    for m in RE_BGER_Q.finditer(query):
        case, e = m.groups()
        if e:
            resolved, _ = resolve_against_corpus(f"{case} E. {e}")
            seeds.extend(resolved)
        resolved, _ = resolve_against_corpus(case)
        seeds.extend(resolved)
    return dedup_preserve_order(seeds)


def _resolve_each(cits: list[str]) -> list[str]:
    out: list[str] = []
    for c in cits:
        resolved, _ = resolve_against_corpus(c)
        out.extend(resolved)
    return out


def _safe_search_laws(query_en: str, query_de: str | None, top_k: int = 200):
    try:
        from .retrieval import search_laws
        return search_laws(query_en, query_de, top_k=top_k)
    except Exception:
        return []


def _safe_search_court(query_en: str, query_de: str | None, top_k_e: int = 200):
    try:
        from .index_court import search_court
        return search_court(query_en, query_de, top_k_e=top_k_e)
    except Exception:
        return []


def _safe_translate(query_en: str) -> str:
    try:
        from .translate import translate_to_de
        return translate_to_de(query_en) or query_en
    except Exception:
        return query_en


def _safe_rerank(query: str, candidate_cits: list[str]) -> list[tuple[str, float]]:
    """Rerank using laws_de text snippets when available."""
    idx = get_index()
    try:
        from .retrieval import build_laws_index
        from .index_court import build_court_index
        from .rerank import cross_encoder_rerank
    except Exception:
        return [(c, 0.0) for c in candidate_cits]

    text_lookup: dict[str, str] = {}
    try:
        laws = build_laws_index()
        for cit, txt in zip(laws.citations, laws.texts):
            text_lookup[cit] = txt
    except Exception:
        pass
    try:
        court = build_court_index()
        for cit, txt in zip(court.e_citations, court.e_texts):
            if cit not in text_lookup:
                text_lookup[cit] = txt
    except Exception:
        pass

    pairs = [(c, text_lookup.get(c, "")) for c in candidate_cits]
    try:
        return cross_encoder_rerank(query, pairs, top_k=len(pairs))
    except Exception:
        return [(c, 0.0) for c in candidate_cits]


def _safe_llm_verify(query: str, ranked: list[tuple[str, float]], budget: int = 60):
    try:
        from .index_court import build_court_index
        from .retrieval import build_laws_index
        from .rerank import llm_verify
    except Exception:
        return [c for c, _ in ranked]
    text_lookup: dict[str, str] = {}
    try:
        laws = build_laws_index()
        for cit, txt in zip(laws.citations, laws.texts):
            text_lookup[cit] = txt
    except Exception:
        pass
    try:
        court = build_court_index()
        for cit, txt in zip(court.e_citations, court.e_texts):
            if cit not in text_lookup:
                text_lookup[cit] = txt
    except Exception:
        pass
    pairs = [(c, text_lookup.get(c, "")) for c, _ in ranked]
    try:
        return llm_verify(query, pairs, budget=budget)
    except Exception:
        return [c for c, _ in ranked]


def run_pipeline(
    query_id: str,
    query: str,
    topic: str | None = None,
    use_llm: bool = True,
) -> list[str]:
    seeds = extract_seed_citations(query)
    defaults = _resolve_each(get_defaults(topic))

    query_de = _safe_translate(query)

    law_results = _safe_search_laws(query, query_de, top_k=200)
    court_results = _safe_search_court(query, query_de, top_k_e=200)

    candidate_cits = dedup_preserve_order(
        [c for c, _ in law_results] + [c for c, _ in court_results]
    )

    if not candidate_cits:
        merged = dedup_preserve_order(seeds + defaults)
        return granularity_filter(merged)[:20]

    reranked = _safe_rerank(query, candidate_cits)
    rerank_scores = [s for _, s in reranked]
    top_for_llm = reranked[:100]
    if use_llm:
        llm_kept = _safe_llm_verify(query, top_for_llm, budget=60)
    else:
        llm_kept = [c for c, _ in top_for_llm]

    n_emit = calibrate_emit_count(rerank_scores, query, topic)

    fixed = dedup_preserve_order(seeds + defaults)
    remaining_budget = max(0, n_emit - len(fixed))

    pool_after_llm = [c for c in llm_kept if c not in set(fixed)]
    rest = pool_after_llm[:remaining_budget]

    final = dedup_preserve_order(fixed + rest)
    final = granularity_filter(final)
    return final[:n_emit]
