"""Path B: 2-stage indexing for court_considerations.csv (2.4GB, 1.99M E.-level rows).

Stage 1: judgment-level (140K aggregated docs) — dense top-K.
Stage 2: expand selected judgments to all their E. rows → cross-rank.
"""
from __future__ import annotations
import csv
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import BGE_M3_DIR, COURT_CSV, WORK_DIR

RE_BGE = re.compile(r"^BGE\s+\d+\s+[IVX]+\s+\d+")
RE_BGER_NEW = re.compile(r"^\d+[A-Za-z]_\d+/\d+")
RE_BGER_LEGACY = re.compile(r"^([A-Z])\s*\d+/\d+\s+\d{2}\.\d{2}\.\d{4}")
RE_BGER_LEGACY_C = re.compile(r"^\d+[A-Z]\.\d+/\d+\s+\d{2}\.\d{2}\.\d{4}")


def judgment_id_from_citation(citation: str) -> Optional[str]:
    """Strip the ' E. X.Y' suffix; keep the judgment identifier itself."""
    m = RE_BGE.match(citation)
    if m: return m.group(0)
    m = RE_BGER_NEW.match(citation)
    if m: return m.group(0)
    m = RE_BGER_LEGACY.match(citation)
    if m: return m.group(0)
    m = RE_BGER_LEGACY_C.match(citation)
    if m: return m.group(0)
    return None


@dataclass
class CourtIndexes:
    judgment_ids: list[str]
    judgment_concat_texts: list[str]
    judgment_dense: object       # faiss.IndexFlatIP
    e_citations: list[str]
    e_to_judgment_idx: list[int]
    e_texts: list[str]
    e_dense: object              # faiss.IndexIVFPQ or IndexFlatIP


_INDEX: Optional[CourtIndexes] = None


def _aggregate_judgments(court_csv: Path, e_cap_per_judgment: int = 30,
                         char_cap_per_judgment: int = 4000) -> tuple[list[str], list[str], list[str], list[int], list[str]]:
    """Stream the CSV once. Return (judgment_ids, judgment_texts, e_citations, e_to_jidx, e_texts).

    Memory caution: 2.4GB on disk; aggregate text into chunks while iterating.
    """
    bucket_texts: dict[str, list[str]] = defaultdict(list)
    e_citations: list[str] = []
    e_to_jid: list[str] = []
    e_texts: list[str] = []

    with open(court_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            cit, text = row[0], row[1] or ""
            jid = judgment_id_from_citation(cit)
            if jid is None:
                continue
            e_citations.append(cit)
            e_to_jid.append(jid)
            e_texts.append(text)
            if len(bucket_texts[jid]) < e_cap_per_judgment:
                bucket_texts[jid].append(text)

    judgment_ids = list(bucket_texts.keys())
    j_id_to_idx = {j: i for i, j in enumerate(judgment_ids)}
    judgment_texts: list[str] = []
    for j in judgment_ids:
        concat = "\n".join(bucket_texts[j])
        if len(concat) > char_cap_per_judgment:
            concat = concat[:char_cap_per_judgment]
        judgment_texts.append(concat)

    e_to_jidx = [j_id_to_idx[j] for j in e_to_jid]
    return judgment_ids, judgment_texts, e_citations, e_to_jidx, e_texts


def build_court_index(force_rebuild: bool = False,
                      e_index_quantized: bool = True) -> CourtIndexes:
    global _INDEX
    if _INDEX is not None and not force_rebuild:
        return _INDEX

    cache_dir = WORK_DIR / "court_index"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "j_ids": cache_dir / "judgment_ids.pkl",
        "j_dense": cache_dir / "judgment.faiss",
        "j_texts": cache_dir / "judgment_texts.pkl",
        "e_meta": cache_dir / "e_meta.pkl",
        "e_dense": cache_dir / "e.faiss",
    }

    if not force_rebuild and all(p.exists() for p in paths.values()):
        import faiss
        with open(paths["j_ids"], "rb") as f:
            judgment_ids = pickle.load(f)
        with open(paths["j_texts"], "rb") as f:
            judgment_texts = pickle.load(f)
        j_dense = faiss.read_index(str(paths["j_dense"]))
        with open(paths["e_meta"], "rb") as f:
            e_meta = pickle.load(f)
        e_dense = faiss.read_index(str(paths["e_dense"]))
        _INDEX = CourtIndexes(
            judgment_ids=judgment_ids, judgment_concat_texts=judgment_texts,
            judgment_dense=j_dense,
            e_citations=e_meta["e_citations"], e_to_judgment_idx=e_meta["e_to_jidx"],
            e_texts=e_meta["e_texts"], e_dense=e_dense,
        )
        return _INDEX

    judgment_ids, judgment_texts, e_citations, e_to_jidx, e_texts = \
        _aggregate_judgments(COURT_CSV)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(BGE_M3_DIR))
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
            try: model.half()
            except Exception: pass
    except Exception:
        pass

    import faiss
    j_emb = model.encode(judgment_texts, batch_size=32, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True).astype("float32")
    j_dense = faiss.IndexFlatIP(j_emb.shape[1])
    j_dense.add(j_emb)

    e_emb = model.encode(e_texts, batch_size=64, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True).astype("float32")
    if e_index_quantized and len(e_emb) > 50_000:
        nlist = 4096
        quantizer = faiss.IndexFlatIP(e_emb.shape[1])
        e_dense = faiss.IndexIVFPQ(quantizer, e_emb.shape[1], nlist, 64, 8)
        e_dense.train(e_emb)
        e_dense.add(e_emb)
        e_dense.nprobe = 32
    else:
        e_dense = faiss.IndexFlatIP(e_emb.shape[1])
        e_dense.add(e_emb)

    with open(paths["j_ids"], "wb") as f:
        pickle.dump(judgment_ids, f)
    with open(paths["j_texts"], "wb") as f:
        pickle.dump(judgment_texts, f)
    faiss.write_index(j_dense, str(paths["j_dense"]))
    with open(paths["e_meta"], "wb") as f:
        pickle.dump({"e_citations": e_citations, "e_to_jidx": e_to_jidx, "e_texts": e_texts}, f)
    faiss.write_index(e_dense, str(paths["e_dense"]))

    _INDEX = CourtIndexes(
        judgment_ids=judgment_ids, judgment_concat_texts=judgment_texts,
        judgment_dense=j_dense, e_citations=e_citations,
        e_to_judgment_idx=e_to_jidx, e_texts=e_texts, e_dense=e_dense,
    )
    return _INDEX


_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer(str(BGE_M3_DIR))
        try:
            import torch
            if torch.cuda.is_available():
                _ENCODER = _ENCODER.to("cuda")
                try: _ENCODER.half()
                except Exception: pass
        except Exception:
            pass
    return _ENCODER


def _encode_query(text: str) -> np.ndarray:
    enc = _get_encoder()
    return enc.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")


def search_court(
    query_en: str,
    query_de: str | None = None,
    top_k_judgment: int = 50,
    top_k_e: int = 200,
) -> list[tuple[str, float]]:
    idx = build_court_index()
    q_en = _encode_query(query_en)
    D_en, I_en = idx.judgment_dense.search(q_en, top_k_judgment)
    selected_jidx = set(int(i) for i in I_en[0] if i >= 0)
    if query_de and query_de != query_en:
        q_de = _encode_query(query_de)
        D_de, I_de = idx.judgment_dense.search(q_de, top_k_judgment)
        selected_jidx.update(int(i) for i in I_de[0] if i >= 0)

    e_indices = [i for i, jidx in enumerate(idx.e_to_judgment_idx) if jidx in selected_jidx]
    if not e_indices:
        return []

    e_emb_sub: list[np.ndarray] = []
    if hasattr(idx.e_dense, "reconstruct_n"):
        try:
            e_emb_full = idx.e_dense.reconstruct_n(0, idx.e_dense.ntotal)
            e_emb_sub_arr = e_emb_full[e_indices]
        except Exception:
            from sentence_transformers import SentenceTransformer  # noqa
            enc = _get_encoder()
            sub_texts = [idx.e_texts[i] for i in e_indices]
            e_emb_sub_arr = enc.encode(sub_texts, normalize_embeddings=True,
                                       convert_to_numpy=True, batch_size=64).astype("float32")
    else:
        enc = _get_encoder()
        sub_texts = [idx.e_texts[i] for i in e_indices]
        e_emb_sub_arr = enc.encode(sub_texts, normalize_embeddings=True,
                                   convert_to_numpy=True, batch_size=64).astype("float32")

    scores = e_emb_sub_arr @ q_en[0]
    if query_de and query_de != query_en:
        scores_de = e_emb_sub_arr @ q_de[0]
        scores = scores + scores_de
    order = np.argsort(-scores)[:top_k_e]
    return [(idx.e_citations[e_indices[k]], float(scores[k])) for k in order]
