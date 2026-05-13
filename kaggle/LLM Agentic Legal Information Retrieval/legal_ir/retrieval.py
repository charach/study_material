"""Path A: BM25 + BGE-M3 dense retrieval over laws_de.csv with RRF fusion."""
from __future__ import annotations
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import BGE_M3_DIR, LAWS_CSV, WORK_DIR

_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+")


def tokenize_de(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class LawsIndex:
    citations: list[str]
    texts: list[str]
    bm25: object
    dense_index: object  # faiss.IndexFlatIP
    embedding_dim: int


_INDEX: Optional[LawsIndex] = None


def build_laws_index(force_rebuild: bool = False) -> LawsIndex:
    global _INDEX
    if _INDEX is not None and not force_rebuild:
        return _INDEX

    cache_dir = WORK_DIR / "laws_index"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = cache_dir / "bm25.pkl"
    faiss_path = cache_dir / "dense.faiss"
    meta_path = cache_dir / "meta.pkl"

    df = pd.read_csv(LAWS_CSV)
    citations = df["citation"].astype(str).tolist()
    texts = df["text"].fillna("").astype(str).tolist()

    if not force_rebuild and bm25_path.exists() and faiss_path.exists() and meta_path.exists():
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        import faiss
        dense_index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _INDEX = LawsIndex(citations=meta["citations"], texts=meta["texts"],
                           bm25=bm25, dense_index=dense_index,
                           embedding_dim=dense_index.d)
        return _INDEX

    from rank_bm25 import BM25Okapi
    tokenized = [tokenize_de(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(BGE_M3_DIR))
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
            try:
                model.half()
            except Exception:
                pass
    except Exception:
        pass

    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype("float32")

    import faiss
    dense_index = faiss.IndexFlatIP(emb.shape[1])
    dense_index.add(emb)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    faiss.write_index(dense_index, str(faiss_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"citations": citations, "texts": texts}, f)

    _INDEX = LawsIndex(citations=citations, texts=texts, bm25=bm25,
                       dense_index=dense_index, embedding_dim=emb.shape[1])
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
                try:
                    _ENCODER.half()
                except Exception:
                    pass
        except Exception:
            pass
    return _ENCODER


def _encode_query(text: str) -> np.ndarray:
    enc = _get_encoder()
    v = enc.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    return v


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for rank_list, w in zip(rankings, weights):
        for r, cit in enumerate(rank_list):
            scores[cit] = scores.get(cit, 0.0) + w / (k + r + 1)
    return scores


def search_laws(
    query_en: str,
    query_de: str | None = None,
    top_k: int = 200,
    bm25_n: int = 500,
    dense_n: int = 500,
    rrf_weights: tuple[float, float, float] = (0.6, 0.3, 0.3),
) -> list[tuple[str, float]]:
    idx = build_laws_index()

    bm25_query_text = query_de or query_en
    bm25_scores = idx.bm25.get_scores(tokenize_de(bm25_query_text))
    bm25_top = np.argsort(bm25_scores)[::-1][:bm25_n]
    bm25_ranking = [idx.citations[i] for i in bm25_top]

    rankings = [bm25_ranking]
    weights = [rrf_weights[1]]

    q_en = _encode_query(query_en)
    D, I = idx.dense_index.search(q_en, dense_n)
    rankings.append([idx.citations[i] for i in I[0]])
    weights.append(rrf_weights[0])

    if query_de and query_de != query_en:
        q_de = _encode_query(query_de)
        D, I = idx.dense_index.search(q_de, dense_n)
        rankings.append([idx.citations[i] for i in I[0]])
        weights.append(rrf_weights[2])

    fused = reciprocal_rank_fusion(rankings, weights=weights)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    return ranked[:top_k]
