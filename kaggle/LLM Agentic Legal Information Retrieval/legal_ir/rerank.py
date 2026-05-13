"""Cross-encoder reranker + LLM verifier (both local, fail-soft)."""
from __future__ import annotations
import re
from typing import Optional

from .config import BGE_RERANKER_DIR, QWEN_DIR

_RERANK = None
_RERANK_FAILED = False


def _get_reranker():
    global _RERANK, _RERANK_FAILED
    if _RERANK is not None or _RERANK_FAILED:
        return _RERANK
    try:
        from sentence_transformers import CrossEncoder
        _RERANK = CrossEncoder(str(BGE_RERANKER_DIR))
        try:
            import torch
            if torch.cuda.is_available():
                _RERANK.model.to("cuda")
        except Exception:
            pass
        return _RERANK
    except Exception:
        _RERANK_FAILED = True
        return None


def cross_encoder_rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """candidates: [(citation, text), ...]"""
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return [(c, 0.0) for c, _ in candidates[:top_k]]
    pairs = [[query, text] for _, text in candidates]
    try:
        scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    except TypeError:
        scores = reranker.predict(pairs)
    scored = list(zip([c for c, _ in candidates], (float(s) for s in scores)))
    scored.sort(key=lambda kv: -kv[1])
    return scored[:top_k]


_LLM = None
_LLM_TOK = None
_LLM_FAILED = False


def _get_llm():
    global _LLM, _LLM_TOK, _LLM_FAILED
    if _LLM is not None or _LLM_FAILED:
        return _LLM, _LLM_TOK
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _LLM_TOK = AutoTokenizer.from_pretrained(str(QWEN_DIR))
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _LLM = AutoModelForCausalLM.from_pretrained(
            str(QWEN_DIR), torch_dtype=dtype, device_map="auto",
        )
        _LLM.eval()
        return _LLM, _LLM_TOK
    except Exception:
        _LLM_FAILED = True
        return None, None


_NUM_LINE = re.compile(r"^\s*\(?\d+\)?[\.\)]?\s*(.+?)\s*$")


def _parse_llm_response(text: str, candidate_set: set[str]) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _NUM_LINE.match(s)
        if m:
            s = m.group(1).strip()
        for cand in candidate_set:
            if cand == s or cand in s:
                out.append(cand)
                break
    seen: set[str] = set()
    deduped = []
    for c in out:
        if c in seen: continue
        seen.add(c); deduped.append(c)
    return deduped


def llm_verify(
    query: str,
    candidates: list[tuple[str, str]],
    budget: int = 60,
    max_new_tokens: int = 800,
) -> list[str]:
    """Returns the subset of candidate citations the LLM judges relevant.

    Soft-fails to all candidates if LLM unavailable.
    """
    candidate_set = {c for c, _ in candidates}
    model, tok = _get_llm()
    if model is None:
        return [c for c, _ in candidates[:budget]]

    truncated = candidates[:budget]
    numbered = "\n".join(
        f"{i+1}. {cit} :: {(text or '')[:160]}"
        for i, (cit, text) in enumerate(truncated)
    )
    prompt = (
        "You are a Swiss law expert. Given the legal question and candidate "
        "citations, return ONLY the citation strings that are directly relevant, "
        "one per line. Do not add any citations not in the candidate list.\n\n"
        f"Question:\n{query}\n\nCandidates:\n{numbered}\n\nRelevant citations:\n"
    )
    try:
        import torch
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
        if next(model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=0.0,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse_llm_response(gen, candidate_set)
    except Exception:
        return [c for c, _ in truncated]
