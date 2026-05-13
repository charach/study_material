"""EN→DE translation via NLLB-200 distilled. Lazy-loaded; falls back to identity."""
from __future__ import annotations
from typing import Optional

from .config import NLLB_DIR

_model = None
_tokenizer = None
_load_failed = False


def _load() -> bool:
    global _model, _tokenizer, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(str(NLLB_DIR), src_lang="eng_Latn")
        _model = AutoModelForSeq2SeqLM.from_pretrained(str(NLLB_DIR))
        try:
            import torch
            if torch.cuda.is_available():
                _model = _model.to("cuda").half()
        except Exception:
            pass
        return True
    except Exception:
        _load_failed = True
        return False


def translate_to_de(text: str, max_new_tokens: int = 256) -> str:
    if not text or not _load():
        return text
    try:
        import torch
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if next(_model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        forced = _tokenizer.convert_tokens_to_ids("deu_Latn")
        with torch.no_grad():
            out = _model.generate(**inputs, forced_bos_token_id=forced,
                                   max_new_tokens=max_new_tokens, num_beams=1)
        return _tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    except Exception:
        return text


def translate_batch(texts: list[str], batch_size: int = 4) -> list[str]:
    return [translate_to_de(t) for t in texts]
