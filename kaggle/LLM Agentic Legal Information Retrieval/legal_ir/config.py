"""Path/config for both local dev and Kaggle notebook.

Kaggle: set `KAGGLE_ENV=1` (or detect /kaggle/input automatically).
Local : reads from project ./data/ + ~/.cache/legal_ir/ for built indexes.
"""
from __future__ import annotations
import os
from pathlib import Path

_KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = _KAGGLE_INPUT.exists() or os.environ.get("KAGGLE_ENV") == "1"

if ON_KAGGLE:
    DATA_DIR = _KAGGLE_INPUT / "llm-agentic-legal-information-retrieval"
    BGE_M3_DIR = _KAGGLE_INPUT / "bge-m3"
    BGE_RERANKER_DIR = _KAGGLE_INPUT / "bge-reranker-v2-m3"
    QWEN_DIR = _KAGGLE_INPUT / "qwen25-7b-instruct"
    NLLB_DIR = _KAGGLE_INPUT / "nllb-200-distilled"
    WORK_DIR = Path("/kaggle/working")
else:
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    CACHE_DIR = Path.home() / ".cache" / "legal_ir"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BGE_M3_DIR = CACHE_DIR / "bge-m3"
    BGE_RERANKER_DIR = CACHE_DIR / "bge-reranker-v2-m3"
    QWEN_DIR = CACHE_DIR / "qwen25-7b-instruct"
    NLLB_DIR = CACHE_DIR / "nllb-200-distilled"
    WORK_DIR = ROOT / "build"
    WORK_DIR.mkdir(parents=True, exist_ok=True)

LAWS_CSV = DATA_DIR / "laws_de.csv"
COURT_CSV = DATA_DIR / "court_considerations.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
