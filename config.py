"""
Central config for the project.

Put all environment-dependent values here:
- paths (datasets, artifacts, DBs, FAISS indices)
- model defaults (SentenceTransformer, CLIP, TinyLlama)
- runtime controls (threads, tokenizers parallelism)
- chatbot generation parameters

All scripts should import from this file rather than hardcoding paths/constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATASET_DIR: Path = PROJECT_ROOT / "dataset"
DEFAULT_CSV_PATH: Path = DATASET_DIR / "full_dataset.csv"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
DEFAULT_DB_PATH: Path = ARTIFACTS_DIR / "recipes.db"

FAISS_TITLE_INDEX_PATH: Path = ARTIFACTS_DIR / "faiss_title.index"
FAISS_INSTRUCTIONS_INDEX_PATH: Path = ARTIFACTS_DIR / "faiss_instructions.index"


# =============================================================================
# Task 2 (YouCookII) paths
# =============================================================================

YOUCOOK_ROOT: Path = PROJECT_ROOT / "YouCookII_downscaled"
YOUCOOK_ARTIFACTS_DB: Path = ARTIFACTS_DIR / "youcook_clips.db"
YOUCOOK_FAISS_INDEX: Path = ARTIFACTS_DIR / "youcook_clip.index"

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


# =============================================================================
# Task 1 defaults (RecipeNLG: SQLite + FAISS + SBERT)
# =============================================================================

DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNKSIZE: int = 2000
DEFAULT_LIMIT: int = 300          # 0 means "all" (handled in scripts)
DEFAULT_EMBED_BATCH: int = 128
DEFAULT_VERIFY_QUERY: str = "chicken garlic pasta"
DEFAULT_TOP_K: int = 5


# =============================================================================
# Task 2 defaults (YouCookII: CLIP + clips)
# =============================================================================

DEFAULT_YOUCOOK_SPLIT: str = "val"
DEFAULT_YOUCOOK_MAX_VIDEOS: int = 5     # 0 means "all"
DEFAULT_CLIP_LEN_SEC: float = 6.0
DEFAULT_STRIDE_SEC: float = 6.0
DEFAULT_FRAMES_PER_CLIP: int = 2

DEFAULT_CLIP_MODEL: str = "openai/clip-vit-base-patch32"
DEFAULT_TASK2_VERIFY_QUERY: str = "chop onions"
DEFAULT_TASK2_TOP_K: int = 5


# =============================================================================
# Assignment 9 defaults (Chatbot / TinyLlama)
# =============================================================================

CHAT_MODEL_ID: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Generation parameters
CHAT_MAX_NEW_TOKENS: int = 512
CHAT_TEMPERATURE: float = 0.2
CHAT_TOP_P: float = 0.9
CHAT_REPETITION_PENALTY: float = 1.15

CHAT_MAX_HISTORY_TURNS: int = 6

# Explicit role tokens
CHAT_TOK_SYSTEM: str = "<|system|>"
CHAT_TOK_CONTEXT: str = "<|context|>"
CHAT_TOK_USER: str = "<|user|>"
CHAT_TOK_ASSISTANT: str = "<|assistant|>"

CHAT_SYSTEM_PROMPT: str = (
    "You are a helpful cooking assistant. "
    "Use the provided context to answer questions. "
    "If the context is empty, give general cooking advice in English. "
    "Be concise and clear."
)


# =============================================================================
# Runtime controls (threads / env vars)
# =============================================================================

TOKENIZERS_PARALLELISM: str = "false"
MAX_NUM_THREADS: str = "1"
FAISS_NUM_THREADS: int = 1


def apply_env() -> None:
    """
    Apply environment variables early (before importing heavy libs if possible).
    Safe to call multiple times.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", TOKENIZERS_PARALLELISM)

    os.environ.setdefault("OMP_NUM_THREADS", MAX_NUM_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", MAX_NUM_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", MAX_NUM_THREADS)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", MAX_NUM_THREADS)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", MAX_NUM_THREADS)


@dataclass(frozen=True)
class ResolvedPaths:
    # Task 1
    csv_path: str
    artifacts_dir: str
    db_path: str
    faiss_title_index: str
    faiss_instructions_index: str
    # Task 2
    youcook_root: str
    youcook_db_path: str
    youcook_faiss_index: str


def get_default_paths() -> ResolvedPaths:
    return ResolvedPaths(
        csv_path=str(DEFAULT_CSV_PATH),
        artifacts_dir=str(ARTIFACTS_DIR),
        db_path=str(DEFAULT_DB_PATH),
        faiss_title_index=str(FAISS_TITLE_INDEX_PATH),
        faiss_instructions_index=str(FAISS_INSTRUCTIONS_INDEX_PATH),
        youcook_root=str(YOUCOOK_ROOT),
        youcook_db_path=str(YOUCOOK_ARTIFACTS_DB),
        youcook_faiss_index=str(YOUCOOK_FAISS_INDEX),
    )

def pick_device_torch(preferred: str = "auto") -> str:
    import torch

    if not preferred:
        preferred = "auto"

    preferred = preferred.lower()
    if preferred != "auto":
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"