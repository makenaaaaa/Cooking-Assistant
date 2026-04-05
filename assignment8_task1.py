#!/usr/bin/env python3
"""
Dataset (not included): Download and put in project folder (project/dataset/full_dataset.csv)
Quick run: python assignment8_task1.py --csv dataset/full_dataset.csv --limit 300
Full run: python assignment8_task1.py --csv dataset/full_dataset.csv --limit 0
Outputs (not commited):
    artifacts/recipes.db
    artifacts/faiss_title.index
    artifacts/faiss_instructions.index
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import config

config.apply_env()

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from sentence_transformers import SentenceTransformer
import faiss

faiss.omp_set_num_threads(config.FAISS_NUM_THREADS)

# ---------------------------
# Helpers: text normalization
# ---------------------------

def _to_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)

def normalize_listish_field(value: str, join_with: str = "\n") -> str:
    """
    Many RecipeNLG fields are stored as strings that look like Python lists.
    Example: "['step 1', 'step 2']" or "['salt', 'pepper']"
    If it looks like a list, parse and join; otherwise return as-is.
    """
    s = _to_str(value).strip()
    if not s:
        return ""

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                parts = [str(p).strip() for p in parsed if str(p).strip()]
                return join_with.join(parts)
        except Exception:
            pass
    return s

def normalize_ner(value: str) -> str:
    """
    Store NER as JSON string when possible. If already JSON-like or list-like, convert.
    Otherwise store raw.
    """
    s = _to_str(value).strip()
    if not s:
        return "[]"

    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    return json.dumps([s], ensure_ascii=False)

# ---------------------------
# SQLite: schema and inserts
# ---------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS recipes (
    recipe_id INTEGER PRIMARY KEY,
    title TEXT,
    ingredients TEXT,
    instructions TEXT,
    source TEXT,
    url TEXT,
    named_entities TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS recipe_fts
USING fts5(
    recipe_id UNINDEXED,
    title,
    instructions,
    ingredients,
    content=''
);

CREATE INDEX IF NOT EXISTS idx_recipes_source ON recipes(source);
"""

INSERT_RECIPE_SQL = """
INSERT OR REPLACE INTO recipes
(recipe_id, title, ingredients, instructions, source, url, named_entities)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

INSERT_FTS_SQL = """
INSERT INTO recipe_fts (recipe_id, title, instructions, ingredients)
VALUES (?, ?, ?, ?);
"""

def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA_SQL)
    return conn

def insert_batch(
    conn: sqlite3.Connection,
    rows: List[Tuple[int, str, str, str, str, str, str]],
    fts_rows: List[Tuple[int, str, str, str]],
) -> None:
    with conn:
        conn.executemany(INSERT_RECIPE_SQL, rows)
        conn.executemany(INSERT_FTS_SQL, fts_rows)


# ---------------------------
# FAISS: building indices
# ---------------------------

@dataclass
class FaissStores:
    title_index: faiss.Index
    instr_index: faiss.Index
    dim: int

def build_faiss_stores(dim: int) -> FaissStores:
    title_base = faiss.IndexFlatIP(dim)
    instr_base = faiss.IndexFlatIP(dim)

    title_index = faiss.IndexIDMap2(title_base)
    instr_index = faiss.IndexIDMap2(instr_base)
    return FaissStores(title_index=title_index, instr_index=instr_index, dim=dim)

def add_embeddings(index: faiss.Index, embeddings: np.ndarray, ids: np.ndarray) -> None:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    if ids.dtype != np.int64:
        ids = ids.astype(np.int64)
    index.add_with_ids(embeddings, ids)

def save_faiss(index: faiss.Index, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(index, path)


# ---------------------------
# CSV reading / processing
# ---------------------------

def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in ("id", "recipe_id")]
    if candidates:
        return candidates[0]
    if "Unnamed: 0" in df.columns:
        return "Unnamed: 0"
    return None

def iter_csv(
    csv_path: str,
    chunksize: int,
    limit: Optional[int],
) -> Iterable[pd.DataFrame]:
    seen = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if limit is None:
            yield chunk
            continue

        remaining = limit - seen
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()
        yield chunk
        seen += len(chunk)

def process_chunk(
    chunk: pd.DataFrame,
    id_col: Optional[str],
) -> Tuple[List[Tuple[int, str, str, str, str, str, str]],
           List[Tuple[int, str, str, str]],
           np.ndarray,
           List[str],
           List[str]]:
    for col in ["title", "ingredients", "directions", "link", "source", "NER"]:
        if col not in chunk.columns:
            raise ValueError(f"Missing required column '{col}'. Found columns: {list(chunk.columns)}")

    if id_col is not None:
        raw_ids = chunk[id_col].astype(str).fillna("").tolist()
        ids = []
        for r in raw_ids:
            r = r.strip()
            if r.isdigit():
                ids.append(int(r))
            else:
                ids.append(abs(hash(r)) % (2**31))
        recipe_ids = np.array(ids, dtype=np.int64)
    else:
        recipe_ids = np.arange(len(chunk), dtype=np.int64)

    titles = [normalize_listish_field(v, join_with=" ") for v in chunk["title"].tolist()]
    ingredients = [normalize_listish_field(v, join_with="\n") for v in chunk["ingredients"].tolist()]
    instructions = [normalize_listish_field(v, join_with="\n") for v in chunk["directions"].tolist()]
    urls = [_to_str(v) for v in chunk["link"].tolist()]
    sources = [_to_str(v) for v in chunk["source"].tolist()]
    ners = [normalize_ner(v) for v in chunk["NER"].tolist()]

    rows = []
    fts_rows = []
    for i in range(len(chunk)):
        rid = int(recipe_ids[i])
        rows.append((rid, titles[i], ingredients[i], instructions[i], sources[i], urls[i], ners[i]))
        fts_rows.append((rid, titles[i], instructions[i], ingredients[i]))

    return rows, fts_rows, recipe_ids, titles, instructions


# ---------------------------
# Retrieval verification
# ---------------------------

def keyword_search(conn: sqlite3.Connection, query: str, top_k: int = 5) -> List[Tuple[int, str]]:
    sql = """
    SELECT r.recipe_id, r.title
    FROM recipe_fts f
    JOIN recipes r ON r.recipe_id = f.recipe_id
    WHERE recipe_fts MATCH ?
    LIMIT ?;
    """
    cur = conn.execute(sql, (query, top_k))
    return [(int(rid), title) for (rid, title) in cur.fetchall()]

def faiss_search(
    index: faiss.Index,
    model: SentenceTransformer,
    query: str,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
    scores, ids = index.search(q_emb, top_k)
    results = []
    for rid, score in zip(ids[0], scores[0]):
        if rid == -1:
            continue
        results.append((int(rid), float(score)))
    return results

def fetch_titles(conn: sqlite3.Connection, recipe_ids: List[int]) -> List[Tuple[int, str]]:
    if not recipe_ids:
        return []
    qmarks = ",".join(["?"] * len(recipe_ids))
    sql = f"SELECT recipe_id, title FROM recipes WHERE recipe_id IN ({qmarks});"
    rows = conn.execute(sql, tuple(recipe_ids)).fetchall()
    title_map = {int(r): t for (r, t) in rows}
    return [(rid, title_map.get(rid, "")) for rid in recipe_ids]


# ---------------------------
# Main
# ---------------------------

def main():
    defaults = config.get_default_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=defaults.csv_path, help="Path to RecipeNLG CSV file")
    parser.add_argument("--db", default=defaults.db_path, help="Output SQLite DB path")
    parser.add_argument("--out_dir", default=defaults.artifacts_dir, help="Directory to save FAISS indices")
    parser.add_argument("--model", default=config.DEFAULT_EMBEDDING_MODEL,
                        help="Sentence-BERT model name or local path")
    parser.add_argument("--chunksize", type=int, default=config.DEFAULT_CHUNKSIZE, help="CSV streaming chunk size")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT,
                        help="Process only first N recipes (for testing). Use 0 for all.")
    parser.add_argument("--embed_batch", type=int, default=config.DEFAULT_EMBED_BATCH, help="Embedding batch size")
    parser.add_argument("--verify_query", default=config.DEFAULT_VERIFY_QUERY, help="Query to verify retrieval")
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K, help="Top-k retrieval results to display")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit

    conn = init_db(args.db)

    device = config.pick_device_torch()
    model = SentenceTransformer(args.model, device=device)
    print("SentenceTransformer device:", device)

    dummy = model.encode(["test"], normalize_embeddings=True, convert_to_numpy=True)
    dim = int(dummy.shape[1])
    stores = build_faiss_stores(dim)

    total_processed = 0
    first_chunk = True
    id_col: Optional[str] = None

    print(f"Reading CSV: {args.csv}")
    for chunk in tqdm(iter_csv(args.csv, chunksize=args.chunksize, limit=limit), desc="Processing chunks"):
        if first_chunk:
            id_col = detect_id_column(chunk)
            first_chunk = False
            if id_col:
                print(f"Detected recipe id column: {id_col}")
            else:
                print("No explicit id column detected; generating sequential IDs (not recommended for full runs).")

        rows, fts_rows, recipe_ids, titles, instructions = process_chunk(chunk, id_col=id_col)

        insert_batch(conn, rows, fts_rows)

        title_emb = model.encode(
            titles,
            batch_size=args.embed_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        instr_emb = model.encode(
            instructions,
            batch_size=args.embed_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        add_embeddings(stores.title_index, title_emb, recipe_ids)
        add_embeddings(stores.instr_index, instr_emb, recipe_ids)

        total_processed += len(rows)

    print(f"\nDone. Total processed: {total_processed}")

    # Save FAISS indices (filenames centrally defined in config)
    title_index_path = os.path.join(args.out_dir, os.path.basename(str(config.FAISS_TITLE_INDEX_PATH)))
    instr_index_path = os.path.join(args.out_dir, os.path.basename(str(config.FAISS_INSTRUCTIONS_INDEX_PATH)))

    save_faiss(stores.title_index, title_index_path)
    save_faiss(stores.instr_index, instr_index_path)

    print(f"Saved FAISS title index: {title_index_path}")
    print(f"Saved FAISS instructions index: {instr_index_path}")
    print(f"Saved SQLite DB: {args.db}")

    print("\n=== Verification ===")
    q = args.verify_query

    print(f"\nKeyword (FTS5) results for: {q!r}")
    kw = keyword_search(conn, q, top_k=args.top_k)
    for rid, title in kw:
        print(f"  - [{rid}] {title}")

    print(f"\nSemantic (FAISS title) results for: {q!r}")
    sem_title = faiss_search(stores.title_index, model, q, top_k=args.top_k)
    sem_ids = [rid for rid, _ in sem_title]
    sem_titles = fetch_titles(conn, sem_ids)
    score_map = {rid: score for rid, score in sem_title}
    for rid, title in sem_titles:
        print(f"  - [{rid}] {title}  (score={score_map.get(rid, 0.0):.4f})")

    print(f"\nSemantic (FAISS instructions) results for: {q!r}")
    sem_instr = faiss_search(stores.instr_index, model, q, top_k=args.top_k)
    semi_ids = [rid for rid, _ in sem_instr]
    semi_titles = fetch_titles(conn, semi_ids)
    score_map2 = {rid: score for rid, score in sem_instr}
    for rid, title in semi_titles:
        print(f"  - [{rid}] {title}  (score={score_map2.get(rid, 0.0):.4f})")

    conn.close()


if __name__ == "__main__":
    main()