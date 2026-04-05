# retrievers.py
from __future__ import annotations

import os
import sqlite3
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

import config

@dataclass
class RecipeResult:
    recipe_id: int
    title: str
    ingredients: str
    instructions: str
    source: str
    url: str
    score: float


class RecipeRetriever:
    """
    Loads:
      - SQLite recipe DB
      - FAISS title index
      - SentenceTransformer model
    Provides:
      - semantic_search(query)
      - optional keyword fallback
      - format_context(results)
    """

    def __init__(
        self,
        db_path: str = str(config.DEFAULT_DB_PATH),
        faiss_title_index_path: str = str(config.FAISS_TITLE_INDEX_PATH),
        embedding_model_name: str = config.DEFAULT_EMBEDDING_MODEL,
        device: str = "auto",
        faiss_threads: int = config.FAISS_NUM_THREADS,
    ) -> None:
        config.apply_env()

        self.db_path = db_path
        self.faiss_title_index_path = faiss_title_index_path
        self.device = config.pick_device_torch(device)

        faiss.omp_set_num_threads(faiss_threads)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON;")

        self.title_index = faiss.read_index(self.faiss_title_index_path)

        # Load embedding model
        self.model = SentenceTransformer(embedding_model_name, device=self.device)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _fetch_recipes_by_ids(self, ids: List[int]) -> Dict[int, Tuple]:
        """
        Returns map: recipe_id -> row tuple
        """
        if not ids:
            return {}
        qmarks = ",".join(["?"] * len(ids))
        sql = f"""
        SELECT recipe_id, title, ingredients, instructions, source, url
        FROM recipes
        WHERE recipe_id IN ({qmarks});
        """
        rows = self.conn.execute(sql, tuple(ids)).fetchall()
        return {int(r[0]): r for r in rows}

    def keyword_fallback(self, query: str, top_k: int = 5) -> List[RecipeResult]:
        """
        Optional fallback using SQLite FTS table created in task1 preprocessing.
        """
        sql = """
        SELECT r.recipe_id, r.title, r.ingredients, r.instructions, r.source, r.url
        FROM recipe_fts f
        JOIN recipes r ON r.recipe_id = f.recipe_id
        WHERE recipe_fts MATCH ?
        LIMIT ?;
        """
        rows = self.conn.execute(sql, (query, top_k)).fetchall()
        out: List[RecipeResult] = []
        for r in rows:
            out.append(
                RecipeResult(
                    recipe_id=int(r[0]),
                    title=str(r[1] or ""),
                    ingredients=str(r[2] or ""),
                    instructions=str(r[3] or ""),
                    source=str(r[4] or ""),
                    url=str(r[5] or ""),
                    score=0.0,
                )
            )
        return out

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.15,
        use_keyword_fallback: bool = True,
    ) -> List[RecipeResult]:
        """
        Encode query -> FAISS search -> fetch full metadata from SQLite.
        Uses cosine similarity via IP on normalized embeddings (same as your preprocessing).
        """
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)

        scores, ids = self.title_index.search(q_emb, top_k)
        ids_list = [int(i) for i in ids[0].tolist() if int(i) != -1]
        score_list = [float(s) for s in scores[0].tolist()]

        # Filter weak matches
        filtered: List[Tuple[int, float]] = []
        for rid, sc in zip(ids_list, score_list):
            if sc >= min_score:
                filtered.append((rid, sc))

        if not filtered and use_keyword_fallback:
            return self.keyword_fallback(query, top_k=top_k)

        recipe_rows = self._fetch_recipes_by_ids([rid for rid, _ in filtered])

        results: List[RecipeResult] = []
        for rid, sc in filtered:
            row = recipe_rows.get(rid)
            if not row:
                continue
            _, title, ingredients, instructions, source, url = row
            results.append(
                RecipeResult(
                    recipe_id=rid,
                    title=str(title or ""),
                    ingredients=str(ingredients or ""),
                    instructions=str(instructions or ""),
                    source=str(source or ""),
                    url=str(url or ""),
                    score=sc,
                )
            )
        return results

    @staticmethod
    def format_context(results: List[RecipeResult], max_recipes: int = 3) -> str:
        """
        Convert retrieved recipes into a compact context block for RAG prompting.
        """
        if not results:
            return ""

        chunks: List[str] = []
        for r in results[:max_recipes]:
            chunks.append(
                "=== Recipe ===\n"
                f"Title: {r.title}\n"
                f"Source: {r.source}\n"
                f"Ingredients:\n{r.ingredients}\n"
                f"Instructions:\n{r.instructions}\n"
            )
        return "\n".join(chunks)

@dataclass
class VideoResult:
    clip_id: int
    video_id: str
    video_path: str
    start_sec: float
    end_sec: float
    caption: Optional[str]
    score: float


class VideoRetriever:
    """
    Loads:
      - SQLite clip DB
      - FAISS clip index
      - CLIP model
    Provides:
      - search(query) returning (video_id, timestamps, score)
    """

    def __init__(
        self,
        db_path: str = str(config.YOUCOOK_ARTIFACTS_DB),
        faiss_index_path: str = str(config.YOUCOOK_FAISS_INDEX),
        clip_model_name: str = config.DEFAULT_CLIP_MODEL,
        device: str = "auto",
        faiss_threads: int = config.FAISS_NUM_THREADS,
    ) -> None:
        config.apply_env()

        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.device = config.pick_device_torch(device)

        faiss.omp_set_num_threads(faiss_threads)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.index = faiss.read_index(self.faiss_index_path)

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.model.eval()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _fetch_clips_by_ids(self, clip_ids: List[int]) -> Dict[int, Tuple]:
        if not clip_ids:
            return {}
        qmarks = ",".join(["?"] * len(clip_ids))
        sql = f"""
        SELECT clip_id, video_id, video_path, start_sec, end_sec, caption
        FROM clips
        WHERE clip_id IN ({qmarks});
        """
        rows = self.conn.execute(sql, tuple(clip_ids)).fetchall()
        return {int(r[0]): r for r in rows}

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        txt = self.model.get_text_features(**inputs)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return txt.detach().cpu().numpy().astype(np.float32)[0]

    def search(self, query: str, top_k: int = 5) -> List[VideoResult]:
        q_emb = self.encode_text(query)
        q = np.ascontiguousarray(q_emb.reshape(1, -1), dtype=np.float32)

        scores, ids = self.index.search(q, top_k)
        clip_ids = [int(i) for i in ids[0].tolist() if int(i) != -1]
        score_list = [float(s) for s in scores[0].tolist()]

        rows = self._fetch_clips_by_ids(clip_ids)

        out: List[VideoResult] = []
        for cid, sc in zip(clip_ids, score_list):
            r = rows.get(cid)
            if not r:
                continue
            _, video_id, video_path, s, e, caption = r
            out.append(
                VideoResult(
                    clip_id=cid,
                    video_id=str(video_id),
                    video_path=str(video_path),
                    start_sec=float(s),
                    end_sec=float(e),
                    caption=(str(caption) if caption is not None else None),
                    score=sc,
                )
            )
        return out