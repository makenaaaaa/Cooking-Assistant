#!/usr/bin/env python3
"""
Dataset (not included): Download and put in project folder (project/YouCookII_downscaled)
    YouCookII_downscaled/
      videos/
      splits/
      annotations/
Quick run: python assignment8_task2.py --root YouCookII_downscaled --split val --max_videos 3 --verify_query onion
Run test split (no captions): python assignment8_task2.py --root YouCookII_downscaled --split test --max_videos 3 --verify_query "chop onions"
Full run: python assignment8_task2.py --root YouCookII_downscaled --split val --max_videos 0
Outputs (not commited):
    artifacts/youcook_clips.db
    artifacts/youcook_clip.index
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
config.apply_env()

import numpy as np
from tqdm import tqdm

import cv2
import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------
# SQLite schema
# ---------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS clips (
    clip_id INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL,
    video_path TEXT NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    frame_times_json TEXT NOT NULL,
    caption TEXT
);

CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id);
"""

INSERT_CLIP_SQL = """
INSERT OR REPLACE INTO clips
(clip_id, video_id, video_path, start_sec, end_sec, frame_times_json, caption)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    return conn

# ---------------------------
# FAISS index
# ---------------------------

@dataclass
class FaissStore:
    index: faiss.Index
    dim: int

def build_faiss(dim: int) -> FaissStore:
    # cosine similarity = inner product after L2 normalization
    base = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIDMap2(base)
    faiss.omp_set_num_threads(config.FAISS_NUM_THREADS)
    return FaissStore(index=idx, dim=dim)

def add_to_faiss(store: FaissStore, emb: np.ndarray, clip_ids: np.ndarray) -> None:
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    clip_ids = np.ascontiguousarray(clip_ids, dtype=np.int64)
    store.index.add_with_ids(emb, clip_ids)

def save_faiss(store: FaissStore, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(store.index, path)


# ---------------------------
# Dataset discovery
# ---------------------------

VIDEO_EXTS = config.VIDEO_EXTS

def read_split_ids(dataset_root: str, split: str) -> Optional[List[str]]:
    split_path = os.path.join(dataset_root, "splits", f"{split}_list.txt")
    if not os.path.exists(split_path):
        return None
    ids: List[str] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                ids.append(v)
    return ids

def discover_videos(dataset_root: str, split: str, max_videos: int) -> List[Tuple[str, str]]:
    split_to_folder = {"train": "training", "val": "validation", "test": "testing"}

    videos_root = os.path.join(dataset_root, "videos")
    if not os.path.exists(videos_root):
        return []

    mapped = split_to_folder.get(split, split)
    preferred_root = os.path.join(videos_root, mapped)
    search_root = preferred_root if os.path.exists(preferred_root) else videos_root

    video_map: Dict[str, str] = {}
    for root, _, files in os.walk(search_root):
        for fn in files:
            lower = fn.lower()
            if lower.endswith(VIDEO_EXTS):
                vid = os.path.splitext(fn)[0]
                video_map[vid] = os.path.join(root, fn)

    if not video_map:
        return []

    ids = read_split_ids(dataset_root, split)
    results: List[Tuple[str, str]] = []

    def normalize_to_vid(line: str) -> str:
        s = line.strip().replace("\\", "/")
        if not s:
            return ""
        base = os.path.basename(s)
        stem = os.path.splitext(base)[0]
        return stem

    if ids is not None:
        for raw in ids:
            vid = normalize_to_vid(raw)
            if not vid:
                continue

            path = video_map.get(vid)
            if path is None:
                vid_lower = vid.lower()
                for k, v in video_map.items():
                    if vid_lower == k.lower() or vid_lower in k.lower():
                        path = v
                        vid = k
                        break

            if path is not None:
                results.append((vid, path))

            if max_videos > 0 and len(results) >= max_videos:
                break

        if results:
            return results

    items = list(video_map.items())
    if max_videos > 0:
        items = items[:max_videos]
    return [(vid, path) for vid, path in items]


# ---------------------------
# Annotations
# ---------------------------

def load_annotations(dataset_root: str, split: str) -> Dict[str, List[Tuple[float, float, str]]]:
    ann_dir = os.path.join(dataset_root, "annotations")
    if split == "test":
        ann_path = os.path.join(ann_dir, "youcookii_annotations_test_segments_only.json")
    else:
        ann_path = os.path.join(ann_dir, "youcookii_annotations_trainval.json")

    if not os.path.exists(ann_path):
        return {}

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, List[Tuple[float, float, str]]] = {}

    if isinstance(data, dict) and "database" in data and isinstance(data["database"], dict):
        for vid, vinfo in data["database"].items():
            anns = vinfo.get("annotations", [])
            triples: List[Tuple[float, float, str]] = []
            for a in anns:
                seg = a.get("segment") or a.get("timestamps")
                sent = a.get("sentence") or a.get("description") or ""
                if seg and len(seg) == 2:
                    triples.append((float(seg[0]), float(seg[1]), str(sent)))
            if triples:
                out[str(vid)] = triples
        return out

    if isinstance(data, list):
        for item in data:
            vid = item.get("video_id") or item.get("id")
            anns = item.get("annotations", [])
            if not vid:
                continue
            triples: List[Tuple[float, float, str]] = []
            for a in anns:
                seg = a.get("segment") or a.get("timestamps")
                sent = a.get("sentence") or a.get("description") or ""
                if seg and len(seg) == 2:
                    triples.append((float(seg[0]), float(seg[1]), str(sent)))
            if triples:
                out[str(vid)] = triples
        return out

    return {}

def caption_for_clip(
    vid: str,
    start_sec: float,
    end_sec: float,
    ann_map: Dict[str, List[Tuple[float, float, str]]],
) -> Optional[str]:
    if vid not in ann_map:
        return None
    sents: List[str] = []
    for a_s, a_e, sent in ann_map[vid]:
        overlap = max(0.0, min(end_sec, a_e) - max(start_sec, a_s))
        if overlap > 0:
            sent = sent.strip()
            if sent:
                sents.append(sent)
    if not sents:
        return None
    return " ".join(sents)

# ---------------------------
# Clip generation + frame sampling
# ---------------------------

def video_duration_seconds(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps and fps > 0 and nframes and nframes > 0:
        return float(nframes / fps)
    return 0.0

def sample_frame_at_time(cap: cv2.VideoCapture, t_sec: float) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec * 1000.0))
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return None
    return frame_bgr

def sliding_windows(duration: float, clip_len: float, stride: float) -> List[Tuple[float, float]]:
    if duration <= 0:
        return []
    windows: List[Tuple[float, float]] = []
    t = 0.0
    while t < duration:
        end = min(t + clip_len, duration)
        if end - t >= max(0.5, clip_len * 0.5):
            windows.append((t, end))
        if t + stride >= duration:
            break
        t += stride
    return windows

def frame_times_in_clip(start: float, end: float, n_frames: int) -> List[float]:
    if n_frames <= 1:
        return [0.5 * (start + end)]
    ts: List[float] = []
    span = max(1e-6, end - start)
    for i in range(n_frames):
        alpha = (i + 0.5) / n_frames
        ts.append(start + alpha * span)
    return ts

# ---------------------------
# CLIP embedding
# ---------------------------

@torch.no_grad()
def clip_image_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    frames_rgb: List[np.ndarray],
    device: str,
) -> np.ndarray:
    pil_images = [Image.fromarray(img) for img in frames_rgb]
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    img_feats = model.get_image_features(**inputs)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    clip_feat = img_feats.mean(dim=0, keepdim=True)
    clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
    return clip_feat.detach().cpu().numpy().astype(np.float32)[0]

@torch.no_grad()
def clip_text_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    text: str,
    device: str,
) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    txt = model.get_text_features(**inputs)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.detach().cpu().numpy().astype(np.float32)[0]

# ---------------------------
# Retrieval / verification
# ---------------------------

def faiss_search(store: FaissStore, query_emb: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    q = np.ascontiguousarray(query_emb.reshape(1, -1), dtype=np.float32)
    scores, ids = store.index.search(q, top_k)
    out: List[Tuple[int, float]] = []
    for rid, sc in zip(ids[0], scores[0]):
        if rid == -1:
            continue
        out.append((int(rid), float(sc)))
    return out

def fetch_clip_rows(conn: sqlite3.Connection, clip_ids: List[int]) -> Dict[int, Tuple]:
    if not clip_ids:
        return {}
    qmarks = ",".join(["?"] * len(clip_ids))
    sql = f"SELECT clip_id, video_id, video_path, start_sec, end_sec, caption FROM clips WHERE clip_id IN ({qmarks});"
    rows = conn.execute(sql, tuple(clip_ids)).fetchall()
    return {int(r[0]): r for r in rows}

def keyword_caption_search(conn: sqlite3.Connection, keyword: str, top_k: int) -> List[Tuple]:
    sql = """
    SELECT clip_id, video_id, start_sec, end_sec, caption
    FROM clips
    WHERE caption IS NOT NULL AND caption LIKE ?
    LIMIT ?;
    """
    return conn.execute(sql, (f"%{keyword}%", top_k)).fetchall()

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=str(config.YOUCOOK_ROOT),
                        help="Path to downscaled YouCookII dataset root folder")
    parser.add_argument("--split", default=config.DEFAULT_YOUCOOK_SPLIT,
                        choices=["train", "val", "test", "all"],
                        help="Which split list to use if available; 'all' scans all videos")

    parser.add_argument("--max_videos", type=int, default=config.DEFAULT_YOUCOOK_MAX_VIDEOS,
                        help="Process only first N videos (0 = all)")
    parser.add_argument("--clip_len", type=float, default=config.DEFAULT_CLIP_LEN_SEC,
                        help="Clip length in seconds (sliding window)")
    parser.add_argument("--stride", type=float, default=config.DEFAULT_STRIDE_SEC,
                        help="Stride in seconds for sliding window")
    parser.add_argument("--frames_per_clip", type=int, default=config.DEFAULT_FRAMES_PER_CLIP,
                        help="Number of frames sampled per clip")

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--db", default=str(config.YOUCOOK_ARTIFACTS_DB), help="SQLite output path")
    parser.add_argument("--faiss_out", default=str(config.YOUCOOK_FAISS_INDEX), help="FAISS index output path")

    parser.add_argument("--clip_model", default=config.DEFAULT_CLIP_MODEL, help="CLIP model name")
    parser.add_argument("--verify_query", default=config.DEFAULT_TASK2_VERIFY_QUERY,
                        help="Text query to verify retrieval")
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TASK2_TOP_K)

    args = parser.parse_args()

    device = config.pick_device_torch(args.device)
    print("Device:", device)

    conn = init_db(args.db)

    ann_map = load_annotations(args.root, args.split)
    print(f"Loaded annotations for {len(ann_map)} videos (0 means none found).")

    if args.split == "all":
        vids = []
        for sp in ["train", "val", "test"]:
            vids.extend(discover_videos(args.root, split=sp, max_videos=0))
        if args.max_videos > 0:
            vids = vids[:args.max_videos]
    else:
        vids = discover_videos(args.root, split=args.split, max_videos=args.max_videos)

    if not vids:
        raise SystemExit("No videos found. Check --root and dataset structure.")

    print(f"Found {len(vids)} videos to process.")

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()

    dim = int(model.config.projection_dim)
    store = build_faiss(dim)

    next_clip_id = 0
    all_embs: List[np.ndarray] = []
    all_ids: List[int] = []

    for vid, vpath in tqdm(vids, desc="Videos"):
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[warn] cannot open video: {vpath}")
            continue

        dur = video_duration_seconds(cap)
        if dur <= 0:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            _ = cap.read()
            dur = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        windows = sliding_windows(dur, clip_len=args.clip_len, stride=args.stride)
        if not windows:
            cap.release()
            continue

        for (s, e) in windows:
            ts = frame_times_in_clip(s, e, args.frames_per_clip)
            frames_rgb: List[np.ndarray] = []

            for tsec in ts:
                frame_bgr = sample_frame_at_time(cap, tsec)
                if frame_bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)

            if not frames_rgb:
                continue

            emb = clip_image_embedding(model, processor, frames_rgb, device=device)
            emb = np.ascontiguousarray(emb, dtype=np.float32)

            cap_caption = caption_for_clip(vid, s, e, ann_map)

            frame_times_json = json.dumps(ts)
            conn.execute(
                INSERT_CLIP_SQL,
                (next_clip_id, vid, vpath, float(s), float(e), frame_times_json, cap_caption),
            )

            all_embs.append(emb)
            all_ids.append(next_clip_id)
            next_clip_id += 1

        conn.commit()
        cap.release()

    if not all_embs:
        raise SystemExit("No clip embeddings produced. Try smaller clip_len/stride or check videos.")

    embs = np.stack(all_embs, axis=0).astype(np.float32)
    ids = np.array(all_ids, dtype=np.int64)
    add_to_faiss(store, embs, ids)
    save_faiss(store, args.faiss_out)

    print(f"Saved FAISS index: {args.faiss_out}")
    print(f"Saved SQLite DB: {args.db}")
    print(f"Total clips embedded: {len(all_ids)}")

    print("\n=== Verification ===")
    print(f"Text query (CLIP): {args.verify_query!r}")
    q_emb = clip_text_embedding(model, processor, args.verify_query, device=device)
    hits = faiss_search(store, q_emb, top_k=args.top_k)

    hit_ids = [cid for cid, _ in hits]
    rows = fetch_clip_rows(conn, hit_ids)
    for cid, score in hits:
        r = rows.get(cid)
        if not r:
            continue
        _, video_id, _, s, e, cap_caption = r
        print(
            f"  - clip_id={cid} video={video_id} [{s:.1f}s–{e:.1f}s] score={score:.4f}"
            + (f" | caption={cap_caption[:80]}..." if cap_caption else "")
        )

    if args.split != "test":
        print(f"\nCaption keyword search (LIKE): {args.verify_query!r}")
        kw = keyword_caption_search(conn, args.verify_query, top_k=args.top_k)
        if not kw:
            print("  (no caption matches)")
        else:
            for clip_id, video_id, s, e, cap_caption in kw:
                print(f"  - clip_id={clip_id} video={video_id} [{s:.1f}s–{e:.1f}s] caption={str(cap_caption)[:100]}...")
    else:
        print("\nCaption keyword search: skipped (test annotations are segments-only)")

    conn.close()

if __name__ == "__main__":
    main()