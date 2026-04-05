This project builds a small cooking assistant pipeline with:
1) **Recipe retrieval** (SQLite + FAISS + SentenceTransformer embeddings)
2) **Video clip retrieval** (SQLite + FAISS + CLIP text embeddings)
3) **Chatbot generation** (TinyLlama) with a **RAG-style context block** injected into the prompt

Datasets and large artifacts are not committed. You generate the DB/indices locally and keep them in `artifacts/`.

---

## Repository Structure
```
project/
├── config.py
├── assignment8_task1.py
├── assignment8_task2.py
├── assignment_9.py
├── retrievers.py
├── pipeline_demo.py
├── dataset/               # NOT committed
├── YouCookII_downscaled/  # NOT committed
└── artifacts/             # NOT committed
```

## Central Configuration (`config.py`)

`config.py` is the single place to define environment-specific settings:
- Paths (datasets, artifacts directory, DBs, FAISS indices)
- Default model IDs (SentenceTransformer, CLIP, TinyLlama)
- Thread/env settings (`apply_env()`)
- Device selection (`pick_device_torch()`)

All scripts import `config` and use these values as defaults, while still allowing CLI overrides.

## Components

### Recipe Index Builder — `assignment8_task1.py`

**Goal:** Build:
- `artifacts/recipes.db` (SQLite DB with an FTS table)
- `artifacts/faiss_title.index` and `artifacts/faiss_instructions.index` (FAISS cosine indices)

**Input:**
- RecipeNLG CSV: `dataset/full_dataset.csv`

**Output:**
- `artifacts/recipes.db`
- `artifacts/faiss_title.index`
- `artifacts/faiss_instructions.index`

**How it works:**
1. Stream the CSV in chunks
2. Normalize fields (ingredients/instructions/NER)
3. Insert metadata into SQLite + FTS5 table
4. Encode text with SentenceTransformer (same model later used for query embedding)
5. Add normalized embeddings to FAISS (cosine similarity via inner-product on L2-normalized vectors)

### YouCookII Clip Index Builder — `assignment8_task2.py`

**Goal:** Build:
- `artifacts/youcook_clips.db` (clip metadata)
- `artifacts/youcook_clip.index` (FAISS index over CLIP image embeddings)

**Input:**
- YouCookII downscaled dataset directory: `YouCookII_downscaled/`

**Output:**
- `artifacts/youcook_clips.db`
- `artifacts/youcook_clip.index`

**How it works:**
1. Discover videos by split (train/val/test)
2. Slide a window over each video to create short clips
3. Sample frames per clip and encode frames using CLIP image encoder
4. Mean-pool frame embeddings to produce a clip embedding
5. Store clip metadata in SQLite and embeddings in FAISS

### Cooking Chatbot — `assignment_9.py`

**Goal:** Run a TinyLlama-based interactive chatbot using a prompt format with explicit role tokens.

**Key idea (RAG-ready prompt):**
The prompt is built as:
- `<|system|>` system instructions
- `<|context|>` retrieved context (initially empty in Assignment 9)
- `<|user|>` user message
- `<|assistant|>` generation start

Output validation removes leaked role tokens and handles empty responses.

### Retrievers — `retrievers.py`

This file contains two classes that **load** the prebuilt indices/DBs:

#### `RecipeRetriever`
Loads:
- `artifacts/recipes.db`
- `artifacts/faiss_title.index`
- SentenceTransformer embedding model (same as in Task 1)

Provides:
- `semantic_search(query, top_k=...)`
- optional FTS keyword fallback
- `format_context(results)` to produce a context block for RAG prompting

#### `VideoRetriever`
Loads:
- `artifacts/youcook_clips.db`
- `artifacts/youcook_clip.index`
- CLIP model/processor

Provides:
- `search(query, top_k=...)` using CLIP text encoder → FAISS search → fetch timestamps/metadata

### End-to-End Demo Pipeline — `pipeline_demo.py`

Demonstrates full interaction between components:

1. **User query** (text)
2. **RecipeRetriever** finds top-K recipes (semantic search)
3. Retrieved recipes are formatted into a **context block**
4. A TinyLlama prompt is constructed with:
   - system prompt
   - context block
   - user query
5. **TinyLlama** generates an answer
6. The answer is split into a few steps (simple heuristic)
7. Each step is used as a query for **VideoRetriever**
8. The script prints video IDs + clip timestamps per step

## Setup

### Requirements
Install dependencies in your environment (requirements.txt in main directory).

### Datasets (NOT committed)
1) RecipeNLG:
- Download the CSV and place it at:
  - `dataset/full_dataset.csv`

2) YouCookII downscaled:
- Place the dataset folder at:
  - `YouCookII_downscaled/`
  with subfolders:
  - `videos/`, `splits/`, `annotations/`

## How to Run

### Step 1 - Build recipe DB + FAISS indices
Quick run: `python assignment8_task1.py --csv dataset/full_dataset.csv --limit 300`

Full run: `python assignment8_task1.py --csv dataset/full_dataset.csv --limit 0`

### Step 2 - Build YouCookII DB + FAISS index
Quick run: `python assignment8_task2.py --root YouCookII_downscaled --split val --max_videos 3 --verify_query onion`

Run test split (no captions): `python assignment8_task2.py --root YouCookII_downscaled --split test --max_videos 3 --verify_query "chop onions"`

Full run: `python assignment8_task2.py --root YouCookII_downscaled --split val --max_videos 0`

### Step 3 - Run chatbot alone
`python assignment_9.py`

### Step 4 - Run end-to-end pipeline
`python pipeline_demo.py`