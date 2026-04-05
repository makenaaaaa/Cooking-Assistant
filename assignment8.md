## Layout

```text
+----------------------------------------------------------------------------------+
| Header: "Recipe + Video Retrieval (RAG-ready)"                                   |
+-------------------------------+--------------------------------------------------+
| LEFT: Chat                    | RIGHT: Retrieval Results                         |
| - Chat history                | Tabs: [Recipes] [Video Clips] [Inspection]       |
| - User input box              |                                                  |
| - Buttons: Retrieve / Clear   | [Recipes Tab]                                    |
|                               |  - Top-K list (title + score)                    |
|                               |  - Expand details (ingredients/instructions)     |
|                               |                                                  |
|                               | [Video Clips Tab]                                |
|                               |  - Top-K clips (video_id + time + score)         |
|                               |  - Clip preview area (player/thumbnail/path)     |
|                               |                                                  |
|                               | [Inspection Tab]                                 |
|                               |  - Similarity scores (Top-K)                     |
|                               |  - Token budget / context size                   |
|                               |  - Injected context items                        |
+-------------------------------+--------------------------------------------------+

```

### User Flow

1. User asks a question in the chat, e.g. "I want a quick spicy noodle dish without peanuts.”

2. System runs retrieval:
    - Recipe retrieval: query embedding → FAISS (title + instructions) → Top-K recipes
    - Video retrieval: CLIP text embedding → FAISS → Top-K clips

3. UI displays results:
    - Ranked recipes with similarity scores and expandable details
    - Ranked clips with timestamps, scores, and optional captions/preview

4. User clicks “Generate Answer”: Selected recipes/clips are injected as context for a grounded answer (RAG)

### Components

1. Chat window
    - Chat history
    - Query input
    - Buttons:
        - Retrieve: run Top-K retrieval only
        - Clear: reset session
        - Generate Answer: RAG response based on selected context

2. Automated Top-K recipe retrieval
    - Uses embeddings computed in Task 1:
        - title index: faiss_title.index
        - instructions index: faiss_instructions.index
    - Combines results:
        - union of candidate IDs, normalize scores

3. Automated Top-K clip retrieval
    - Uses CLIP embeddings computed in Task 2:
        - clip index: youcook_clip.index
    - Displays:
        - video_id, start_sec–end_sec, score
        - caption if available, otherwise omit

### Inspection Tab

1. Similarity score visualization
    - Bar chart of Top-K recipe similarity scores
    - Bar chart of Top-K clip similarity scores
    - Purpose: see confidence and identify weak/ambiguous retrieval.

2. Context size / token budget display (RAG readiness)
    - Estimated context size for selected items (recipes + captions)
    - Token budget meter (e.g., 8k/16k)
    - Purpose: detect truncation risk and guide what to include.

3. "Injected context” preview
    - Shows exactly what would be sent to a future LLM:
        - Selected recipe snippets (title + ingredient list + key instruction steps)
        - Selected clip references (video_id + timestamps + optional caption)
    - Purpose: makes retrieval auditable and easier to debug.

4. Retrieval settings (transparency controls)
    - Top-K sliders (recipes/clips)
    - Recipe weighting (title vs instructions)
    - Clip settings summary (clip_len / stride / frames_per_clip)
    - Device/model names used
    - Purpose: reproducibility across machines and runs.