# pipeline_demo.py
from __future__ import annotations

import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from retrievers import RecipeRetriever, VideoRetriever


def build_tinyllama_prompt(user_query: str, recipe_context: str) -> str:
    """
    Use the same role-token structure as assignment 9.
    """
    eos = "</s>"
    ctx = recipe_context if recipe_context else "No external context provided."
    prompt = (
        f"{config.CHAT_TOK_SYSTEM}\n{config.CHAT_SYSTEM_PROMPT}{eos}\n"
        f"{config.CHAT_TOK_CONTEXT}\n{ctx}{eos}\n"
        f"{config.CHAT_TOK_USER}\n{user_query}{eos}\n"
        f"{config.CHAT_TOK_ASSISTANT}\n"
    )
    return prompt


def generate_answer(prompt: str, device: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(config.CHAT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.CHAT_MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.CHAT_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.CHAT_TEMPERATURE,
            top_p=config.CHAT_TOP_P,
            repetition_penalty=config.CHAT_REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # stop at EOS
    if tokenizer.eos_token and tokenizer.eos_token in text:
        text = text.split(tokenizer.eos_token)[0]

    # strip leaked role tokens
    for tok in [config.CHAT_TOK_USER, config.CHAT_TOK_SYSTEM, config.CHAT_TOK_CONTEXT, config.CHAT_TOK_ASSISTANT]:
        if tok in text:
            text = text.split(tok)[0]

    return text.strip()


def extract_steps(generated_text: str, max_steps: int = 5) -> List[str]:
    """
    Simple heuristic: grab numbered lines if present, else split into sentences.
    """
    lines = [l.strip() for l in generated_text.splitlines() if l.strip()]
    numbered = []
    for l in lines:
        if re.match(r"^\d+[\).\:-]\s+", l):
            numbered.append(re.sub(r"^\d+[\).\:-]\s+", "", l).strip())
    if numbered:
        return numbered[:max_steps]

    # fallback: sentence split
    sents = re.split(r"(?<=[.!?])\s+", generated_text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents[:max_steps]


def main():
    config.apply_env()
    device = config.pick_device_torch()

    user_query = input("User query: ").strip()
    if not user_query:
        raise SystemExit("Empty query.")

    # Retrieve recipes
    rr = RecipeRetriever(device=device)
    recipes = rr.semantic_search(user_query, top_k=5, use_keyword_fallback=True)

    print("\n=== Recipe Retrieval Results ===")
    for r in recipes[:5]:
        print(f"- [{r.recipe_id}] {r.title} (score={r.score:.4f}) source={r.source}")

    recipe_context = rr.format_context(recipes, max_recipes=3)

    # Build prompt and generate response
    prompt = build_tinyllama_prompt(user_query, recipe_context)
    answer = generate_answer(prompt, device=device)

    print("\n=== Generated Answer ===")
    print(answer)

    # Extract steps and retrieve video clips per step
    steps = extract_steps(answer, max_steps=5)
    print("\n=== Extracted Steps ===")
    for i, s in enumerate(steps, 1):
        print(f"{i}. {s}")

    vr = VideoRetriever(device=device)

    print("\n=== Video Retrieval (per step) ===")
    for i, step in enumerate(steps, 1):
        hits = vr.search(step, top_k=3)
        print(f"\nStep {i}: {step}")
        if not hits:
            print("  (no hits)")
            continue
        for h in hits:
            print(
                f"  - video={h.video_id} clip_id={h.clip_id} "
                f"[{h.start_sec:.1f}s–{h.end_sec:.1f}s] score={h.score:.4f}"
            )

    rr.close()
    vr.close()


if __name__ == "__main__":
    main()