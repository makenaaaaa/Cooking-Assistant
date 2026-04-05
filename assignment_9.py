"""
Assignment #9
"""

from __future__ import annotations

from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
config.apply_env()

# ==============================================================================
# CONVERSATION MANAGER
# ==============================================================================

class ConversationManager:
    """
    Manages conversation history and prompt construction.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.history: List[Dict[str, str]] = []
        self.eos = tokenizer.eos_token if tokenizer.eos_token else "</s>"

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > config.CHAT_MAX_HISTORY_TURNS:
            self.history = self.history[-config.CHAT_MAX_HISTORY_TURNS:]

    def get_approx_token_count(self) -> int:
        full_text = "".join([t["content"] for t in self.history])
        return len(self.tokenizer.encode(full_text))

    def build_full_prompt(self, retrieved_context: str = "") -> str:
        parts = []

        # 1) System prompt
        parts.append(f"{config.CHAT_TOK_SYSTEM}\n{config.CHAT_SYSTEM_PROMPT}{self.eos}\n")

        # 2) Context block (reserved for RAG retrieval)
        context_text = retrieved_context if retrieved_context else "No external context provided."
        parts.append(f"{config.CHAT_TOK_CONTEXT}\n{context_text}{self.eos}\n")

        # 3) Conversation history
        for turn in self.history:
            if turn["role"] == "user":
                parts.append(f"{config.CHAT_TOK_USER}\n{turn['content']}{self.eos}\n")
            elif turn["role"] == "assistant":
                parts.append(f"{config.CHAT_TOK_ASSISTANT}\n{turn['content']}{self.eos}\n")

        # 4) Trigger assistant generation
        parts.append(f"{config.CHAT_TOK_ASSISTANT}\n")
        return "".join(parts)

# ==============================================================================
# OUTPUT VALIDATION
# ==============================================================================

def validate_and_clean_output(
    generated_tokens: torch.Tensor,
    prompt_token_len: int,
    tokenizer,
) -> str:
    """
    Output validation to prevent token leakage and handle degenerates.
    """
    new_tokens = generated_tokens[prompt_token_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Stop at EOS token
    if tokenizer.eos_token and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]

    forbidden_tokens = [
        config.CHAT_TOK_USER,
        config.CHAT_TOK_SYSTEM,
        config.CHAT_TOK_CONTEXT,
        config.CHAT_TOK_ASSISTANT,
        "<|user|>",
        "<|system|>",
    ]
    for tok in forbidden_tokens:
        if tok in response:
            response = response.split(tok)[0]

    cleaned = response.strip()
    if not cleaned:
        return "[System: The model returned an empty response. Please try rephrasing your question.]"
    return cleaned

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    print("Initializing Cooking Chatbot...")

    device = config.pick_device_torch()
    print(f"Running on: {device}")

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.CHAT_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            config.CHAT_MODEL_ID,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        print(f"Model loaded: {config.CHAT_MODEL_ID}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    chat_manager = ConversationManager(tokenizer)

    print("\n" + "=" * 60)
    print(" COOKING CHATBOT (Assignment #9 - Project Part 2)")
    print(" Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except EOFError:
            break

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        chat_manager.add_turn("user", user_input)

        prompt = chat_manager.build_full_prompt(retrieved_context="")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_token_len = inputs.input_ids.shape[1]

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

        final_response = validate_and_clean_output(outputs[0], prompt_token_len, tokenizer)
        chat_manager.add_turn("assistant", final_response)

        print(f"Assistant: {final_response}")
        print(f"[Debug] History tokens: ~{chat_manager.get_approx_token_count()}")
        print("-" * 60)


if __name__ == "__main__":
    main()

# ==============================================================================
# TASK 3: RAG ARCHITECTURE PREPARATION
# ==============================================================================
"""
RAG Extension Plan:

## 1. Injecting Recipe Text 
We will integrate the SQLite database and FAISS index created in Assignment #8:
1. Encode user query using the same Sentence-BERT model (all-MiniLM-L6-v2)
2. Retrieve Top-K recipe IDs from FAISS index
3. Query SQLite: "SELECT title, ingredients, instructions FROM recipes WHERE id IN (...)"
4. Format as: "Recipe: {title}\\nIngredients: {ingredients}\\nInstructions: {instructions}"
5. Pass into "chat_manager.build_full_prompt(retrieved_context=formatted_recipes)"

## 2. Including Video Metadata 
After generating the assistant's text response:
1. Extract cooking action keywords from the response
2. Use CLIP text encoder to retrieve relevant video clips
3. Display video metadata to user: "[Video: {video_id}, {start}s-{end}s]"


## 3. Context Control Strategy
To prevent exceeding TinyLlama's 2048 token limit:
1. Monitor total tokens: 'system_tokens + context_tokens + history_tokens < 2048'
2. If approaching limit:
   a) Reduce Top-K recipes (e.g., from 3 to 1)
   b) Truncate recipe instructions to first 2-3 steps
   c) Remove oldest conversation turns from history
3. Track usage with 'chat_manager.get_approx_token_count()'
4. Implement priority: System Prompt (highest) > Context > History (lowest)
"""
