"""
Assignment #11: User Interface
"""
from __future__ import annotations

import os
import re
import html
import gradio as gr
import torch
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import existing project modules
import config
from retrievers import RecipeRetriever, VideoRetriever

# Apply environment settings
config.apply_env()

# GLOBAL MODEL LOADING
print("Loading models... (this may take a minute)")
DEVICE = config.pick_device_torch()

tokenizer = AutoTokenizer.from_pretrained(config.CHAT_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    config.CHAT_MODEL_ID,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    low_cpu_mem_usage=True,
).to(DEVICE)
model.eval()

recipe_retriever = RecipeRetriever(device=DEVICE)
video_retriever = VideoRetriever(device=DEVICE)

print(f"Models loaded on {DEVICE}")

# LOGIC
def extract_steps(generated_text: str, max_steps: int = 25) -> List[str]:
    """Extracts cooking steps from the generated text."""
    lines = [l.strip() for l in generated_text.splitlines() if l.strip()]
    numbered = []
    for l in lines:
        # Regex to find "1. " or "1) " or "1- "
        if re.match(r"^\d+[\).\:-]\s+", l):
            clean_line = re.sub(r"^\d+[\).\:-]\s+", "", l).strip()
            numbered.append(clean_line)
    
    if numbered:
        return numbered[:max_steps]
    
    # Fallback: split by sentence if no numbers found
    sents = re.split(r"(?<=[.!?])\s+", generated_text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents[:max_steps]

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def build_prompt(history: List[Dict[str, str]], context: str) -> str:
    parts = []
    parts.append(f"{config.CHAT_TOK_SYSTEM}\n{config.CHAT_SYSTEM_PROMPT}</s>\n")
    ctx_text = context if context else "No external context provided."
    parts.append(f"{config.CHAT_TOK_CONTEXT}\n{ctx_text}</s>\n")
    for turn in history:
        if turn['role'] == 'user':
            parts.append(f"{config.CHAT_TOK_USER}\n{turn['content']}</s>\n")
        elif turn['role'] == 'assistant':
            parts.append(f"{config.CHAT_TOK_ASSISTANT}\n{turn['content']}</s>\n")
    parts.append(f"{config.CHAT_TOK_ASSISTANT}\n")
    return "".join(parts)

def format_playlist_as_html(playlist_display: List[List[str]]) -> str:
    """Creates a HTML table for the playlist."""
    if not playlist_display:
        return "<p style='color:gray; padding:10px;'>No related clips found yet.</p>"
    
    html_str = """
    <table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.9em;">
        <thead>
            <tr style="background-color: #f0f0f0; text-align: left;">
                <th style="padding: 8px; border-bottom: 2px solid #ddd; width: 5%;">#</th>
                <th style="padding: 8px; border-bottom: 2px solid #ddd; width: 75%;">Instruction Match</th>
                <th style="padding: 8px; border-bottom: 2px solid #ddd; width: 10%;">Time</th>
                <th style="padding: 8px; border-bottom: 2px solid #ddd; width: 10%;">Score</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, row in enumerate(playlist_display):
        desc, time, score = row
        html_str += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 8px;"><b>{i+1}</b></td>
            <td style="padding: 8px;">{html.escape(desc)}</td>
            <td style="padding: 8px;">{html.escape(time)}</td>
            <td style="padding: 8px;">{html.escape(score)}</td>
        </tr>
        """
    html_str += "</tbody></table>"
    return html_str

def generate_response(message: str, history_state, playlist_state):
    if history_state is None:
        history_state = []
    if playlist_state is None:
        playlist_state = []

    # Retrieve Recipes
    recipes = recipe_retriever.semantic_search(message, top_k=3, use_keyword_fallback=True)
    context_str = recipe_retriever.format_context(recipes, max_recipes=2)

    # Update History & Prompt
    history_state.append({"role": "user", "content": message})
    full_prompt = build_prompt(history_state, context_str)
    total_tokens = count_tokens(full_prompt)

    # Generate Text
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
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

    response_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=False)

    # Cleanup response tokens
    if tokenizer.eos_token and tokenizer.eos_token in response_text:
        response_text = response_text.split(tokenizer.eos_token)[0]
    
    for tok in [config.CHAT_TOK_USER, config.CHAT_TOK_SYSTEM, config.CHAT_TOK_CONTEXT, config.CHAT_TOK_ASSISTANT]:
        if tok in response_text:
            response_text = response_text.split(tok)[0]
    
    response_text = response_text.strip()
    history_state.append({"role": "assistant", "content": response_text})
    total_tokens += count_tokens(response_text)

    # Retrieve Videos based on Steps
    steps = extract_steps(response_text)
    
    playlist_data = [] 
    playlist_display = [] 
    seen_clips = set()

    # Use extracted steps for searching, or fallback to the full message
    search_queries = steps if steps else [message]
    
    for i, step_text in enumerate(search_queries):
        # Human readable step number
        step_num = i + 1 if steps else 0
        
        # Search: fetch top 3 candidates to be safe, but will only pick ONE
        hits = video_retriever.search(step_text, top_k=3)
        
        for h in hits:
            # Stricter score to avoid irrelevant clips
            if h.score < 0.23: 
                continue

            # Skip duplicates (if this clip was used for a previous step)
            if h.clip_id in seen_clips:
                continue
            
            # Found match
            seen_clips.add(h.clip_id)
            
            # Add to playlist
            playlist_data.append({
                "path": h.video_path,
                "start": h.start_sec,
                "end": h.end_sec,
                "score": h.score
            })
            
            # Description alignment
            prefix = f"[Step {step_num}] " if steps else "[Query] "
            clean_step = step_text.replace("\n", " ")
            
            display_desc = prefix + clean_step
            playlist_display.append([display_desc, f"{h.start_sec:.0f}-{h.end_sec:.0f}s", f"{h.score:.2f}"])
            
            # Break inner loop immediately to ensure only 1 clip per step
            break 

    # Format HTML Table
    html_table = format_playlist_as_html(playlist_display)
    
    # Update dropdown choices
    choices = []
    if playlist_display:
        for i, row in enumerate(playlist_display):
            desc_preview = row[0]
            if len(desc_preview) > 60:
                desc_preview = desc_preview[:57] + "..."
            choices.append(f"{i+1}: {desc_preview} ({row[1]})")
    else:
        choices = ["No videos found"]

    # Format Chat for Gradio
    chat_tuples = []
    for i in range(0, len(history_state), 2):
        u = history_state[i]['content']
        a = history_state[i+1]['content'] if i+1 < len(history_state) else ""
        chat_tuples.append((u, a))

    return (
        "",               # Clear msg_input
        chat_tuples,      # chatbot
        history_state,    # history state
        context_str,      # context display
        full_prompt,      # prompt display
        str(total_tokens),# token display
        html_table,       # playlist HTML
        gr.Dropdown(choices=choices, value=None, interactive=True), # Update dropdown
        playlist_data     # playlist state
    )

def play_video(selection_str: str, playlist_state):
    """
    Handles dropdown selection.
    Returns an HTML string with a <video> tag using #t=start,end.
    """
    if not selection_str or not playlist_state:
        return None
    
    try:
        idx = int(selection_str.split(":")[0]) - 1
        if 0 <= idx < len(playlist_state):
            data = playlist_state[idx]
            abs_path = os.path.abspath(data["path"])
            
            if os.path.exists(abs_path):
                video_src = f"/file={abs_path}#t={data['start']},{data['end']}"
                
                html_code = f"""
                <video width="100%" controls autoplay>
                    <source src="{video_src}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """
                return html_code
            else:
                return "<p style='color:red'>Video file not found on server.</p>"
    except Exception as e:
        print(f"Selection error: {e}")
        return f"<p style='color:red'>Error: {e}</p>"
    return None

# UI
with gr.Blocks(title="Cooking Bot UI", theme=gr.themes.Soft()) as demo:
    
    history_state = gr.State()
    playlist_state = gr.State()

    gr.Markdown("# 🍳 AI Cooking Assistant")

    with gr.Row():
        # Left: Chat
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                msg_input = gr.Textbox(show_label=False, placeholder="Ask about a recipe...", scale=4)
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            # System Inspection
            with gr.Accordion("System Inspection", open=False):
                token_display = gr.Textbox(label="Total Token Usage", interactive=False)
                system_prompt_display = gr.Textbox(
                    label="System Prompt", 
                    lines=4, 
                    max_lines=10, 
                    interactive=False, 
                    show_copy_button=True
                )
                context_display = gr.Textbox(
                    label="Retrieved Context", 
                    lines=6, 
                    max_lines=15, 
                    interactive=False, 
                    show_copy_button=True
                )

        # Right: Video
        with gr.Column(scale=1):
            gr.Markdown("### 🎥 Video Explorer")
            
            # HTML component for custom video player
            video_player = gr.HTML(
                value="<div style='height:300px; background:#eee; display:flex; align-items:center; justify-content:center; color:#777;'>Video Player</div>",
                label="Player"
            )
            
            gr.Markdown("#### Related Clips")
            playlist_html = gr.HTML(value="<p style='color:gray'>No videos loaded.</p>")
            
            video_selector = gr.Dropdown(
                label="Select Clip to Play", 
                choices=[], 
                interactive=True
            )

    # Wiring
    submit_triggers = [msg_input.submit, submit_btn.click]
    for trigger in submit_triggers:
        trigger(
            fn=generate_response,
            inputs=[msg_input, history_state, playlist_state],
            outputs=[
                msg_input, 
                chatbot, 
                history_state, 
                context_display, 
                system_prompt_display, 
                token_display, 
                playlist_html,    # Updates table
                video_selector,   # Updates choices
                playlist_state
            ]
        )

    video_selector.change(
        fn=play_video,
        inputs=[video_selector, playlist_state],
        outputs=[video_player]
    )

if __name__ == "__main__":
    allowed = [os.path.abspath(config.YOUCOOK_ROOT), os.path.abspath(config.PROJECT_ROOT)]
    print("Please wait for the public link...")
    demo.launch(share=True, allowed_paths=allowed)