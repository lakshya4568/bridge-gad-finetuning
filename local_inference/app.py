"""Minimal Gradio UI for the local bridge-planning adapter."""

import gradio as gr
from pathlib import Path

from model_loader import ModelConfig, generate_text, load_model_and_tokenizer


CONFIG = ModelConfig(
    base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
    base_model_path=Path("./base_model"),
    adapter_path=Path("./adapter"),
    download_base_model=True,
    require_cuda=True,
    load_in_4bit=True,
)

print("Loading local base model and LoRA adapter...")
MODEL, TOKENIZER = load_model_and_tokenizer(CONFIG)
print("Model ready.")


def respond(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    if not prompt.strip():
        return "Enter a question first."
    return generate_text(
        MODEL,
        TOKENIZER,
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )


with gr.Blocks(title="Railway Bridge Assistant") as demo:
    gr.Markdown("# Railway Bridge Assistant\nLocal Qwen model with your LoRA adapter")
    prompt = gr.Textbox(
        label="Prompt",
        lines=6,
        placeholder="Ask about scour, waterway, foundations, or CAD parameters...",
    )
    with gr.Row():
        max_new_tokens = gr.Slider(32, 1024, value=400, step=1, label="Max new tokens")
        temperature = gr.Slider(0.0, 1.5, value=0.1, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
    submit = gr.Button("Generate", variant="primary")
    output = gr.Markdown(label="Answer")
    submit.click(respond, [prompt, max_new_tokens, temperature, top_p], output)
    prompt.submit(respond, [prompt, max_new_tokens, temperature, top_p], output)


if __name__ == "__main__":
    demo.launch()
