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
    formatted_prompt = (
        f"{prompt}\n\n"
        "Answer for a human reader. Start with a short plain-language explanation, "
        "then add a clearly labeled Formula section, define every variable, and "
        "finish with a practical bridge-design note when relevant. Use clean "
        "Markdown and LaTeX equations. Do not return JSON unless the user "
        "explicitly asks for JSON. Do not respond with only an equation."
    )
    system_prompt = (
        "You are an expert railway bridge planning assistant grounded in the "
        "IRICEN training material. Give the exact bridge formula requested, "
        "explain what it means in plain language, define its variables, and use "
        "Markdown. Render equations with $...$ or $$...$$. Never invent "
        "alternative formulas. For normal alluvial scour, use exactly "
        "D = 0.473 * (Qf / f)^(1/3), with f = 1.76 * sqrt(m). For an "
        "abutment use 1.25 * D; for a pier nose use 2.00 * D. Never replace "
        "these with pressure, temperature, hydraulic-gradient, or polynomial "
        "formulas."
    )
    return generate_text(
        MODEL,
        TOKENIZER,
        formatted_prompt,
        system_prompt=system_prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )


CSS = """
.formula-app { max-width: 1180px; margin: 0 auto; }
.answer-panel { min-height: 420px; border: 1px solid #d8dee9; border-radius: 12px; padding: 18px; }
.formula-note { color: #5b6472; font-size: 0.92rem; }
"""

with gr.Blocks(title="Railway Bridge Assistant") as demo:
    with gr.Column(elem_classes="formula-app"):
        gr.Markdown(
            "# Railway Bridge Assistant\n"
            "Ask about bridge planning, scour, waterways, foundations, or CAD parameters."
        )
        gr.Markdown(
            "Formulas are rendered as Markdown/LaTeX. For example: "
            "$$D = 0.473\\left(\\frac{Q_f}{f}\\right)^{1/3}$$",
            elem_classes="formula-note",
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Question",
                    lines=8,
                    placeholder="Example: What is Lacey's formula for normal scour depth?",
                )
                with gr.Row():
                    max_new_tokens = gr.Slider(32, 1024, value=400, step=1, label="Max tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.1, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                submit = gr.Button("Generate answer", variant="primary")
            with gr.Column(scale=1):
                output = gr.Markdown(label="Answer", elem_classes="answer-panel")

        gr.Examples(
            examples=[
                ["State Lacey's normal scour depth formula and define every variable."],
                ["What are the abutment and pier scour multipliers? Show the equations."],
                ["Return CAD JSON for an RCC single-cell box culvert with key dimensions."],
            ],
            inputs=prompt,
            label="Example questions",
        )

    submit.click(respond, [prompt, max_new_tokens, temperature, top_p], output)
    prompt.submit(respond, [prompt, max_new_tokens, temperature, top_p], output)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
