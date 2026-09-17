"""Smoke-test local model loading, device placement, and bridge-domain responses.

Run from this directory with:
    uv run python test_local.py

The first run may download dependencies and the base model. The adapter must
already exist in ./adapter/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from model_loader import ModelConfig, generate_text, load_model_and_tokenizer, select_device


CONFIG = ModelConfig(
    base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
    base_model_path=Path("./base_model"),
    adapter_path=Path("./adapter"),
    download_base_model=True,
    require_cuda=True,
    load_in_4bit=True,
)

TESTS = [
    {
        "name": "Lacey normal scour formula",
        "prompt": "What is Lacey's formula for normal depth of scour D below the foundation design discharge level in alluvial rivers?",
        "required": ["0.473", "Qf", "f", "1.76"],
    },
    {
        "name": "Scour multipliers",
        "prompt": "What are the scour multipliers for an abutment and the nose of a pier?",
        "required": ["1.25", "2.00"],
    },
    {
        "name": "Regime waterway formula",
        "prompt": "What is Lacey's regime linear waterway formula, and what is the standard coefficient result?",
        "required": ["4.83", "sqrt", "Q"],
    },
    {
        "name": "Foundation grip length",
        "prompt": "What grip length is required in ordinary soil, hard rock, and soft rock?",
        "required": ["1.75", "0.3", "1.5"],
    },
    {
        "name": "CAD JSON response",
        "prompt": "Return only a JSON object for an RCC single-cell box culvert with clear span, clear height, wall thickness, and slab thickness in millimetres.",
        "required": ["clear_span", "clear_height", "wall_thickness", "slab_thickness"],
        "json": True,
    },
]


def print_runtime_info(model) -> None:
    device = select_device()
    first_parameter_device = next(model.parameters()).device
    print("Runtime checks")
    print("- torch version:", torch.__version__)
    print("- CUDA available:", torch.cuda.is_available())
    print("- selected device:", device)
    print("- first model parameter device:", first_parameter_device)
    print("- model device map:", getattr(model, "hf_device_map", "not exposed"))
    if torch.cuda.is_available():
        print("- GPU:", torch.cuda.get_device_name(0))
        print("- allocated VRAM (GB):", round(torch.cuda.memory_allocated() / 1e9, 2))
        print("- reserved VRAM (GB):", round(torch.cuda.memory_reserved() / 1e9, 2))
        if first_parameter_device.type != "cuda":
            raise RuntimeError(
                f"Model parameter is on {first_parameter_device}, not CUDA. "
                "GPU loading did not succeed."
            )


def validate_response(test: dict, response: str) -> list[str]:
    lowered = response.lower()
    missing = [item for item in test["required"] if item.lower() not in lowered]
    if test.get("json"):
        candidate = response.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            missing.append("valid JSON object")
    return missing


def main() -> int:
    print("Loading model and adapter for local smoke test...")
    model, tokenizer = load_model_and_tokenizer(CONFIG)
    print_runtime_info(model)
    print()

    failures = 0
    for index, test in enumerate(TESTS, start=1):
        response = generate_text(
            model,
            tokenizer,
            test["prompt"],
            max_new_tokens=350,
            temperature=0.0,
            top_p=0.9,
        )
        missing = validate_response(test, response)
        status = "PASS" if not missing else "CHECK"
        print(f"[{status}] Test {index}: {test['name']}")
        print("Prompt:", test["prompt"])
        print("Response:\n", response)
        if missing:
            failures += 1
            print("Missing expected content:", ", ".join(missing))
        print("-" * 80)

    if failures:
        print(f"Completed with {failures} validation warning(s). Review the responses above.")
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
