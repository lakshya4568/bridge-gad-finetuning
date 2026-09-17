"""Shared local loader and generation helpers for the bridge adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass(frozen=True)
class ModelConfig:
    base_model_path: Path = Path("./base_model")
    adapter_path: Path = Path("./adapter")
    load_in_4bit: bool = True
    max_input_tokens: int = 2048


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_paths(config: ModelConfig) -> None:
    if not config.base_model_path.exists():
        raise FileNotFoundError(
            f"Base model directory not found: {config.base_model_path}. "
            "Place the local Qwen2.5-1.5B-Instruct files there."
        )
    if not config.adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {config.adapter_path}. "
            "Place adapter_model.safetensors and adapter_config.json there."
        )
    if not (config.adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Missing adapter_config.json in {config.adapter_path}."
        )
    required_tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    missing_tokenizer_files = [
        name
        for name in required_tokenizer_files
        if not (config.base_model_path / name).exists()
    ]
    if missing_tokenizer_files:
        raise FileNotFoundError(
            "Missing tokenizer files in "
            f"{config.base_model_path}: {', '.join(missing_tokenizer_files)}."
        )


def load_model_and_tokenizer(config: ModelConfig | None = None):
    config = config or ModelConfig()
    _validate_paths(config)
    device = select_device()

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }

    if device == "cuda" and config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(
        base_model,
        config.adapter_path,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
