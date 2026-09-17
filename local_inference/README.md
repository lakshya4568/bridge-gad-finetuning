# Local Bridge Inference

This project loads the local `Qwen/Qwen2.5-1.5B-Instruct` base model and the trained LoRA adapter from disk. It contains no training code and does not download model files at runtime.

## Layout

```text
local_inference/
├── app.py
├── inference.ipynb
├── model_loader.py
├── pyproject.toml
├── adapter/
└── base_model/
```

Copy the downloaded adapter files into `adapter/`. Copy the complete local base-model directory into `base_model/`. The base model must be the same `Qwen/Qwen2.5-1.5B-Instruct` model used during training.

Expected adapter files include `adapter_model.safetensors` and `adapter_config.json`.

Expected base-model files include `config.json`, `tokenizer.json`, `tokenizer_config.json`,
the Qwen chat-template file if supplied separately, and model weight files such as `*.safetensors`.

The tokenizer is part of the base model and is required at runtime. The copied tokenizer
files in the adapter export are optional duplicates; the loader intentionally uses the
tokenizer from `base_model/` so the base model and tokenizer stay together.

## Run

From this directory:

```bash
uv run python app.py
```

The app prints a local URL. Open it in a browser and enter a bridge-planning or CAD prompt. CUDA is selected automatically when available; otherwise the loader uses CPU.

For a quick loader check without launching the UI:

```bash
uv run python -c "from model_loader import load_model_and_tokenizer; load_model_and_tokenizer(); print('model loaded')"
```

The application uses NF4 4-bit loading on CUDA by default. Set `load_in_4bit=False` in `app.py` if you need a non-quantized CUDA load and have enough VRAM.
