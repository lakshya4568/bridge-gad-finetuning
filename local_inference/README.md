# Local Bridge Inference

This project loads the local `Qwen/Qwen2.5-1.5B-Instruct` base model and the trained LoRA adapter. On the first run, it downloads the base model into `base_model/` if that directory is incomplete. Later runs reuse those local files. The adapter is always expected locally.

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

Copy the downloaded adapter files into `adapter/`. The base model is downloaded automatically on first load unless you set `download_base_model=False` in the config and provide it yourself. The base model must be the same `Qwen/Qwen2.5-1.5B-Instruct` model used during training.

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

## Test GPU/CPU and responses

Run the smoke-test script:

```bash
uv run python test_local.py
```

It reports CUDA availability, selected device, model parameter placement, GPU
name and VRAM usage when CUDA is available. It then tests Lacey's scour formula,
scour multipliers, regime waterway, foundation grip length, and a CAD JSON
response. The script exits with status `1` when expected content is missing.

The checks are only smoke tests. They do not certify engineering correctness;
validate final calculations with deterministic engineering code and applicable
railway standards.

The application uses NF4 4-bit loading on CUDA by default. Set `load_in_4bit=False` in `app.py` if you need a non-quantized CUDA load and have enough VRAM. The downloaded base model and adapter weights are ignored by git; only code and configuration are committed.

The app and smoke test set `require_cuda=True`. This prevents an accidental CPU run and raises a clear error when CUDA is unavailable. Set `require_cuda=False` only when you intentionally want CPU inference.
