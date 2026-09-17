"""
train_text_qlora.py — Flow A: QLoRA fine-tune Qwen2.5-1.5B-Instruct on the
bridge-planning dataset. Targets an 8GB VRAM GPU.

Install:
  pip install unsloth  ( pulls torch/transformers/trl/peft/bitsandbytes )

Run:
  python train_text_qlora.py
"""
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

MAX_SEQ_LEN = 2048
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
OUT_DIR = "bridge-qlora-1.5b"

# ---------------------------------------------------------------- 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,          # QLoRA: NF4 quantization
    dtype=None,
)

# ---------------------------------------------------------------- 2. Attach LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                        # try 32 if underfitting
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",   # big VRAM saver
    random_state=42,
)

# ---------------------------------------------------------------- 3. Dataset
ds = load_dataset("json", data_files={
    "train": "train.jsonl", "validation": "val.jsonl"
})

def to_text(example):
    """Apply chat template; training masks prompt tokens via
    train_on_responses_only below."""
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(to_text, remove_columns=ds["train"].column_names)

# ---------------------------------------------------------------- 4. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args=SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,        # effective batch 16
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),     # 8GB consumer GPU = fp16 LoRA params
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    ),
    # Only compute loss on assistant tokens
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ---------------------------------------------------------------- 5. Train
gpu_mem = lambda: torch.cuda.memory_allocated() / 1e9
print(f"VRAM after setup: {gpu_mem():.2f} GB")
trainer_stats = trainer.train()
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"Train time: {trainer_stats.metrics['train_runtime']:.0f}s for "
      f"{trainer_stats.metrics['epoch']:.2f} epochs")

# ---------------------------------------------------------------- 6. Smoke test
FastLanguageModel.for_inference(model)
msgs = [{"role": "user", "content":
         "Compute Q50 for a catchment of 5.2 sq km, L = 3.1 km, H = 85 m, "
         "R = 14 cm, red soil, F = 0.79, tc-rainfall ratio 0.30."}]
inputs = tokenizer.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
out = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))

# ---------------------------------------------------------------- 7. Export
model.save_pretrained("bridge-lora-adapter")            # LoRA only (~50 MB)
model.save_pretrained_merged("bridge-merged-16bit", tokenizer)  # full merge
# Optional GGUF for llama.cpp/Ollama local serving:
# model.save_pretrained_gguf("bridge-gguf", tokenizer, quantization_method="q4_k_m")
