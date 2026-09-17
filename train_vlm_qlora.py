"""
train_vlm_qlora.py — Flow B: QLoRA fine-tune Qwen2.5-VL-2B-Instruct for
GAD/CAD bridge-drawing parameter extraction. Targets an 8GB VRAM GPU.

Task: image of an engineering drawing -> strict JSON of parameters
      (clear span, clear height, wall thickness, slab thickness, etc.)

Dataset format expected at vlm_data/train.jsonl:
  {"image": "path/to/drawing.png", "messages": [
      {"role": "user", "content": "<image>Extract all structural parameters from this bridge drawing as JSON."},
      {"role": "assistant", "content": "{\"structure_type\": \"multi-cell box culvert\", \"clear_span_mm\": 3000, ...}"}
  ]}

Install:
  pip install unsloth
Run:
  python train_vlm_qlora.py
"""
import torch
from datasets import load_dataset
from unsloth import FastVisionModel   # not FastLanguageModel!
from unsloth.trainers import SFTTrainer as VisionSFTTrainer  # vision-aware trainer

MODEL_NAME = "unsloth/Qwen2.5-VL-2B-Instruct-bnb-4bit"
OUT_DIR = "bridge-vlm-qlora"
MAX_PIXELS = 1024 * 28 * 28   # cap image tokens for 8GB VRAM

# ---------------------------------------------------------------- 1. Load VLM
model, processor = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,          # QLoRA NF4 on the language + vision weights
)

# ---------------------------------------------------------------- 2. LoRA
# KEY: freeze the vision tower. Small dataset -> ViT already sees lines well.
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,     # vision tower frozen
    finetune_language_layers=True,    # LLM side trained
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ---------------------------------------------------------------- 3. Dataset
ds = load_dataset("json", data_files={
    "train": "vlm_data/train.jsonl",
    "validation": "vlm_data/val.jsonl",
})

from unsloth.chat_templates import convert_to_conversation
def to_convo(example):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "path": example["image"]},
            {"type": "text", "text": (
                "Extract all structural parameters from this engineering drawing "
                "and respond with ONLY a JSON object. Use keys: structure_type, "
                "clear_span_mm, clear_height_mm, wall_thickness_mm, slab_thickness_mm, "
                "n_cells (if applicable), units, source_view. No other text.")}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": example["json"]}]}]}

ds = ds.map(to_convo, remove_columns=ds["train"].column_names)

from unsloth.trainers import UnslothVisionDataCollator
collator = UnslothVisionDataCollator(model, processor)

# ---------------------------------------------------------------- 4. Trainer
FastVisionModel.for_training(model)
trainer = VisionSFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=collator,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args={
        "output_dir": OUT_DIR,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,      # effective batch 16
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "report_to": "none",
        "seed": 42,
        "max_length": 2048,
    },
)
trainer.train()
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ---------------------------------------------------------------- 5. Export
model.save_pretrained("bridge-vlm-lora-adapter")
model.save_pretrained_merged("bridge-vlm-merged", processor)

# ---------------------------------------------------------------- 6. Inference
FastVisionModel.for_inference(model)
from PIL import Image
image = Image.open("vlm_data/test/drawing.png")
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Extract all structural parameters from this "
     "engineering drawing as JSON."}]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, return_dict_in=True,
    tokenizer_kwargs={"dtype": torch.bfloat16})
inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
inputs["pixel_values"] = processor.image_processor(
    [image], return_tensors="pt")["pixel_values"].to("cuda")
out = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print(processor.decode(out[0], skip_special_tokens=True))
