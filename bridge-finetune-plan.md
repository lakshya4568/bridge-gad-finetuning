# Bridge Domain Model — Fine-Tuning Master Plan
**Target: 1–2B parameter model, 8GB VRAM, LoRA/QLoRA, VLM-first with text fallback**

---

## 0. Context Recap

| Asset | What it gives us |
|---|---|
| IRICEN Railway Bridge Planning book (144 pp) | Design discharge Q50 formulas (0.278·C·I₅₀·A), Lacey's waterway/scour formulas (Pw = 4.83·Q^0.5, D = 0.473·(Qf/f)^⅓), SUH parameters (tp, qp, W50, TB...), method statements (span arrangement, foundation depth), 3 fully worked numeric examples, girder type/cost tables |
| CeADAR fine-tuning guide (114 pp) | 7-stage pipeline, LoRA/QLoRA/DoRA math, hyperparameter ranges, evaluation methodology |
| Your CAD project | GAD/PDF/DWG bridge + culvert drawings with parameters (clear span, clear height, wall thickness, slab thickness) → **this is your VLM training image source** |

**Key decision:** The book is text-only → text SFT is guaranteed. Your CAD/GAD drawing collection is what makes a VLM possible. The VLM is the higher-value target (it plugs directly into your CAD ingestion pipeline), but it needs ~300–1000 paired image→JSON examples before it beats prompting a big model.

---

## 1. The Multi-Flow (Decision Diagram)

```
                        ┌─────────────────────────────┐
                        │  PHASE 0: DATA AUDIT (1 day) │
                        └──────────────┬──────────────┘
                                       │
                    Do you have GAD/CAD drawing images
                    you can pair with parameter labels?
                                       │
              ┌────────────────────────┴────────────────────────┐
              │ YES (or can make ≥300 pairs)      NO / not yet    │
              ▼                                                  ▼
   ┌─────────────────────┐                          ┌──────────────────────┐
   │ FLOW B: VLM         │                          │ FLOW A: TEXT-ONLY    │
   │ Qwen2.5-VL-2B       │                          │ Qwen2.5-1.5B-Instruct│
   │ QLoRA 4-bit         │                          │ QLoRA 4-bit          │
   └──────────┬──────────┘                          └──────────┬───────────┘
              │                                                │
   B1. Render/collect drawings                     A1. Parse book → chunks
   B2. Label: image → JSON params                  A2. Generate instruction pairs
       (clear span, wall thk,                     A3. Train QLoRA text model
        span type, dimensions)                    A4. Eval: held-out worked
   B3. Train QLoRA VLM                                 examples, numeric tolerance
   B4. Eval: parameter                            A5. Export GGUF / merge
       extraction accuracy                             │
       vs ground truth                                  │
              │                                                │
              └────────────────┬───────────────────────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ FLOW C: HYBRID DEPLOYMENT     │
                  │ VLM extracts drawing params   │
                  │ → text LLM runs planning      │
                  │   method statements / Q50     │
                  │   calculations                │
                  │ → feeds your CAD constraint   │
                  │   graph solver                │
                  └──────────────────────────────┘
```

**Run Flow A first regardless** — it's a 1-day job that validates your whole training pipeline, and its output model answers "how do I compute design discharge for a 2.54 km² red-soil catchment in sub-zone 3i" with the book's exact method. Then build Flow B data while Flow A trains.

---

## 2. Model Selection

| Path | Model | Params | Why this one | QLoRA VRAM @ seq 1024 |
|---|---|---|---|---|
| **VLM (primary)** | Qwen2.5-VL-2B-Instruct | 2.2B | Best-in-class document/diagram/OCR understanding at this size; native dynamic resolution; Unsloth support | ~5–6.5 GB |
| VLM (tiny fallback) | SmolVLM2-500M / 2.2B | 0.5 / 2.2B | Extremely memory-efficient; 500M trains in <4GB if 2B is tight | ~3–6 GB |
| **Text (primary)** | Qwen2.5-1.5B-Instruct | 1.5B | Strong instruction following, math-capable for Q50/scour calcs, proven fine-tune target | ~4–5 GB |
| Text (alt) | Llama-3.2-1B-Instruct | 1.3B | Slightly smaller, good reasoning per param | ~3.5–4.5 GB |
| Bigger (if you get 12–16GB later) | Qwen2.5-VL-3B / 7B | 3–7B | Meaningful accuracy jump for complex GADs | 7–16 GB |

**Rules of thumb used:** 4-bit quantized weights ≈ 0.55 GB per B params; + LoRA adapters/optimizer ~0.3–0.5 GB; + activations with gradient checkpointing ~1–2 GB at seq 1024. Never train the vision tower on a small dataset — freeze it, train language + attention only.

---

## 3. Flow A — Text-Only Fine-Tuning (run this week)

### A1. Dataset from the book
Target: **800–1500 instruction pairs**, split 90/10 train/val. Five pair types:

1. **Formula QA** — "What is Lacey's formula for normal scour depth?" → formula + when it applies (SSC 4.6.3 vs 4.6.4).
2. **Method-statement steps** — "List the steps to finalize span arrangement" → the 8-step procedure.
3. **Worked-example rebuilds** — the book's illustrations (2.54 km² catchment → Q50 = 36.3 m³/s; 294 km² Kaveri → SUH → Q50 = 477 m³/s) turned into "given inputs, compute" Q→A pairs with the full chain of calculation. **This is the highest-value pair type.**
4. **Norm/criteria lookup** — freeboard by discharge, vertical clearance bands, scour multipliers (1.25D abutment / 2D pier), grip length rules.
5. **Choice rationale** — PSC vs steel girder tradeoffs, foundation type selection (open/pile/well).

You cannot hand-write 1500 pairs. Strategy: `build_bridge_dataset.py` extracts and chunks the book, you fill a `formula_bank.json` + `examples.json` with ~40 rich seed entries, and the script programmatically expands them (input perturbation: vary catchment area, rainfall, soil type → recompute answers with the actual formulas as ground truth). Because the answers are **computed by formula**, not hallucinated, label quality is guaranteed.

### A2. Training config (per CeADAR guide + standard QLoRA practice)

| Hyperparameter | Value | Reason |
|---|---|---|
| Quantization | NF4 + double quant, bnb 4-bit | QLoRA standard |
| LoRA rank | 16 (try 32 if underfit) | CeADAR case studies use 32/α=32; 16 is enough for narrow domains |
| LoRA alpha | 16 (= rank) | α/r = 1 keeps update scale stable |
| LoRA dropout | 0.05 | Small dataset regularization |
| Target modules | q, k, v, o, gate, up, down proj | Full attention + MLP coverage |
| LR | 2e-4 | Standard for LoRA SFT |
| Scheduler | cosine + 3% warmup | Stable late-training decay |
| Batch | 2 per-device × 8 grad accum = 16 effective | 8GB constraint |
| Epochs | 3 (watch val loss; stop when flat) | Small data overfits fast |
| Max seq len | 2048 | Worked examples are long |
| Train on responses only | yes | Don't learn to mimic questions |

### A3. Evaluation
- Held-out worked examples: **numeric tolerance eval** — |predicted Q50 − true Q50| / true Q50 < 5% counts as pass.
- Formula recall: exact-formula match on 30 held-out formula questions.
- Compare base vs fine-tuned on the same set. Expect base model to fail catastrophically on Lacey/IRB-specific norms — that's your improvement signal.

---

## 4. Flow B — VLM Fine-Tuning (start data work in parallel)

### B1. Image–label pairs
For your CAD pipeline the most valuable task is **drawing → structured parameter JSON**:

```json
{
  "structure_type": "multi-cell box culvert",
  "clear_span_mm": 3000,
  "clear_height_mm": 2400,
  "wall_thickness_mm": 300,
  "slab_thickness_mm": 250,
  "n_cells": 3,
  "units": "mm",
  "source_view": "GAD - longitudinal section"
}
```

Sources: (a) your own CAD app exports rendered to PNG with known parameters — **free, unlimited, perfectly labeled synthetic data**; (b) real GAD PDFs/DWGs you can access, labeled manually in Label Studio or a simple Streamlit labeler; (c) augmentation: rotate ±5°, vary line weights, add scan noise, change scales — VLMs must be robust to drawing style.

Target: start with 300–500 pairs, aim for 1000. Below ~200 pairs, LoRA on a 2B VLM won't reliably generalize.

### B2. Training config
Same QLoRA skeleton as Flow A but:
- **Freeze vision tower** (`finetune_vision_layers=False`) — your labels are language-shaped; ViT already sees lines fine.
- Train language layers + attention modules only, r=16, α=16.
- Images ≤ 1024px on long edge; batch 1–2 with grad accum 8.
- Enforce **JSON-only outputs** in the chat template during training so inference is parseable.

### B3. Evaluation
Field-level accuracy per parameter (exact match and ±2% tolerance), plus end-to-end: extracted params → fed into PlaneGCS/constraint graph → does reconstructed geometry validate?

---

## 5. Flow C — How the two models compose with your CAD system

```
GAD PDF/DWG ──► [VLM Qwen2.5-VL-2B] ──► params JSON ──► your constraint graph / PlaneGCS
                                                      │
User: "compute Q50 & span" ──► [text LLM] ──► method statement + computed values
                                                      │
                                              parametric drawing update
```

The VLM replaces brittle OCR+heuristics for drawing ingestion; the text model encodes the IRICEN planning methodology that currently lives only in the book. RAG over the book remains useful for exact citation-heavy answers — fine-tuning and RAG are complementary, not competing (fine-tuning teaches *behavior/method*, RAG supplies *verbatim context*).

---

## 6. Execution Timeline

| Week | Milestone |
|---|---|
| 1 | Flow A dataset built (script + seed data), first QLoRA run on your 8GB GPU |
| 2 | Eval harness, hyperparameter pass (rank 16 vs 32, epochs), export GGUF |
| 2–4 | VLM data: CAD-app synthetic exports + manual labeling of real GADs |
| 4 | First VLM QLoRA run + field-accuracy eval |
| 5+ | Flow C integration into the CAD ingestion pipeline |

## 7. Risks & Mitigations
- **Overfitting on small data** → 3 epochs max, dropout 0.05, early stop on val loss.
- **VLM numeric hallucination** → JSON-constrained decoding + post-hoc sanity checks (span > 0, thickness < span, etc.).
- **Table extraction from PDF is messy** → worked-example numbers are hand-seeded into `examples.json`; script handles expansion, not extraction.
- **8GB OOM spikes** → gradient checkpointing on, batch 1, grad accum up, seq len down to 1024 first.
