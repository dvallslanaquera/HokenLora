# CLAUDE.md — hoken-lora

Operating manual for Claude Code (and humans) working in this repo. Read this first,
then `ARCHITECTURE.md` for the design and `ROADMAP.md` for the plan.

---

## 1. What this project is

Fine-tune a **small Japanese summarizer** for the **life-insurance / finance** domain by
**distilling** a large teacher (GLM-5.2) into **Qwen2.5-3B-Instruct** with **QLoRA**, then
ship the **LoRA adapter to the Hugging Face Hub**.

- **Task:** abstractive summarization of Japanese insurance/financial disclosures
  (有価証券報告書, ディスクロージャー誌, EDINET/TDnet releases, FSA reports, 約款).
- **Distillation method:** **sequence-level KD** — the teacher writes reference summaries,
  the student is trained with **SFT on those completions**. (No logit/soft-label KD; the
  teacher is remote and closed-weights to us.)
- **Teacher + judge:** **GLM-5.2** served over **Ollama Cloud** (hosted OpenAI-compatible
  endpoint). It (a) generates synthetic training summaries and (b) acts as the LLM judge
  at eval time only.
- **Student:** `Qwen/Qwen2.5-3B-Instruct`, trained with QLoRA on a **single GTX 1060**.
- **Baselines to beat/compare:** base Qwen2.5-3B-Instruct (before/after anchor),
  **Llama-3-ELYZA-JP-8B**, **Qwen3-8B-Instruct** (both ~8B → "3B student vs two 8B references").
- **Tracking:** Weights & Biases. **Deliverable:** LoRA adapter + model card + dataset card on HF.

---

## 2. Hard constraints (do not design around wishful hardware)

The training box is a **GTX 1060 — assume 6 GB, Pascal (compute capability 6.1)**. This
dictates almost every training decision:

- **No bf16.** Pascal has no bf16 path → **use fp16** everywhere (`fp16=True, bf16=False`).
- **No Flash-Attention 2** (needs Ampere+). Use `attn_implementation="sdpa"` or `"eager"`.
- **No Unsloth.** Unsloth requires CUDA capability ≥ 7.0 → **won't run on the 1060**. Use the
  plain HF stack (transformers + peft + trl + bitsandbytes).
- **bitsandbytes on Pascal is a known gotcha.** 4-bit NF4 QLoRA *does* run on Pascal, but
  pin a known-good version and verify on a tiny run before a long one. If `paged_adamw_8bit`
  errors, fall back to `adamw_torch` (more VRAM) or `adafactor`.
- **Token budget is the real bottleneck, not just VRAM.** Training seq_len is **1024 total**
  (input passage + summary). Long filings **must be chunked** so `prompt + doc + summary ≤ 1024`
  tokens (measure with the Qwen tokenizer). Whole-document summaries at inference use a
  **map-reduce / hierarchical** pass — never assume a full 有報 fits in context.
- **Training will be slow.** Keep the SFT set modest (single-digit-thousands of pairs),
  seq short, batch 1 + grad-accum. A **free Colab/Kaggle T4** is the sanctioned escape hatch
  if the 1060 is too slow — same code, just flip `bf16=True` there.
- **Windows 11 native.** bitsandbytes ≥ 0.43 ships Windows wheels; native works, WSL2 optional.
  Avoid Triton-dependent paths.

If the card is the **3 GB** variant, QLoRA of a 3B model won't fit for training — say so and
we drop to a ~1.5B student or move training to the cloud.

---

## 3. Decisions locked

| Area | Decision |
|---|---|
| Student | `Qwen/Qwen2.5-3B-Instruct` |
| Teacher / judge | GLM-5.2 via Ollama Cloud (hosted, OpenAI-compatible) |
| Method | Sequence-level distillation → **SFT on teacher completions** |
| Quantization | 4-bit **NF4** + double-quant, fp16 compute |
| LoRA | **r=16, α=32, dropout=0.05**, targets = **all linear** (q,k,v,o,gate,up,down) |
| Train shape | seq_len 1024, batch 1 × grad-accum 16, gradient checkpointing, `paged_adamw_8bit` |
| Tracking | Weights & Biases |
| Ship | LoRA adapter → HF Hub (adapter only, not merged, not raw source docs) |

## 4. Open items

Resolved: GPU = **6 GB**; baselines = **Llama-3-ELYZA-JP-8B** + **Qwen3-8B-Instruct**.

Still pending (not blocking Phase 0):
- [ ] Exact **Ollama Cloud model tag** for GLM-5.2 + its context window.
- [ ] HF **repo id** (+ public vs private) — user will share once created.

---

## 5. Tech stack

- **Python** 3.10–3.11
- **PyTorch** CUDA build that includes `sm_61` (current stable PyTorch does)
- **transformers, peft, trl** (SFTTrainer), **bitsandbytes** (≥0.43), **accelerate, datasets**
- **wandb**, **huggingface_hub**
- Teacher client: **`ollama`** python lib or **`openai`** client pointed at the Ollama Cloud host
- Ingest: **pymupdf (fitz)**, **pdfplumber**, EDINET API (v2) client, **lxml** (XBRL)
- JP metrics: **fugashi + unidic-lite** (MeCab) for ROUGE tokenization, **rouge-score**,
  **bert-score** (Japanese/multilingual model)
- Config: YAML in `configs/` (plain dataclasses/pydantic, no Hydra needed)

Secrets live in `.env` (never commit): `OLLAMA_CLOUD_HOST`, `OLLAMA_API_KEY`, `HF_TOKEN`,
`WANDB_API_KEY`, `EDINET_API_KEY`. Ship a `.env.example`.

---

## 6. Repo layout (target)

```
hoken-lora/
├── CLAUDE.md ARCHITECTURE.md ROADMAP.md README.md
├── pyproject.toml            # or requirements.txt
├── .env.example  .gitignore
├── configs/                  # data.yaml generate.yaml train_qlora.yaml eval.yaml
├── data/
│   ├── raw/                  # downloaded filings           (gitignored)
│   ├── interim/              # extracted + cleaned text      (gitignored)
│   └── processed/            # {train,val,test}.jsonl chat   (gitignored)
├── src/hoken_lora/
│   ├── ingest/               # EDINET/TDnet/FSA fetch + PDF/XBRL → text
│   ├── chunk/                # tokenizer-aware chunking
│   ├── generate/             # GLM-5.2 teacher summaries (Ollama Cloud)
│   ├── dataset/              # build chat-format SFT jsonl + splits
│   ├── train/                # QLoRA SFT
│   ├── eval/                 # judge + ROUGE/BERTScore + baseline runners
│   └── utils/
├── scripts/                  # thin CLI wrappers over src/
├── notebooks/
└── outputs/                  # adapters, logs               (gitignored)
```

## 7. Planned entrypoints (implement as thin CLIs)

> None of these exist yet — they are the intended surface. Keep each a thin wrapper that
> reads a YAML config and calls into `src/hoken_lora/`.

```bash
python -m hoken_lora.ingest.fetch      --config configs/data.yaml       # download filings
python -m hoken_lora.ingest.extract    --config configs/data.yaml       # PDF/XBRL -> text
python -m hoken_lora.dataset.build     --config configs/generate.yaml   # chunk + teacher summaries -> jsonl
python -m hoken_lora.train.sft         --config configs/train_qlora.yaml # QLoRA SFT (W&B)
python -m hoken_lora.eval.run          --config configs/eval.yaml       # judge + metrics vs baselines
python -m hoken_lora.train.push_to_hub --config configs/train_qlora.yaml # upload adapter + card
```

---

## 8. Conventions & gotchas

- **Reproducibility:** every run is driven by a committed YAML config; log the resolved config
  and git SHA to W&B. Seed everything.
- **Chat formatting:** build training text with the **Qwen chat template** via
  `tokenizer.apply_chat_template`; mask the prompt, compute loss on the summary only
  (use TRL's completion-only / `DataCollatorForCompletionOnlyLM` equivalent).
- **Data provenance & licensing:** source filings are public but **do not redistribute raw
  documents or teacher-generated corpora blindly**. Ship a **dataset card that describes how to
  regenerate**, and upload only the **adapter** to HF.
- **Judge ≠ ground truth.** GLM-5.2 is *both* teacher and judge → biased toward its own style.
  Headline numbers come from a **small human-verified gold test set** + reference-based metrics,
  not judge-only. See `ARCHITECTURE.md §Evaluation`.
- **Verify before you commit compute:** run 5–10 training steps and one eval example end-to-end
  before any multi-hour run.
- Prefer editing configs over hardcoding; keep functions pure and testable; no secrets in code.
