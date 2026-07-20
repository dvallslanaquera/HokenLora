# ROADMAP.md — hoken-lora

Phased plan to ship a QLoRA adapter that distills GLM-5.2 into Qwen2.5-3B for Japanese
insurance/finance summarization. Each phase has **deliverables** and an **acceptance gate**;
don't start the next phase until the gate is green. See `ARCHITECTURE.md` for the how.

Legend: 🎯 deliverable · ✅ acceptance gate · ⚠️ risk to watch

---

## Phase 0 — Setup & confirmation  *(fast)*
- Remaining open items (`CLAUDE.md §4`, non-blocking): exact GLM-5.2 Ollama Cloud tag +
  context window; HF repo id + public/private (user will share).
- Repo scaffolding: `src/hoken_lora/` packages, `configs/`, `.env.example`, `.gitignore`
  (ignore `data/`, `outputs/`, `.env`), `pyproject.toml`/`requirements.txt`, `git init`.
- Create accounts/keys wired via `.env`: Ollama Cloud, HF, W&B, EDINET.
- 🎯 Runnable env; `import torch; torch.cuda.is_available()` True; a "hello" call to
  GLM-5.2 on Ollama Cloud returns text; W&B logs a dummy run.
- ✅ **Smoke test passes:** load `Qwen2.5-3B-Instruct` in **4-bit NF4** and run **10 SFT steps**
  on 20 dummy examples on the 1060 without OOM. ⚠️ *This validates the Pascal/bitsandbytes path
  before any real work — if it fails here, escalate to the Colab/Kaggle T4 plan.*

## Phase 1 — Data ingestion
- Implement EDINET v2 fetch + TDnet/FSA/insurer PDF collection with a manifest.
- PDF/XBRL → clean text; boilerplate stripping; store `data/interim/`.
- Pick an initial scope: e.g., N companies × key sections (事業等のリスク, 経営成績,
  保険 約款 抜粋, ディスクロージャー誌 要点) → a few hundred source docs.
- 🎯 `data/interim/` populated + manifest with provenance/license notes.
- ✅ Spot-check 10 extracted docs: text is clean, Japanese intact, tables not garbled.
- ⚠️ Table-heavy PDFs; scanned/image PDFs (may need OCR or exclusion); site terms of use.

## Phase 2 — Synthetic dataset (teacher distillation)
- Tokenizer-aware chunking (≈600–800 input tokens).
- Generate GLM-5.2 summaries via Ollama Cloud with the versioned rubric prompt; cache by hash;
  apply quality gates (length, language, faithfulness guard, dedupe).
- Build chat-format JSONL; **split by document**; hand-verify/edit the **gold test set** (~50–100).
- 🎯 `data/processed/{train,val,test}.jsonl` + `gold.jsonl`; target **~2–5k** train pairs to start.
- ✅ Manual read of 20 random pairs: summaries faithful, concise, no leaked chunks across splits.
- ⚠️ Teacher hallucinating numbers; cost/rate limits; near-duplicate chunks inflating the set.

## Phase 3 — First QLoRA training run
- Implement `train.sft` with the **Balanced** config (`configs/train_qlora.yaml`); loss on
  assistant turn only; W&B logging; checkpoint + adapter save.
- Run 1–2 epochs; watch loss, GPU mem, tokens/sec.
- 🎯 A saved LoRA adapter under `outputs/` + a complete W&B run.
- ✅ Train loss decreases smoothly; a few generated summaries look **coherently better than the
  base 3B** on held-out val (eyeball).
- ⚠️ OOM (drop seq_len/grad-accum first); `paged_adamw_8bit` failing on Pascal (fall back);
  overfitting on tiny data (fewer epochs / more data).

## Phase 4 — Evaluation harness
- Implement `eval.run`: ROUGE-JA (MeCab), BERTScore, length ratio; GLM-5.2 pairwise judge with
  **position-swap**; run **all baselines** (base 3B, ELYZA, Qwen3) through the same map-reduce
  inference wrapper.
- Report on **gold set** (headline) + full test set; log comparison tables to W&B.
- 🎯 A metrics table: fine-tuned vs base-3B vs ELYZA vs Qwen3 across all signals.
- ✅ **Primary criterion:** fine-tuned **beats base 3B** on gold ROUGE-L, BERTScore, and judge
  win-rate; gap to 8B baselines is materially reduced.
- ⚠️ Judge/teacher self-bias (gold set + reference metrics are the guard); JP tokenization for
  ROUGE (must be MeCab, not whitespace); position bias in the judge.

## Phase 5 — Iteration & small sweeps
- Using W&B, sweep the levers that matter on the 1060: **rank {8,16,32}**, **target modules
  (attention-only vs all-linear)**, **epochs**, **learning rate**, **data size/quality**.
- Optional data upgrades: better teacher prompts, harder faithfulness filtering, more sources.
- 🎯 A best-config chosen from evidence, not vibes.
- ✅ Best config beats the Phase-4 baseline run on the gold set; results are reproducible from
  its committed YAML.
- ⚠️ Chasing judge score while gold/reference metrics stall — trust the gold set.

## Phase 6 — Release
- Finalize the best adapter; write the **model card** (config, data provenance, eval table,
  intended use, **limitations**: JP summarization only, may err on numbers, not advice) and a
  **dataset card** (regeneration recipe, no raw filings).
- Push **adapter only** to the HF Hub; verify it loads on top of `Qwen2.5-3B-Instruct` from a
  clean environment.
- 🎯 Public (or private) HF adapter repo + cards + a short README with the eval table.
- ✅ Fresh-env load + one inference reproduces a sample from the eval.
- ⚠️ Accidentally publishing secrets, raw copyrighted docs, or a merged model instead of the adapter.

---

## Cross-cutting risks & mitigations

| Risk | Mitigation |
|---|---|
| Pascal can't run 4-bit QLoRA / bnb version issues | Phase-0 smoke test; pin bnb; fp16; T4 fallback |
| GTX 1060 too slow for real epochs | Small dataset, seq_len 1024, batch1×GA; Colab/Kaggle T4 escape hatch |
| Long filings exceed 1024 tokens | Chunk for training; map-reduce summarization at inference |
| Teacher = judge self-bias | Human-verified gold set + reference metrics + optional 2nd judge |
| Data leakage across splits | Split by source document id |
| Copyright/ToS on filings | Adapter-only release; regeneration-recipe dataset card; provenance manifest |
| Teacher cost/rate limits | Hash caching, batching, retry/backoff, capped dataset size |

## Definition of done
A HF-hosted LoRA adapter over Qwen2.5-3B-Instruct that, on a human-verified Japanese
insurance/finance gold set, **beats the base 3B and closes most of the gap to the 8B baselines**
on ROUGE-L, BERTScore, and GLM-5.2 judge win-rate — with every stage reproducible from committed
configs and tracked in W&B.
