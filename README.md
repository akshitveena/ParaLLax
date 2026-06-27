# RiDAE — Reasoning-inspired Denoising Autoencoder

RiDAE learns a **geometric latent space of reasoning approach** by training an
encoder to reconstruct reasoning chains after they have been deliberately damaged.
The most valuable training signal is an empirically-verified phenomenon: **LLMs
sometimes reach the correct answer via a conceptually wrong approach** (Type B,
"wrong-approach-right-answer"), which becomes dominant on hard problems
(4% on GSM8K → 52% on OmniMath).

After training, the 64-dim bottleneck code `z` encodes conceptual approach
*independently of answer correctness*. We can then locate any LLM's reasoning in
this space, separate brittle from robust approaches, and — the headline result —
measure the geometric gap between what a model **thinks** and what it **says**
(`‖z_thinking − z_response‖`), the first direct measure of reasoning unfaithfulness.

## Architecture

```
corrupted reasoning chain
  → [ENCODER]    all-MiniLM-L6-v2 (fine-tuned, mean-pooled)   384-d
  → [BOTTLENECK] 384 → 256 → 128 → 64   = z   (interpretable code)
  → [DECODER]    64 → 128 → 256 → 384   = reconstructed embedding

Losses (all train simultaneously):
  L_reconstruct : 1 − cos(decoder(z_corrupted), encoder(original))   (TSDAE-style)
  L_MNR         : in-batch multiple-negatives ranking on (z_corr, z_orig)
  L_triplet     : push Type A and Type B apart in z-space (margin 0.5)
  L_total = L_reconstruct + L_MNR + 0.3 · L_triplet
```

Three corruptions force the learning: **approach** (replace stated framing),
**step** (delete/shuffle an interior step), **conclusion** (perturb the `\boxed{}`).

## Build note — two corrections applied

This build follows the file/code spec in `RiDAE_ClaudeCode.docx`, **with the two
corrections from the "Corrected Edition" `RiDAE_Roadmap.docx`**, which override it:

1. **No GSM8K in training.** Its Type B cases are arithmetic slips, not conceptual
   divergences. Hard datasets only (OmniMath, OlympiadBench, MATH L4+L5).
2. **MNR loss added** alongside reconstruction + triplet (batch size 64), to give
   `z` real inter-candidate similarity geometry.

## Setup

```bash
conda create -n ridae python=3.11 -y
conda activate ridae
pip install -r requirements.txt
```

## Quick start — verify the pipeline offline (no API key)

```bash
python scripts/make_synthetic_corpus.py        # tiny FAKE corpus (plumbing only)
python main/data_pipeline.py --raw data/raw/candidates_raw.jsonl --dataset synthetic
python experiments/baseline_umap.py            # control: generic-encoder UMAP + probe
python main/train.py --epochs 3 --batch_size 16 # trains on MPS/CPU in ~minutes
python main/analyse.py                          # UMAP, probes, interpolation
```

> The synthetic corpus is **not research data** — it exists only to exercise every
> component end-to-end. Replace it with real Claude-generated candidates below.

## Real data — Claude extended-thinking generation

Requires `export ANTHROPIC_API_KEY=...`. Generation uses the Batch API (~50% off).

```bash
# 1. generate candidates (thinking + response kept separately) — per dataset
python api_generation/claude_generate.py --dataset omnimath      --limit 300
python api_generation/claude_generate.py --dataset olympiadbench --limit 300
python api_generation/claude_generate.py --dataset math          --limit 400

# 2. assign process scores (LLM-as-judge) → enables Type B detection
python api_generation/score_candidates.py --in data/raw/candidates_raw.jsonl \
                                          --out data/raw/candidates_raw.jsonl

# 3. classify + build contrastive pairs
python main/data_pipeline.py --raw data/raw/candidates_raw.jsonl

# 4. baseline → train → analyse
python experiments/baseline_umap.py
python main/train.py            # batch 64, 10 epochs, early stopping
python main/analyse.py
```

Pipeline order is **generate → score → data_pipeline → train → analyse**. Scoring
must run before `data_pipeline` because Type B = answer correct *and* process
score < 10.

## Layout

```
main/data_pipeline.py        Candidate/ContrastivePair, answer extraction, A/B/D typing
main/corruption.py           approach / step / conclusion corruptions
main/ridae.py                encoder + bottleneck + decoder + reconstruction/MNR/triplet
main/train.py                training loop (by-problem split, warmup, early stopping)
main/analyse.py              UMAP, linear probe, dimension probe, interpolation
experiments/baseline_umap.py   the "run-first" control
api_generation/claude_generate.py  Claude extended-thinking generation (Batch API)
api_generation/score_candidates.py LLM-as-judge process scorer (Batch API)
scripts/make_synthetic_corpus.py   offline pipeline test corpus
```

## Status

All modules verified end-to-end on the synthetic corpus (MPS). Next real step:
generate the OmniMath pilot with Claude, then run the baseline on real candidates.
See `RiDAE_Roadmap.docx` for the full 6-phase plan (~8 weeks, target ICLR 2027).
