# RiDAE Runbook — how to run it yourself

## 0. Every new terminal session (do this first)

```bash
cd ~/Projects/ridae
conda activate ridae
set -a; source .env; set +a      # exports ANTHROPIC_API_KEY + GROQ_API_KEY for this shell
```

Check the keys loaded:
```bash
echo "anthropic: ${ANTHROPIC_API_KEY:+set}  groq: ${GROQ_API_KEY:+set}"
```

---

## 1. Smoke test FIRST (a few cents — always do this before the real run)

Confirms the live APIs, dataset field names, think/response split, and costs are sane.

```bash
# Claude: 2 hardest OmniMath problems WITH extended thinking
python api_generation/claude_generate.py --dataset omnimath --limit 2 --hardest --extended_thinking

# Groq: 2 OmniMath problems (QwQ inline think) — appended to the same raw file
python api_generation/groq_generate.py --dataset omnimath --limit 2 --append

# Verify answers with the Claude LLM-judge (deterministic-first; only the symbolic
# residual hits the API). Needed for OmniMath/OlympiadBench symbolic answers.
python api_generation/verify_answers.py --in data/raw/candidates_raw.jsonl \
                                        --out data/raw/candidates_raw.jsonl

# Process + sanity-check (advisory validation for this tiny sample)
python main/data_pipeline.py --raw data/raw/candidates_raw.jsonl --allow_small
```

Look for: non-zero candidates, a Type-B or two, sane token/cost numbers, no errors.
The Claude step uses the Batch API and may take a few minutes (it polls until done).

---

## 2. The real corpus  (~$22 spend, $30 budget — the $30 plan)

The FIRST command has no `--append` (it overwrites the smoke-test data for a clean start);
every command after it uses `--append`.

```bash
# --- Claude (paid, ~$19) ---
python api_generation/claude_generate.py --dataset omnimath --limit 100 --hardest --extended_thinking
python api_generation/claude_generate.py --dataset math      --limit 200 --append

# --- Groq (free) ---
python api_generation/groq_generate.py   --dataset omnimath      --limit 300 --append
python api_generation/groq_generate.py   --dataset olympiadbench --limit 300 --append

# --- verify answers (Claude LLM-judge; deterministic-first, ~$1-2) ---
python api_generation/verify_answers.py   --in data/raw/candidates_raw.jsonl \
                                          --out data/raw/candidates_raw.jsonl

# --- optional: let Claude label the few 'unknown'-approach candidates ---
python api_generation/score_candidates.py --in data/raw/candidates_raw.jsonl \
                                          --out data/raw/candidates_raw.jsonl

# --- process (REAL validation — no --allow_small; it HARD-STOPS if the corpus is off) ---
python main/data_pipeline.py --raw data/raw/candidates_raw.jsonl
```

Watch the cost line each generator prints. Expected ~2,700 candidates, ~568 Type B.
If `data_pipeline` stops on a validation FAIL, fix generation before training (that's by design).

---

## 3. Train + analyse (free, on your M3)

```bash
python experiments/baseline_umap.py            # control: generic encoder
python main/train.py                            # v1 (uniform sampling)
python main/compute_hardness.py                 # score hard negatives
python main/train.py                            # v1' (hard-negative mining now ACTIVE)
python main/analyse.py --version v1             # all 8 figures -> outputs/
```

Results land in `outputs/` (figures + `analysis_results.json`). The headline is
`outputs/fig7_thinking_response_gap.png` — the geometric thinking-vs-response gap,
strongest claim on Claude ET data.

---

## 4. Iterate (Roadmap phases)

- Re-run §3 labelling the version (`--version v2`, `v3`) to grow the Fig-8 ablation.
- If the baseline UMAP shows weak A/B separation, try a different encoder:
  `python main/train.py --encoder tbs17/MathBERT` (math vocab) or
  `--encoder nomic-ai/nomic-embed-text-v1.5` (long context).

---

## Cost guardrails

- Smoke test: a few cents.
- Real Claude run: ~$19 (ET OmniMath ~$13.68 + MATH ~$5.76). Groq: free.
- Keep ~$7.76 buffer for retries. To add free OmniMath signal, increase the **Groq**
  `--limit` (cheap), not the Claude ET `--limit` (expensive).

## Pipeline order (always)

generate (claude + groq) → verify_answers → [score_candidates] → data_pipeline →
baseline → train → compute_hardness → train → analyse
