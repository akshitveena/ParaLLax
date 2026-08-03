# Phase 2 hardening — cluster runbook (2a / 2b / 2c)

Runs the three GPU experiments from the Phase-2 spec. 2d (API) and 2e (CPU, already done) stay
local. Everything here is **offline-safe**: weights are staged once on the login node, and every
job exports `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` because compute nodes usually have no
internet.

## 0. Ship the repo + corpus

The corpus is small; ship it with the code.

```bash
rsync -av --exclude '.git' --exclude 'checkpoints' \
  ~/Projects/ridae/ user@cluster:/scratch/$USER/ridae/
```

`data/step_cache.pt` (1,700 records with `steps_text`) and `data/processed_pb/candidates.jsonl`
are both **required** — `prep.sh` verifies them.

## 1. Once, on the LOGIN node

```bash
export RIDAE_ROOT=/scratch/$USER/ridae
export HF_HOME=/scratch/$USER/hf        # shared storage — NOT $HOME if it has a quota
bash $RIDAE_ROOT/cluster/prep.sh
```

Stages `all-MiniLM-L6-v2` (small) and `math-shepherd-mistral-7b-prm` (**~14–28 GB — this is the
slow step**), creates the `ridae` env, and validates the corpus.

Edit the commented `--partition` / `--account` lines at the top of each `.sbatch` to match your
cluster before submitting.

## 2. Submit

```bash
cd $RIDAE_ROOT
sbatch cluster/job_2a.sbatch   # 5-seed e2e (array 0-4)          -> Phase 2a
sbatch cluster/job_2b.sbatch   # 5-seed recon-only e2e (array)   -> Phase 2b
sbatch cluster/job_2c.sbatch   # external PRM scoring            -> Phase 2c
```

2a and 2b are **array jobs**: all 5 seeds run in parallel, so wall-clock is one seed (~1 h), not
five. They are independent and can be submitted together. 2c is independent of both.

## 3. Collect

```bash
python experiments/eval_e2e.py --aggregate --out experiments/results_e2e
```

Prints the 2a/2b table (mean ± std across seeds) and writes `e2e_summary.csv`. Pull results back:

```bash
rsync -av user@cluster:/scratch/$USER/ridae/experiments/results_e2e/ ./experiments/results_e2e/
rsync -av user@cluster:/scratch/$USER/ridae/experiments/results_prm/ ./experiments/results_prm/
```

Then re-derive every 2c statistic locally, free, without the GPU:

```bash
python experiments/prm_external.py analyze --scores experiments/results_prm/scores.json
```

## What each job produces

| job | produces | feeds |
|---|---|---|
| 2a | 5 e2e checkpoints + `F_e2e_seed*.json` | Table 1 e2e row gains error bars |
| 2b | 5 recon-only checkpoints + `R_e2e_seed*.json` | removes the three "frozen-encoder only" caveats |
| 2c | `results_prm/scores.json` + raw-vs-controlled table | new Results paragraph + Table 1 row |

## Two things that will bite you

**2b's model selection.** `--heads none` removes the chain loss, so the trainer *must not* select
on `val chain_f1` — it would be selecting on noise. `--heads none` automatically switches
selection to **val L_denoise**. Don't override it.

**2c's sanity gate.** `analyze` first checks the PRM's per-step scores against ProcessBench's
human step labels. If step-level AUC is at chance, the scoring format (tag id / prompt layout /
token alignment) is wrong and the job **exits non-zero** rather than reporting a confident wrong
number. If it fires, fix the format — do not pass `--force` to paper over it.

## Model choice note (do not "fall back")

The spec offers `Qwen/Qwen2.5-Math-PRM-7B` as a fallback. **Don't use it here.** ProcessBench is
Qwen's own benchmark, so a Qwen math-PRM plausibly saw ProcessBench-adjacent data — inflating its
*raw* score for reasons unrelated to confounds and corrupting exactly the raw-vs-controlled
contrast 2c exists to measure. Math-Shepherd is a different lab and predates ProcessBench. The
fallback was hardware-motivated; on an A100 the hardware reason disappears.

## Appendix A.4

Every script logs wall-clock and GPU name into its JSON output; `--aggregate` totals them per
variant.
