# Mechanistic interpretability results (M1) — paper-ready

Drop-in results for a "Where the confound lives" section. All numbers verified by
`experiments/mechinterp_m1.py` (CPU, ~1 min, frozen Phase-1a step-SDAE checkpoint, the same
1,700-solution ProcessBench corpus and 80/20 split used throughout the paper). M2 (the 7B
residual-stream ablation) is a separate A100 run, status noted at the end.

## Framing (the claim this section earns)

We show statistically that PRM Type-B evaluation is difficulty-confounded. M1 asks the mechanistic
follow-up *inside our own model*: our pipeline is per-step MiniLM embeddings **H** → Transformer
bottleneck step-codes **Z** → attention-pooled chain vector **c**. At which stage does difficulty
become linearly available, and does removing it behave like our statistical control? This turns
"the detector reads difficulty" into a statement about *where in the computation* difficulty is
represented.

## M1b — aggregation manufactures the confound

Probe R² (5-fold ridge for continuous targets; logistic accuracy for dataset) at each stage:

| stage | log_length R² | n_steps R² | latex_density R² | dataset acc |
|---|---|---|---|---|
| H (per-step MiniLM, pooled) | 0.239 | **0.072** | 0.612 | 0.628 |
| Z (bottleneck, pooled)      | 0.642 | **0.986** | 0.520 | 0.622 |
| c (attention-pooled chain)  | 0.640 | 0.982 | 0.530 | 0.620 |

**Finding.** The two confounds that carry the difficulty signal — total length and step count
(length is the dominant one, point-biserial r = +0.415 with the Type-B label) — are **nearly
invisible in the per-step encoding H (n_steps R² = 0.07) and become almost perfectly decodable
after aggregation (n_steps R² = 0.99 at Z).** This is expected and mechanistically clean: MiniLM
encodes each step independently, so it *cannot* represent total response length or step count;
those quantities only exist once the bottleneck sees the whole sequence. **The confound is not in
the features — it is created by the aggregation.** This is the mechanistic form of the paper's
pooling critique. (Latex density is already high at H, R² = 0.61 — consistent, as notation density
is a genuinely per-step surface property.)

## M1c — the confound is distributed, not a single direction (honest negative)

A/B f1_B measured on the chain vector **c** three ways (same probe, same split):

| representation of c | f1_B |
|---|---|
| raw (no control) | 0.638 |
| full 4-confound residualized (paper's protocol) | 0.523 |
| single length-direction projected out | 0.638 |

**Finding.** Projecting out the top linear length direction from **c** leaves the A/B result
unchanged (0.638 → 0.638), whereas full residualization drops it to 0.523. So the difficulty
confound in **c** is **not a single linear direction** — it is distributed across the
multi-variable confound subspace. Two consequences worth stating: (i) this *justifies* residualizing
against the full confound set rather than a targeted single-direction edit; (ii) the length
direction is approximately orthogonal to the A/B axis in **c** (removing it does not touch the
validity signal), which is why residualization removes confound without destroying signal.

## Attention analysis — the chain head reads signal, not surface

The chain head attention-pools step-codes. Per-candidate attention weights vs:

| quantity | value |
|---|---|
| corr(attention weight, step length) | +0.138 (mild) |
| mean attention on human-labelled **error** steps | 0.168 |
| mean attention on non-error steps | 0.142 |

**Finding.** The learned attention **up-weights error-containing steps (0.168 > 0.142)** with only
a weak surface-length bias (+0.14). The step-structured representation attends to the validity
signal, not merely to long steps — evidence that its advantage over pooling is substantive, not a
length artifact.

## What M1 establishes for the paper

1. The difficulty confound is **manufactured by aggregation**, not present in per-step features —
   a mechanistic account of why pooled representations are confounded (ties to E2's pooled→floor
   result).
2. The confound is **distributed**, so statistical residualization is the right tool; a single
   causal direction-edit does not replace it (honest, and it motivates M2's layer-wise search in
   the 7B).
3. The step-structured model's attention **reads the validity signal**, supporting that its lift
   over pooling is genuine.

## Caveats to state
- M1 is on our own small model (frozen 22M MiniLM + 1.3M-param SDAE); it localizes the confound in
  *our pipeline*, not in a large PRM (that is M2).
- The single-direction ablation is linear; a nonlinear edit was not attempted.

## Status of M2 (for completeness, not yet a result)
M2 — where difficulty lives inside the 7B Math-Shepherd PRM, and whether mean-ablating that
direction drops its raw f1_B toward its controlled value — is queued on the A100
(`experiments/mechinterp_m2.py`). A Kaggle T4×2 attempt failed for stack reasons (its transformers
tokenizes the step tag incompatibly → 96% of solutions skipped, gate at chance; and device_map
OOM'd on the ablation pass), so M2 runs on the validated A100 stack. **Do not cite M2 numbers until
that run completes and its baseline gate reproduces ~0.735.**

## Reproducibility
`python experiments/mechinterp_m1.py` — prints all tables above and writes
`experiments/results_mechinterp/m1_where_confound.png` (R²-by-stage figure).
