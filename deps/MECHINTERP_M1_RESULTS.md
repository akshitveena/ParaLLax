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

## M2 — where difficulty lives inside the 7B PRM, and can it be removed?

`experiments/mechinterp_m2.py`, bf16 on A100. Validated stack (candidate tokens [648,387], step
tag 12902, 12/1700 skipped) and the **baseline reproduces the paper's 2c numbers exactly**
(raw f1_B 0.429, controlled 0.075, step-label gate 0.735) — so the mechanistic numbers are
trustworthy. Method: cache residual-stream activations every 4th layer at each step tag over the
1,700 solutions; linear-probe each layer for log length and step count; mean-ablate the peak length
direction (mean-ablation, not zero-ablation) and re-score.

**Probe — difficulty is strongly and linearly represented in the PRM's residual stream, peaking
mid-network.** log_length probe R² by layer:

| layer | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| R² | −0.66 | 0.69 | 0.91 | **0.92** | 0.92 | 0.92 | 0.91 | 0.91 | 0.84 |

Length/step-count are almost perfectly decodable (R² ≈ 0.92) across layers 8–28. A SOTA process
reward model internally encodes response length/difficulty richly — the mechanistic counterpart of
the statistical confound. (Figure: R²-by-layer, `experiments/results_mechinterp/m2_where.png`.)

**Ablation — the confound is distributed, not a single direction (honest negative).**

| | raw f1_B | controlled f1_B | step-label gate |
|---|---|---|---|
| baseline | 0.429 | 0.075 | 0.735 |
| length-direction mean-ablated at layer 12 | 0.433 | 0.076 | 0.732 |
| Δ | +0.005 | +0.001 | −0.003 |

Mean-ablating the single peak length direction does **not** move the raw score toward its
controlled value (Δ ≈ 0), and the step-label gate is unchanged (no competence removed). Difficulty
is decodable from a high-R² but multi-dimensional, redundant subspace, and ~20 downstream layers
can reconstruct it, so a single-direction single-layer edit cannot remove its influence on the
score. **This agrees with M1c**: in both the 22M encoder (M1) and the 7B PRM (M2), the difficulty
confound is distributed — statistical residualization cannot be replaced by a targeted causal edit.

**Net.** M2 is a clean *positive* (difficulty is linearly, richly represented in the PRM, R² 0.92,
localized to mid-layers) plus a coherent *negative* (not a single removable direction), the latter
consistent across both models. It is the plan's "distributed" outcome, reported as pre-committed.
The overall M1+M2 story: the confound is *created by aggregation* in our model and *richly
represented* in the 7B, and in neither case is it a single direction — which is precisely why the
paper controls for it statistically.

## M3 — resolving M2's null: subspace ablation, self-repair (Hydra), head attribution

`experiments/mechinterp_m3.py` (A100, bf16, validated stack; baseline reproduces raw 0.429 / ctl
0.075 / gate 0.735). Motivated by M2's single-direction null.

**M3a — difficulty is a ~16-dim subspace, but it is ENTANGLED with the PRM's competence.**
Mean-ablating the top-k PLS difficulty directions at the peak layer (12); raw f1_B *and the
step-error gate* vs k:

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| raw f1_B | 0.417 | 0.462 | 0.447 | 0.365 | **0.127** | 0.000 | 0.000 |
| step gate | 0.713 | 0.703 | 0.704 | 0.661 | **0.637** | 0.566 | 0.484 |

(baseline: raw 0.429, controlled 0.075, gate 0.735.) One direction barely moves the score;
~16 directions drive raw f1_B down to the controlled floor (0.127 ≈ 0.075) — so difficulty is a
**~16-dimensional subspace**, not one direction (M2). **But the step-error gate falls in tandem**
(0.735 → 0.637 at k=16, → 0.48 at k=64): you cannot remove the difficulty subspace without
degrading the PRM's genuine step-scoring competence. This is the plan's "entangled with capability"
outcome — difficulty and competence share representational structure in Math-Shepherd; the confound
is **not a cleanly separable direction**. (This is the *opposite* of the small model, where the
length direction in c was ~orthogonal to the A/B axis — an interesting model-scale contrast.)

**M3b — self-repair (Hydra effect): the network reconstructs difficulty downstream.** With a 64-dim
ablation live at layer 12, downstream length-probe R² barely drops (0.917→0.889 at L16; 0.836→0.805
at L32). Difficulty is re-encoded downstream after removal — the Hydra effect (McGrath et al. 2023).
A mechanistic reason single-point ablation of the score's difficulty dependence is hard.

**M3c — no single culprit head.** Per-head write onto the length direction at layer 12 is spread
thin (top head 0.010, tailing off across heads 19, 17, 26, 21, 31, …). Difficulty is written by many
heads, none dominant — consistent with the subspace and self-repair findings.

**Combined M2+M3 — why statistical control is *necessary*, not just convenient.** In the 7B PRM,
difficulty is (i) richly linearly represented (R² 0.92, M2), (ii) a ~16-dim subspace, not one
direction (M3a), (iii) written by many heads with no single culprit (M3c), (iv) actively
reconstructed downstream when removed — the Hydra effect (M3b), and (v) **entangled with the
verifier's step-scoring competence** — removing it degrades the gate (M3a). These five together
make the point sharper than a clean ablation would: the difficulty confound in a SOTA PRM is
distributed, self-repairing, and inseparable from capability, so it **cannot be excised by a
targeted internal edit**. Confound control therefore has to be done statistically at evaluation
time — the paper's protocol is not merely a convenience but a mechanistic necessity.

## Reproducibility
`python experiments/mechinterp_m1.py` (M1, CPU), `mechinterp_m2.py` and `mechinterp_m3.py` (M2/M3,
A100). Figures in `experiments/results_mechinterp/` (`m1_where_confound.png`, `m2_where.png`,
`m3_analysis.png`).
