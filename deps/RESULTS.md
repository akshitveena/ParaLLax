# RiDAE — Results Ledger

Persistent record of benchmarks (numbers were otherwise only in stdout/tmp logs).
All runs in conda env `ridae` (py3.11); prefix `HF_HUB_OFFLINE=1` when no network.
Rigorous numbers are **confound-controlled + leakage-free** unless noted.

---

## Phase 0 — Phenomenon + judge (validated)

- **Wrong-approach-right-answer scales with difficulty** (ProcessBench, human labels, answer-correct subset):
  GSM8K 4% · MATH 19% · OlympiadBench 32% · OmniMath 52%.
- **Mechanism judge vs ProcessBench humans:** Cohen's κ ≈ 0.60, precision 0.84, recall 0.73 (F1 ≈ 0.78).
  Disagreements are mostly *non-causal / self-corrected* errors — our definition is a stricter causal subset.
- PRM800K checked and **ruled out** for the phenomenon (0% — its construction curates good paths).

## Phase 1 — Representation (the core result)

Training corpus: ProcessBench, 1,700 answer-correct solutions (A=1,179 / B=521), human per-step error labels.
Metric: **confound-controlled, leakage-free held-out A/B f1_B** (length/latex/#steps/dataset residualized).

| model | f1_B |
|---|---|
| pooled AE (mean-pool → 64-d)              | **0.286**  (≈ raw-SBERT ceiling; pooling destroys validity) |
| step-structured SDAE+PRM, **frozen** encoder | **0.436** |
| step-structured SDAE+PRM, **e2e** (unfrozen + text corruption) | **0.576** |

Reproduce: `main/train_sdae.py` (frozen) / `main/train_sdae_e2e.py` (e2e, ~166 min CPU) →
`main/diagnose_sdae_e2e.py`.

### Phase 1a — Multi-seed error bars on the core ablation (#1)

5 seeds (each varies init + data order + corruption + 80/20 split together), same
confound-controlled leakage-free probe. `experiments/multiseed_ablation.py`.

| model | f1_B (mean ± std, n=5) |
|---|---|
| raw-SBERT (mean-pooled)      | **0.291 ± 0.045** |
| step-structured SDAE, frozen | **0.497 ± 0.047** |

Gap +0.205 ± 0.062; all 5 seeds positive, CIs non-overlapping. **Honest correction:** the
single-run frozen number in the table above (0.436) was a low draw — the multi-seed mean is
0.497 ± 0.047. Step-structure over raw pooling is the robust, seed-stable effect.

### Phase 1b — Corruption-composition ablation (#2): what carries the signal?

Train the frozen step-SDAE (denoise + PRM + chain) under 5 corruption regimes × 3 seeds; the
denoising *target* is always the clean step sequence, only the corrupted *input* changes.
`experiments/corruption_composition.py`.

| training corruption | f1_B (mean ± std) | Δ vs mask |
|---|---|---|
| **none** (AE + heads, zero corruption) | 0.527 ± 0.037 | −0.015 |
| mask (the default)                     | 0.541 ± 0.052 |  +0.000 |
| shuffle                                | 0.515 ± 0.056 | −0.026 |
| noise                                  | 0.500 ± 0.040 | −0.042 |
| all (mask+shuffle+noise)               | 0.530 ± 0.043 | −0.012 |

**All five regimes are statistically indistinguishable** (every mean in 0.50–0.54, every std
~0.04–0.06, CIs fully overlapping). Critically, **`none` ≈ `mask`**: removing the denoising
corruption entirely costs nothing. → The Type-B signal is **not** produced by the denoising
objective; it is carried by the **step-structured supervised heads** (per-step PRM + chain A/B).
This corroborates the toxicology finding and honestly reframes the contribution as a
*step-structured supervised representation*, not a denoising autoencoder — the "denoising" is
architecturally present but empirically inert for validity discrimination.

### Phase 1c — Content-preservation probe (#1c): is the latent empty, or just unused?

Phase 1b showed the *classifier's* f1_B is flat across corruption regimes — the denoising
objective is inert **for classification**. That does not prove the latent `z` is empty of
content: a flat score only proves the classifier doesn't *need* reconstruction. Phase 1c checks
`z` **directly**, with the two supervised heads switched OFF.

- **Variant R** (recon-only, `--heads none`, L_denoise alone): trained fresh, 5 seeds.
- **Variant F** (full, denoise+PRM+chain): the Phase-1a frozen checkpoints — *same* embeddings,
  *same* per-seed splits as R, so R-vs-F isolates one variable (heads on vs off). (We do **not**
  use `sdae_e2e_best.pt`: it bundles a fine-tuned MiniLM → different embedding distribution →
  not comparable. This matched-frozen F is stricter than the spec's n=1 fallback.)
- **Probe 1 (retrieval):** `r_i =` decode(encode(corrupted_i)); target `t_i =` clean step
  embeddings (the L_denoise target — a frozen MiniLM quantity, so co-adaptation collapse is
  impossible by construction). Candidate-level mean-pool → N=340 val, chance R@1 ≈ 0.003.
- **Probe 2 (specificity):** mean pairwise cosine among `r_i` (collapse detector); targets as ceiling.

`experiments/content_probe.py` (5 seeds).

| metric | R (recon-only) | F (full, frozen) | chance |
|---|---|---|---|
| Recall@1      | **0.996 ± 0.004** | 0.068 ± 0.079 | 0.003 |
| Recall@10     | **0.998 ± 0.002** | 0.275 ± 0.219 | 0.03 |
| Median rank   | **1.0 ± 0.0**     | 45.6 ± 34.8   | ~170 |
| MRR           | **0.997 ± 0.003** | 0.139 ± 0.120 | ~0.02 |
| Decode spec ↓ | 0.299 ± 0.013     | 0.697 ± 0.132 | (target ceiling 0.229) |
| Val L_denoise ↓ | 0.194 ± 0.002   | 0.529 ± 0.044 | — |

**The latent is FULL, not empty — and the heads actively evict content.** Recon-only recovers
its own clean target essentially perfectly (R@1 = 0.996, median rank 1) with near-ceiling
specificity (decode spec 0.299 vs target 0.229 — almost no collapse). The full model retains
little content (R@1 = 0.068, median rank 46), its decodes partially collapse (0.697), and its
reconstruction is 2.7× worse (L_denoise 0.529 vs 0.194). So Phase 1b's "inert" was a fact about
the *classifier*, not the *memory*: `z` holds the content, the A/B classifier simply doesn't use
it, **and training the PRM/chain heads actively trades that content away.** This is the
detection↔generation tension measured, not argued — content-invariance (what validity detection
rewards) and content-preservation (what generation needs) pull the one bottleneck in opposite
directions. Sanity: L_denoise fell 0.62→0.19 every R seed; targets non-degenerate (spec 0.229);
`r_i` std non-zero.

### Phase 1d — Validity in the content-preserving latent (#1d): is it there without supervision?

Phase 1c: content survives with the validity heads off (Variant R, Recall@1 0.996). Phase 1d
asks the other half — is any A/B *validity* signal still inside that content-preserving `z`, even
though nothing ever trained it there? Variant R never had a classifier, so this is a held-out
linear probe on frozen `z`, the identical confound-controlled protocol behind F's 0.497 (clean
candidates in, no corruption, no grad; length/latex/#steps/dataset residualized; fit on train,
eval on held-out val). `experiments/phase1d_validity_probe.py`, 5 seeds.

| representation | f1_B | AUC | accuracy |
|---|---|---|---|
| pooled raw-SBERT (floor, Phase 1a) | 0.291 ± 0.045 | — | — |
| **R (recon-only) latent**          | **0.416 ± 0.032** | **0.693 ± 0.030** | 0.718 ± 0.016 |
| F (full) latent                    | 0.497 ± 0.047 | 0.673 ± 0.031 | 0.739 ± 0.020 |

(F reproduces its Phase-1a 0.497 ± 0.047 exactly — the probe protocol is identical.)

**Outcome (a): validity is largely *latent*, not manufactured by supervision.** R's z — trained
only to reconstruct, never shown an A/B label — probes to f1_B 0.416, **+0.125 above the pooled
floor** and only 0.081 below the fully-supervised F. (Both gaps are confirmed by the Phase-2e
paired bootstrap: R−pooled 95% CI [+0.040, +0.214]; F−R [+0.007, +0.157]. An earlier draft of
this line justified the gap with "≈2.8σ of the floor" using seed-variance — that was an
overclaim; held-out *sampling* variance dominates and the paired test is the valid one.) Strikingly, R's **AUC
(0.693) slightly exceeds F's (0.673)**: the content-preserving latent *ranks* A/B at least as
well as the supervised one — F's f1 edge is a better-calibrated threshold, not better
separability. Read with Phase 1c, the heads pay a large content cost (Recall@1 0.996→0.068) for a
small, threshold-level classification gain (f1 0.416→0.497) and **essentially no new separability
(AUC tied)**. This is direct evidence for the factored direction — a lightweight validity
*readout* on top of a content-preserving encoder (PRM as input feature, not a competing loss
head) — rather than baking A/B into the bottleneck by gradient pressure.

## Phase W1 — Difficulty-only null model + stratified control (audit W1/W2/W12/W13)

The null model the paper was missing. `experiments/difficulty_baseline.py`, 5 seeds, CPU.
Base rate = **0.306** Type-B (n=1,700); AUPRC chance = 0.306.

| representation | f1_B | AUC | AUPRC |
|---|---|---|---|
| **difficulty-only (4 confounds, no text)** | **0.515 ± 0.008** | 0.791 ± 0.024 | 0.610 ± 0.034 |
| raw-SBERT pooled (uncontrolled) | 0.580 ± 0.032 | 0.830 ± 0.027 | 0.740 ± 0.039 |
| frozen step-SDAE (uncontrolled)  | 0.624 ± 0.042 | 0.816 ± 0.032 | 0.708 ± 0.047 |

**W1 — the critique's missing number, and it lands.** Difficulty alone (length/latex/#steps/
dataset, no text at all) reaches f1_B **0.515** — most of the way to what *uncontrolled* detectors
score (0.58–0.62). So the bulk of a naive detector's apparent Type-B skill is difficulty. Every
uncontrolled number must be read against 0.515, not raw-SBERT's 0.291. (This is why the
confound-*controlled* numbers — frozen 0.497, e2e 0.591 — are the honest headline: they are what
survives *after* removing this.)

**W2 — stratified within-dataset, the model-free control (no residualization assumption):**

| dataset | n | %B | difficulty-only f1_B | step-SDAE f1_B |
|---|---|---|---|---|
| gsm8k        | 200 | 3.5% | 0.000 | 0.000 (only ~7 pos — uninformative) |
| math         | 500 | 18.8% | 0.087 | **0.448** |
| olympiadbench| 500 | 32.2% | 0.331 | **0.662** |
| omnimath     | 500 | 51.8% | 0.717 | **0.802** |

**This is the result that rescues the representation.** *Within* a single dataset — where the
dataset difficulty-proxy is constant — difficulty-only largely collapses (math 0.087, olympiad
0.331) while the step-SDAE retains large signal (0.448, 0.662, 0.802). So the step-structured
representation is **not** merely reading difficulty: its signal survives where difficulty range is
narrow. Crucially this is a *model-free* confirmation of the linear residualization (audit W2's
concern that linear control misses nonlinear difficulty dependence) — two independent methods,
same conclusion. Note most of difficulty-only's aggregate 0.515 comes from the *dataset* variable
(the coarse 4→52% base-rate proxy); strip it and difficulty-within-dataset is weak.

**W13 — confound justification (point-biserial r with the Type-B label):** length **+0.415**
(dominant), n_steps +0.115 (weak), latex_density −0.010 (inert). Length is the real confound;
latex is kept for completeness but carries ~no marginal signal — state this rather than asserting
all four matter equally.

## Phase W3 — Does the judge read difficulty? (the sharpest attack)

If the LLM judge (κ=0.60) calls solutions flawed *more when the problem is hard*, the judge-labelled
splits are difficulty-confounded and a detector recovering them is partly recovering difficulty —
circular. Tested directly on the 99 problems with BOTH human and judge labels
(`data/processbench_calib.jsonl`). `experiments/judge_confound_check.py`.

| labeller | confound→B AUC | std length coef |
|---|---|---|
| human | 0.665 | 0.691 |
| judge | 0.700 | 0.672 |

judge − human confound→AUC gap = **+0.034, 95% CI [−0.090, +0.129]** (bootstrap, spans 0);
labeller agreement 0.80.

**The attack is answered: the judge is NOT more difficulty-driven than humans.** Both labellers show
the *same* moderate difficulty-dependence (AUC ~0.67–0.70, near-identical length coefficients) —
which is expected and correct, because Type-B genuinely rises with difficulty (4%→52% across
splits). The point is that the judge adds **no extra** difficulty bias beyond what human labels
already carry; the CI on the gap spans zero. So judge labels are clean *on the difficulty axis* and
a detector recovering them is not circularly recovering judge-injected difficulty. Caveat: n=99,
wide CI (±0.11) — reported as "no detectable contamination," not "provably none."

## Phase E2 — Encoder scale ladder (audit E2/W7): does it survive at scale?

Core ablation (pooled vs step-structured, confound-controlled f1_B, 5 seeds) at three encoder
sizes. `experiments/encoder_ladder.py`. Kills "22M-on-a-laptop".

| encoder | dim | pooled (controlled) | step-SDAE (controlled) | lift |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (22M)  | 384  | 0.291 ± 0.045¹ | 0.497 ± 0.047¹ | +0.206 |
| all-mpnet-base-v2 (110M)| 768  | 0.270 ± 0.037 | 0.523 ± 0.043 | +0.253 |
| gte-large (335M)        | 1024 | **0.011 ± 0.009** | 0.511 ± 0.049 | +0.500 |

¹MiniLM row = the Phase-1a 5-seed numbers (the ladder's 1-seed MiniLM check reproduced them:
pooled 0.288, step 0.486).

**The clean-win outcome: the lift is not a small-encoder artifact.** The step-structured controlled
f1_B is flat and healthy (0.497 → 0.523 → 0.511) from 22M to 335M, and pooled-controlled never
clears the difficulty floor at any scale (0.291, 0.270, 0.011). Scaling the encoder 15× does not let
pooling recover the validity axis. "22M-on-a-laptop" is answered: the step-structure > pooling result
holds across encoder sizes, and the lift *grows* (+0.21 → +0.25 → +0.50).

**Honest correction (a predicted number that the data refuted).** An earlier draft of this row
claimed the gte-large pooled collapse to 0.011 showed "stronger encoders are *more* confounded — the
control erases a strong signal." A direct check refuted that: gte-large pooled is a **weak Type-B
detector even UNCONTROLLED** (f1_B = 0.305, barely above the 0.306 base rate) — lower, not higher,
than MiniLM pooled uncontrolled (0.580). So the 0.011 is not "control erasing strong signal"; gte's
mean-pooled vector barely carries Type-B signal in the first place, and what little it has is
confound. The defensible cross-scale claims are therefore: (a) pooled-controlled ≤ floor at every
scale, (b) step-structured ≈ 0.50 at every scale, (c) the gap survives scaling. Do **not** claim
"more confounded with scale."

**Not a broken embedding (the gap is real):** on the *same* gte-large embeddings the step-SDAE
reaches 0.511, so the embeddings do carry Type-B signal — it is specifically *mean-pooling* gte that
fails to expose it (gte is a retrieval encoder; its mean-pool may simply be ill-suited to this
classification, which itself is a fair point about pooled baselines). The method's scale-robustness
(0.50 at every size) is the load-bearing E2 result; the pooled magnitudes are the foil.

## Phase 2a / 2b — e2e multi-seed + content/validity under the unfrozen encoder (DONE)

A100, 5 seeds each, fully unfrozen encoder + heads. `cluster/run_phase2.sh` →
`experiments/eval_e2e.py`. Harness first validated by reproducing the published single-run
**f1_B = 0.5761** on the old checkpoint (probe protocol identical to `diagnose_sdae_e2e.py`).

### 2a — e2e headline, now with error bars

| | validity f1_B | AUC | accuracy |
|---|---|---|---|
| e2e F, single run (old Table 1) | 0.576 | — | — |
| **e2e F, 5 seeds**              | **0.591 ± 0.050** | 0.741 ± 0.019 | 0.781 ± 0.025 |

**The headline holds.** The single-run 0.576 sits inside the 5-seed band; the mean is if anything
slightly higher (0.591). No 0.436→0.497-style correction needed this time. Table 1's e2e row can
now carry ±0.050.

### 2b — content/validity under the unfrozen encoder (removes the frozen-only caveat)

Retrieval on the held-out split (N=340), R-e2e vs F-e2e, 5 seeds each:

| metric | R-e2e (recon-only) | F-e2e (full) | frozen R / F (Phase 1c/1d) |
|---|---|---|---|
| Recall@1     | **0.981 ± 0.006** | 0.007 ± 0.005 | 0.996 / 0.068 |
| median rank  | **1.0 ± 0.0**     | 110.5 ± 19.0  | 1.0 / 45.6 |
| decode spec ↓| 0.370 ± 0.005     | 0.882 ± 0.057 | 0.299 / 0.697 |
| val L_denoise↓| 0.247 ± 0.003    | 0.588 ± 0.014 | 0.194 / 0.529 |
| validity f1_B | 0.432 ± 0.025    | 0.591 ± 0.050 | 0.416 / 0.497 |
| validity AUC  | 0.692 ± 0.023    | 0.741 ± 0.019 | 0.693 / 0.673 |

**The content result replicates and strengthens.** Recon-only preserves content almost perfectly
(R@1 0.981, median rank 1); the full model evicts it *harder* than in the frozen regime (R@1
0.007 vs frozen 0.068; median rank 110 vs 46; decode spec 0.882 = near-total collapse). Giving the
encoder freedom to specialise makes the eviction worse, not better. The frozen-encoder caveat on
Phase 1c is removed — the tension is not a frozen-features artifact.

**The validity result CHANGES under e2e — reported, not suppressed (spec §2b).** In the frozen
regime, recon-only z had AUC (0.693) *equal to or above* the supervised F (0.673): supervision
added no separability, only a threshold. **Under the unfrozen encoder that flips: F-e2e AUC (0.741)
now clearly exceeds R-e2e (0.692), and F's f1 lead over R widens to 0.16 (vs 0.08 frozen).**
Interpretation: a frozen MiniLM cannot reshape its features toward validity, so supervision can
only re-weight what is already there (no AUC gain); once the encoder is trainable, supervision
genuinely *reshapes* features to create validity structure — separability the recon-only objective
does not produce on its own. So the honest, sharpened claim across both regimes:

> Content and validity compete for one bottleneck. Recon-only always keeps content (R@1 ≥ 0.98).
> Supervision always evicts it, and the more plastic the encoder, the harder it evicts. What the
> encoder's plasticity *buys* for that cost is regime-dependent: nothing separability-wise when
> frozen (only a threshold), but real, measurable validity structure (AUC +0.05) when unfrozen.

This is a *stronger* two-regime result than "the pattern survives" — it quantifies what unfreezing
trades content away *for*.

## Phase 2c — External PRM under our confound protocol (highest-impact result)

An established open PRM — `peiyi9979/math-shepherd-mistral-7b-prm`, full precision (bf16, no
quantization), a different lab than ProcessBench's — scored through our *exact* confound protocol.
`experiments/prm_external.py`, 1,688/1,700 solutions (12 dropped to truncation, recorded not
padded), 2.1 min on an A100. **Sanity gate passed:** the PRM's per-step scores predict
ProcessBench *human* step labels at AUC 0.735 — the scoring format is correct, so the numbers below
are trustworthy.

| aggregation | raw f1_B | ctrl f1_B | raw AUC | ctrl AUC |
|---|---|---|---|---|
| min (weakest-step, primary) | 0.429 | **0.075** | 0.761 | 0.666 |
| mean (secondary)            | 0.494 | **0.131** | 0.794 | 0.688 |

**The difficulty-artifact critique generalises to a strong, widely-used 7B PRM — outcome 1, the
strongest possible upgrade to the paper's impact.** Residualising length / LaTeX-density / #steps /
dataset collapses Math-Shepherd's Type-B f1_B from ~0.43–0.49 to **0.075–0.131** (Δ ≈ −0.35). So a
SOTA process reward model's apparent ability to flag wrong-approach-right-answer is *almost entirely
a confound artifact*: control for difficulty and it is near-useless at the operating point.

Two honest nuances:
- **AUC deflates less than f1** (0.76→0.67, Δ−0.10). The PRM keeps a *weak* confound-independent
  ranking signal (ctrl AUC 0.666 > 0.5), but its thresholded discrimination is destroyed. Correct
  claim: strong PRMs retain faint rank signal after control; their headline f1-style numbers do not
  survive it.
- **The min-vs-mean robustness check clears cleanly** — both aggregations deflate almost identically
  (Δf1 −0.353 vs −0.363), so the effect is not an artifact of the weakest-step `min` aggregator (and
  there is no quantization here to blame either). Full-precision, both aggregators, same story.

This reframes the paper's contribution from "our small model inflates under naive evaluation" to
"**difficulty confounds inflate PRM Type-B evaluation across the board, including SOTA verifiers**" —
a claim about how the field measures verifiers, not just about our model.

## Phase 2e — Bootstrap confidence intervals

10,000 resamples on the held-out split, resampled at the PROBLEM level. *Honest note:* this
corpus has 1,700 records / 1,700 unique problems (max 1 candidate per problem), so problem-level
grouping **degenerates to record-level** — implemented correctly, but it is not a safeguard we
actually needed here. `experiments/bootstrap_ci.py`, no retraining. (0.2 min, CPU.)

| representation | f1_B | marginal 95% CI |
|---|---|---|
| pooled raw-SBERT      | 0.291 ± 0.045 | [0.189, 0.390] |
| frozen step-SDAE (F)  | 0.497 ± 0.047 | [0.402, 0.583] |
| recon-only latent (R) | 0.416 ± 0.032 | [0.317, 0.508] |

Marginal CIs are wide because the held-out split is only ~340 records (~30% B) — **sampling
variance, not seed variance, is the dominant uncertainty in this project.** But overlapping
marginal CIs are an overly conservative test. Since every representation shares the same val
split per seed, the valid test is the **paired difference** on shared resamples:

| comparison | Δ | 95% CI | verdict |
|---|---|---|---|
| frozen F − pooled | +0.206 | [+0.113, +0.302] | **significant** |
| R − pooled        | +0.125 | [+0.040, +0.214] | **significant** (marginals overlapped) |
| F − R             | +0.081 | [+0.007, +0.157] | **significant, barely** |

All three headline gaps survive. Two of them (R−pooled, F−R) would have been wrongly softened by
the naive overlapping-CI heuristic. **Claim calibration:** F's edge over R is real but *small*
(lower bound +0.007) — so Phase 1d's reading should be stated as "supervision buys a small but
reliable f1 gain **and no separability gain** (AUC 0.673 vs 0.693, F not ahead) at a large
content cost," not "supervision buys nothing."

**Known limitation this does NOT fix:** checkpoints are selected on val chain_f1 and the probe is
evaluated on that same held-out split (true in every phase, 1a–2a alike). Bootstrapping resamples
the same optimistically-selected split, so it cannot correct selection optimism. Uniform across
phases ⇒ internal comparisons stay valid, but this should be disclosed in Limitations.

## Phase 2 — PRM process-verifier benchmark

Held-out ProcessBench val (`main/eval_prm_e2e.py`):
- step-level error **AUC = 0.799**
- clean-chain "no error" acc = 0.772
- first-error exact localization = 0.277 (within ±1 = 0.436)
- ProcessBench-style F1 (harmonic) = 0.407

Honest: strong as a step-error *ranker*; modest as an *exact* localizer; small model (1.3M head on MiniLM).

## Phase 3 — Utility

**Correctness-reranking (OpenR1, R1 traces) — NEGATIVE.** `main/openr1_rerank.py` (250 mixed problems):
random 0.495 · self-consistency 0.604 · **PRM-rerank 0.476** · oracle 1.000.
Diagnostic `main/openr1_diag.py`: AUC(P_sound→correct) = 0.478 ≈ noise → **distribution shift**
(ProcessBench short solutions → long R1 `<think>` traces; PRM reads ~all R1 traces as flawed, P_sound≈0.33).
Conclusion: transfer failure, *not* a representation failure; uninformative about true utility.

**Type-B mining (the right reframe) — POSITIVE.** `main/eval_typeb_retrieval.py`, held-out val, base rate 0.276:
- precision@10 = **1.000** (3.6× base) · @20 = 0.950 · @50 = 0.740
- AUPRC = 0.711 · ROC-AUC = 0.840

Reranking *for Type B* (find wrong-approach-right-answer) keeps the phenomenon as the target and works —
a trustworthy miner (corpus-building / faithfulness flagging).

## Phase 4 — Geometry (domain-independent validity)

`main/geometry_sdae.py` → figures in `outputs_geometry/{geom_by_type,geom_by_subject,geom_by_dataset}.png`.
- z separates A/B visibly; subjects **intermix** (B-cluster is not one topic); a difficulty gradient exists
  (easy GSM8K → hard OmniMath), so the *raw* separation is partly difficulty.
- Probes: A/B AUC 0.948 (raw), subject acc 0.711 (chance 0.14), dataset acc 0.581 (chance 0.25).

**Confound-controlled domain transfer** (`main/domain_transfer_controlled.py`, difficulty/dataset/length
residualized, leave-one-subject-out):

| held-out subject | raw AUC | resid AUC |
|---|---|---|
| algebra | 0.963 | 0.896 |
| combinatorics | 0.988 | 0.954 |
| geometry | 0.946 | 0.909 |
| number_theory | 0.946 | 0.908 |
| other | 0.923 | 0.794 |
| probability | 0.955 | 0.924 |
| **MEAN** | **0.954** | **0.898** |

**Validity transfers across held-out subjects even after difficulty is removed** (only −0.056).
→ the validity axis is **domain-independent beyond difficulty**.
Caveat: these AUCs carry representation leakage (encoder saw the candidates); the rigorous *magnitude*
of validity is the 0.576 f1_B above. Two honest claims: strength = 0.576, generality = domain-independent.

## Phase 5 — Novel reasoning

Not started. Oracle-gated (needs a correctness oracle + a text decoder). Longest horizon.

---

## Key artifacts

- Data: `data/processed_pb/` (1,700 labeled), `data/step_cache.pt` (12,303 step embeddings + labels + text)
- Checkpoints: `checkpoints/checkpoints_sdae_e2e/sdae_e2e_best.pt` (the working model)
- Figures: `outputs_geometry/*.png`
- Model: `main/sdae_prm.py` (step-transformer + decoder + PRM head + chain head) — a *representation*
  (denoising AE latent) with verification/A-B *readouts*, not a bare labeler.

## Deferred / stranded

- `dose_response.py` — toxicology dose-study, built for the **old pooled** model; needs adapting to the
  step-structured SDAE to complete Phase-4 characterization.
- `train_supervised.py` — pooled SupCon control; not run (user's prior evidence that pooling fails).
