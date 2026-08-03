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

## Phase 2a / 2b — e2e multi-seed + content/validity under the unfrozen encoder (QUEUED)

Code complete and validated; awaiting the cluster run (`cluster/`, SLURM array jobs, 5 seeds each).

**Harness validation.** `experiments/eval_e2e.py` run on the existing e2e checkpoint (seed 42)
returns **f1_B = 0.5761**, reproducing the published single-run 0.576 exactly — confirming the new
harness's probe protocol is identical to `diagnose_sdae_e2e.py`'s. It additionally produces AUC
(0.706) and accuracy (0.771), which the original single run never reported.

**Preliminary, n=1 — to be superseded by 2b's 5 seeds.** Same checkpoint, retrieval on the
held-out split (N=340, matching Phase 1c):

| | validity f1_B | R@1 | median rank | decode spec |
|---|---|---|---|---|
| frozen F (Phase 1c/1a, n=5) | 0.497 | 0.068 | 45.6 | 0.697 |
| e2e F (n=1, seed 42)        | 0.576 | 0.012 | 63.5 | 0.764 |

The e2e model is *better* at validity and *worse* at content on every retrieval measure, with a
more collapsed decoder — i.e. a **dose-response between validity specialisation and content
eviction**: the freer the encoder is to specialise, the more content it discards. If 2b's 5 seeds
hold this, the Phase-1c tension result strengthens under e2e rather than merely surviving it.
Treat as a preview only (n=1 vs n=5).

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
