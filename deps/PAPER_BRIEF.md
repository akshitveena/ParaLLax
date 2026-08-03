# RiDAE — Paper Brief (self-contained spec for drafting)

Paste this whole file into an LLM and ask: "Write a workshop-length empirical-study paper
draft from this brief. Do not overclaim beyond what the numbers and caveats state." All
numbers below are from real runs; keep them exact. Honesty is the paper's strength — obey the
guardrails at the end.

## 1. One-sentence contribution
Representing "wrong-approach-right-answer" (a solution that reaches the correct final answer via
unsound reasoning; "Type B") requires **per-step structure**: mean-pooling a chain into one
vector destroys the signal, a step-structured denoising autoencoder with a per-step reward head
recovers it, and the resulting validity axis is **domain-independent** — but it is a **detector,
not a generator**, for reasons we characterize.

## 2. The phenomenon
Type B = final answer correct AND the reasoning contains a genuine error (compensating error,
answer insensitive to the flaw, unsound-but-lucky step, unfaithful/post-hoc, or spurious).
Distinct from: correct+sound (A), and wrong-answer (C/D). Matters for faithfulness/safety —
"right answer for the wrong reason."

Human-labeled prevalence (ProcessBench, answer-correct subset) scales sharply with difficulty:
GSM8K 4% (7/200) · MATH 19% (94/500) · OlympiadBench 32% (161/500) · OmniMath 52% (259/500).

## 3. Data
- **ProcessBench** (Qwen/ProcessBench): step-segmented model solutions with human labels of the
  first erroneous step (-1 if clean). We keep **1,700 answer-correct** solutions across the four
  splits: **A=1,179** (label -1) / **B=521** (label ≥0). Per-step labels: step j good for j<label,
  error at j==label, ignore for j>label.
- Confounds explicitly controlled in every evaluation: response length, LaTeX density, #steps,
  source dataset (these correlate with difficulty and with Type-B rate).

## 4. Method
- **Encoder:** all-MiniLM-L6-v2 (384-d), applied **per step** (each reasoning step → one vector);
  crucially NO pooling across steps.
- **Bottleneck:** 2-layer Transformer over the step-embedding sequence (d_model 256) →
  contextualized step-codes.
- **Decoder:** reconstructs the clean step-embedding sequence (denoising AE). Corruption is at the
  step level (mask step vectors) or text level (delete words per step); the e2e variant uses a
  fixed frozen-embedding reconstruction target (collapse-safe) so the encoder can be unfrozen.
- **PRM head:** per-step error logit (BCE on ProcessBench step labels).
- **Chain head:** attention-pooled step-codes → A/B logit.
- Losses: L_denoise (1-cos) + L_PRM + L_chain. ~1.3M params (transformer+heads) on MiniLM (22M).
- Two variants: **frozen** step-encoder; **e2e** (encoder unfrozen + text-level corruption).
- **Baseline (the thing we beat):** the same corpus encoded by mean-pooling the whole chain into
  one 384-d SBERT vector → 384→256→128→64 bottleneck denoising AE ("pooled RiDAE"), and raw SBERT.

## 5. Evaluation protocol (rigor)
- **Confound-controlled, leakage-free:** residualize [length, LaTeX, #steps, dataset] out of the
  representation, fit the probe on the model's train split, evaluate on its held-out val split.
- Report f1_B (Type-B as positive), AUC, and the raw vs residualized gap.
- Judge (used to define/validate Type B where needed) is an LLM "mechanism judge" validated
  against ProcessBench humans.

## 6. Results (exact)
**Judge validation** (mechanism judge vs ProcessBench humans, 100-problem balanced sample):
Cohen's κ = 0.60, precision 0.84, recall 0.73, F1 ≈ 0.78. Disagreements characterized: our
"errors persist to the answer" definition is a stricter, causal subset of ProcessBench's
"any first-error step" (under-counts are mostly self-corrected/non-causal errors).

**Core ablation — confound-controlled, leakage-free held-out f1_B:**
| representation | f1_B |
|---|---|
| pooled AE (mean-pool → 64-d) | 0.286 |
| raw SBERT (pooled, reference) | 0.286 |
| step-structured SDAE+PRM, frozen encoder | 0.436 |
| step-structured SDAE+PRM, e2e (unfrozen + text corruption) | **0.576** |
(Raw, non-controlled f1_B for pooled ≈ 0.50–0.56; the gap collapses to ~0.29 under control, while
the step model *retains* 0.436→0.576 — the improvement is concentrated in the confound-independent
validity signal.)

**PRM as a process verifier** (held-out ProcessBench val, e2e): step-error AUC 0.799;
clean-chain "no error" accuracy 0.772; first-error exact localization 0.277 (within ±1 = 0.436);
ProcessBench-style F1 (harmonic of error/clean accuracy) 0.407.

**Utility — Type-B mining** (held-out; rank answer-correct candidates by P(Type-B)): base rate
0.276; precision@10 = 1.00, @20 = 0.95, @50 = 0.74; AUPRC 0.711; ROC-AUC 0.840.

**Geometry** (attention-pooled step-codes; UMAP + probes): A/B AUC 0.948 (raw), subject accuracy
0.711 (chance 0.14), dataset accuracy 0.581 (chance 0.25). Subjects intermix (B-cluster is not one
topic); a difficulty gradient exists (easy→hard = left→right).
**Domain transfer** (train A/B probe on all-but-one subject, test on held-out subject),
raw → confound-controlled (difficulty removed): mean **0.954 → 0.898** (per-subject 0.79–0.95).
=> validity axis is domain-independent *beyond difficulty* (drop of only 0.056). Caveat: absolute
AUCs carry representation leakage (encoder saw the candidates); the rigorous *magnitude* is 0.576.

**Mechanism — toxicology / dose-response** (corrupt step-sequences at rising doses): reconstruction
error ≈ 0.547 and nearly flat under all corruptions; chain A/B AUC **invariant to step-shuffle**
(0.867→0.868) and to vector noise (0.867→0.869); only 90% word-deletion dents it (→0.755).
=> the model is a **per-step validity aggregator** ("is there an error-looking step?"),
**order-invariant, not relational**; and the reconstruction objective is weak — the supervised
heads carry the signal.

## 7. Negative results (report them; they are findings)
- **Reranking for correctness, OOD (OpenR1 DeepSeek-R1 traces, 250 problems):** random 0.495,
  self-consistency 0.604, PRM-rerank 0.476 (below random), oracle 1.0; AUC(P_sound→correct)=0.478
  (noise). Cause: distribution shift (long R1 `<think>` traces are far from ProcessBench solutions —
  the PRM scores nearly all R1 traces as flawed, P_sound≈0.33) + validity≠correctness on Type B.
- **Generation from the latent (decode step-codes → text via T5):** generic boilerplate, no content.
  Cause: the validity representation is content-invariant by design.
- **Controllable "Type-B knob" (T5 conditioned on a valid/flawed control):** no consistent control
  effect. Cause: Type-B is a subtle content-specific logical property, not a surface style — the same
  reason it is hard to detect.

## 8. Two deep findings (discussion)
1. **A validity detector and a generator want opposite representations:** detection needs
   content-invariance, generation needs content-preservation; one latent cannot be both.
2. **Type-B is subtle logic, not surface style:** valid and Type-B solutions are near-identical
   except for one hidden error — which is simultaneously why it is an interesting phenomenon, why it
   resists cheap detection (0.576, not 0.95), and why it resists controllable generation.

## 9. Related work (position against)
- **ProcessBench** (Qwen) — phenomenon + human step labels.
- **Process Reward Models** — Lightman et al. "Let's Verify Step by Step" (PRM800K); Math-Shepherd.
  Our PRM is small and below SOTA critics; the contribution is the *representation* finding, not a
  SOTA verifier.
- **Unfaithful CoT** — Turpin et al.; Lanham et al. (the faithfulness framing for Type B).
- **TSDAE** (Wang et al.) — denoising sentence-embedding AE lineage; we show pooled TSDAE-style
  reconstruction is insufficient for validity.
- **`real_mhcot`** (author's prior line) — a *pooled* thought-embedding denoising SAE + reranking.
  **This work is the step-structured successor that fixes its pooling ceiling** and adds per-step
  PRM supervision + the Type-B target. State this overlap explicitly.
- SBERT/MiniLM (Reimers & Gurevych).

## 10. Limitations (state plainly)
Small scale (1,700 examples, 1.3M-param head, ProcessBench only); PRM below SOTA (F1 0.41 vs
60–80 for strong critics); the denoising AE is weak (supervised heads carry the signal); the model
is an order-invariant aggregator, NOT a relational reasoning model; generation is not achievable
(detector, not generator); reranking-for-correctness fails out-of-distribution; correctness labels
on hard datasets rely on an LLM/deterministic verifier with some noise.

## 11. Suggested structure
1. Intro (phenomenon, faithfulness motivation, question: can we represent Type B?).
2. Related work. 3. Phenomenon + judge validation. 4. Method (step-structured SDAE+PRM,
confound-controlled protocol). 5. Results (pooling-vs-structure ablation; PRM benchmark; mining;
geometry/domain-transfer; toxicology mechanism). 6. Negative results (generation, OOD reranking).
7. Discussion (the two deep findings). 8. Limitations. 9. Conclusion.

## 12. Title options
- "Representing Wrong-Approach-Right-Answer: Why Pooling Fails and Step-Structure Works"
- "A Domain-Independent Representation of Reasoning Validity"
- "RiDAE: Step-Structured Denoising for Detecting Flawed-but-Correct Reasoning"

## 13. Target venue
Workshop (NeurIPS/ICML reasoning · math-AI · interpretability) or ACL/EMNLP findings. Frame as an
**empirical study / analysis paper**, not a systems/SOTA paper.

## HONESTY GUARDRAILS (instruct the drafting LLM to obey)
- Do NOT claim "novel reasoning generation," "relational reasoning," or SOTA verification — the
  evidence contradicts all three.
- Quote the **confound-controlled 0.576** as the headline magnitude, not the raw ~0.95.
- Present the negatives (generation, OOD reranking) as characterized findings, not omit them.
- Note the representation-leakage caveat on the 0.898 domain-transfer.
- Position honestly against `real_mhcot` as the step-structured fix, not as a wholly new paradigm.
