# RiDAE Workshop Paper — Complete Knowledge Base

Everything needed to write the paper: the story, the ideas, the method, every result with
interpretation, the related work, the figures, a section-by-section guide, a draft abstract,
and reviewer-defense. Pair with `PAPER_BRIEF.md` (terse spec) and `RESULTS.md` (numbers).

===============================================================================
PART A — THE STORY (read this first; it is the spine of the paper)
===============================================================================
A model can reach the **right answer through wrong reasoning** — "Type B" (wrong-approach-
right-answer). It is a faithfulness problem: the model looks correct but is right for the wrong
reasons. We ask a simple question: **can we learn a representation that captures this?**

The journey (this IS the paper's arc):
1. The naive approach — encode the whole reasoning chain into one vector (a TSDAE-style denoising
   autoencoder) — **fails**. The Type-B signal is not there.
2. We find *why*: **mean-pooling destroys it.** A flawed step is 1/N of an average; validity is a
   per-step, relational property, so pooling washes it out. Semantics survives pooling (that's why
   TSDAE works for similarity); validity does not.
3. We fix it with a **step-structured** denoising autoencoder + a per-step reward (PRM) head:
   0.286 (pooled) → 0.436 (step, frozen) → **0.576** (step, end-to-end), all confound-controlled.
4. We show the recovered validity axis is **domain-independent** — it transfers across held-out
   subjects even after difficulty is removed.
5. We characterize *what* the model learned (toxicology / dose-response): it is a **per-step
   validity aggregator**, order-invariant — not a relational reasoner. Honest, and it explains the
   modest magnitude.
6. We show the representation is a **detector, not a generator** — and explain the deep reason:
   validity detection needs content-*invariance*, generation needs content-*preservation*; and
   Type-B is subtle *logic*, not surface *style*.

**The one-line takeaway:** *representing right-answer-wrong-reasoning requires per-step structure,
the signal is domain-independent once you control for confounds, and it is fundamentally a
detection signal — not a generative one.*

===============================================================================
PART B — MOTIVATION & POSITIONING (the "why care")
===============================================================================
- **Faithfulness / safety:** "right answer for the wrong reason" is exactly the failure alignment
  cares about — a model that appears competent but reasons unsoundly. Detecting it matters.
- **The gap:** process-error detection (PRMs, ProcessBench) treats *any* step error the same. Type
  B — answer-correct *despite* a flaw — is a distinct, under-studied slice, and it is the one where
  "the answer looks right" masks the failure.
- **A methodological gap:** verification/validity results are rarely confound-controlled. We show a
  lot of apparent "validity" is just length/difficulty, and isolate the causal signal.
- **Prevalence is real and scales with difficulty** (human-labeled, ProcessBench answer-correct):
  4% (GSM8K) → 19% (MATH) → 32% (OlympiadBench) → 52% (OmniMath). On hard problems it is the
  *majority* of correct answers. This number is a strong opener.

===============================================================================
PART C — CONCEPTUAL FOUNDATIONS (define these precisely; reviewers will check)
===============================================================================
**Type taxonomy** (over answer × reasoning):
- A = correct answer, sound reasoning.
- **B = correct answer, UNSOUND reasoning** ← the target (wrong-approach-right-answer).
- C = wrong answer, sound path (rare; folded out).
- D = wrong answer.

**What "wrong approach" means (a key conceptual contribution — three orthogonal axes):**
- **Validity** (sound / flawed) — is there a genuine invalid step? THIS is what "wrong" means.
- **Canonicality** (canonical / alternative) — same as a reference approach or different? "Different
  ≠ wrong." A valid different route (sound_alternative) is Type A.
- **Generality** (general / brittle) — does it generalize? "Brittle ≠ wrong." A valid narrow trick
  is still sound.
- **Claim:** a solution is Type B iff its reasoning is *invalid* (or unfaithful/spurious) — not
  because it is different, brittle, or merely incomplete.

**Mechanism taxonomy** (how a correct answer arises; used to define/validate B):
- sound_canonical, sound_alternative → A.
- flawed_lucky (real error, compensating/insensitive/lucky), unfaithful (stated steps don't produce
  the answer; hidden/post-hoc), spurious (guessed, back-filled) → B.
- Incompleteness alone (valid but not fully proven) is NOT B — it's sound-but-partial.

**Our Type-B vs ProcessBench's "first-error step":** ProcessBench labels the first step containing
*any* error. Ours requires the error to *persist causally into the answer*. So ours is a stricter,
causal subset; the disagreements are mostly *self-corrected* errors (flagged by ProcessBench,
recovered by the model) — an honest, characterizable difference.

===============================================================================
PART D — THE DEEP IDEAS (these elevate the paper above an ablation)
===============================================================================
1. **Semantics survives pooling; validity does not.** Meaning is a "bag" property robust to
   averaging (why TSDAE/SBERT work). Validity is relational/per-step — a single flawed step is 1/N
   of the average and washes out. This is *why* pooling is the ceiling.
2. **Detection and generation want opposite representations.** Detecting validity requires content-
   *invariance* (abstract away *what* is said, keep *whether it's sound*). Generation requires
   content-*preservation*. One latent cannot be both — so a validity latent is content-poor and
   cannot be a generative source. (We show this empirically: decoding it yields boilerplate.)
3. **Type-B is subtle logic, not surface style.** A valid and a Type-B solution are near-identical
   except for one hidden error. This *single fact* explains three things: why the phenomenon is
   interesting, why it resists cheap detection (0.576, not 0.95), and why it resists controllable
   generation (no surface signal for a control to latch onto).
4. **Confound-controlled verification reveals inflated skill.** Raw, a pooled representation looks
   decent (~0.56 f1_B); control for length/difficulty/dataset and it collapses to 0.286, while the
   step-structured model *retains* 0.436→0.576. The improvement is concentrated in the confound-
   *independent* validity signal — the part that actually matters.
5. **What "step-structured" buys is aggregation, not relation.** The dose-response study shows the
   model is order-invariant — it aggregates per-step validity, it does not model reasoning *flow*.
   Being honest about this is a strength, and it's a genuine mechanistic finding.

===============================================================================
PART E — METHOD (enough to reproduce)
===============================================================================
**Data:** ProcessBench (Qwen/ProcessBench), 1,700 answer-correct solutions across GSM8K/MATH/
OlympiadBench/OmniMath. A=1,179 (label -1) / B=521 (label ≥0). Per-step target: good for j<label,
error at j==label, ignore for j>label.

**Architecture:**
- Encoder: all-MiniLM-L6-v2 (384-d), applied PER STEP (no cross-step pooling).
- Bottleneck: 2-layer Transformer (d_model 256, 4 heads) over the step-embedding sequence →
  contextualized step-codes. Positional encoding; a learned MASK token for masked steps.
- Decoder: 2-layer MLP per step → reconstructs the clean step embedding (denoising AE; 1-cos loss).
- PRM head: per-step linear → error logit (BCE on step labels; masked to labeled steps).
- Chain head: attention-pool step-codes → A/B logit (BCE).
- Loss = L_denoise + L_PRM + L_chain. ~1.3M params on MiniLM (22M).

**Variants:** frozen step-encoder; **e2e** = encoder unfrozen + text-level word-deletion corruption
+ a FIXED frozen-embedding reconstruction target (prevents collapse when unfreezing).

**Evaluation protocol (the rigor — emphasize this):**
- Confound set: response length, LaTeX density, #steps, source dataset.
- **Confound-controlled, leakage-free:** residualize the confounds out (fit on train split),
  fit the probe on train, evaluate on the model's held-out val split. Report f1_B / AUC and the
  raw-vs-residualized gap.

**The judge (for defining/validating B where human labels are absent):** an LLM "mechanism judge"
(5-way taxonomy above) validated against ProcessBench humans: κ=0.60, precision 0.84, recall 0.73.

===============================================================================
PART F — RESULTS (every number + how to interpret it)
===============================================================================
**F1. Phenomenon + judge.** Prevalence 4→52% with difficulty. Judge vs humans κ=0.60, precision
0.84; disagreements characterized (our causal subset vs their any-error).

**F2. Core ablation (confound-controlled, leakage-free f1_B):** pooled 0.286 → step-frozen 0.436
→ step-e2e **0.576**. Interpretation: pooling ≈ raw-SBERT ceiling; per-step structure roughly
doubles the causal validity signal; unfreezing the encoder + text corruption adds more. Show the
raw-vs-controlled contrast (raw pooled ~0.56 collapses to 0.286; the step model retains).

**F3. PRM as a verifier (held-out val):** step-error AUC 0.799; clean-chain acc 0.772; first-error
exact 0.277 (±1: 0.436); ProcessBench-style F1 0.407. Honest: strong ranker, weak exact localizer,
small model — NOT a SOTA-verifier claim.

**F4. Utility — Type-B mining (held-out):** base rate 0.276; precision@10 = 1.00, @20 = 0.95,
@50 = 0.74; AUPRC 0.711; AUC 0.840. Interpretation: the *high-confidence* Type-B calls are reliable
→ a usable miner (corpus-building / faithfulness flagging). This is utility that KEEPS the
phenomenon as the target (unlike correctness-reranking).

**F5. Geometry.** A/B AUC 0.948 (raw), subject acc 0.711 (chance 0.14), dataset acc 0.581 (chance
0.25); subjects intermix (B-cluster is not one topic); a difficulty gradient exists.
**Domain transfer** (leave-one-subject-out A/B), raw → confound-controlled: mean 0.954 → **0.898**
(per-subject 0.79–0.95). => validity axis is domain-independent BEYOND difficulty. Caveat:
absolute AUCs carry representation leakage; the rigorous *magnitude* is 0.576.

**F6. Mechanism — toxicology.** Reconstruction ≈ 0.547, ~flat under corruption; chain AUC invariant
to step-shuffle (0.867→0.868) and vector-noise (0.867→0.869); only 90% word-deletion dents it
(→0.755). => per-step aggregator, order-invariant, NOT relational; reconstruction is weak (heads
carry the signal).

===============================================================================
PART G — NEGATIVE RESULTS & THE BOUNDARY (include; they are findings)
===============================================================================
- **Reranking-for-correctness, OOD (OpenR1 R1 traces):** random 0.495, self-consistency 0.604,
  ours 0.476 (below random), oracle 1.0; AUC(P_sound→correct)=0.478. Cause: distribution shift
  (long R1 <think> traces are OOD; the PRM scores nearly all as flawed) + validity≠correctness on B.
- **Generation from the latent:** decoding step-codes → boilerplate. Cause: content-free latent.
- **Controllable Type-B knob:** no consistent control effect. Cause: Type-B is subtle logic, not
  style.
These support the Part-D ideas (detector≠generator; subtle-logic).

===============================================================================
PART H — RELATED WORK (with the positioning line for each)
===============================================================================
- ProcessBench (Qwen) — phenomenon + human step labels; we study its answer-correct-with-error
  slice specifically and add confound control.
- PRMs: Lightman et al. "Let's Verify Step by Step" (PRM800K); Math-Shepherd; Skywork-PRM — general
  step verification; ours is small & below SOTA, contribution is the representation/confound finding.
- Unfaithful CoT: Turpin et al.; Lanham et al. — the faithfulness framing for Type B.
- TSDAE (Wang et al.) — denoising sentence-embedding AE; we show pooled TSDAE-style reconstruction
  is insufficient for validity.
- real_mhcot (author's prior line) — POOLED thought-embedding denoising SAE + reranking; **this is
  the step-structured successor that fixes its pooling ceiling.** State explicitly.
- SBERT/MiniLM (Reimers & Gurevych) — the encoder.
(Verify every citation before submission — LLM drafts hallucinate references.)

===============================================================================
PART I — FIGURES & TABLES
===============================================================================
- Fig 1: prevalence vs difficulty (4→52%) — the opener.
- Fig 2: the pipeline (per-step encode → transformer → decoder + PRM/chain heads).
- Table 1: core ablation (pooled 0.286 → frozen 0.436 → e2e 0.576), raw vs confound-controlled.
- Fig 3: UMAP of step-codes by type (A/B separation) + by subject (intermix).
- Table 2: domain transfer raw vs confound-controlled (0.954 → 0.898).
- Fig 4: toxicology dose-response (recon error + chain AUC vs dose) — the mechanism.
- Table 3: PRM benchmark + mining (AUC 0.80; precision@k).
- (Optional) Table 4: negative results summary.

===============================================================================
PART J — SECTION-BY-SECTION GUIDE + DRAFT ABSTRACT
===============================================================================
1. **Intro:** open with the prevalence number and the faithfulness stakes; pose "can we represent
   Type B?"; preview the arc (pooling fails → step-structure → domain-independent → detector-not-
   generator); list contributions.
2. **Related work** (Part H).
3. **The phenomenon & judge** (Part C + F1): taxonomy, prevalence, judge validation.
4. **Method** (Part E): architecture + the confound-controlled protocol (spend words on the protocol
   — it's a differentiator).
5. **Results:** ablation (F2) → PRM+mining (F3,F4) → geometry/domain-transfer (F5) → mechanism (F6).
6. **Negative results & discussion** (Part G + D): the two deep ideas.
7. **Limitations** (Part L).
8. **Conclusion.**

**Draft abstract (edit freely):**
> Large models often reach the correct answer through unsound reasoning — "wrong-approach-right-
> answer" — a faithfulness failure that is the majority of correct answers on hard problems
> (52% on OmniMath, human-labeled). We ask whether this can be *represented*. A standard pooled
> denoising autoencoder fails: mean-pooling averages away the per-step signal that distinguishes
> sound from flawed reasoning. A step-structured denoising autoencoder with a per-step reward head
> recovers it, roughly doubling the confound-controlled, leakage-free detection signal (0.286 →
> 0.576 f1_B), and the recovered validity axis is domain-independent — it transfers across held-out
> subjects even after difficulty is removed. A dose-response analysis shows the model is a per-step
> validity *aggregator*, not a relational reasoner, and we show the representation is a *detector,
> not a generator*: it yields a reliable Type-B miner (precision@10 = 1.0) but cannot be decoded or
> steered to generate reasoning, because validity detection requires content-invariance while
> generation requires content-preservation. We argue right-answer-wrong-reasoning is a distinct,
> faithfulness-relevant target that generic error detection conflates, and that confound-controlled
> evaluation is necessary to measure it.

===============================================================================
PART K — ANTICIPATED REVIEWER OBJECTIONS + RESPONSES
===============================================================================
- "Small scale / toy model." → True; frame as a controlled *study* not a systems paper; scale is
  future work (see MAINCONF_PLAN). The *finding* (pooling ceiling + domain-independence) is the
  point, and it's rigorously controlled.
- "Step > pooling is obvious." → It isn't, quantitatively, once confound-controlled; and we show
  the *mechanism* (aggregation not relation) and that *semantics survives pooling but validity
  doesn't* — a non-obvious dissociation.
- "PRM is below SOTA." → We don't claim a SOTA verifier; the contribution is representational +
  methodological (confound control) + the Type-B framing.
- "Incremental over real_mhcot." → real_mhcot was pooled and un-benchmarked; we identify pooling as
  the ceiling and fix it, add PRM supervision, confound control, domain-transfer, and the Type-B
  target.
- "Why did generation fail — is the method broken?" → No; we *explain* it (content-invariance vs
  content-preservation; subtle-logic-not-style) and it's a genuine finding, not a bug.
- "Confounds — did you really control them?" → Yes; residualized length/LaTeX/#steps/dataset,
  leakage-free (probe fit on train, eval on val); report raw-vs-controlled gaps.

===============================================================================
PART L — HONESTY GUARDRAILS / WHAT NOT TO CLAIM
===============================================================================
- Do NOT claim: novel-reasoning generation; relational reasoning; SOTA verification; that step-
  structure "understands reasoning flow" (it's order-invariant aggregation).
- DO quote the confound-controlled 0.576 as the headline magnitude (not raw ~0.95).
- DO keep the negatives and the leakage caveat (0.898) in the paper.
- DO position honestly vs real_mhcot.
- The rigor + honesty is the moat. Reviewers reward controlled negatives over inflated positives.
