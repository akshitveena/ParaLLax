# ParaLLax — Follow-up Experiments Plan

Scope: the six reviewer/collaborator asks against `ParaLLax_MATHAI_submission_review_version.pdf`.
Nothing here is executed — these are runnable designs + reference code (see `experiments/`) meant to
drop into the real repo (`/workspace/ridae/experiments/` on your container, not present in this
workspace). Every script has an `# ==== ADAPTER ====` block you wire into your existing loaders/models.

**Guiding rule (non-negotiable):** none of these experiments are allowed to *assume* their conclusion.
Two of the six asks (factored architecture "in main text", surgical erasure) are phrased as settled
wins but are flagged by the paper *itself* as untested and possibly negative. We build them to be
falsifiable and report whatever comes out. No fabricated numbers anywhere — draft text uses
`<PLACEHOLDER>` tokens until a run fills them.

---

## 0. The headline correction (blocks E1/E6 until resolved)

Your terminal run of `rl_policy_gaming.py` is being read as a null result. It is not. It is the
predicted exploit, mislabelled, produced by a mis-specified scorer.

| scorer | selected length vs random | reading |
|---|---|---|
| Uncontrolled (raw SBERT) | **−85.9 tokens (−27%)** | reward moves systematically with a difficulty confound |
| Controlled (residualized) | **≈ random (+2.4)** | control removes the length preference |

Why the sign is *shorter*, not longer:
- Fig 7 shows adding apparent difficulty **raises** P(Type B). A policy rewarded to look **sound**
  therefore strips apparent difficulty ⇒ shorter. length↔label point-biserial is **+0.415**.
- "Longer, LaTeX-heavy filler" would be the exploit only if the reward were **P(Type B)** (a Type-B
  *miner*, or a policy told to look impressively hard) — not the deployment threat model.

Two defects to fix before this can go in the paper:
1. **Wrong scorer.** raw SBERT is the *floor* (0.291 controlled), not the length-riding verifier. The
   paper's length-shortcut evidence is Math-Shepherd-7B and RLHFlow-8B (they collapse under control)
   and the difficulty-only null. Attack **those**, plus your own uncontrolled Step-SDAE.
2. **No validity measurement.** The run reports length/latex/steps but not the **gold Type-A rate** of
   the selected set. Reward-hacking is proven only if apparent difficulty shifts **while gold validity
   does not improve**. Add it — it is the whole point.

→ Implemented in `experiments/e1_policy_gaming_v2.py`. Report signed Δdifficulty (sign-agnostic) AND
Δ(gold Type-A rate). The claim that survives peer review is: *"optimizing the uncontrolled verifier
moves apparent difficulty by X with no gain in true validity (Δgold = Y, CI ...), whereas the
controlled verifier shows neither shift; control closes the exploit channel."*

---

## E1 / E6 — Closed-loop policy gaming (the "game-changer")

**Claim to support:** ParaLLax is an exploitable channel, not just a measurement artifact — a policy
optimized against the uncontrolled verifier games apparent difficulty without improving validity, and
the confound-controlled verifier suppresses this.

**Three escalating designs (cheapest → most convincing), all in `e1_policy_gaming_v2.py`:**

1. **Best-of-N selection (cheap, CPU, primary).** For each problem, take its N answer-correct
   candidates; select argmax under each scorer (uncontrolled PRM, controlled PRM, random). Report, per
   scorer: mean length / latex / n_steps **and mean gold Type-A rate** of selected candidates, 5 seeds,
   paired bootstrap CI vs random. Exploit = difficulty shift with Type-A rate statistically ≈ random.
2. **Text-level adversarial attack (the missing "text-level Fig 7", strongest single result).** Take
   *sound* (gold Type-A) solutions; apply validity-preserving difficulty inflations (append redundant
   LaTeX-heavy but logically inert steps; re-typeset numbers as LaTeX; pad restatements). A gold judge
   re-verifies validity is unchanged. Measure verifier score shift. Uncontrolled score should move
   monotonically with inflation dose; controlled should stay flat. This is Fig 7 at the token level and
   needs no policy training.
3. **PPO / GRPO loop (optional, GPU, confirmatory).** Small policy (e.g. Qwen2.5-Math-1.5B) rewarded by
   each verifier on a fixed problem set; log KL-controlled training curves of reward, length, latex, and
   **held-out gold validity**. Provided as a scaffold — expensive; only run if reviewers demand a live
   agent. Best-of-N + text attack already close the loop.

**Kill criteria (report honestly if hit):** if the uncontrolled verifier's selected set shows *higher*
gold Type-A rate, that is genuine validity signal, not gaming — say so. If controlled also shifts
difficulty, control is incomplete — say so.

---

## E2 — Factored architecture, and whether it earns the main text

**Ask:** move Appendix N's fix (content-preserving encoder Recall@1≈0.996 + non-competing validity
readout) into the main Method.

**Integrity flag (this is the important one).** Appendix N already flags this as *untested*:
> "under full plasticity such a readout would have to recover separability that supervision currently
> creates by reshaping the encoder itself, which the frozen-regime result suggests a fixed latent may
> not supply on its own. Testing this is the specific next experiment our negative results identify."

The numbers say why to be cautious:
- Frozen recon-only latent (Variant R): validity **AUC 0.693**, f1B 0.416.
- Shared-bottleneck end-to-end (Variant F-e2e): **AUC 0.741**, controlled f1B **0.591**.

So a frozen encoder + readout may land at ~0.69 AUC — **below** the 0.741 the shared e2e model buys by
reshaping the encoder. Promoting the factored design to the main text as "the solution" **before**
showing it matches e2e would be exactly the overclaim the paper is currently careful to avoid.

**So E2 is a test, not a coronation.** `e2_factored_architecture.py` builds three encoder regimes and
one readout, and the main-text move is **conditional on the result**:
- (A) Frozen recon-only encoder + MLP readout (gradients stop at encoder).
- (B) Recon encoder + **separate LoRA/adapter path** trained on validity only, structurally isolated
  from the reconstruction decoder (so validity can reshape features *without* touching the recon
  bottleneck) — the honest attempt to get e2e-level validity while keeping content.
- (C) Shared-bottleneck e2e (reproduce Variant F-e2e as the baseline to beat).
- Report for each: content preservation (Recall@1, target ≈0.996), controlled f1B (target ≥0.591),
  validity AUC (target ≥0.741).

**Decision rule for the draft (`paper/factored_architecture_maintext.md`):**
- If (A) or (B) reaches content ≈0.996 **and** f1B ≥ 0.591 → promote to main Method; Appendix N becomes
  the motivation. **✅ full ask satisfied.**
- If it preserves content but validity drops to ~0.69 → main text presents it as *the content-preserving
  variant with a stated validity cost*, not a strict improvement. Honest, still a contribution.
- If it fails both → stays an appendix direction. Do not move it.

---

## E3 — Second multi-model corpus

**Ask:** because PRM800K has 0.3% Type-B, build a second corpus by sampling long-CoT models
(DeepSeek-R1, Qwen2.5-Math) on AIME/MATH-500 and show ParaLLax holds across models.

**Pipeline (`e3_second_corpus.py`):** sample K solutions/problem → keep answer-correct → segment into
steps → mechanism judge (same rubric as κ=0.60 judge) labels Type A/B → persist matched corpus. Then
re-run the three ParaLLax tests on the new corpus **and cross-model** (train on ProcessBench, test on
new corpus): difficulty-only null, uncontrolled vs controlled detectors, per-step vs pooled.

**Caveats to bake in (the paper already found these):**
- Appendix O: R1 `<think>` traces are far OOD for the PRM (scores ≈0.33 uniformly). Step segmentation
  and judge prompts must be adapted for long CoT — don't reuse ProcessBench segmentation blindly.
- AIME is tiny (~30 problems/year). MATH-500 carries the count. Report N honestly; this is a
  *replication on generated data*, and its defensibility rests on Appendix K (judge not
  difficulty-biased) — cite that explicitly.
- Pre-register the direction: the effect **replicates** if uncontrolled ≈ difficulty-null and per-step
  controlled signal survives on the new corpus too. If it doesn't replicate, that bounds the claim to
  ProcessBench — report it.

---

## E4 — Non-linear confound control

**Ask:** replace linear residualization/probes with non-linear (Kernel Ridge / MLP) to prove the
surviving 0.591 isn't non-linear difficulty leakage.

**Design (`e4_nonlinear_control.py`):**
- **Non-linear null first.** Fit an MLP (and KRR) on the 4 confounds → Type-B. Its f1B is the new null
  bar (linear null = 0.515). If the non-linear null jumps a lot, more of the "signal" was always
  difficulty.
- **Non-linear residualization with cross-fitting.** For each representation dim, fit KRR/MLP
  `dim ~ f(confounds)` on train, subtract prediction on val → non-linear residuals; re-fit the probe.
  **Cross-fit / hold out** the residualizer — an over-powerful residualizer can regress out *everything*
  (including validity) and produce a spuriously deflated number. Report residualizer's own R² on a
  held-out fold so over-fitting is visible.
- **Report both** linear (0.591) and non-linear controlled f1B side by side. Signal robust ⇔ small drop.
- Guardrail: match residualizer capacity across the signal model and the null so the comparison is fair
  (don't give the null a bigger MLP than the residualizer).

---

## E5 — Make the causal result surgical

**Ask:** find a representation that removes difficulty while preserving validity **and** general verifier
competence (current ablation drops the step gate 0.735→0.637 alongside difficulty).

**Design (`e5_surgical_erasure.py`):** three erasure operators, evaluated on the same axis:
- Baseline: top-k PLS/PCA difficulty subspace ablation (reproduces paper: not surgical).
- **LEACE** (least-squares concept erasure) — minimal-collateral linear erasure of the difficulty concept.
- **Validity-preserving oblique erasure** — erase difficulty *within the null space of the validity
  subspace*: remove difficulty variance, then add back its projection onto the protected validity
  direction(s). Equivalent to conditional erasure that leaves the validity readout's inputs invariant.
- Optional: a learned linear map minimizing difficulty decodability **subject to** preserving the
  validity probe output and the step gate (two-term constrained objective).

**Metric of "surgical":** difficulty R² ↓ to control target **and** ΔAUROC(validity) ≈ 0 **and**
Δ(step gate) ≈ 0. Current baseline: step gate Δ = −0.098.

**Integrity flag.** The paper found difficulty is redundantly written and self-repairing (Hydra effect).
If difficulty and validity are genuinely entangled in this verifier, **no linear surgery can separate
them** — in which case the "not surgical" result is *fundamental*, and reporting "we tried LEACE +
oblique erasure and the step gate still falls" is a **stronger** mechanistic claim than the current one.
E5 is designed so a negative outcome is a real result, not a dead end.

---

## Cross-cutting requirements

- **Seeds & CIs:** keep the paper's convention — 5 seeds, paired-difference bootstrap CIs on every gap.
- **Leakage:** fit every residualizer/probe/erasure on train only; evaluate on held-out val (the paper
  is careful about this — Table 4 even annotates encoder leakage).
- **Pre-registration:** for E1/E3/E5 write the predicted direction + kill criteria into the script header
  before running, so a null is reportable rather than silently re-tuned.
- **No number invention:** `paper/*.md` uses `<PLACEHOLDER>`; fill only from a real run.

## Suggested run order
1. E1 Best-of-N + text attack (cheapest, highest payoff, closes the loop).
2. E4 non-linear control (cheap, hardens the 0.591 claim reviewers will poke).
3. E2 factored architecture (medium; decides a main-text change).
4. E5 surgical erasure (medium; may return an honest negative).
5. E3 second corpus (most expensive: generation + judging), then cross-model replication.
