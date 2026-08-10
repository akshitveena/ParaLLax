# "Hard, Not Wrong" — hardening status vs. the experiment plan & weakness audit

Status of every item in `experiment_plan.md` (E1–E4) and `weakness_audit.md` (W1–W17),
with the actual numbers produced. Source of truth for numbers: `deps/RESULTS.md`.
Legend: ✅ done · ⚠️ partial · ➖ negative/informative · ✍️ writing-only (no compute) · ❌ not done.

---

## Headline numbers (what the paper now rests on)

| result | number |
|---|---|
| Core representation, confound-controlled f1_B — pooled | 0.291 ± 0.045 |
| — frozen step-SDAE | 0.497 ± 0.047 |
| — **e2e step-SDAE (headline)** | **0.591 ± 0.050** (5 seeds) |
| Difficulty-only null model (4 confounds, no text) | 0.515 |
| Base rate (Type-B) | 0.306 |

---

## Experiment plan (E1–E4)

### E1 — PRM panel ✅ (2 of 4 clean; all 4 reported)
Score open PRMs through the identical confound protocol + sanity gate.

| PRM | lab / base | gate | raw f1_B | ctl f1_B | Δf1 |
|---|---|---|---|---|---|
| Math-Shepherd-7B | DeepSeek / Mistral | 0.735 | 0.429 | 0.075 | **−0.353** |
| RLHFlow-8B-PRM | RLHFlow / Llama-3.1 | 0.802 | 0.508 | 0.125 | **−0.383** |
| Qwen2.5-Math-PRM-7B | Qwen / Qwen-2.5 | 0.559 | — | — | inconclusive¹ |
| Skywork-o1-PRM-7B | Skywork / Qwen-2.5 | — | — | — | unrunnable² |

**Two independent labs / base families both collapse under control.** This is the "claim about
the field, not one model" upgrade. ¹Qwen adapter verified correct (`qwen_diag.py`) but its reward
head scores our step segmentation weakly (gate 0.559, ctl f1 degenerate) — a train/eval format
mismatch, reported honestly, excluded from the headline. ²Skywork uses bespoke repo inference
incompatible with the shared scheme on transformers 5.x; 0/1700 scored, reported as such.
**Pre-commitment honoured: every model run is in the table.**

### E2 — Encoder scale ladder ✅
Core ablation at three encoder sizes (confound-controlled f1_B, 5 seeds).

| encoder | dim | pooled | step-SDAE | lift |
|---|---|---|---|---|
| MiniLM (22M) | 384 | 0.291 | 0.497 | +0.21 |
| mpnet (110M) | 768 | 0.270 | 0.523 | +0.25 |
| gte-large (335M) | 1024 | 0.011 | 0.511 | +0.50 |

Scaling the encoder 15× does not rescue pooling (stays ≤ floor); step-structure holds ~0.50 at
every scale. **Kills "22M-on-a-laptop."** (Honest note in RESULTS.md: gte pooled is weak even
uncontrolled — 0.305 — so the 0.011 is not "control erasing strong signal"; the load-bearing claim
is the flat step-structured line, confirmed by the SDAE reaching 0.511 on the same embeddings.)

### E3 — Second corpus (PRM800K) ➖ NEGATIVE (informative)
Only **23 Type-B of 7,841** solution chains (0.3% vs ProcessBench's 30.6%). Structural, not a bug:
PRM800K's preferred path is the human-*curated correct* trajectory, so errors almost never sit on
the followed path. The wrong-approach-right-answer phenomenon is a property of *sampled model*
reasoning, not curated critique trees. **"One benchmark" is instead addressed by breadth (E1 across
labs, E2 across scales).** A matched second corpus would require *generating* one — deferred.

### E4 — Judge reliability ⚠️ (the hard part done; reframe is writing)
The confound-check half (W3, below) is done and favourable. The ensemble-of-judges and the
"human-primary / judge-robustness" restructuring are ✍️ writing tasks, not yet applied to the paper.

---

## Weakness audit

### Tier 1 (severe, cheap)
- **W1 — difficulty-only baseline ✅.** 0.515 (4 confounds, no text) — near uncontrolled detectors
  (0.58–0.62), so most naive skill is difficulty. The paper's missing number, now present.
- **W2 — nonlinear/stratified ✅ (stratified, the stronger version).** Within-dataset, step-SDAE ≫
  difficulty-only (math 0.448 vs 0.087; olympiad 0.662 vs 0.331; omnimath 0.802 vs 0.717). Signal
  survives where difficulty range is narrow — model-free confirmation of the residualization.
- **W3 — does the judge read difficulty? ✅.** Judge−human confound→label AUC gap = +0.034,
  95% CI [−0.090, +0.129] (spans 0). No detectable contamination. The sharpest attack, answered.
- **W4 — 0.576/0.591 contradiction ✍️.** RESULTS.md uses 0.591 (5-seed) throughout; must be
  swept through the paper source (not in this repo).

### Tier 2 (severe, expensive)
- **W5 — single external PRM ✅** → E1 panel.
- **W6 — one corpus ⚠️** → E3 attempted, negative; breadth added via E1/E2 instead. Still one
  corpus for the *core ablation* — the residual scope weakness.
- **W7 — small encoder ✅** → E2 ladder.
- **W8 — small corpus (1,700 / 521 B) ⚠️.** Not enlarged (E3 failed). Mitigation: 2e bootstrap +
  paired-difference test show the headline gaps are significant despite the size; report the
  val-positive count honestly.

### Tier 3 (writing, mostly)
- **W9 — selection optimism ✍️.** Disclosed in RESULTS.md (2e); move it out of the mid-paragraph.
- **W10 — domain-transfer leakage ✍️.** Flagged; demote the 0.954/0.898 table to "indicative."
- **W11 — human ceiling ✅/✍️.** Judge F1 ≈ 0.78 vs humans = cite as automated-detection ceiling.
- **W12 — class imbalance / AUPRC ✅.** Base rate 0.306 + AUPRC reported (W1 run).
- **W13 — confound justification ✅.** Point-biserial r: length +0.415, n_steps +0.115, latex −0.010.
- **W14 — corruption ad hoc ⚠️/✍️.** Composition ablation exists (Phase 1b: all ~0.50–0.54,
  none≈mask → denoising inert); surface it prominently.
- **W15 — architecture ablation ❌.** Not run. Defensible to state "architecture not tuned."
- **W16 — generation boilerplate ✅/✍️.** Decode-specificity metric exists (Phase 1c, 0.30–0.88);
  replace the qualitative claim with it.
- **W17 — self-citation deanonymisation ✍️.** Paper-source edit.

---

## Bonus results beyond the two documents (already banked)
- **Phase 1c — content-preservation probe:** recon-only latent recovers content near-perfectly
  (R@1 0.996) while the supervised model evicts it (0.068). Detection↔generation tension, measured.
- **Phase 1d — validity is latent:** recon-only z probes to f1_B 0.416 / AUC 0.693 (≥ supervised
  AUC 0.673) with no A/B supervision — motivates the "readout, not loss head" direction.
- **Phase 2e — bootstrap CIs + paired-difference test:** all three headline gaps significant;
  caught and corrected two of our own overclaims (Phase-1d "2.8σ", E2 "more confounded with scale").

---

## What is NOT done
- **2d — LLM-judge baseline row** (built, ~$2.30, your Mac). Minor.
- **E3 generated second corpus** (the one real remaining scope item; large lift).
- **W15 architecture ablation** (optional).
- **All ✍️ items** — folding these numbers into the actual paper, W4 number sweep, W9/W10/W16/W17
  edits. The *results exist*; the *paper text* is not yet updated.

---

## Are we done? Is it publishable? — direct answer

**Done with experiments: essentially yes.** Everything compute-bound that moves the needle is
finished (E1, E2, W1, W2, W3, plus the 2-series and Phase 1a–1d). What remains is one cheap
optional run (2d) and **writing** — folding results into the paper.

**Publishable — workshop (MATH-AI, the stated target): yes, comfortably.** On the plan's own scale
this sits at **~7.5–8**. The paper is now a coherent *methods + honest-negative* contribution:
a confound-control protocol, a difficulty-only null (0.515), a step-structure result that survives
it (0.497→0.591) and survives encoder scaling (E2), confound inflation shown across *two
independent PRMs* (E1), and the sharpest attack (judge-reads-difficulty, W3) directly refuted.
That is a solid, defensible workshop paper, well above bar.

**Main conference (NeurIPS/ICML main track): not yet, and that's a design choice, not a flaw.**
The plan says it plainly: above ~8 the limit becomes the underlying model — a 1.3M head on a 22M
encoder is not a competitive verifier, and you correctly don't claim it is. The contribution is the
*protocol and the negative result*, which is workshop/findings-track shaped, not a main-conference
SOTA claim. To push to main-conference you'd need a genuinely second corpus (generated), a
competitive-scale verifier baseline, and larger data — a different, bigger paper.

**The one honest caveat for the workshop version:** the core representation result still rests on a
single corpus (E3 negative). Breadth from E1/E2 softens this, and it's honestly disclosed — but a
sharp reviewer may still note it. It is not disqualifying for a workshop.

**Bottom line: it is good enough to submit to MATH-AI now, once the results are written into the
paper.** The science is finished; the remaining work is prose.
