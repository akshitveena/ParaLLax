# RiDAE → Main Conference (NeurIPS / ICML) — Project Plan

Companion to `PAPER_BRIEF.md` (the workshop draft spec) and `RESULTS.md` (the results ledger).
This is the roadmap to turn the pilot into a main-conference submission. Do the workshop paper
FIRST (see below); this plan runs in parallel and afterward.

---

## 0. Sequencing (do not skip)
1. **Now:** submit the current work to a **non-archival** workshop (NeurIPS/ICML reasoning ·
   math-AI · interpretability, or ACL/EMNLP findings). Source = `PAPER_BRIEF.md`.
2. **Parallel:** secure GPU compute; start Workstream 1 (scale).
3. **Months 1–6:** execute this plan; submit the extended paper to main NeurIPS/ICML.

## 1. The main-conf thesis (the hook — don't fight SOTA head-to-head)
> **Reasoning-validity verification is confounded. The *causal* signal for right-answer-wrong-
> reasoning (Type B) requires step-structure, is domain-independent, and existing PRMs / LLM-judges
> do not isolate it. We give a confound-controlled benchmark + method that does.**

This *reframes how verifiers are measured* (a "changes the field" contribution) instead of
competing on raw PRM F1 (which we would lose against large models). Your two strongest assets —
**confound-controlled evaluation** and **domain-independence** — become the core.

## 2. Gap analysis
| dimension | pilot (have) | main-conf (need) |
|---|---|---|
| data | 1,700 ProcessBench answer-correct | 10k–50k+ labeled steps, multi-source, cross-model |
| encoder | MiniLM-22M + 1.3M head | a real encoder (0.5–1B fine-tuned, or PRM hidden states) |
| baselines | none | SOTA PRMs + strong LLM-judges, under the confound protocol |
| downstream | mining (in-dist) | a *win* (reranking-at-scale beats SC+PRM, OR Type-B synthesis) |
| benchmark | ProcessBench only | a curated, cross-model, human-validated Type-B benchmark |
| compute | M3 (16GB) | cloud/cluster GPUs |

## 3. Workstreams (prioritized)

### WS1 — Scale (kills the "toy" critique) — needs GPUs
- **Data:** full ProcessBench (all splits, both correct & incorrect); **PRM800K**; **cross-model
  Type-B** — run several models (Qwen2.5, gpt-oss, R1/QwQ, Llama-3) on the *same* problems, label
  with the validated mechanism judge (+ a human-checked subset). Optionally MR-GSM8K, BIG-Bench
  Mistake. Target ≥10k, ideally 50k labeled steps.
- **Encoder:** replace MiniLM with (a) a fine-tuned 0.5–1B encoder, or (b) hidden states from an
  open PRM (e.g., Qwen2.5-Math-PRM), or (c) a strong sentence encoder (gte-large / e5-large).
  Keep the step-structured transformer + PRM/chain heads on top.
- Re-run the **core ablation** (pooled vs step-structured) and **domain-transfer** at scale.

### WS2 — Baselines + the confound critique (the novel core)
- Run **SOTA PRMs** (Qwen2.5-Math-PRM, Math-Shepherd, Skywork-PRM) and **LLM-judges** (GPT-4o /
  Claude as Type-B detectors) on the SAME confound-controlled, leakage-free protocol.
- Headline claim to establish: their apparent skill **drops under confound control** and/or
  **does not transfer across domains**, while the step-structured causal signal does. This
  contrast is the paper.

### WS3 — A downstream win (pick ONE, make it convincing)
- (a) **Reranking-at-scale, in-distribution:** best-of-N from a matched generator → validity
  rerank beats self-consistency AND a strong PRM baseline on final-answer accuracy.
- (b) **Type-B synthesis for faithfulness:** prompt a strong LLM to produce sound vs subtly-flawed-
  but-correct solutions; show the detector reliably separates them and the synthesized set is
  human-validated. Framed as adversarial faithfulness data — a compelling "so what" for safety.

### WS4 — Benchmark
- Curate a **cross-model, human-validated Type-B test set** (a named benchmark others can cite).
  Scale labels with the judge; human-validate a subset; report judge–human agreement.

### WS5 — Framing & writing
- Position Type B as a *distinct, faithfulness-relevant* target that generic error-detection
  conflates. Argue confound-controlled verification as the *standard*. Lean into rigor + honest
  negatives (reviewers respect this).

## 4. Compute budget (rough)
- Fine-tune a 0.5–1B encoder + transformer on 10–50k examples: single A100, a few days →
  **~$100–500** cloud, or free on an academic cluster.
- Cross-model generation + judge labeling: GPU inference or API → **~$300–1,500** depending on
  scale/models.
- **Total realistic: ~$500–2,000.** The M3 cannot do WS1/WS2 training — GPU access is the #1 gate.

## 5. Timeline (aggressive but honest, ~6 months)
- **M1:** assemble + generate + label the scaled corpus; stand up the GPU pipeline.
- **M2:** scale the encoder + re-run core ablation & domain-transfer at scale.
- **M3:** implement + run all baselines under the confound protocol (the critique).
- **M4:** the downstream win (WS3).
- **M5:** benchmark (WS4) + ablations + start writing.
- **M6:** polish, additional experiments reviewers will ask for, submit.

## 6. Risks & mitigations
- **No GPU access** → biggest risk. Mitigate: cloud rental, Colab Pro+/A100, university cluster, or
  a collaborator with compute.
- **Baselines too strong head-to-head** → don't compete on F1; win on confound-control +
  domain-independence + the causal frame.
- **Downstream doesn't beat baselines** → pivot to the synthesis/faithfulness angle (WS3b).
- **Type-B labels noisy at scale** → human-validate a subset; report agreement; use judge+
  deterministic ensemble.
- **Reviewer "incremental over real_mhcot"** → position explicitly as the step-structured fix +
  confound-critique + benchmark; real_mhcot was pooled and un-benchmarked.

## 7. Success criteria (what makes it accept-worthy)
1. The confound-critique holds **at scale** across ≥3 strong verifiers (their skill is
   confound-inflated; the step-structured causal signal survives).
2. **Domain-independence at scale** (transfer across held-out domains after confound removal).
3. **One clear downstream win** (accuracy ↑ or a validated synthesis capability).
4. A **benchmark** others can use.
5. Honest, thorough characterization (the negatives + mechanism analysis stay in).

## 8. Honest probability
With GPU access + ~6 focused months + the confound-critique framing: a **plausible** main-conf
submission (not a lock — the venue is competitive). Without scale/compute: it stays a workshop
paper. The workshop-first path guarantees a result either way.
