# ParaLLax — *Hard, Not Wrong*

### Reasoning verifiers don't judge correctness. They read difficulty.

We took a state-of-the-art 7B process reward model, found the direction in its
residual stream that encodes problem *difficulty*, and tried to surgically remove it.
**It grew back.** Downstream layers reconstructed it within four layers — a Hydra
effect — and what little we could excise took the verifier's genuine error-detection
ability down with it. Difficulty isn't a bug you can patch out of these models. It's
load-bearing.

That is the punchline. Here is the arc that gets there.

---

## The problem: *hard* looks like *wrong*

LLMs sometimes reach the **right answer through a conceptually wrong approach**
("wrong-approach–right-answer," Type B) — and this gets more common the harder the
problem is (4% on GSM8K → 52% on OmniMath). A good reasoning verifier should flag it.
The trouble: **a verifier that simply keys on difficulty will look like it detects bad
reasoning**, because bad reasoning and hard problems co-occur. Nobody was controlling
for that.

ParaLLax is three things: a **detector** that isolates the validity signal, a
**confound-controlled evaluation protocol** that shows most apparent verifier skill is
difficulty, and a **mechanistic account** of *why* — where difficulty lives inside a
verifier and why you can't remove it.

---

## Headline results

| result | number |
|---|---|
| **Difficulty-only null** (4 confounds, *no text*) reaches | **f1\_B 0.515** — near naive detectors |
| Step-structured detector, confound-controlled (5-seed) | pooled **0.29** → frozen **0.50** → e2e **0.59** |
| Open 7B PRMs under the *identical* protocol (Math-Shepherd, RLHFlow) | raw f1\_B **0.43–0.51 → 0.08–0.13** (Δ ≈ −0.35) |
| Difficulty linearly decodable inside the 7B PRM | **R² = 0.92** (mid-network) |
| Removing difficulty: a **~16-dim, self-repairing subspace** | entangled with the verifier's competence |

Full ledger: [`deps/RESULTS.md`](deps/RESULTS.md). Mech-interp write-up:
[`deps/MECHINTERP_M1_RESULTS.md`](deps/MECHINTERP_M1_RESULTS.md). Reviewer-audit status:
[`deps/HARDENING_STATUS.md`](deps/HARDENING_STATUS.md).

---

## 1 — The detector

A **step-structured denoising autoencoder** that keeps per-step structure instead of
mean-pooling it away (pooling destroys the validity axis — it never clears the 0.29 floor).

```
reasoning chain, one embedding per step
  → per-step MiniLM/SBERT embeddings              (384-d)
  → 2-layer Transformer bottleneck                (256-d step-codes)   ← relational mixing
  → decoder (denoising)  + PRM head (per-step error)  + attention-pooled chain head (A/B)
```

Trained on **ProcessBench** human per-step error labels (1.7K answer-correct solutions).
Confound-controlled, leakage-free held-out **f1\_B rises 0.29 → 0.50 → 0.59** as the encoder
unfreezes — a 2× gain over the pooled baseline. The same PRM head is a competent process
verifier (step-error AUC 0.80) and a Type-B data miner (precision@10 = 1.0).

## 2 — The confound critique, generalized to the field

- **The missing null model:** a logistic classifier on four confounds
  (length / #steps / LaTeX-density / dataset) with **no text** reaches f1\_B **0.515** —
  most of what naive detectors "know" is difficulty.
- **A PRM panel:** score open 7B PRMs (Math-Shepherd, RLHFlow — different labs, different
  base models) through the *same* residualized protocol. Headline f1\_B **collapses ~0.43–0.51
  → 0.08–0.13** under control. The critique isn't about one model; it's about how the field
  measures verifiers.
- **Scale-invariant:** holds across a 22M → 335M encoder ladder.
- **Not label circularity:** the κ = 0.60 LLM-judge's difficulty-slope matches humans'
  (bootstrap gap CI spans 0).

## 3 — Where the confound lives (and why you can't cut it out)

Inside Math-Shepherd-7B's residual stream:

- **Probe:** difficulty is linearly decodable at **R² = 0.92**, peaking mid-network.
- **Subspace ablation:** one direction does nothing; ~16 directions drop the score to the
  controlled floor — but the step-error gate falls in lockstep. Difficulty is **entangled
  with competence**.
- **Self-repair (Hydra):** ablate it at the peak layer and downstream layers re-encode it
  (R² recovers to ~0.90).
- **Steering:** *adding* the direction monotonically inflates the Type-B score — the causal
  bookend.

**Conclusion:** confound control cannot be replaced by a targeted internal edit — it is a
mechanistic necessity, not a statistical convenience.

---

## Repository

```
main/            core model (sdae_prm.py), training (frozen + e2e), data pipeline, judge
experiments/     every experiment (each self-documenting):
                   difficulty_baseline.py   confound null + stratified control (W1/W2)
                   prm_panel.py             the 7B PRM panel (E1)
                   encoder_ladder.py        22M→335M scale ladder (E2)
                   judge_confound_check.py  judge-reads-difficulty test (W3)
                   mechinterp_m1/m2/m3.py   mech-interp: probe / ablation / subspace+Hydra
                   mechinterp_steer.py      activation steering
                   bootstrap_ci.py          paired-difference CIs
deps/            RESULTS.md · MECHINTERP_M1_RESULTS.md · HARDENING_STATUS.md
```

## Setup

```bash
conda create -n ridae python=3.11 -y && conda activate ridae
pip install -r requirements.txt
```

Core detector experiments run on CPU in minutes; the 7B PRM panel and mech-interp
(M2/M3/steer) need a single A100-40GB (bf16, no quantization).

## Honest limitations

- The core representation result rests on one corpus (ProcessBench); PRM800K structurally
  lacks the phenomenon (curated paths → only 23 Type-B), so a matched second corpus would have
  to be *generated*.
- The detector is a small head on a small encoder — the contribution is the **protocol and the
  mechanistic account**, not a competitive verifier.
- Checkpoints are selected on the same held-out split the probe evaluates (uniform across
  phases; disclosed).

*(Repository URL remains `github.com/akshitveena/ridae`; "ParaLLax" is the project/paper name.)*
