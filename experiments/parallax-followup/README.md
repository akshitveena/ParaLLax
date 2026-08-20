# ParaLLax follow-up — reviewer asks #1–#6

Runnable designs + reference code for the six asks. **Nothing here has been executed.** The real
ParaLLax code lives in your container (`/workspace/ridae/experiments/`), not in this workspace, so each
script has an `# ==== ADAPTER ====` block: fill those stubs against your existing loaders/models and the
experiment logic below them runs unchanged.

## Read first
`PLAN.md` — the designs, the integrity flags, and **§0: why the pasted `rl_policy_gaming.py` run is a
mislabelled positive, not a null.** Start there.

## Map: ask → file
| ask | file | cost |
|---|---|---|
| #1 policy gaming (RL/BoN) + #6 close the loop | `experiments/e1_policy_gaming_v2.py` | CPU (BoN, text attack); GPU optional (PPO) |
| #2 factored architecture in main text | `experiments/e2_factored_architecture.py` + `paper/factored_architecture_maintext.md` | medium |
| #3 second multi-model corpus | `experiments/e3_second_corpus.py` | high (generation + judging) |
| #4 non-linear confound control | `experiments/e4_nonlinear_control.py` | cheap |
| #5 surgical causal erasure | `experiments/e5_surgical_erasure.py` | medium |

## Suggested order
E1 (BoN + text attack) → E4 → E2 → E5 → E3. Rationale in PLAN.md.

## Three things that are not optional
1. **Fill `<PLACEHOLDER>`s from real runs only.** No invented numbers reach the paper.
2. **Two asks are conditional, by the paper's own admission:** the factored architecture (Appendix N
   flags it may not match e2e validity) and surgical erasure (the Hydra/self-repair finding may make it
   impossible). Both scripts are built so a negative result is reportable and, in E5's case, a *stronger*
   claim than the current paper makes. Don't tune them into a win.
3. **E1's exploit is a difficulty *shift* with unchanged gold validity — direction-agnostic.** For a
   soundness reward the shift is toward *shorter*, not longer; measure the gold Type-A rate, not just
   surface stats. See PLAN.md §0.

## Deps
`numpy scipy scikit-learn torch sentence-transformers` (+ `trl` only for E1's optional PPO scaffold,
`sympy` for E3 answer-matching, a served DeepSeek-R1 / Qwen2.5-Math for E3 generation).
