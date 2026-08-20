"""
E1 — Closed-loop policy gaming, corrected.  (Reviewer asks #1 and #6.)

Fixes the mis-specified `rl_policy_gaming.py` run:
  * scores with the length-confounded verifiers (Math-Shepherd-7B / RLHFlow-8B / your own
    UNCONTROLLED Step-SDAE), NOT raw SBERT (which is the *floor*, controlled f1B 0.291).
  * measures the GOLD Type-A rate of selected candidates — the measurement that turns
    "apparent difficulty shifted" into "reward-hacked without improving validity".
  * is sign-agnostic about the exploit direction. A soundness-maximizing policy against a
    verifier that reads length-as-flaw games toward SHORTER (paper Fig 7: +difficulty raises
    P(TypeB); length↔label point-biserial = +0.415), so do NOT hard-code "longer = gaming".

PRE-REGISTRATION (write result honestly whatever happens):
  H1 (exploit): uncontrolled verifier's selected set shifts apparent difficulty (|Δlength| large,
       paired-bootstrap CI excludes 0) while gold Type-A rate is statistically indistinguishable
       from random selection (CI includes 0).
  H2 (control works): controlled verifier's selected set shows neither a difficulty shift nor a
       Type-A change vs random.
  KILL: if the uncontrolled set has a HIGHER gold Type-A rate (CI excludes 0 upward), that is real
       validity signal, not gaming — report it as such. If the controlled set also shifts
       difficulty, control is incomplete — report that too.

Three designs, cheapest first. Best-of-N (Design 1) + the text-level attack (Design 2) already
close the loop on CPU; the PPO scaffold (Design 3) is optional/GPU and only for a live-agent demand.
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from typing import Callable, Sequence
import numpy as np

# ============================== ADAPTER ==============================
# Wire these to your existing repo. Return plain numpy / python types so the logic below is
# framework-agnostic. Everything downstream depends ONLY on these signatures.

@dataclass
class Candidate:
    problem_id: str
    text: str
    steps: list[str]          # segmented reasoning steps
    length: float             # response length in tokens (the paper's confound #1)
    latex_density: float      # LaTeX density (confound #2)
    n_steps: int              # step count (confound #3)
    dataset: str              # source dataset (confound #4)
    gold_type_a: int          # 1 == sound (Type A), 0 == Type B. From human/gold judge ONLY.

def load_answer_correct_candidates() -> list[Candidate]:
    """All answer-correct ProcessBench candidates, >= N per problem where possible.
    Best-of-N groups by `problem_id`. gold_type_a must be the GOLD label, never the verifier's."""
    raise NotImplementedError

# A scorer maps a candidate -> a scalar the policy MAXIMIZES (higher == "looks more sound").
# For a PRM that outputs P(TypeB), pass `1 - P(TypeB)`. Keep the reward orientation identical
# across scorers so the comparison is apples-to-apples.
Scorer = Callable[[Candidate], float]

def uncontrolled_verifier() -> tuple[str, Scorer]:
    """The length-confounded verifier under attack. Use Math-Shepherd-7B / RLHFlow-8B, and ALSO
    your own Step-SDAE evaluated WITHOUT residualization. Return (name, scorer)."""
    raise NotImplementedError

def controlled_verifier() -> tuple[str, Scorer]:
    """Same verifier with the 4-confound residualization applied to its representation before the
    validity head (the paper's confound-controlled protocol). Return (name, scorer)."""
    raise NotImplementedError

# For Design 2 only:
def inflate_difficulty(text: str, dose: int) -> str:
    """Return `text` with `dose` validity-preserving difficulty inflations applied: append
    logically-inert LaTeX-heavy restatement steps, re-typeset bare numbers as $...$, pad
    equivalent reformulations. MUST NOT change the mathematical argument."""
    raise NotImplementedError

def gold_validity_unchanged(original: str, inflated: str) -> bool:
    """Gold judge confirms the inflation did not change soundness. Drop any pair where it did."""
    raise NotImplementedError

def to_candidate(problem_id: str, text: str, dataset: str, gold_type_a: int) -> Candidate:
    """Segment/measure a raw solution string into a Candidate (reuse your ProcessBench pipeline)."""
    raise NotImplementedError
# ============================ END ADAPTER ============================


def _paired_bootstrap_ci(diffs: np.ndarray, iters: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% CI of paired differences, bootstrapped over the unit of analysis (problems)."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.array([rng.choice(diffs, n, replace=True).mean() for _ in range(iters)])
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# --------------------------- Design 1: Best-of-N ---------------------------
def best_of_n(cands: list[Candidate], scorer: Scorer, n: int, rng: np.random.Generator) -> list[Candidate]:
    """Per problem, sample n candidates and return argmax under `scorer` (random tie-break)."""
    by_problem: dict[str, list[Candidate]] = {}
    for c in cands:
        by_problem.setdefault(c.problem_id, []).append(c)
    selected = []
    for group in by_problem.values():
        if len(group) < 2:
            continue
        pool = list(rng.choice(group, size=min(n, len(group)), replace=False))
        scores = np.array([scorer(c) for c in pool])
        selected.append(pool[int(rng.choice(np.flatnonzero(scores == scores.max())))])
    return selected


def _surface(sel: list[Candidate]) -> dict[str, float]:
    return {
        "length": float(np.mean([c.length for c in sel])),
        "latex_density": float(np.mean([c.latex_density for c in sel])),
        "n_steps": float(np.mean([c.n_steps for c in sel])),
        "gold_type_a_rate": float(np.mean([c.gold_type_a for c in sel])),  # <-- the key addition
    }


def run_best_of_n(n: int = 4, seeds: Sequence[int] = range(5)) -> dict:
    cands = load_answer_correct_candidates()
    scorers = {"random": (lambda c: 0.0), "uncontrolled": uncontrolled_verifier()[1],
               "controlled": controlled_verifier()[1]}
    # Random must break ties uniformly -> handled inside best_of_n via equal scores.
    rows: dict[str, list[dict]] = {k: [] for k in scorers}
    per_seed_selected: dict[str, list[list[Candidate]]] = {k: [] for k in scorers}
    for s in seeds:
        rng = np.random.default_rng(s)
        for name, sc in scorers.items():
            sel = best_of_n(cands, sc, n, np.random.default_rng(s))  # same subsample seed across scorers
            per_seed_selected[name].append(sel)
            rows[name].append(_surface(sel))
    # aggregate mean +/- std across seeds
    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for name, rs in rows.items():
        agg[name] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                     for k in rs[0]}
    # paired bootstrap: exploit metric = Δlength and Δgold_type_a of a scorer vs random, per problem,
    # averaged over seeds. (Uses seed 0's problem alignment for the paired unit.)
    def paired_vs_random(name: str, field: str) -> tuple[float, float, float]:
        # align by problem_id within each seed, then pool paired diffs
        diffs = []
        for s_idx in range(len(list(seeds))):
            r = {c.problem_id: c for c in per_seed_selected["random"][s_idx]}
            for c in per_seed_selected[name][s_idx]:
                if c.problem_id in r:
                    diffs.append(getattr(c, field) - getattr(r[c.problem_id], field))
        return _paired_bootstrap_ci(np.array(diffs, float))
    exploit = {name: {"delta_length": paired_vs_random(name, "length"),
                      "delta_gold_type_a": paired_vs_random(name, "gold_type_a")}
               for name in ("uncontrolled", "controlled")}
    return {"n": n, "surface": agg, "exploit_vs_random": exploit}


# --------------------- Design 2: text-level adversarial attack ---------------------
def run_text_attack(doses: Sequence[int] = (0, 1, 2, 4, 8), seed: int = 0) -> dict:
    """Fig-7-at-the-token-level: inflate apparent difficulty of GOLD-SOUND solutions with
    validity-preserving edits and watch each verifier's score. Uncontrolled should move
    monotonically with dose; controlled should stay flat. Strongest single result, no training."""
    cands = [c for c in load_answer_correct_candidates() if c.gold_type_a == 1]
    unc_name, unc = uncontrolled_verifier()
    ctl_name, ctl = controlled_verifier()
    curves = {unc_name: {}, ctl_name: {}}
    kept = 0
    for c in cands:
        base_ok = True
        for dose in doses:
            inflated = inflate_difficulty(c.text, dose) if dose else c.text
            if dose and not gold_validity_unchanged(c.text, inflated):
                base_ok = False
                break
        if not base_ok:
            continue
        kept += 1
        for dose in doses:
            inflated = inflate_difficulty(c.text, dose) if dose else c.text
            cc = to_candidate(c.problem_id, inflated, c.dataset, gold_type_a=1)
            curves[unc_name].setdefault(dose, []).append(unc(cc))
            curves[ctl_name].setdefault(dose, []).append(ctl(cc))
    def summarize(cv):
        return {d: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) for d, v in sorted(cv.items())}
    return {"n_solutions_kept": kept,
            "doses": list(doses),
            "uncontrolled_score_by_dose": summarize(curves[unc_name]),
            "controlled_score_by_dose": summarize(curves[ctl_name]),
            "note": "monotone rise in uncontrolled + flat controlled == exploit channel confirmed; "
                    "validity held constant by construction (gold_validity_unchanged filter)."}


# --------------------------- Design 3: PPO/GRPO scaffold (optional, GPU) ---------------------------
def ppo_scaffold_notes() -> str:
    return (
        "OPTIONAL / GPU. Only run if reviewers demand a live agent; Designs 1+2 already close the loop.\n"
        "  policy: small math LM (e.g. Qwen2.5-Math-1.5B-Instruct).\n"
        "  reward: verifier soundness score (1 - P(TypeB)); run twice, uncontrolled vs controlled.\n"
        "  algo: PPO or GRPO with a KL penalty to the SFT policy (else length collapses trivially).\n"
        "  log every step: reward, mean length, latex_density, AND held-out GOLD Type-A rate on a\n"
        "    frozen eval set of problems the policy is NOT trained on.\n"
        "  claim: under the uncontrolled reward, length/latex drift while gold Type-A is flat\n"
        "    (reward up, validity flat = hacking); under the controlled reward, no drift.\n"
        "  Use trl.PPOTrainer / GRPO; keep batch small, this is a demonstration not a SOTA run."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["bon", "text", "ppo"], default="bon")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="e1_results.json")
    args = ap.parse_args()
    if args.design == "bon":
        res = run_best_of_n(n=args.n)
    elif args.design == "text":
        res = run_text_attack()
    else:
        print(ppo_scaffold_notes()); raise SystemExit(0)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
