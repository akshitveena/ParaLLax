"""
E3 — Generate a second, multi-model corpus, and test whether ParaLLax replicates.
(Reviewer ask #3.)

PRM800K cannot serve as a second corpus (Appendix J: 0.3% Type-B, because its preferred path is the
human-curated correct trajectory). Appendix J's own conclusion: "a matched second corpus would have
to be GENERATED (sample solutions, verify answers, judge validity) rather than found." This builds it.

Pipeline:
  sample K solutions/problem from long-CoT models (DeepSeek-R1, Qwen2.5-Math) on AIME + MATH-500
  -> keep answer-correct  -> segment into steps  -> mechanism judge (same rubric as the κ=0.60 judge)
  labels Type A/B  -> persist matched corpus (same schema as the ProcessBench corpus).

Then re-run the three ParaLLax tests on the NEW corpus AND cross-model (train on ProcessBench, test on
new corpus): difficulty-only null, uncontrolled vs controlled detectors, per-step vs pooled.

CAVEATS baked in (the paper already found these):
  * Appendix O: R1 <think> traces are far OOD for the PRM (P(sound) ~0.33 uniformly). Do NOT reuse
    ProcessBench step segmentation blindly; adapt for long CoT (segment the post-<think> solution, or
    the think trace with a long-CoT-aware splitter). Segmentation choice is logged as a config.
  * AIME is tiny (~30 problems/yr). MATH-500 carries the count. N is reported honestly.
  * Defensibility rests on Appendix K (judge not difficulty-biased). Re-run the K judge-confound check
    on the NEW corpus's judge labels; if the new judge shows difficulty bias, the corpus is compromised.
  PRE-REGISTER: effect REPLICATES iff on the new corpus uncontrolled ~ difficulty-null and per-step
  controlled signal survives. If it doesn't, the claim is bounded to ProcessBench — report that.
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np

# ============================== ADAPTER ==============================
@dataclass
class Problem:
    problem_id: str
    statement: str
    gold_answer: str
    dataset: str            # 'AIME' | 'MATH-500'

@dataclass
class Solution:
    problem_id: str
    model: str              # 'deepseek-r1' | 'qwen2.5-math' | ...
    text: str
    extracted_answer: str
    steps: list[str]        # post-segmentation
    type_label: int | None  # 1 == Type A, 0 == Type B, filled by the judge; None until judged
    dataset: str

def load_problems() -> list[Problem]:
    """AIME (recent years) + MATH-500 problems with gold answers."""
    raise NotImplementedError

def sample_solutions(problem: Problem, model: str, k: int, temperature: float) -> list[str]:
    """Sample k raw completions from `model` for `problem`. Long-CoT models: keep the full <think>...
    </think> answer. Provider-agnostic (vLLM / API)."""
    raise NotImplementedError

def extract_answer(text: str) -> str:
    """Pull the final boxed/⧉ answer from a completion (reuse your MATH answer extractor)."""
    raise NotImplementedError

def answers_match(pred: str, gold: str) -> bool:
    """Math-equivalence check (sympy-based), same as your ProcessBench answer-correctness filter."""
    raise NotImplementedError

def segment_steps_long_cot(text: str) -> list[str]:
    """Long-CoT-aware step segmentation. Document the choice (post-<think> solution vs think trace)."""
    raise NotImplementedError

def mechanism_judge(problem: Problem, solution_text: str, steps: list[str]) -> int:
    """The SAME LLM mechanism judge (rubric from Appendix G Table 2): returns 1 (Type A / sound) or
    0 (Type B / correct-answer-but-unsound). Must use the identical prompt/rubric as the κ=0.60 judge."""
    raise NotImplementedError

# Reuse E4's / the repo's evaluators for the replication step:
def run_parallax_tests(corpus_train, corpus_test, cross_model: bool) -> dict:
    """Difficulty-only null, uncontrolled vs controlled detectors, per-step vs pooled — on `corpus_test`.
    If cross_model, fit on ProcessBench (corpus_train) and evaluate on the new corpus (corpus_test)."""
    raise NotImplementedError

def judge_confound_check(corpus) -> dict:
    """Re-run Appendix K: is the judge reading difficulty on THIS corpus? Return {'auc_gap':..,'ci':..}."""
    raise NotImplementedError
# ============================ END ADAPTER ============================


def build_corpus(models=("deepseek-r1", "qwen2.5-math"), k: int = 8, temperature: float = 0.8,
                 seed: int = 0) -> list[Solution]:
    problems = load_problems()
    rng = np.random.default_rng(seed)   # (only for any tie-breaking; generation seeded by provider)
    corpus: list[Solution] = []
    stats = {"sampled": 0, "answer_correct": 0, "type_a": 0, "type_b": 0}
    for prob in problems:
        for model in models:
            for raw in sample_solutions(prob, model, k, temperature):
                stats["sampled"] += 1
                ans = extract_answer(raw)
                if not answers_match(ans, prob.gold_answer):
                    continue                                    # keep ONLY answer-correct (Type A or B)
                stats["answer_correct"] += 1
                steps = segment_steps_long_cot(raw)
                label = mechanism_judge(prob, raw, steps)
                stats["type_a" if label == 1 else "type_b"] += 1
                corpus.append(Solution(prob.problem_id, model, raw, ans, steps, label, prob.dataset))
    print("[build_corpus]", json.dumps(stats, indent=2))
    if stats["answer_correct"]:
        print(f"[build_corpus] Type-B fraction = {stats['type_b']/stats['answer_correct']:.3f} "
              f"(ProcessBench = 0.306; PRM800K = 0.003)")
    return corpus


def replicate(new_corpus: list[Solution], processbench_corpus) -> dict:
    jc = judge_confound_check(new_corpus)   # gate: if judge is difficulty-biased here, corpus is suspect
    within = run_parallax_tests(new_corpus, new_corpus, cross_model=False)
    cross = run_parallax_tests(processbench_corpus, new_corpus, cross_model=True)
    return {
        "judge_confound_check": jc,
        "within_new_corpus": within,
        "cross_model_processbench_to_new": cross,
        "prereg_verdict": ("REPLICATES iff within_new_corpus shows uncontrolled ~ difficulty-null and "
                           "per-step controlled signal survives; cross-model transfer strengthens it. "
                           "If not, bound the claim to ProcessBench and say so."),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="e3_corpus.jsonl")
    ap.add_argument("--results", default="e3_replication.json")
    args = ap.parse_args()
    corpus = build_corpus(k=args.k, temperature=args.temperature)
    with open(args.out, "w") as f:
        for sol in corpus:
            f.write(json.dumps(asdict(sol)) + "\n")
    # Replication requires your ProcessBench corpus handle; wire it in run_parallax_tests' adapter.
    print(f"Wrote {len(corpus)} solutions -> {args.out}. "
          f"Run replicate(corpus, processbench_corpus) to fill {args.results}.")
