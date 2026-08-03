"""
prm800k_check.py — does 'wrong-path-right-answer' exist in PRM800K? (human labels, FREE)

PRM800K (tasksource/PRM800K mirror) is a TREE of step-completions: at each step the
labeler rates candidate completions (-1 bad / 0 neutral / +1 good) and picks a
`chosen_completion` to continue. We reconstruct the CHOSEN solution path, decide if it
reached the correct final answer (vs `ground_truth_answer`), and flag human Type B as:

    answer-correct  AND  the chosen path contains a step rated -1

No API spend — this reads the human ratings only. Caveat: because the labeler SELECTS
good steps, the chosen path is curated sound, so this base rate is a LOWER bound on the
phenomenon (the opposite of ProcessBench's error-curated construction).

Run:
    python api_generation/prm800k_check.py --limit 4000
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402


def _as_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return str(x.get("text") or x.get("completion") or "")
    return "" if x is None else str(x)


def reconstruct(label: dict):
    """Return (list[(text, rating)], finish_reason) for the CHOSEN path."""
    steps = []
    for st in label.get("steps", []) or []:
        comps = st.get("completions") or []
        ci = st.get("chosen_completion")
        if ci is not None and 0 <= ci < len(comps):
            steps.append((_as_text(comps[ci].get("text")), comps[ci].get("rating")))
        elif st.get("human_completion"):
            steps.append((_as_text(st["human_completion"]), 1))   # human-written = treated correct
    return steps, label.get("finish_reason")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=4000, help="solutions to scan (0 = all)")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("tasksource/PRM800K", split=args.split, streaming=True)

    n = n_qc = n_solution = n_correct = 0
    finish = Counter()
    type_b = 0                 # correct + has a -1 chosen step
    correct_with_bad_examples = []
    for row in ds:
        if args.limit and n >= args.limit:
            break
        if row.get("is_quality_control_question") or row.get("is_initial_screening_question"):
            n_qc += 1
            continue
        n += 1
        q = row.get("question") or {}
        gt = str(q.get("ground_truth_answer", "")).strip()
        steps, fr = reconstruct(row.get("label") or {})
        finish[str(fr)] += 1
        if fr != "solution" or not steps:
            continue                                       # never reached a final answer
        n_solution += 1
        solution = "\n".join(t for t, _ in steps)
        final = T.extract_answer(solution)[0]
        correct = bool(final) and T.answers_match(final, gt)[0]
        if not correct:
            continue
        n_correct += 1
        ratings = [r for _, r in steps if r is not None]
        if any(r == -1 for r in ratings):                  # wrong-path-right-answer (human)
            type_b += 1
            if len(correct_with_bad_examples) < 4:
                bad_i = next(i for i, (_, r) in enumerate(steps) if r == -1)
                correct_with_bad_examples.append(
                    (q.get("problem", "")[:90], bad_i, steps[bad_i][0][:120]))

    print("=" * 66)
    print("  PRM800K — does wrong-path-right-answer exist? (human labels, free)")
    print("=" * 66)
    print(f"  scanned {n} solutions (skipped {n_qc} QC/screening)")
    print(f"  finish_reason: {dict(finish)}")
    print(f"  reached a final answer (finish=solution): {n_solution}")
    print(f"  of those, ANSWER-CORRECT (vs ground_truth): {n_correct}")
    print(f"  of correct, chosen path has a -1 step  -> Type B: {type_b}  "
          f"({100*type_b/max(n_correct,1):.1f}% of correct)")
    if correct_with_bad_examples:
        print("\n  sample wrong-path-right-answer (human -1 in a correct chosen path):")
        for prob, i, txt in correct_with_bad_examples:
            print(f"    - [{prob}...] bad@step{i}: {txt}...")
    print("=" * 66)
    print("  NOTE: chosen paths are labeler-SELECTED good, so this is a LOWER BOUND;")
    print("  the deterministic answer-matcher also undercounts correct (LaTeX). ")


if __name__ == "__main__":
    main()
