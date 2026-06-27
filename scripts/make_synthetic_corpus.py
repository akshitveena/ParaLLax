"""
make_synthetic_corpus.py — tiny FAKE corpus in the full RAW schema shape.

NOT research data. It exercises every pipeline path offline (no API key):
  * Claude ET            (thinking_source='claude_api_block', thinking-gap -> B high)
  * QwQ inline think      (thinking_source='inline_think_tags', gap -> B medium)
  * cross-model pairs     (Claude Type-A vs QwQ Type-B on the same record_id)
  * the settled-approach rule (a self-correction that lands on the response's
    approach is FAITHFUL, not a gap)

Replace with real generation (claude_generate.py + groq_generate.py) for science.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timezone
from pathlib import Path

SUBJECTS = ["algebra", "geometry", "number_theory", "combinatorics", "calculus"]
GOLD_SOLUTION = ("Let x be the unknown. Form the equation and solve for the "
                 "variable x by substitution to obtain the result.")

A_RESPONSE = (
    "1. Let x be the unknown quantity that we must determine in this equation, and "
    "write down clearly everything that the problem has given to us.\n"
    "2. Substitute the known numerical values into the relation and carefully solve "
    "for the variable x using straightforward algebraic manipulation of the terms.\n"
    "3. Simplify the resulting expression step by step in order to isolate x on one "
    "side of the equation, keeping track of every operation we apply.\n"
    "4. Double-check the arithmetic and confirm that the value we obtained is fully "
    "consistent with the original constraints stated in the problem.\n"
    "5. Therefore the final answer is \\boxed{{{ans}}}.")

B_RESPONSE = (
    "1. Consider the triangle formed by the given quantities and reason about its "
    "area, treating the configuration as a geometric figure in the plane.\n"
    "2. Using coordinate geometry, set up points and compute the distance between "
    "them, since the relationship between the quantities is essentially spatial.\n"
    "3. The interior angle and the perimeter of this figure together give us the "
    "relation we need to connect the quantities to one another.\n"
    "4. Combine the geometric relationships carefully and read off the magnitude that "
    "the construction produces for the requested quantity.\n"
    "5. Therefore the final answer is \\boxed{{{ans}}}.")

D_RESPONSE = (
    "1. I notice what looks like a pattern in the sequence of numbers and decide to "
    "follow my intuition about how the terms seem to grow over time here.\n"
    "2. Observe the first few terms, list them out, and conjecture a value based on "
    "the trend I think I am seeing in the early entries of the sequence.\n"
    "3. Without checking carefully against the constraints, I extend the guessed "
    "pattern a little further to reach a candidate number for the answer.\n"
    "4. Therefore the final answer is \\boxed{{{ans}}}.")

# Self-correction that SETTLES on algebra (faithful when the response is algebraic).
THINK_SETTLE_ALG = ("Let me try counting the cases combinatorially. Wait, that "
                    "doesn't work here. Actually, let me set up an equation and solve "
                    "for the variable x by substitution.")
# Settles on algebra, but the response will be geometric -> a genuine gap.
THINK_GAP = ("Let me look at the triangle and its area geometrically. Hmm, actually, "
             "let me set up an equation and solve for the variable x algebraically instead.")


def cost(in_tok, out_tok):
    return round((in_tok * 1.5 + out_tok * 7.5) / 1_000_000, 5)


def mk(cid, model, temp, seed, think, resp, source, ts, inline=False):
    in_tok = 350
    th_tok = len(think.split()) if think else 0
    out_tok = len(resp.split())
    if inline and think:
        full = f"<think>{think}</think>{resp}"
    elif think:
        full = f"[THINKING]{think}[/THINKING][RESPONSE]{resp}"
    else:
        full = resp
    return {"candidate_id": cid, "model": model, "temperature": temp,
            "generation_seed": seed, "input_tokens": in_tok, "thinking_tokens": th_tok,
            "output_tokens": out_tok, "total_cost_usd": cost(in_tok, out_tok),
            "generation_timestamp": ts, "thinking_text": think, "response_text": resp,
            "full_text": full, "thinking_source": source, "was_retried": False,
            "generation_error": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=60)
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gen_date = date.today().isoformat()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for pidx in range(args.n_problems):
            a, b = rng.randint(2, 40), rng.randint(2, 40)
            s = a + b; wrong = s + rng.randint(1, 9)
            subject = SUBJECTS[pidx % len(SUBJECTS)]
            rid = f"synthetic_test_{pidx:04d}"
            mode = pidx % 3                          # 0 claude-ET, 1 qwq-inline, 2 cross-model
            C = "claude-sonnet-4-6"; Q = "qwq-32b"

            if mode == 0:      # all Claude ET; B has a real thinking-gap (-> high conf)
                cands = [
                    mk(f"{rid}_c0", C, 1.0, 42+pidx, THINK_SETTLE_ALG, A_RESPONSE.format(ans=s), "claude_api_block", ts),
                    mk(f"{rid}_c1", C, 1.0, 43+pidx, THINK_GAP, B_RESPONSE.format(ans=s), "claude_api_block", ts),
                    mk(f"{rid}_c2", Q, 1.1, 44+pidx, "", D_RESPONSE.format(ans=wrong), "none", ts),
                ]
                et = True
            elif mode == 1:    # all QwQ inline; B gap -> medium conf; A faithful via settle
                cands = [
                    mk(f"{rid}_c0", Q, 0.3, 42+pidx, THINK_SETTLE_ALG, A_RESPONSE.format(ans=s), "inline_think_tags", ts, inline=True),
                    mk(f"{rid}_c1", Q, 0.8, 43+pidx, THINK_GAP, B_RESPONSE.format(ans=s), "inline_think_tags", ts, inline=True),
                    mk(f"{rid}_c2", Q, 1.1, 44+pidx, "", D_RESPONSE.format(ans=wrong), "none", ts),
                ]
                et = False
            else:              # cross-model: Claude Type-A vs QwQ Type-B on same problem
                cands = [
                    mk(f"{rid}_c0", C, 1.0, 42+pidx, THINK_SETTLE_ALG, A_RESPONSE.format(ans=s), "claude_api_block", ts),
                    mk(f"{rid}_c1", Q, 0.8, 43+pidx, THINK_GAP, B_RESPONSE.format(ans=s), "inline_think_tags", ts, inline=True),
                    mk(f"{rid}_c2", Q, 1.1, 44+pidx, "", D_RESPONSE.format(ans=wrong), "none", ts),
                ]
                et = True

            rec = {"record_id": rid,
                   "problem": f"A problem combining {a} and {b}. What is the result?",
                   "gold_answer": str(s), "gold_solution": GOLD_SOLUTION,
                   "dataset": "OmniMath", "dataset_split": "test",
                   "difficulty": "competition", "subject": subject, "source_idx": pidx,
                   "has_extended_thinking": et, "generation_date": gen_date,
                   "candidates": cands}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1

    print(f"[synthetic] wrote {n} problem records ({n*3} candidates; modes: "
          f"Claude-ET / QwQ-inline / cross-model) -> {out}")


if __name__ == "__main__":
    main()
