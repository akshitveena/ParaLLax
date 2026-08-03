"""
approach_judge.py — Claude conceptual-approach judge (Type A vs B labeling).

The keyword classifier labels APPROACH by surface vocabulary, which is unreliable on
competition math (mostly 'mixed'/'unknown') — and approach is a CONCEPTUAL property
(the composition of the reasoning), not the words. So this step asks Claude, for each
CORRECT candidate, whether its reasoning used the SAME conceptual approach as the
reference (gold) solution, or a genuinely DIFFERENT one. That verdict drives Type A vs B:

    correct + SAME approach as gold      -> Type A
    correct + DIFFERENT approach as gold -> Type B  (wrong-approach-right-answer)

It writes `approach_matches_gold` (bool) back onto each judged candidate; data_pipeline
honours that override. It DOES NOT touch `answer_correct` — so it composes with a file
already processed by verify_answers.py without re-spending on verification.

Only CORRECT candidates are judged (Type A/B only exists among correct answers), and
only when a gold_solution is present to compare against — keeping cost minimal.

Run AFTER verify_answers.py, BEFORE data_pipeline.py:
    python api_generation/approach_judge.py --in data/raw/candidates_raw.jsonl \
                                            --out data/raw/candidates_raw.jsonl
Requires ANTHROPIC_API_KEY.  (Later we'll merge this with verify_answers into one call.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402

MODEL = "claude-sonnet-4-6"
JUDGE_SYSTEM = (
    "You are a mathematics reasoning analyst. You are given a problem, a REFERENCE "
    "(canonical) solution, and a STUDENT solution that reached the correct answer. "
    "Decide whether the student used the SAME underlying conceptual approach/strategy "
    "as the reference, or a genuinely DIFFERENT one. Judge the METHOD (e.g. algebraic "
    "vs geometric vs combinatorial vs induction vs generating-functions vs casework), "
    "NOT wording, notation, ordering, or formatting. Two differently-worded solutions "
    "that follow the same strategy are SAME. Reply with EXACTLY one token: SAME or DIFFERENT."
)


def build_prompt(problem: str, gold_solution: str, response: str) -> str:
    return (f"PROBLEM:\n{problem}\n\nREFERENCE SOLUTION:\n{gold_solution[:3000]}\n\n"
            f"STUDENT SOLUTION:\n{response[:3000]}\n\n"
            "Did the student use the SAME conceptual approach as the reference, or a "
            "DIFFERENT one? Answer SAME or DIFFERENT.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--poll_seconds", type=int, default=30)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment."); sys.exit(1)

    records = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]

    requests, index = [], {}
    n_skip_wrong, n_skip_nogold = 0, 0
    for ri, rec in enumerate(records):
        gold_solution = str(rec.get("gold_solution", "")).strip()
        for ci, cand in enumerate(rec.get("candidates", [])):
            # Only correct candidates matter for A vs B; need answer_correct from verify step.
            if not cand.get("answer_correct"):
                n_skip_wrong += 1
                continue
            if not gold_solution:
                n_skip_nogold += 1          # can't compare approach without a reference
                continue
            resp = cand.get("response_text") or T.split_think_response(cand.get("full_text", ""))[1]
            if not resp.strip():
                continue
            cid = f"approach-{ri}-{ci}"
            index[cid] = (ri, ci)
            requests.append({"custom_id": cid, "params": {
                "model": MODEL, "max_tokens": args.max_tokens, "system": JUDGE_SYSTEM,
                "messages": [{"role": "user",
                              "content": build_prompt(rec.get("problem", ""), gold_solution, resp)}]}})

    print(f"[approach] correct candidates to judge: {len(requests)}  "
          f"(skipped {n_skip_wrong} wrong, {n_skip_nogold} no-gold-solution)")
    if not requests:
        print("[approach] nothing to judge. (Did you run verify_answers first? "
              "It sets answer_correct.)")
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                                  encoding="utf-8")
        return

    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"[approach] batch {batch.id} submitted — polling every {args.poll_seconds}s")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(args.poll_seconds)

    same = diff = 0
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        ri, ci = index[result.custom_id]
        text = "".join(blk.text for blk in result.result.message.content
                       if getattr(blk, "type", None) == "text").strip().upper()
        is_same = text.startswith("SAME")
        # SAME -> approach matches gold -> Type A ; DIFFERENT -> diverges -> Type B
        records[ri]["candidates"][ci]["approach_matches_gold"] = bool(is_same)
        records[ri]["candidates"][ci]["approach_judge"] = "same" if is_same else "different"
        same += int(is_same); diff += int(not is_same)

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                              encoding="utf-8")
    print(f"[approach] judged {same+diff}: SAME(->Type A)={same}  DIFFERENT(->Type B)={diff}")
    print(f"[approach] projected Type-B rate among judged correct: "
          f"{100*diff/max(same+diff,1):.0f}%")
    print(f"[approach] wrote {args.out}")
    print(f"[approach] NEXT: python main/data_pipeline.py --raw {args.out} --out_dir /tmp/pilot_proc --allow_small")


if __name__ == "__main__":
    main()
