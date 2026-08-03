"""
verify_bestofn.py — Claude-verify correctness of generated best-of-N candidates.

For hard datasets (OmniMath/OlympiadBench) the deterministic matcher badly undercounts
symbolic answers, so the `correct` labels from generation are unreliable. This fixes them:
deterministic-first (free), then Claude (Batch API, YES/NO) only on the residual it can't
resolve. Updates each candidate's `correct` label in place.

Prints the number of Claude calls (and a cost estimate) BEFORE submitting.

    python api_generation/verify_bestofn.py --in data/raw/qwen_bestofn.jsonl \
                                            --out data/raw/qwen_bestofn.jsonl
Requires ANTHROPIC_API_KEY.
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
    "You are a mathematics answer grader. Given a reference (gold) answer and a candidate's "
    "final answer, decide whether they are mathematically EQUIVALENT — the same value(s) or "
    "the same set of solutions — ignoring formatting, notation, variable names, ordering, and "
    "trivial rephrasing. Reply with EXACTLY one token: YES or NO."
)
# Batch API Sonnet: $0.75/M in, $3.75/M out (50% off). ~1000 in + 8 out per call.
COST_PER_CALL = (1000 * 0.75 + 8 * 3.75) / 1e6


def build_prompt(problem, gold, cand, tail):
    return (f"PROBLEM:\n{problem[:1500]}\n\nREFERENCE ANSWER:\n{gold}\n\n"
            f"CANDIDATE FINAL ANSWER:\n{cand}\n\n(end of the candidate's solution:\n...{tail}\n)\n\n"
            "Are the candidate's answer and the reference mathematically equivalent? YES or NO.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/qwen_bestofn.jsonl")
    ap.add_argument("--out", default="data/raw/qwen_bestofn.jsonl")
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--poll_seconds", type=int, default=30)
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment."); sys.exit(1)

    records = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]

    requests, index = [], {}
    n_det = n_noans = 0
    for ri, rec in enumerate(records):
        gold = str(rec.get("gold_answer", ""))
        for ci, c in enumerate(rec.get("candidates", [])):
            resp = c.get("response_text") or ""
            ans = c.get("answer") or T.extract_answer(resp)[0]
            if not ans:
                c["correct"], c["correct_method"] = False, "none"; n_noans += 1; continue
            det, method = T.answers_match(ans, gold)
            if det:
                c["correct"], c["correct_method"] = True, method; n_det += 1; continue
            cid = f"v-{ri}-{ci}"; index[cid] = (ri, ci)
            requests.append({"custom_id": cid, "params": {
                "model": MODEL, "max_tokens": args.max_tokens, "system": JUDGE_SYSTEM,
                "messages": [{"role": "user",
                              "content": build_prompt(rec.get("problem", ""), gold, ans, resp[-300:])}]}})

    est = len(requests) * COST_PER_CALL
    print(f"[verify] deterministic-correct={n_det}  no-answer={n_noans}  "
          f"-> Claude judge needed for {len(requests)}")
    print(f"[verify] estimated Batch API cost: ${est:.2f}")
    if requests and not args.yes:
        if input("[verify] proceed? [y/N] ").strip().lower() != "y":
            print("[verify] aborted."); return

    if requests:
        import anthropic
        client = anthropic.Anthropic()
        batch = client.messages.batches.create(requests=requests)
        print(f"[verify] batch {batch.id} submitted — polling every {args.poll_seconds}s")
        while True:
            b = client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(args.poll_seconds)
        yes = 0
        for result in client.messages.batches.results(batch.id):
            ri, ci = index[result.custom_id]
            verdict = False
            if result.result.type == "succeeded":
                text = "".join(blk.text for blk in result.result.message.content
                               if getattr(blk, "type", None) == "text").strip().upper()
                verdict = text.startswith("YES")
            records[ri]["candidates"][ci]["correct"] = verdict
            records[ri]["candidates"][ci]["correct_method"] = "llm_judge"
            yes += int(verdict)
        print(f"[verify] Claude judged {len(index)}: {yes} correct, {len(index)-yes} wrong")

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                              encoding="utf-8")
    total_correct = sum(bool(c.get("correct")) for r in records for c in r.get("candidates", []))
    total = sum(len(r.get("candidates", [])) for r in records)
    print(f"[verify] wrote {args.out}  |  {total_correct}/{total} candidates correct (verified)")
    print(f"[verify] NEXT: python main/layer_eval.py --data {args.out}")


if __name__ == "__main__":
    main()
