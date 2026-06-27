"""
verify_answers.py — LLM-judge answer verification (OmniMath "Omni-Judge" style).

Competition answers are often symbolic or multi-form (e.g. 'f(x)=0, f(x)=1-x, ...')
that string/numeric matching cannot verify. This step decides answer_correct with
Claude — but DETERMINISTIC-FIRST: the free matcher (textutils.answers_match) settles
the easy cases, and Claude is only called on the residual it can't resolve. That
keeps cost to ~$1-2 on the full corpus.

For each candidate it writes back into the RAW file:
    answer_correct       (bool)
    answer_match_method  ('exact'|'float'|'normalised'|'llm_judge'|'none')

Run BETWEEN generation and data_pipeline (recommended for OmniMath/OlympiadBench):
    python api_generation/verify_answers.py --in data/raw/candidates_raw.jsonl \
                                           --out data/raw/candidates_raw.jsonl
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
    "You are a mathematics answer grader. Given a reference (gold) answer and a "
    "candidate's final answer, decide whether they are mathematically EQUIVALENT — "
    "the same value(s) or the same set of solutions — ignoring formatting, notation, "
    "variable names, ordering, and trivial rephrasing. Reply with EXACTLY one token: "
    "YES or NO."
)


def build_prompt(problem: str, gold: str, cand_answer: str, resp_tail: str) -> str:
    return (f"PROBLEM:\n{problem}\n\nREFERENCE ANSWER:\n{gold}\n\n"
            f"CANDIDATE FINAL ANSWER:\n{cand_answer}\n\n"
            f"(end of the candidate's solution, for context:\n...{resp_tail}\n)\n\n"
            "Are the candidate's answer and the reference answer mathematically "
            "equivalent? Answer YES or NO.")


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
    n_det, n_noans, n_skip = 0, 0, 0
    for ri, rec in enumerate(records):
        gold = str(rec.get("gold_answer", ""))
        for ci, cand in enumerate(rec.get("candidates", [])):
            if cand.get("generation_error"):
                n_skip += 1
                continue
            resp = cand.get("response_text") or T.split_think_response(cand.get("full_text", ""))[1]
            ans, _ = T.extract_answer(resp)
            if not ans:                                   # no answer at all -> wrong
                cand["answer_correct"], cand["answer_match_method"] = False, "none"
                n_noans += 1
                continue
            det, method = T.answers_match(ans, gold)
            if det:                                       # free matcher already settles it
                cand["answer_correct"], cand["answer_match_method"] = True, method
                n_det += 1
                continue
            # Residual -> ask Claude.
            cid = f"verify-{ri}-{ci}"
            index[cid] = (ri, ci)
            requests.append({"custom_id": cid, "params": {
                "model": MODEL, "max_tokens": args.max_tokens, "system": JUDGE_SYSTEM,
                "messages": [{"role": "user", "content":
                              build_prompt(rec.get("problem", ""), gold, ans, resp[-300:])}]}})

    print(f"[verify] deterministic-correct={n_det}  no-answer={n_noans}  "
          f"error-skipped={n_skip}  -> Claude judge needed for {len(requests)}")

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
            records[ri]["candidates"][ci]["answer_correct"] = verdict
            records[ri]["candidates"][ci]["answer_match_method"] = "llm_judge"
            yes += int(verdict)
        print(f"[verify] Claude judged {len(index)} residual answers: {yes} correct, "
              f"{len(index)-yes} wrong")

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                              encoding="utf-8")
    print(f"[verify] wrote {args.out}")
    print(f"[verify] NEXT: python api_generation/score_candidates.py  (optional)  "
          f"then  python main/data_pipeline.py --raw {args.out}")


if __name__ == "__main__":
    main()
