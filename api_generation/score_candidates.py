"""
score_candidates.py — Claude approach disambiguator (schema Section 6).

The free keyword classifier in approach_analysis.py resolves most candidates. This
script spends a little Claude budget ONLY on the residual uncertain cases — the
ones the schema says to escalate: answer_correct == True but the response approach
is 'unknown' (so we can't tell Type A from Type B). For each, Claude labels the
conceptual approach of the response (and, for ET candidates, the thinking block).
The labels are written back into the RAW file as `approach_in_response` /
`approach_in_thinking`, which data_pipeline.py then honours.

Run BETWEEN claude_generate.py and data_pipeline.py (optional but recommended):
    python api_generation/score_candidates.py --in data/raw/candidates_raw.jsonl \
                                              --out data/raw/candidates_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402
from approach_analysis import classify_approach  # noqa: E402
from schema import APPROACHES  # noqa: E402

MODEL = "claude-sonnet-4-6"
JUDGE_SYSTEM = (
    "You are a mathematics reasoning analyst. Given a problem and a solution, "
    "identify the single dominant conceptual approach used. Answer with EXACTLY one "
    "label from this set and nothing else: "
    + ", ".join(a for a in APPROACHES if a not in ("mixed", "unknown")) + ", mixed."
)
_LABELS = set(APPROACHES)


def needs_claude(problem_rec: dict, cand: dict) -> bool:
    """Uncertain = correct answer AND keyword classifier returns 'unknown'."""
    resp = cand.get("response_text") or T.split_think_response(cand.get("full_text", ""))[1]
    ans = cand.get("answer_extracted") or T.extract_answer(resp)[0]
    correct, _ = T.answers_match(ans, problem_rec.get("gold_answer", ""))
    if not correct:
        return False
    approach, _ = classify_approach(resp)
    return approach == "unknown"


def parse_label(text: str) -> str:
    t = (text or "").strip().lower()
    for lab in _LABELS:
        if re.search(rf"\b{re.escape(lab)}\b", t):
            return lab
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--max_tokens", type=int, default=16)
    ap.add_argument("--poll_seconds", type=int, default=30)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment."); sys.exit(1)

    records = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]

    # Build the request list only for uncertain candidates.
    requests, index = [], {}
    for ri, rec in enumerate(records):
        for ci, cand in enumerate(rec.get("candidates", [])):
            if not needs_claude(rec, cand):
                continue
            resp = cand.get("response_text") or T.split_think_response(cand.get("full_text", ""))[1]
            cid = f"score-{ri}-{ci}-resp"      # custom_id: ^[a-zA-Z0-9_-]{1,64}$
            index[cid] = (ri, ci, "approach_in_response")
            requests.append({"custom_id": cid, "params": {
                "model": MODEL, "max_tokens": args.max_tokens, "system": JUDGE_SYSTEM,
                "messages": [{"role": "user", "content":
                              f"PROBLEM:\n{rec.get('problem','')}\n\nSOLUTION:\n{resp}\n\nApproach label:"}]}})
            if rec.get("has_extended_thinking") and cand.get("thinking_text"):
                cidt = f"score-{ri}-{ci}-think"
                index[cidt] = (ri, ci, "approach_in_thinking")
                requests.append({"custom_id": cidt, "params": {
                    "model": MODEL, "max_tokens": args.max_tokens, "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content":
                                  f"PROBLEM:\n{rec.get('problem','')}\n\nSOLUTION:\n{cand['thinking_text']}\n\nApproach label:"}]}})

    if not requests:
        print("[score] no uncertain candidates — keyword classifier covered everything. "
              "Nothing to do.")
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                                  encoding="utf-8")
        return

    print(f"[score] {len(requests)} uncertain approach labels to resolve via Claude")
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"[score] batch {batch.id} submitted")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(args.poll_seconds)

    filled = 0
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        ri, ci, field = index[result.custom_id]
        text = "\n".join(b.text for b in result.result.message.content
                         if getattr(b, "type", None) == "text")
        label = parse_label(text)
        if label != "unknown":
            records[ri]["candidates"][ci][field] = label
            filled += 1

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                              encoding="utf-8")
    print(f"[score] filled {filled} approach labels -> {args.out}")
    print(f"[score] NEXT: python main/data_pipeline.py --raw {args.out}")


if __name__ == "__main__":
    main()
