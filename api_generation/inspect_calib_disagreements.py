"""
inspect_calib_disagreements.py — read WHY our judge and ProcessBench humans disagree.

processbench_calibrate.py saved slim per-item verdicts but not the judge's reasoning.
Anthropic keeps batch results ~29 days, so we RE-RETRIEVE the batch by id (no re-spend)
to recover each judge reply, and join it with the human-flagged step text from
ProcessBench. Then we print the two disagreement cells so they can be adjudicated:

  UNDER-count (human:B / ours:A): show the HUMAN-flagged step + our judge's reasoning
                                  -> is it self-correction, greyzone, or a real miss?
  OVER-count  (human:A / ours:B): show OUR claimed flaw
                                  -> are we hallucinating, or catching an error humans missed?

custom_id 'pb-{i}' == line i of the calib jsonl (written in the same order).

Run:
    python api_generation/inspect_calib_disagreements.py \
        --calib data/processbench_calib.jsonl \
        --batch_id msgbatch_017PMffxkFTJ2DLpFycMTGE6 \
        --splits omnimath,olympiadbench
Requires ANTHROPIC_API_KEY (read-only batch retrieval).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/processbench_calib.jsonl")
    ap.add_argument("--batch_id", required=True)
    ap.add_argument("--splits", default="omnimath,olympiadbench")
    ap.add_argument("--reasoning_chars", type=int, default=700)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment."); sys.exit(1)

    rows = [json.loads(l) for l in Path(args.calib).read_text().splitlines() if l.strip()]

    # id -> steps, from ProcessBench (deterministic lookup by id; no re-sampling)
    from datasets import load_dataset
    steps_by_id = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        for r in load_dataset("Qwen/ProcessBench", split=split):
            steps_by_id[r["id"]] = r["steps"]

    # recover judge replies from the batch (free)
    import anthropic
    client = anthropic.Anthropic()
    judge_text = {}
    for result in client.messages.batches.results(args.batch_id):
        i = int(result.custom_id.split("-", 1)[1])
        if result.result.type == "succeeded":
            judge_text[i] = "".join(blk.text for blk in result.result.message.content
                                    if getattr(blk, "type", None) == "text").strip()
        else:
            judge_text[i] = f"[{result.result.type}]"

    under = [(i, r) for i, r in enumerate(rows)
             if r.get("human_AB") == "B" and r.get("ours_AB") == "A"]
    over = [(i, r) for i, r in enumerate(rows)
            if r.get("human_AB") == "A" and r.get("ours_AB") == "B"]

    def tail(txt):
        return txt[-args.reasoning_chars:] if txt else "(no reasoning)"

    print("#" * 74)
    print(f"  UNDER-COUNT — human flagged an error, we called it SOUND   ({len(under)})")
    print("  question: self-correction / definitional greyzone / real miss?")
    print("#" * 74)
    for i, r in under:
        idx = r.get("human_label_idx", -1)
        steps = steps_by_id.get(r["id"], [])
        flagged = steps[idx] if 0 <= idx < len(steps) else "(step text unavailable)"
        print(f"\n=== {r['id']}  human error@step {idx}  ours={r.get('our_mechanism')} ===")
        print(f"PROBLEM: {r['problem'][:200]}")
        print(f"HUMAN-FLAGGED STEP {idx}:\n  {flagged[:500]}")
        print(f"OUR JUDGE (why we said sound), tail:\n  ...{tail(judge_text.get(i,''))}")

    print("\n" + "#" * 74)
    print(f"  OVER-COUNT — human said clean, we claimed a FLAW   ({len(over)})")
    print("  question: are we hallucinating, or did humans miss it?")
    print("#" * 74)
    for i, r in over:
        print(f"\n=== {r['id']}  human=clean  ours={r.get('our_mechanism')} ===")
        print(f"PROBLEM: {r['problem'][:200]}")
        print(f"OUR JUDGE (the flaw we claim), tail:\n  ...{tail(judge_text.get(i,''))}")

    print("\n" + "=" * 74)
    print(f"  {len(under)} under + {len(over)} over = {len(under)+len(over)} to adjudicate.")
    print("  Tally each as: self_correction | greyzone | real_miss | human_error")
    print("=" * 74)


if __name__ == "__main__":
    main()
