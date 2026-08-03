"""
processbench_ingest.py — turn ProcessBench (human-labeled) into RiDAE RAW records.

ProcessBench gives, per solution: problem, step-segmented `steps`, `final_answer_correct`,
and `label` = first human-flagged error step (-1 if clean). We keep ANSWER-CORRECT
solutions and map the HUMAN label straight into our schema via the two overrides
data_pipeline already honours — no LLM judge, no spend, human ground truth:

    answer_correct   = True                     (we keep only final_answer_correct)
    type_b_mechanism = 'sound_canonical'  if label == -1   -> Type A
                       'flawed_lucky'     if label >= 0     -> Type B

So the training corpus carries HUMAN A/B labels. One candidate per problem (ProcessBench
is single-solution), so there are no within-problem contrastive pairs — reconstruction +
MNR still train fully; the triplet term is simply 0 (train.py handles that). The labels
exist for the POST-training probes (what did z learn), per the two-things split.

Run:
    python api_generation/processbench_ingest.py --splits omnimath,olympiadbench,math,gsm8k
    python main/data_pipeline.py --raw data/raw/processbench_raw.jsonl \
                                 --out_dir data/processed_pb --allow_small
    python main/train.py --data_dir data/processed_pb
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="omnimath,olympiadbench,math,gsm8k")
    ap.add_argument("--limit_per_split", type=int, default=0, help="0 = all answer-correct")
    ap.add_argument("--out", default="data/raw/processbench_raw.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    gen_date = date.today().isoformat()
    n_rec = n_a = n_b = 0
    idx = 0
    with out.open("w", encoding="utf-8") as fh:
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            ds = load_dataset("Qwen/ProcessBench", split=split)
            kept = 0
            for r in ds:
                if not r["final_answer_correct"]:
                    continue
                if args.limit_per_split and kept >= args.limit_per_split:
                    break
                kept += 1
                steps = list(r["steps"])
                response = "\n".join(steps)
                human_B = r["label"] >= 0
                mech = "flawed_lucky" if human_B else "sound_canonical"
                n_a += int(not human_B); n_b += int(human_B)
                cand = {
                    "candidate_id": f"{r['id']}_c0",
                    "model": f"pb/{r.get('generator', 'unknown')}",
                    "temperature": 0.0, "reasoning_effort": None, "generation_seed": 0,
                    "input_tokens": 0, "thinking_tokens": 0, "output_tokens": 0,
                    "total_cost_usd": 0.0, "generation_timestamp": "",
                    "thinking_text": "", "response_text": response, "full_text": response,
                    "thinking_source": "none",
                    # ---- human-label injections (honoured by data_pipeline) ----
                    "answer_correct": True,             # we kept only final_answer_correct
                    "answer_match_method": "processbench_human",
                    "type_b_mechanism": mech,           # human A/B -> authoritative
                    "human_error_step": r["label"],     # provenance (kept for reference)
                    "generation_error": None,
                }
                rec = {
                    "record_id": r["id"], "problem": r["problem"],
                    "gold_answer": "", "gold_solution": "",
                    "dataset": f"ProcessBench_{split}", "dataset_split": split,
                    "difficulty": split, "subject": "other", "source_idx": idx,
                    "has_extended_thinking": False, "generation_date": gen_date,
                    "candidates": [cand],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rec += 1; idx += 1

    print(f"[pb-ingest] wrote {n_rec} records -> {out}")
    print(f"[pb-ingest] human labels: Type A (clean)={n_a}  Type B (wrong-path)={n_b}  "
          f"({100*n_b/max(n_rec,1):.0f}% B)")
    print(f"[pb-ingest] NEXT: python main/data_pipeline.py --raw {out} "
          f"--out_dir data/processed_pb --allow_small")


if __name__ == "__main__":
    main()
