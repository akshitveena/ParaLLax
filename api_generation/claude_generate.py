"""
claude_generate.py — generate reasoning candidates with Claude, in the full
RiDAE Dataset Schema (one ProblemRecord per line, 3 candidates each).

Two modes per the schema's budget plan:
  * Extended-thinking (ET): OmniMath hardest-N. Keeps thinking_text + response_text
    separately — the gap between them is the headline unfaithfulness signal.
    ET requires temperature = 1.0 (all 3 candidates), diversity comes from seeds.
  * Standard: MATH L4+L5 / OlympiadBench / OmniMath. Temperature strategy across
    the 3 candidates: c0=0.3, c1=0.8, c2=1.1 (Section 5.2).

Uses the Batch API (~50% off). Requires ANTHROPIC_API_KEY.

Pipeline:  claude_generate.py -> [score_candidates.py] -> main/data_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402

MODEL = "claude-sonnet-4-6"
SYSTEM_PROMPT = (
    "You are solving a competition mathematics problem.\n"
    "Think carefully about the approach before computing.\n"
    "Show your complete reasoning step by step.\n"
    "Each step should be on a new line.\n"
    "State your final answer clearly and enclose it in \\boxed{}.\n"
    "Do not skip steps. Do not summarise. Show everything."
)
TEMP_STRATEGY = [0.3, 0.8, 1.1]                      # Section 5.2 (standard candidates)
PRICE_IN, PRICE_OUT = 1.50, 7.50                    # USD / 1M tokens (schema cost formula)


# --------------------------------------------------------------------------- #
def load_problems(dataset: str, limit: int, hardest: bool = False,
                  min_difficulty: float | None = None,
                  max_difficulty: float | None = None) -> list[dict]:
    """Return problem dicts with the schema's problem-level fields.

    hardest=True (OmniMath only) selects the highest-difficulty problems (Claude-ET).
    min_difficulty / max_difficulty (OmniMath only) select a difficulty BAND — use a
    moderate band (e.g. <=6) for weaker local models so they actually solve enough
    problems to yield Type B (correct-via-wrong-approach), instead of all Type D.
    """
    from datasets import load_dataset
    out: list[dict] = []
    if dataset == "math":
        # lighteval/MATH was removed from the Hub; MATH-500 is the live, clean replacement
        # (direct `answer` field + integer `level`).
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.filter(lambda r: int(r.get("level", 0) or 0) >= 4)   # hard subset (L4-5)
        ds = ds.select(range(min(limit, len(ds))))
        for i, r in enumerate(ds):
            out.append(dict(record_id=f"math_test_{i:04d}", problem=r["problem"],
                            gold_answer=str(r.get("answer", "")),
                            gold_solution=str(r.get("solution", "")), dataset="MATH",
                            dataset_split="test", difficulty=str(r.get("level", "")),
                            subject=_norm_subject(r.get("subject")), source_idx=i))
    elif dataset == "olympiadbench":
        ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
        ds = ds.select(range(min(limit, len(ds))))
        for i, r in enumerate(ds):
            ans = r.get("final_answer")
            gold = ans[0] if isinstance(ans, list) and ans else str(ans)
            out.append(dict(record_id=f"olympiadbench_train_{i:04d}", problem=r["question"],
                            gold_answer=str(gold), gold_solution=str(r.get("solution", "")),
                            dataset="OlympiadBench", dataset_split="train",
                            difficulty="competition", subject=_norm_subject(r.get("subject")),
                            source_idx=i))
    elif dataset == "omnimath":
        ds = load_dataset("KbsdJames/Omni-MATH", split="test")
        rows = list(enumerate(ds))                       # keep original source_idx

        def _diff(item):
            try:
                return float(item[1].get("difficulty"))
            except (TypeError, ValueError):
                return -1.0

        # Difficulty-band filter: keep only problems in [min, max]. Use a MODERATE band
        # for weaker local models (e.g. --max_difficulty 6) so they solve enough to
        # produce Type B; leave unset (or --hardest) for Claude on the hardest tier.
        if min_difficulty is not None:
            rows = [it for it in rows if _diff(it) >= min_difficulty]
        if max_difficulty is not None:
            rows = [it for it in rows if 0 <= _diff(it) <= max_difficulty]
        if hardest:
            rows.sort(key=_diff, reverse=True)           # "hardest N"
        rows = rows[:limit]
        for i, r in rows:
            out.append(dict(record_id=f"omnimath_test_{i:04d}", problem=r["problem"],
                            gold_answer=str(r.get("answer", "")),
                            gold_solution=str(r.get("solution", "")), dataset="OmniMath",
                            dataset_split="test", difficulty=str(r.get("difficulty", "")),
                            subject=_norm_subject(r.get("domain")), source_idx=i))
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return out


def _norm_subject(s) -> str:
    s = (str(s or "")).lower()
    for key in ("algebra", "geometry", "number", "combinator", "calculus", "precalc"):
        if key in s:
            return {"number": "number_theory", "combinator": "combinatorics",
                    "precalc": "calculus"}.get(key, key)
    return "other"


def build_requests(problems, n_candidates, extended_thinking, budget_tokens, max_tokens):
    reqs = []
    for p in problems:
        for c in range(n_candidates):
            params = {"model": MODEL, "max_tokens": max_tokens,
                      "system": SYSTEM_PROMPT,
                      "messages": [{"role": "user", "content": p["problem"]}]}
            if extended_thinking:
                params["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
                params["temperature"] = 1.0            # ET forces temp=1.0
            else:
                params["temperature"] = TEMP_STRATEGY[c % len(TEMP_STRATEGY)]
            # custom_id must match ^[a-zA-Z0-9_-]{1,64}$ — no '#'/':' allowed.
            reqs.append({"custom_id": f"{p['record_id']}__c{c}", "params": params})
    return reqs


def parse_message(message):
    blocks = message.content
    think = "\n".join(b.thinking for b in blocks if getattr(b, "type", None) == "thinking")
    text = "\n".join(b.text for b in blocks if getattr(b, "type", None) == "text")
    return think.strip(), text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "olympiadbench", "omnimath"], required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--extended_thinking", action="store_true",
                    help="enable ET (use for the OmniMath ET subset)")
    ap.add_argument("--hardest", action="store_true",
                    help="OmniMath only: pick the highest-difficulty problems (hardest N)")
    ap.add_argument("--budget_tokens", type=int, default=4000)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--append", action="store_true", help="append instead of overwrite")
    ap.add_argument("--poll_seconds", type=int, default=30)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment."); sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    problems = load_problems(args.dataset, args.limit, hardest=args.hardest)
    by_id = {p["record_id"]: p for p in problems}
    print(f"[generate] {len(problems)} problems x {args.n_candidates} candidates "
          f"(extended_thinking={args.extended_thinking})")

    requests = build_requests(problems, args.n_candidates, args.extended_thinking,
                              args.budget_tokens, args.max_tokens)
    batch = client.messages.batches.create(requests=requests)
    print(f"[generate] batch {batch.id} submitted — polling every {args.poll_seconds}s")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        c = b.request_counts
        print(f"    status={b.processing_status} ok={c.succeeded} err={c.errored} proc={c.processing}")
        if b.processing_status == "ended":
            break
        time.sleep(args.poll_seconds)

    # Collect candidates grouped by record_id.
    gen_date = date.today().isoformat()
    grouped: dict[str, list[dict]] = {pid: [] for pid in by_id}
    in_tok = out_tok = 0
    for result in client.messages.batches.results(batch.id):
        rid, cidx = result.custom_id.rsplit("__c", 1)
        if result.result.type != "succeeded":
            grouped[rid].append({"candidate_id": f"{rid}_c{cidx}", "model": MODEL,
                                 "generation_error": str(result.result.type),
                                 "thinking_text": "", "response_text": "", "full_text": ""})
            continue
        msg = result.result.message
        think, resp = parse_message(msg)
        full = T.assemble_full_text(think, resp)
        u = msg.usage
        it = getattr(u, "input_tokens", 0)
        tt = getattr(u, "output_tokens", 0)  # thinking tokens are billed as output
        in_tok += it; out_tok += tt
        grouped[rid].append({
            "candidate_id": f"{rid}_c{cidx}", "model": MODEL,
            "temperature": 1.0 if args.extended_thinking else TEMP_STRATEGY[int(cidx) % 3],
            "generation_seed": int(cidx),
            "input_tokens": it, "thinking_tokens": (len(think.split()) if think else 0),
            "output_tokens": tt,
            "total_cost_usd": round((it * PRICE_IN + tt * PRICE_OUT) / 1e6, 6),
            "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "thinking_text": think, "response_text": resp, "full_text": full,
            "thinking_source": "claude_api_block" if (args.extended_thinking and think) else "none",
            "was_retried": False,
            "generation_error": None,
        })

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    n, total_cost = 0, 0.0
    with out.open(mode, encoding="utf-8") as fh:
        for pid, cands in grouped.items():
            if not cands:
                continue
            p = by_id[pid]
            total_cost += sum(c.get("total_cost_usd", 0.0) for c in cands)
            rec = {**{k: p[k] for k in ("record_id", "problem", "gold_answer",
                                         "gold_solution", "dataset", "dataset_split",
                                         "difficulty", "subject", "source_idx")},
                   "has_extended_thinking": args.extended_thinking,
                   "generation_date": gen_date, "candidates": cands}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    # PRICE_IN/OUT are already the batch ($1.50/$7.50 per M) rates used per-candidate,
    # so total = sum of per-candidate total_cost_usd. No extra discount (that was the
    # double-counting bug).
    print(f"[generate] wrote {n} problem records -> {out} ({'appended' if args.append else 'overwrote'})")
    print(f"[generate] tokens in={in_tok:,} out={out_tok:,} | batch cost ${total_cost:.2f}")
    print(f"[generate] NEXT: python main/data_pipeline.py --raw {out}")


if __name__ == "__main__":
    main()
