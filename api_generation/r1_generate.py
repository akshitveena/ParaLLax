"""
r1_generate.py — DeepSeek-R1-Distill-14B local generator (Ollama).

Purpose-built for the R1 distil, whose behaviour differs from gpt-oss/Qwen3. The
seven R1-specific rules, all enforced here:

  1. TEMPERATURE IS NON-FUNCTIONAL in thinking mode → diversity comes from the SEED.
     Official sampling is temperature 0.6 / top_p 0.95 (set for compatibility), and
     the 3 candidates per problem differ only by generation_seed (not temperature).
  2. max_tokens = 8192 (R1 chains are long; 3000 would truncate hard problems → Type D).
  3. NO system prompt — R1 partially ignores it. All instructions go in the user turn.
  4. The <think> block is ALWAYS present → thinking_source = 'inline_think_tags' for
     every candidate; the settled-approach rule (data_pipeline) applies in full.
  5. LANGUAGE MIXING: R1 sometimes reasons in Chinese. Such candidates are still valid
     (math is sound) but the English keyword approach-classifier is unreliable on them,
     so data_pipeline flags quality_flags=['language_mixing'] (kept, handled separately).
  6. HARDWARE: ~9GB weights fit a 16GB M3 comfortably (no swap). ~12-18 tok/s → roughly
     2-4 min/candidate → ~30-60h for 300 problems. PILOT 20 first; if correct-rate < 25%
     on your subset, R1-14B isn't worth the time — reconsider.
  7. ROLE: R1-14B is the FREE LOCAL FALLBACK, best on OlympiadBench (32% Type-B, moderate
     difficulty). Reserve the Claude budget for OmniMath's hardest. So --dataset defaults
     to olympiadbench.

Shared with local_generate.py: dataset loaders, num_ctx fix (Ollama defaults to 4096 and
silently truncates), blackhole guard, schema output.

Requires:  ollama pull deepseek-r1:14b
Pipeline:  r1_generate.py -> verify_answers.py -> data_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402
from claude_generate import load_problems  # reuse dataset loaders  # noqa: E402

DEFAULT_MODEL = "deepseek-r1:14b"
DEFAULT_MAX_TOKENS = 8192          # point 2: R1 chains are long
DEFAULT_NUM_CTX = 16384            # critical: Ollama defaults to 4096 -> silent truncation
TEMPERATURE, TOP_P = 0.6, 0.95     # point 1: official; non-functional, seed drives diversity
REPEAT_PENALTY = 1.15
BLACKHOLE_NGRAM, BLACKHOLE_REPEAT = 30, 3

# point 3: NO system prompt — everything in the user turn.
USER_TEMPLATE = (
    "Solve the following competition mathematics problem. Think carefully about the "
    "approach before computing, and show your complete reasoning step by step. State "
    "your final answer enclosed in \\boxed{{}}. Do not skip steps.\n\n"
    "Problem:\n{problem}")


def extract_thinking(msg) -> tuple[str, str]:
    """Return (thinking, response). R1 via Ollama think=True puts the chain in `thinking`;
    fall back to inline <think> tags if needed."""
    think = getattr(msg, "thinking", None) or (msg.get("thinking") if isinstance(msg, dict) else None) or ""
    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "") or ""
    if not think and "<think>" in content:
        think, content = T.split_think_response(content)
    return think.strip(), content.strip()


def generate_one(client, model, problem, max_tokens, num_ctx, seed):
    """One Ollama chat call (seed = the diversity mechanism for R1)."""
    msgs = [{"role": "user", "content": USER_TEMPLATE.format(problem=problem)}]
    resp = client.chat(
        model=model, messages=msgs, think=True,
        options={"temperature": TEMPERATURE, "top_p": TOP_P, "repeat_penalty": REPEAT_PENALTY,
                 "num_predict": max_tokens, "num_ctx": num_ctx, "seed": seed},
    )
    think, response = extract_thinking(resp.message)
    blackhole = T.has_repeated_ngram(think, n=BLACKHOLE_NGRAM, max_repeat=BLACKHOLE_REPEAT)
    return {"thinking": think, "response": response, "blackhole": blackhole,
            "eval_count": getattr(resp, "eval_count", 0) or 0,
            "prompt_eval": getattr(resp, "prompt_eval_count", 0) or 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["omnimath", "olympiadbench", "math"],
                    default="olympiadbench", help="point 7: R1's best fit is OlympiadBench")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--num_ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--hardest", action="store_true",
                    help="OmniMath only: hardest-N (usually leave OFF for R1 — it fails the extreme tier)")
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--checkpoint_every", type=int, default=10)
    args = ap.parse_args()

    import ollama
    client = ollama.Client()
    try:
        have = {m.model for m in client.list().models}
        if not any(args.model.split(":")[0] in h for h in have):
            print(f"[r1] model '{args.model}' not found. Run:  ollama pull {args.model}")
            sys.exit(1)
    except Exception as e:
        print(f"[r1] cannot reach Ollama ({e}). Is it running?  (ollama serve)")
        sys.exit(1)

    problems = load_problems(args.dataset, args.limit, hardest=args.hardest)
    schema_model = args.model.split(":")[0]
    print(f"[r1] {len(problems)} problems x {args.n_candidates} cands | model={args.model} "
          f"| diversity=SEED (temp non-functional) | max_tokens={args.max_tokens} "
          f"| num_ctx={args.num_ctx}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a" if args.append else "w", encoding="utf-8")
    gen_date = date.today().isoformat()
    n_problems = n_blackhole = n_total = n_cjk = 0
    t0 = time.time()

    try:
        for pi, p in enumerate(problems):
            cands = []
            for c in range(args.n_candidates):
                seed = 42 + c                      # point 1: SEED is the diversity axis
                n_total += 1
                try:
                    r = generate_one(client, args.model, p["problem"], args.max_tokens,
                                     args.num_ctx, seed)
                    if r["blackhole"]:             # one reseed retry on a reasoning loop
                        r = generate_one(client, args.model, p["problem"], args.max_tokens,
                                         args.num_ctx, seed=1000 + c)
                except Exception as e:
                    print(f"[r1] WARN {p['record_id']}_c{c}: {str(e)[:80]}")
                    cands.append({"candidate_id": f"{p['record_id']}_c{c}", "model": schema_model,
                                  "generation_error": str(e), "thinking_text": "",
                                  "response_text": "", "full_text": ""})
                    continue

                think, response = r["thinking"], r["response"]
                full = T.assemble_full_text(think, response)
                if r["blackhole"]:
                    n_blackhole += 1
                if T.has_cjk(think):               # point 5: language-mixing observability
                    n_cjk += 1
                cands.append({
                    "candidate_id": f"{p['record_id']}_c{c}", "model": schema_model,
                    "temperature": TEMPERATURE,    # stored, but non-functional for R1
                    "reasoning_effort": None, "generation_seed": seed,
                    "input_tokens": r["prompt_eval"],
                    "thinking_tokens": len(think.split()),    # word proxy (Ollama gives no split)
                    "output_tokens": r["eval_count"], "total_cost_usd": 0.0,
                    "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "thinking_text": think, "response_text": response, "full_text": full,
                    "thinking_source": "inline_think_tags",   # point 4: R1 always inline
                    "blackhole_detected": r["blackhole"], "was_retried": False,
                    "generation_error": None,
                })

            rec = {**{k: p[k] for k in ("record_id", "problem", "gold_answer",
                                         "gold_solution", "dataset", "dataset_split",
                                         "difficulty", "subject", "source_idx")},
                   "has_extended_thinking": False, "generation_date": gen_date,
                   "candidates": cands}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            n_problems += 1
            if (pi + 1) % args.checkpoint_every == 0:
                print(f"[r1] {pi+1}/{len(problems)} problems  "
                      f"(blackholes {n_blackhole}/{n_total}, cjk {n_cjk}, "
                      f"{(time.time()-t0)/60:.1f} min)")
    finally:
        fh.close()

    bh = 100 * n_blackhole / max(n_total, 1)
    print(f"[r1] wrote {n_problems} records -> {out}")
    print(f"[r1] blackholes: {n_blackhole}/{n_total} ({bh:.0f}%) | language-mixing: {n_cjk}/{n_total}")
    print(f"[r1] NEXT: python api_generation/verify_answers.py --in {out} --out {out}")
    print("[r1] PILOT TIP (point 6): if correct-rate < 25% after verify+pipeline, "
          "R1-14B isn't worth the generation time — reconsider difficulty/model.")


if __name__ == "__main__":
    main()
