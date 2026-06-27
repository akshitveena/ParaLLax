"""
local_generate.py — generate reasoning candidates LOCALLY via Ollama (free, no
rate limits, unlimited reasoning depth). Replaces the Groq path for the bulk.

Primary model: gpt-oss:20b (MoE, ~3.6B active, runs on a 16GB M3, ≈o3-mini math).
  * Reasoning-effort diversity (schema-better than temperature): the 3 candidates use
    Reasoning low / medium / high — directly varying how deeply the model explores
    before committing (higher effort -> more unconventional approaches -> more Type B).
  * Full CoT is read from Ollama's separate `thinking` field (think=True). If you only
    read `content` you get the answer with NO reasoning — useless for RiDAE.
  * thinking_source='gpt_oss_cot_block' (cleanly API-separated; tiered as parsed_reasoning
    for claim-strength — generation is still single-pass, not Claude-ET-separate).
  * Blackhole guard: gpt-oss can loop forever in its CoT. Mitigations: temperature 1.0,
    top_p 0.95, repeat_penalty 1.15 (prevention) + post-hoc 30-gram repetition check
    (detection) -> blackhole_detected=True, excluded. One reseed retry on detection.

Secondary model: deepseek-r1:14b (inline <think>; temperature-sweep diversity).

Requires a running Ollama with the model pulled:
    ollama pull gpt-oss:20b        (or: ollama pull deepseek-r1:14b)
Pipeline: local_generate.py -> verify_answers.py -> data_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402
from claude_generate import load_problems  # reuse dataset loaders  # noqa: E402

EFFORT_LEVELS = ["low", "medium", "high"]      # candidates 0/1/2 (gpt-oss)
TEMP_STRATEGY = [0.3, 0.8, 1.1]                # fallback diversity for non-effort models
DEFAULT_MAX_TOKENS = 8000                      # local: no cost; depth preserved
BLACKHOLE_NGRAM, BLACKHOLE_REPEAT = 30, 3      # CoT repetition -> reasoning loop


def is_gpt_oss(model: str) -> bool:
    return "gpt-oss" in model.lower() or "gpt_oss" in model.lower()


def build_messages(problem: str, system_prompt: str, model: str, effort: str):
    sys_text = system_prompt
    if is_gpt_oss(model):
        # gpt-oss reads reasoning effort from the system message (harmony format).
        sys_text = f"Reasoning: {effort}\n\n{system_prompt}"
    return [{"role": "system", "content": sys_text},
            {"role": "user", "content": problem}]


def extract_thinking(msg) -> tuple[str, str, str]:
    """Return (thinking, response, source). Ollama exposes reasoning in `thinking`;
    fall back to reasoning_content, then inline <think>."""
    think = getattr(msg, "thinking", None) or (msg.get("thinking") if isinstance(msg, dict) else None) or ""
    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "") or ""
    if think:
        return think.strip(), content.strip(), "gpt_oss_cot_block"
    rc = getattr(msg, "reasoning_content", None) or (msg.get("reasoning_content") if isinstance(msg, dict) else None)
    if rc:
        return rc.strip(), content.strip(), "parsed_reasoning"
    if "<think>" in content:
        t, b = T.split_think_response(content)
        return t, b, "inline_think_tags"
    return "", content.strip(), "none"


def generate_one(client, model, problem, effort, system_prompt, max_tokens, repeat_penalty,
                 seed, num_ctx):
    """One Ollama chat call. Returns dict(thinking, response, source, eval_count, blackhole).

    CRITICAL: num_ctx (context window) must be set large. Ollama defaults to 4096,
    which silently caps prompt+generation at ~4k tokens — the model gets cut off
    mid-thinking and never writes the answer, regardless of num_predict.
    """
    msgs = build_messages(problem, system_prompt, model, effort)
    resp = client.chat(
        model=model, messages=msgs, think=True,
        options={"temperature": 1.0, "top_p": 0.95, "repeat_penalty": repeat_penalty,
                 "num_predict": max_tokens, "num_ctx": num_ctx, "seed": seed},
    )
    think, response, source = extract_thinking(resp.message)
    blackhole = T.has_repeated_ngram(think, n=BLACKHOLE_NGRAM, max_repeat=BLACKHOLE_REPEAT)
    eval_count = getattr(resp, "eval_count", 0) or 0
    prompt_eval = getattr(resp, "prompt_eval_count", 0) or 0
    return {"thinking": think, "response": response, "source": source,
            "blackhole": blackhole, "eval_count": eval_count, "prompt_eval": prompt_eval}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["omnimath", "olympiadbench", "math"], required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--num_ctx", type=int, default=16384,
                    help="context window — MUST be large (Ollama defaults to 4096 and "
                         "silently truncates the reasoning). 16384 fits 16GB; raise if needed.")
    ap.add_argument("--repeat_penalty", type=float, default=1.15)
    ap.add_argument("--hardest", action="store_true",
                    help="OmniMath only: hardest-N (good for the strong gpt-oss)")
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--checkpoint_every", type=int, default=20)
    args = ap.parse_args()

    SYSTEM_PROMPT = (
        "You are solving a competition mathematics problem. Think carefully about the "
        "approach before computing. Show your complete reasoning step by step. State "
        "your final answer enclosed in \\boxed{}. Do not skip steps.")

    import ollama
    client = ollama.Client()
    # Fail fast if the model isn't pulled.
    try:
        have = {m.model for m in client.list().models}
        if not any(args.model.split(":")[0] in h for h in have):
            print(f"[local] model '{args.model}' not found. Run:  ollama pull {args.model}")
            sys.exit(1)
    except Exception as e:
        print(f"[local] cannot reach Ollama ({e}). Is it running?  (ollama serve)")
        sys.exit(1)

    problems = load_problems(args.dataset, args.limit, hardest=args.hardest)
    use_effort = is_gpt_oss(args.model)
    schema_model = args.model.split(":")[0]
    print(f"[local] {len(problems)} problems x {args.n_candidates} cands | model={args.model} "
          f"| diversity={'reasoning-effort' if use_effort else 'temperature'} | "
          f"max_tokens={args.max_tokens}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a" if args.append else "w", encoding="utf-8")
    gen_date = date.today().isoformat()
    import time
    n_problems = n_blackhole = n_total = 0
    t0 = time.time()

    try:
        for pi, p in enumerate(problems):
            cands = []
            for c in range(args.n_candidates):
                effort = EFFORT_LEVELS[c % 3] if use_effort else "medium"
                temp_label = TEMP_STRATEGY[c % 3]
                n_total += 1
                try:
                    r = generate_one(client, args.model, p["problem"], effort, SYSTEM_PROMPT,
                                     args.max_tokens, args.repeat_penalty, seed=42 + c,
                                     num_ctx=args.num_ctx)
                    # One reseed retry if a reasoning loop was detected.
                    if r["blackhole"]:
                        r = generate_one(client, args.model, p["problem"], effort, SYSTEM_PROMPT,
                                         args.max_tokens, args.repeat_penalty, seed=1000 + c,
                                         num_ctx=args.num_ctx)
                except Exception as e:
                    print(f"[local] WARN {p['record_id']}_c{c}: {str(e)[:80]}")
                    cands.append({"candidate_id": f"{p['record_id']}_c{c}", "model": schema_model,
                                  "generation_error": str(e), "thinking_text": "",
                                  "response_text": "", "full_text": ""})
                    continue

                think, response = r["thinking"], r["response"]
                full = T.assemble_full_text(think, response)
                if r["blackhole"]:
                    n_blackhole += 1
                cands.append({
                    "candidate_id": f"{p['record_id']}_c{c}", "model": schema_model,
                    "temperature": 1.0 if use_effort else temp_label,
                    "reasoning_effort": effort if use_effort else None,
                    "generation_seed": c,
                    "input_tokens": r["prompt_eval"], "thinking_tokens": r["eval_count"],
                    "output_tokens": max(0, r["eval_count"]),
                    "total_cost_usd": 0.0,     # local = free
                    "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "thinking_text": think, "response_text": response, "full_text": full,
                    "thinking_source": r["source"],
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
                bh = 100 * n_blackhole / max(n_total, 1)
                print(f"[local] {pi+1}/{len(problems)} problems  "
                      f"(blackholes {n_blackhole}/{n_total} = {bh:.0f}%, "
                      f"{(time.time()-t0)/60:.1f} min)")
    finally:
        fh.close()

    bh = 100 * n_blackhole / max(n_total, 1)
    print(f"[local] wrote {n_problems} records -> {out}")
    print(f"[local] blackholes: {n_blackhole}/{n_total} ({bh:.0f}%)  "
          f"{'-- >15%: consider Reasoning:medium for high-effort cands' if bh > 15 else ''}")
    print(f"[local] NEXT: python api_generation/verify_answers.py --in {out} --out {out}")


if __name__ == "__main__":
    main()
