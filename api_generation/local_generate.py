"""
local_generate.py — gpt-oss-20B via Apple MLX (M3-ONLY). Engine switched from Ollama
to MLX to fix the 16GB memory squeeze, with every learning from the Ollama runs.

WHY MLX (vs the old Ollama path):
  * gpt-oss-20B (MXFP4, ~12-13GB) was SWAPPING under Ollama on 16GB ("RAM kept
    climbing"). MLX is the most memory-efficient Apple-Silicon runtime AND supports
    KV-CACHE QUANTIZATION (--kv_bits 8) — it shrinks exactly the part of memory that
    grew during long reasoning. Best shot at running gpt-oss without swap on 16GB.
  * NO hidden context cap. The Ollama bug was num_ctx defaulting to 4096, silently
    truncating the CoT at ~4k tokens (the model never reached the answer). MLX uses
    the model's real 128k window; we just set --max_tokens. So that bug cannot recur.

LEARNINGS BAKED IN:
  1. Reasoning effort low/med/high is set the PROPER way — via the chat template's
     `reasoning_effort` argument (the Ollama "Reasoning: x" system-prompt injection
     was ignored). The 3 candidates per problem use low / medium / high.
  2. gpt-oss has NO <think> tags — it uses harmony CHANNELS (analysis = reasoning,
     final = answer). We parse both. thinking_source='gpt_oss_cot_block'.
  3. Blackhole guard: repetition_penalty 1.15 (prevention) + 30-gram CoT check
     (detection) -> blackhole_detected + one reseed retry.
  4. Diversity also via SEED (mx.random.seed) so the 3 candidates differ.
  5. Schema-compatible output -> verify_answers / data_pipeline consume it unchanged.

SETUP (M3 only):  pip install mlx-lm     (do NOT add to requirements.txt — Mac-only;
                  the i5 desktop uses r1_generate.py / Ollama instead.)
MODEL:  default mlx-community/gpt-oss-20b-MXFP4-Q8 (downloads from HF on first load;
        if that repo id 404s, pass --model with another MLX gpt-oss build).

Pipeline:  local_generate.py -> verify_answers.py -> data_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402
from claude_generate import load_problems  # reuse dataset loaders  # noqa: E402

DEFAULT_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"
EFFORT_LEVELS = ["low", "medium", "high"]      # candidates 0/1/2
# max_tokens is the TOTAL budget shared by thinking + answer (one sequential pass).
# Set generous so the model finishes thinking AND lands the \boxed{} answer. There is
# NO num_ctx in MLX (that was the Ollama 4096 trap) — this is the only generation cap.
# kv_bits=8 keeps the KV cache small enough that a 12k budget still fits 16GB.
DEFAULT_MAX_TOKENS = 12000
BLACKHOLE_NGRAM, BLACKHOLE_REPEAT = 30, 3       # CoT repetition -> reasoning loop

# Canonical system prompt — schema Section 5.1, VERBATIM. Used identically across all
# generators so model differences (not prompt wording) drive the variation. Deliberately
# neutral: it does NOT steer toward an approach (diversity comes from temp/effort/seed).
SYSTEM_PROMPT = (
    "You are solving a competition mathematics problem.\n"
    "Think carefully about the approach before computing.\n"
    "Show your complete reasoning step by step.\n"
    "Each step should be on a new line.\n"
    "State your final answer clearly and enclose it in \\boxed{}.\n"
    "Do not skip steps. Do not summarise. Show everything.")

# Harmony channel markers (gpt-oss). analysis = chain-of-thought, final = the answer.
_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|channel\|>|<\|start\|>|$)", re.S)
_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?=<\|return\|>|<\|end\|>|$)", re.S)


def parse_harmony(text: str) -> tuple[str, str]:
    """Return (thinking, response) from gpt-oss harmony output. Robust fallbacks:
    channel markers -> <think> tags -> whole text as response (learning 2)."""
    think, resp = "", ""
    m = _ANALYSIS.search(text)
    if m:
        think = m.group(1).strip()
    m = _FINAL.search(text)
    if m:
        resp = m.group(1).strip()
    if not think and not resp:
        if "<think>" in text:
            think, resp = T.split_think_response(text)
        else:
            resp = text.strip()
    return think, resp


def generate_one(model, tokenizer, problem, effort, max_tokens, kv_bits, seed,
                 temperature, top_p, debug=False):
    """One MLX generation. Returns dict(thinking, response, blackhole, n_tok)."""
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    mx.random.seed(seed)                                  # learning 4: seed diversity
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem}]
    # learning 1: reasoning effort via the chat template (the supported way).
    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, reasoning_effort=effort)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    # NOTE: HIGHER temperature = MORE diverse/creative approaches = more Type-B.
    # Lower temp collapses the 3 candidates together. Keep >=0.7; default 1.0.
    sampler = make_sampler(temp=temperature, top_p=top_p)
    logits_processors = make_logits_processors(repetition_penalty=1.15)   # learning 3
    gen_kwargs = dict(max_tokens=max_tokens, sampler=sampler,
                      logits_processors=logits_processors, verbose=False)
    if kv_bits:
        gen_kwargs["kv_bits"] = kv_bits                   # KV-cache quantization (memory)
    try:
        text = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
    except Exception:
        # gpt-oss uses a RotatingKVCache that mlx-lm can't quantize yet
        # ("RotatingKVCache Quantization NYI"), or older mlx-lm (TypeError). Retry
        # WITHOUT kv quantization — the rotating cache is already memory-bounded.
        if "kv_bits" in gen_kwargs:
            gen_kwargs.pop("kv_bits", None)
            text = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        else:
            raise

    if debug:
        print("\n===== RAW MLX OUTPUT (first 1200 chars) =====")
        print(repr(text[:1200]))
        print("=============================================\n")

    think, response = parse_harmony(text)                 # learning 2: harmony channels
    blackhole = T.has_repeated_ngram(think, n=BLACKHOLE_NGRAM, max_repeat=BLACKHOLE_REPEAT)
    n_tok = len(tokenizer.encode(text)) if text else 0
    return {"thinking": think, "response": response, "blackhole": blackhole, "n_tok": n_tok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["omnimath", "olympiadbench", "math"], required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N problems (useful for resuming with --append)")
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="TOTAL generation budget shared by thinking + answer. Raise if "
                         "responses come back empty (model over-thought); lower for speed. "
                         "No num_ctx in MLX — this is the only cap.")
    ap.add_argument("--kv_bits", type=int, default=0,
                    help="KV-cache quantization. Default 0 (OFF): gpt-oss uses a rotating "
                         "(sliding-window) cache that mlx-lm can't quantize yet AND that is "
                         "already memory-bounded, so we don't need it. Set >0 only if a "
                         "future mlx-lm supports rotating-cache quantization.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="HIGHER = more diverse/creative approaches = more Type-B. Keep >=0.7; "
                         "lowering collapses the 3 candidates together. Diversity also from effort.")
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="nucleus sampling (gpt-oss default 1.0)")
    ap.add_argument("--hardest", action="store_true",
                    help="OmniMath only: hardest-N (reserve for STRONG models, not gpt-oss-20B)")
    ap.add_argument("--max_difficulty", type=float, default=None,
                    help="OmniMath only: keep problems with difficulty <= this. Use ~6 for "
                         "gpt-oss-20B so it solves enough to yield Type B (not all Type D).")
    ap.add_argument("--min_difficulty", type=float, default=None,
                    help="OmniMath only: keep problems with difficulty >= this.")
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--checkpoint_every", type=int, default=10)
    ap.add_argument("--debug", action="store_true", help="dump raw output of the first candidate")
    args = ap.parse_args()

    try:
        from mlx_lm import load
    except ImportError:
        print("[mlx] mlx-lm not installed. On your M3:  pip install mlx-lm   "
              "(do NOT add it to requirements.txt — it is Apple-Silicon only).")
        sys.exit(1)

    print(f"[mlx] loading {args.model} (first run downloads from HF) ...")
    model, tokenizer = load(args.model)

    total_to_load = args.limit + args.offset if args.limit else None
    problems = load_problems(args.dataset, total_to_load, hardest=args.hardest,
                             min_difficulty=args.min_difficulty,
                             max_difficulty=args.max_difficulty)
    if args.offset > 0:
        problems = problems[args.offset:]
    schema_model = "gpt-oss-20b"
    print(f"[mlx] {len(problems)} problems x {args.n_candidates} cands | model={args.model} "
          f"| diversity=reasoning-effort+seed | max_tokens={args.max_tokens} kv_bits={args.kv_bits}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a" if args.append else "w", encoding="utf-8")
    gen_date = date.today().isoformat()
    n_problems = n_blackhole = n_total = 0
    t0 = time.time()

    try:
        for pi, p in enumerate(problems):
            cands = []
            for c in range(args.n_candidates):
                effort = EFFORT_LEVELS[c % 3]
                # FLAT budget for all efforts. (Reverted an earlier 2x bump for high
                # effort: 24k tokens OOM'd Metal because gpt-oss-Q8 weights already sit
                # at the ~12GB GPU ceiling. High effort may truncate at this budget ->
                # resp=0 -> excluded; that's acceptable. To let high-effort finish,
                # raise the GPU limit instead:  sudo sysctl iogpu.wired_limit_mb=14336)
                mt = args.max_tokens
                n_total += 1
                try:
                    r = generate_one(model, tokenizer, p["problem"], effort, mt,
                                     args.kv_bits, seed=42 + c, temperature=args.temperature,
                                     top_p=args.top_p, debug=(args.debug and pi == 0 and c == 0))
                    if r["blackhole"]:                    # one reseed retry on a loop
                        r = generate_one(model, tokenizer, p["problem"], effort, mt,
                                         args.kv_bits, seed=1000 + c, temperature=args.temperature,
                                         top_p=args.top_p)
                except Exception as e:
                    print(f"[mlx] WARN {p['record_id']}_c{c}: {str(e)[:100]}")
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
                    "temperature": args.temperature, "reasoning_effort": effort, "generation_seed": 42 + c,
                    "input_tokens": 0, "thinking_tokens": len(think.split()),
                    "output_tokens": r["n_tok"], "total_cost_usd": 0.0,
                    "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "thinking_text": think, "response_text": response, "full_text": full,
                    "thinking_source": "gpt_oss_cot_block",
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
                print(f"[mlx] {pi+1}/{len(problems)} problems "
                      f"(blackholes {n_blackhole}/{n_total}, {(time.time()-t0)/60:.1f} min)")
    finally:
        fh.close()

    print(f"[mlx] wrote {n_problems} records -> {out}")
    print(f"[mlx] blackholes: {n_blackhole}/{n_total}")
    print(f"[mlx] NEXT: python api_generation/verify_answers.py --in {out} --out {out}")


if __name__ == "__main__":
    main()
