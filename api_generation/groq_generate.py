"""
groq_generate.py — generate reasoning candidates with QwQ-32B on Groq (free tier),
in the full RiDAE Dataset Schema. Produces the 1,800 free candidates the budget
plan relies on: OmniMath (300) + OlympiadBench (300), 3 candidates each.

QwQ differs from Claude ET in ways the schema is explicit about:
  * It thinks INLINE in <think>...</think> within one autoregressive pass, so
    thinking_source = 'inline_think_tags' (a WEAKER unfaithfulness claim than
    Claude's architecturally-separate 'claude_api_block').
  * Boundary is just a text pattern (handled by textutils.split_think_response,
    which tolerates missing/partial tags).
  * Groq returns only total tokens, so thinking_tokens are counted with the Qwen
    tokenizer (item: token counting problem).
  * max_tokens is capped at 3000 (item 1): better a clean 2k candidate than a
    truncated 6k one.
  * Answer is extracted from response_text only — done downstream in data_pipeline.
  * Rate-limit retries set was_retried=True (item 7: retries break reproducibility).

Requires GROQ_API_KEY. Pipeline: groq_generate.py -> main/data_pipeline.py
(append to the same raw file the Claude generator wrote, with --append).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402
from claude_generate import load_problems  # reuse the dataset loaders  # noqa: E402

# Groq deprecated QwQ-32B; qwen/qwen3-32b is its drop-in successor (same size,
# reasoning model, inline <think> tags). Override with --model if Groq's lineup changes.
DEFAULT_GROQ_MODEL = "qwen/qwen3-32b"
DEFAULT_HF_TOKENIZER = "Qwen/Qwen3-32B"   # for accurate thinking-token counts
SYSTEM_PROMPT = (
    "You are solving a competition mathematics problem.\n"
    "Think carefully about the approach before computing.\n"
    "Show your complete reasoning step by step.\n"
    "Each step should be on a new line.\n"
    "State your final answer clearly and enclose it in \\boxed{}.\n"
    "Do not skip steps. Do not summarise. Show everything."
)
TEMP_STRATEGY = [0.3, 0.8, 1.1]      # Section 5.2
GROQ_PRICE_IN, GROQ_PRICE_OUT = 0.29, 0.59   # USD / 1M tokens (qwen3-32b, approx)
# Token budget tension on the FREE tier:
#   * 3000 (the schema's QwQ number) is too small — Qwen3 never finishes thinking.
#   * 8000 exceeds Qwen3-32B's free-tier 6000 tokens-per-minute limit -> 413 rejected.
# 5000 fits under the 6000 TPM ceiling while giving room to finish. The generator also
# AUTO-REDUCES max_tokens if it still hits a TPM limit (handles other models / limits).
DEFAULT_MAX_TOKENS = 5000


# --------------------------------------------------------------------------- #
class _Counter:
    """Qwen tokenizer for accurate thinking-token counts; falls back gracefully."""

    def __init__(self, hf_name=DEFAULT_HF_TOKENIZER):
        self.tok = None
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(hf_name)
        except Exception as e:  # offline / no access — degrade to a word proxy
            print(f"[groq] WARNING: Qwen tokenizer unavailable ({e}); "
                  "thinking_tokens will use a word-count proxy.")

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tok is not None:
            return len(self.tok.encode(text, add_special_tokens=False))
        return int(len(text.split()) * 1.3)


def _is_auth_error(e) -> bool:
    return (getattr(e, "status_code", None) in (401, 403)
            or "invalid_api_key" in str(e).lower()
            or "invalid api key" in str(e).lower()
            or e.__class__.__name__ == "AuthenticationError")


def _call_with_retry(client, model, messages, temperature, max_tokens, max_retries=4):
    """Return (reasoning, content, src_method, usage, was_retried).

    Retries transient/rate errors; on a 413 "request too large" (single request
    exceeds the per-minute token limit) it parses the limit and RETRIES with a
    smaller max_tokens, so it self-adapts to each model's free-tier TPM ceiling.
    """
    was_retried = False
    cur_max = max_tokens
    for attempt in range(max_retries + 1):
        try:
            kwargs = dict(model=model, messages=messages,
                          temperature=temperature, max_tokens=cur_max)
            # Parsed reasoning: thinking comes back in message.reasoning, cleanly
            # separated from message.content. Works for BOTH Qwen3 and gpt-oss (gpt-oss
            # has no <think> tags, so raw parsing would miss its reasoning entirely).
            try:
                resp = client.chat.completions.create(reasoning_format="parsed", **kwargs)
            except TypeError:
                resp = client.chat.completions.create(**kwargs)   # older SDK
            choice = resp.choices[0]
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning", None) or ""
            if reasoning:
                return reasoning.strip(), content.strip(), "parsed", resp.usage, was_retried
            # Fallback: no parsed field but model inlined <think> tags (QwQ-style).
            if "<think>" in content:
                think, body = T.split_think_response(content)
                return think, body, "inline", resp.usage, was_retried
            return "", content.strip(), "none", resp.usage, was_retried
        except Exception as e:
            # Auth/permission errors are not transient — fail fast, don't burn retries.
            if _is_auth_error(e):
                raise
            msg = str(e)
            # 413 "request too large": single request exceeds per-minute token limit.
            # Parse the limit and shrink max_tokens to fit, then retry.
            if "tokens per minute" in msg.lower() or "request too large" in msg.lower():
                m = re.search(r"Limit\s+(\d+)", msg)
                if m:
                    new_max = max(512, int(m.group(1)) - 500)
                    if new_max < cur_max:
                        print(f"[groq] request exceeds TPM; reducing max_tokens "
                              f"{cur_max} -> {new_max} and retrying")
                        cur_max = new_max
                        was_retried = True
                        continue
            if attempt >= max_retries:
                raise
            was_retried = True
            wait = 2 ** attempt
            print(f"[groq] retry {attempt+1}/{max_retries} after error: {msg[:80]} (sleep {wait}s)")
            time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["omnimath", "olympiadbench"], required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="completion cap; Qwen3-32B needs room to finish thinking + answer")
    ap.add_argument("--model", default=DEFAULT_GROQ_MODEL,
                    help="Groq model id (default qwen/qwen3-32b; QwQ-32B is deprecated)")
    ap.add_argument("--hf_tokenizer", default=DEFAULT_HF_TOKENIZER,
                    help="HF tokenizer for thinking-token counts (should match --model)")
    ap.add_argument("--hardest", action="store_true",
                    help="OmniMath only: hardest-N (use for the strong gpt-oss-120b run, "
                         "NOT for mid Qwen3-32B which mostly fails at extreme difficulty)")
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--append", action="store_true",
                    help="append to the raw file (use after claude_generate.py)")
    ap.add_argument("--checkpoint_every", type=int, default=50)
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: set GROQ_API_KEY in your environment."); sys.exit(1)

    from groq import Groq
    client = Groq()
    counter = _Counter(args.hf_tokenizer)
    schema_model = args.model.split("/")[-1]          # record label, e.g. 'qwen3-32b'

    problems = load_problems(args.dataset, args.limit, hardest=args.hardest)
    print(f"[groq] {len(problems)} problems x {args.n_candidates} candidates "
          f"= {len(problems)*args.n_candidates} requests "
          f"(model={args.model}, max_tokens={args.max_tokens}, hardest={args.hardest})")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("a" if args.append else "w", encoding="utf-8")
    gen_date = date.today().isoformat()
    n_problems = 0
    n_err = 0
    in_tok = out_tok = 0
    total_cost = 0.0
    t0 = time.time()

    try:
        for pi, p in enumerate(problems):
            cands = []
            for c in range(args.n_candidates):
                temp = TEMP_STRATEGY[c % len(TEMP_STRATEGY)]
                msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": p["problem"]}]
                try:
                    think, resp, src_method, usage, retried = _call_with_retry(
                        client, args.model, msgs, temp, args.max_tokens)
                except Exception as e:
                    # An invalid key fails EVERY request — abort loudly instead of
                    # silently writing a file full of useless error stubs.
                    if _is_auth_error(e):
                        fh.close()
                        print("\n[groq] FATAL: GROQ_API_KEY is invalid/rejected (401).")
                        print("  Nothing usable was generated. Fix the key, then re-run:")
                        print("    1) get a key at https://console.groq.com  (starts with 'gsk_')")
                        print("    2) put it in .env, then:  set -a; source .env; set +a")
                        print("    3) check it loaded:  echo \"${GROQ_API_KEY:0:4}\"  (should print gsk_)")
                        sys.exit(1)
                    n_err += 1
                    print(f"[groq] WARN: {p['record_id']}_c{c} errored: {str(e)[:80]}")
                    cands.append({"candidate_id": f"{p['record_id']}_c{c}",
                                  "model": schema_model, "temperature": temp,
                                  "generation_error": str(e), "was_retried": True,
                                  "thinking_text": "", "response_text": "", "full_text": ""})
                    continue

                full = T.assemble_full_text(think, resp)
                think_tok = counter.count(think)
                prompt_tok = getattr(usage, "prompt_tokens", 0)
                completion_tok = getattr(usage, "completion_tokens", 0)
                resp_tok = max(0, completion_tok - think_tok)
                cost = (prompt_tok * GROQ_PRICE_IN + completion_tok * GROQ_PRICE_OUT) / 1e6
                in_tok += prompt_tok; out_tok += completion_tok; total_cost += cost
                # Honesty ladder: parsed (Groq reasoning channel) > inline (<think>) > none.
                tsrc = {"parsed": "parsed_reasoning", "inline": "inline_think_tags",
                        "none": "none"}[src_method if think else "none"]

                cands.append({
                    "candidate_id": f"{p['record_id']}_c{c}", "model": schema_model,
                    "temperature": temp, "generation_seed": c,
                    "input_tokens": prompt_tok, "thinking_tokens": think_tok,
                    "output_tokens": resp_tok, "total_cost_usd": round(cost, 6),
                    "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "thinking_text": think, "response_text": resp, "full_text": full,
                    "thinking_source": tsrc,
                    "was_retried": retried, "generation_error": None,
                })

            rec = {**{k: p[k] for k in ("record_id", "problem", "gold_answer",
                                         "gold_solution", "dataset", "dataset_split",
                                         "difficulty", "subject", "source_idx")},
                   "has_extended_thinking": False,      # QwQ inline != Claude ET
                   "generation_date": gen_date, "candidates": cands}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_problems += 1
            if (pi + 1) % args.checkpoint_every == 0:
                print(f"[groq] {pi+1}/{len(problems)} problems  "
                      f"(running cost ${total_cost:.2f}, {(time.time()-t0)/60:.1f} min)")
    finally:
        fh.close()

    print(f"[groq] wrote {n_problems} problem records -> {out} "
          f"({'appended' if args.append else 'overwrote'})")
    print(f"[groq] tokens in={in_tok:,} out={out_tok:,} | cost ${total_cost:.2f}")
    if n_err:
        print(f"[groq] WARNING: {n_err} candidate(s) errored (excluded from training). "
              "If tokens/cost are 0, NONE of this run is usable — investigate before proceeding.")
    if in_tok == 0 and out_tok == 0:
        print("[groq] WARNING: zero tokens consumed — no real candidates were generated.")
    print(f"[groq] NEXT: python main/data_pipeline.py --raw {out}")


if __name__ == "__main__":
    main()
