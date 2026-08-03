"""
gen_qwen_bestofn.py — generate N Qwen2.5-Instruct solutions per problem (Ollama).

Distribution-MATCHED to ProcessBench (Qwen-instruct, short numbered step solutions, no long
<think> dumps) so our validity model can actually read them — the fix for the R1 shift that
sank the OpenR1 rerank. Saves candidates + deterministic correctness for the layer experiment.

Prereq:  ollama pull qwen2.5:7b-instruct   (≈4.7GB q4; runs on a 16GB M3)
    python api_generation/gen_qwen_bestofn.py --limit 120 --n 6 --out data/raw/qwen_bestofn.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402
from claude_generate import load_problems  # noqa: E402

SYS = ("Solve the competition mathematics problem step by step. Number each step on its own "
       "line. Show your work. State the final answer in \\boxed{}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--dataset", default="math", choices=["math", "olympiadbench", "omnimath"])
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_difficulty", type=float, default=None)
    ap.add_argument("--min_difficulty", type=float, default=None)
    ap.add_argument("--out", default="data/raw/qwen_bestofn.jsonl")
    args = ap.parse_args()

    import ollama
    client = ollama.Client()
    try:
        have = {m.model for m in client.list().models}
        if not any(args.model.split(":")[0] in h for h in have):
            print(f"[gen] model '{args.model}' not found. Run: ollama pull {args.model}"); sys.exit(1)
    except Exception as e:
        print(f"[gen] cannot reach Ollama ({e}). Is it running? (ollama serve)"); sys.exit(1)

    kw = {}
    if args.dataset == "omnimath":
        kw = dict(min_difficulty=args.min_difficulty, max_difficulty=args.max_difficulty)
    probs = load_problems(args.dataset, args.limit, **kw)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")
    t0 = time.time(); n_corr = n_cand = 0
    for pi, p in enumerate(probs):
        cands = []
        for k in range(args.n):
            try:
                r = client.chat(model=args.model,
                                messages=[{"role": "system", "content": SYS},
                                          {"role": "user", "content": p["problem"]}],
                                options={"temperature": args.temperature, "seed": k,
                                         "num_predict": 1500, "num_ctx": 4096})
                txt = r.message.content or ""
            except Exception as e:
                txt = ""
                print(f"[gen] warn p{pi} k{k}: {str(e)[:60]}")
            ans = T.extract_answer(txt)[0]
            corr = bool(ans) and T.answers_match(ans, p["gold_answer"])[0]
            cands.append({"response_text": txt, "answer": ans, "correct": bool(corr)})
            n_cand += 1; n_corr += int(corr)
        fh.write(json.dumps({"problem": p["problem"], "gold_answer": p["gold_answer"],
                             "candidates": cands}, ensure_ascii=False) + "\n"); fh.flush()
        if (pi + 1) % 10 == 0:
            print(f"[gen] {pi+1}/{len(probs)} problems  correct {n_corr}/{n_cand} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
    fh.close()
    print(f"[gen] wrote {out}  ({n_corr}/{n_cand} candidates correct by deterministic match)")
    print(f"[gen] NEXT: python main/layer_eval.py --data {out}")


if __name__ == "__main__":
    main()
