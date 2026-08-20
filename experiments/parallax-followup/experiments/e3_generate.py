"""
E3 stage 1 (A100/GPU): generate a second, multi-model corpus. NO judging here — that runs on the
Mac (Claude API), so the API key never touches the box. This stage:
  sample k solutions/problem from 2 long-CoT math models on MATH-500 (+ optional AIME)
  -> keep ONLY answer-correct  -> segment steps  -> dump raw answer-correct solutions to JSONL.

Two A100-feasible models from different groups (edit --models to taste):
  Qwen/Qwen2.5-Math-7B-Instruct              (Qwen)
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    (DeepSeek, long <think> CoT)

    HF_HOME=/workspace/ridae/.hf python experiments/.../e3_generate.py --k 8 --limit 500 --out e3_raw.jsonl
Then on the Mac:  python .../e3_judge_and_test.py --raw e3_raw.jsonl   (Claude judge + ParaLLax tests)
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "main"))
import textutils as T   # extract_answer, answers_match

DEFAULT_MODELS = ["Qwen/Qwen2.5-Math-7B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
SYS = ("Solve the problem step by step. Put your final answer in \\boxed{}.")


def load_problems(limit):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        out.append(dict(problem_id=f"math500-{i}", statement=r["problem"],
                        gold=str(r.get("answer", "")), dataset="MATH-500"))
    return out


def segment_long_cot(text):
    # long-CoT-aware: score the post-</think> solution if present, else the whole text; split on newlines
    body = text.split("</think>")[-1] if "</think>" in text else text
    steps = [s.strip() for s in re.split(r"\n+", body) if len(s.strip()) > 3]
    return steps or [body.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="e3_raw.jsonl")
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    probs = load_problems(args.limit)
    print(f"[E3] {len(probs)} problems x {args.k} samples x {len(args.models)} models", flush=True)
    fh = open(args.out, "w")
    stats = {"sampled": 0, "answer_correct": 0}
    for hf in args.models:
        print(f"[E3] loading {hf}", flush=True)
        tok = AutoTokenizer.from_pretrained(hf)
        model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, device_map="auto").eval()
        t0 = time.time()
        for pi, p in enumerate(probs):
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": p["statement"]}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)
            ids = enc["input_ids"].to(model.device)
            attn = enc["attention_mask"].to(model.device)
            plen = ids.shape[1]
            with torch.no_grad():
                gen = model.generate(input_ids=ids, attention_mask=attn, do_sample=True,
                                     temperature=args.temperature, top_p=0.95,
                                     num_return_sequences=args.k, max_new_tokens=args.max_new,
                                     pad_token_id=tok.eos_token_id)
            for g in gen:
                text = tok.decode(g[plen:], skip_special_tokens=True)
                stats["sampled"] += 1
                ans, _ = T.extract_answer(text)
                ok, _ = T.answers_match(ans, p["gold"])
                if not ok:
                    continue
                stats["answer_correct"] += 1
                fh.write(json.dumps(dict(problem_id=p["problem_id"], model=hf, dataset=p["dataset"],
                                         statement=p["statement"], gold=p["gold"], text=text,
                                         extracted=ans, steps=segment_long_cot(text))) + "\n")
            if (pi + 1) % 25 == 0:
                print(f"  {hf.split('/')[-1]} {pi+1}/{len(probs)}  ok={stats['answer_correct']} "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
        del model; torch.cuda.empty_cache()
    fh.close()
    print(f"[E3] sampled {stats['sampled']}, answer-correct {stats['answer_correct']} -> {args.out}")
    print(f"[E3] NEXT (on your MAC, key present): judge these with the mechanism judge + run ParaLLax tests.")


if __name__ == "__main__":
    main()
