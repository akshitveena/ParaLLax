"""
train_controllable.py — the Type-B knob: validity-controllable reasoning generation.

Factored design (fixes Stage 0's content-free failure): CONTENT comes from the PROBLEM;
VALIDITY is a control prefix derived from our validated A/B labels. We fine-tune T5 to map
  (control, problem) -> reasoning of that validity
on ProcessBench answer-correct solutions:  A (clean) -> "valid",  B (flawed-but-correct) ->
"flawed-but-correct". At inference, toggle the control to generate a sound vs a Type-B solution
for the SAME problem. (Eval — scoring the two with our validity model + an oracle — is separate.)

    python main/train_controllable.py --model t5-base --epochs 5 --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CTRL_VALID = "valid solution"
CTRL_FLAWED = "flawed but correct solution"


def load_data(splits, max_steps_chars=1600):
    from datasets import load_dataset
    data = []
    for split in splits:
        ds = load_dataset("Qwen/ProcessBench", split=split)
        for r in ds:
            if not r["final_answer_correct"]:
                continue
            steps = [str(s) for s in r["steps"]]
            if len(steps) < 2:
                continue
            ctrl = CTRL_VALID if r["label"] < 0 else CTRL_FLAWED
            inp = f"{ctrl}: {r['problem']}"
            tgt = "\n".join(steps)[:max_steps_chars]
            data.append((inp, tgt, "A" if r["label"] < 0 else "B", r["problem"]))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="omnimath,olympiadbench,math,gsm8k")
    ap.add_argument("--model", default="t5-base")
    ap.add_argument("--ckpt_dir", default="checkpoints/controllable")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_in", type=int, default=256)
    ap.add_argument("--max_out", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    dev = torch.device(args.device)
    tok = T5TokenizerFast.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data = load_data([s.strip() for s in args.splits.split(",") if s.strip()])
    rng = np.random.RandomState(42); idx = np.arange(len(data)); rng.shuffle(idx)
    cut = int(0.9 * len(data))
    tr = [data[i] for i in idx[:cut]]; va = [data[i] for i in idx[cut:]]
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    print(f"[ctrl] train={len(tr)} val={len(va)} | {args.model} | control=valid/flawed prefix",
          flush=True)

    def batch(items):
        inp = tok([a for a, _, _, _ in items], max_length=args.max_in, truncation=True,
                  padding=True, return_tensors="pt").to(dev)
        lab = tok([b for _, b, _, _ in items], max_length=args.max_out, truncation=True,
                  padding=True, return_tensors="pt").input_ids.to(dev)
        lab[lab == tok.pad_token_id] = -100
        return inp, lab

    # a fixed held-out problem to eyeball the knob each epoch
    demo = va[0]
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr)); tot = 0.0; nb = 0; t0 = time.time()
        for i in range(0, len(order), args.batch_size):
            items = [tr[j] for j in order[i:i + args.batch_size]]
            inp, lab = batch(items)
            loss = model(**inp, labels=lab).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            outs = {}
            for ctrl in (CTRL_VALID, CTRL_FLAWED):
                ids = tok(f"{ctrl}: {demo[3]}", max_length=args.max_in, truncation=True,
                          return_tensors="pt").to(dev)
                g = model.generate(**ids, max_length=args.max_out, num_beams=3)
                outs[ctrl] = tok.decode(g[0], skip_special_tokens=True)
        print(f"[ctrl] ep{ep} loss={tot/nb:.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
        print(f"   PROBLEM: {demo[3][:110]}")
        print(f"   [valid ]: {outs[CTRL_VALID][:150]}")
        print(f"   [flawed]: {outs[CTRL_FLAWED][:150]}")
        model.save_pretrained(Path(args.ckpt_dir)); tok.save_pretrained(Path(args.ckpt_dir))

    print(f"[ctrl] saved -> {args.ckpt_dir}")
    print("[ctrl] NEXT: generate valid vs flawed on held-out problems and score BOTH with the")
    print("      SDAE validity model — does the knob shift soundness? (eval script next)")


if __name__ == "__main__":
    main()
