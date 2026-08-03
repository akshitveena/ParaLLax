"""
build_step_embeddings.py — cache per-STEP MiniLM embeddings + ProcessBench step labels.

The step-structured SDAE-PRM trains on these. We keep each step as its own vector (NO
cross-step pooling — that pooling was the ~0.29 f1_B ceiling), and we use ProcessBench's
native `steps` (already segmented by the annotators) and `label` (index of the first
erroneous step, -1 if the whole chain is clean) as PRM supervision.

Per-step target (ProcessBench convention):
    step j  ->  0 (good)      for j < label
                1 (error)     for j == label   (the first error)
               -1 (ignore)    for j >  label   (unannotated past the first error)
    label == -1 -> all steps good.

    python main/build_step_embeddings.py --out data/step_cache.pt --device cpu
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def step_labels_from(label: int, n: int) -> np.ndarray:
    y = np.zeros(n, dtype=np.int64)
    if label is None or label < 0:
        return y
    for j in range(n):
        y[j] = 0 if j < label else (1 if j == label else -1)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="omnimath,olympiadbench,math,gsm8k")
    ap.add_argument("--out", default="data/step_cache.pt")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--limit_per_split", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.encoder, device=args.device)

    meta, all_steps, spans = [], [], []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ds = load_dataset("Qwen/ProcessBench", split=split)
        kept = 0
        for r in ds:
            if not r["final_answer_correct"]:
                continue
            if args.limit_per_split and kept >= args.limit_per_split:
                break
            steps = [str(s) for s in r["steps"]]
            if len(steps) < 2:
                continue
            kept += 1
            start = len(all_steps); all_steps.extend(steps)
            spans.append((start, len(all_steps)))
            meta.append({"id": r["id"], "split": split, "n": len(steps),
                         "label": int(r["label"]),
                         "chain": "A" if r["label"] < 0 else "B"})

    print(f"[steps] embedding {len(all_steps)} steps from {len(meta)} candidates ...")
    emb = np.asarray(enc.encode(all_steps, batch_size=128, show_progress_bar=True),
                     dtype=np.float32)

    out = []
    for m, (a, b) in zip(meta, spans):
        out.append({"id": m["id"], "split": m["split"], "chain": m["chain"],
                    "steps_emb": emb[a:b],
                    "steps_text": all_steps[a:b],     # for end-to-end (trainable) encoder
                    "step_labels": step_labels_from(m["label"], m["n"])})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[steps] cached {len(out)} candidates -> {args.out}")
    print(f"[steps] chain dist: {dict(Counter(r['chain'] for r in out))}")
    print(f"[steps] avg steps/candidate: {np.mean([r['steps_emb'].shape[0] for r in out]):.1f}")
    n_err = sum(int((r['step_labels'] == 1).any()) for r in out)
    print(f"[steps] candidates with a labeled error step: {n_err}")


if __name__ == "__main__":
    main()
