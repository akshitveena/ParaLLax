"""
eval_typeb_retrieval.py — reranking FOR Type B (the miner reframe).

Instead of selecting correct answers (which labels flawed-but-correct as 'good' and erases
the phenomenon), rank answer-correct candidates by P(Type-B) and ask: of the model's most
confident Type-B calls, how many are TRULY Type B (per ProcessBench humans)? That's the
mining precision — the useful metric for growing a Type-B corpus / flagging lucky-but-wrong
reasoning. Held-out val, in-distribution, free.

    python main/eval_typeb_retrieval.py --cache data/step_cache.pt \
        --checkpoint checkpoints_sdae_e2e/sdae_e2e_best.pt --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdae_prm import StepSDAE_PRM
from ridae import RiDAE
from train_sdae_e2e import encode_batch, split_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--checkpoint", default="checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score, roc_auc_score

    recs = torch.load(args.cache, weights_only=False)
    _, va = split_recs(recs)                       # held-out (unseen in training)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(args.device); sdae.load_state_dict(ck["sdae"]); sdae.eval()
    dev = torch.device(args.device)

    scores, y = [], []
    with torch.no_grad():
        for i in range(0, len(va), 32):
            X, target, pad, SL, ch = encode_batch(enc, va[i:i + 32], dev, rng=None)
            _, _, cl, _ = sdae(X, pad, None)
            scores += torch.sigmoid(cl).cpu().numpy().tolist()   # P(Type-B)
            y += ch.cpu().numpy().tolist()
    scores, y = np.array(scores), np.array(y)      # y: 1=B, 0=A
    base = float(y.mean())
    ap_ = average_precision_score(y, scores)
    auc = roc_auc_score(y, scores)

    order = np.argsort(-scores)
    print("=" * 58)
    print(f"  Type-B RETRIEVAL (miner) — held-out val, n={len(y)}, {int(y.sum())} true Type-B")
    print("=" * 58)
    print(f"  base rate (random precision) = {base:.3f}")
    print(f"  average precision (AUPRC)    = {ap_:.3f}")
    print(f"  ROC-AUC                      = {auc:.3f}")
    print(f"  precision@k (top-k most-confident Type-B calls):")
    for k in (10, 20, 30, 50, 100):
        if k <= len(y):
            prec = float(y[order[:k]].mean())
            print(f"     @{k:<4} = {prec:.3f}   (lift {prec/max(base,1e-9):.2f}x over base)")
    print("-" * 58)
    print("  read: precision@k >> base rate -> a useful Type-B miner (the flagged ones")
    print("        really are wrong-approach-right-answer). This is utility that KEEPS")
    print("        the phenomenon as the target, unlike correctness-reranking.")


if __name__ == "__main__":
    main()
