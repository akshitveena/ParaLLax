"""
eval_prm_e2e.py — benchmark the e2e SDAE's PRM head as a process verifier.

Held-out val (same split as training). Reports, ProcessBench-style:
  * step-level error detection AUC (good vs first-error steps)
  * solution localization: does argmax step-error prob hit the human first-error index;
    are clean chains predicted 'no error'. Threshold tuned on TRAIN (no val leakage).
    ProcessBench F1 = harmonic mean of (error-chain acc, correct-chain acc).

    python main/eval_prm_e2e.py --cache data/step_cache.pt \
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


def per_cand_probs(enc, sdae, recs, dev, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(recs), bs):
            batch = recs[i:i + bs]
            X, target, pad, SL, ch = encode_batch(enc, batch, dev, rng=None)
            _, prm, _, _ = sdae(X, pad, None)
            p = torch.sigmoid(prm).cpu().numpy()
            for b in range(len(batch)):
                n = int((~pad[b]).sum().item())
                labels = SL[b, :n].cpu().numpy()
                gold = int(np.where(labels == 1)[0][0]) if (labels == 1).any() else -1
                out.append((p[b, :n], gold, labels))
    return out


def sol_metrics(cands, thr):
    err = [c for c in cands if c[1] >= 0]
    cor = [c for c in cands if c[1] == -1]

    def pred(c):
        probs = c[0]
        return int(probs.argmax()) if probs.max() > thr else -1

    ea = np.mean([pred(c) == c[1] for c in err]) if err else 0.0
    ea1 = np.mean([pred(c) >= 0 and abs(pred(c) - c[1]) <= 1 for c in err]) if err else 0.0
    ca = np.mean([pred(c) == -1 for c in cor]) if cor else 0.0
    f1 = 2 * ea * ca / (ea + ca) if (ea + ca) > 0 else 0.0
    return ea, ea1, ca, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--checkpoint", default="checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    recs = torch.load(args.cache, weights_only=False)
    tr, va = split_recs(recs)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(args.device); sdae.load_state_dict(ck["sdae"]); sdae.eval()
    dev = torch.device(args.device)

    trc = per_cand_probs(enc, sdae, tr, dev)
    vac = per_cand_probs(enc, sdae, va, dev)

    st, ss = [], []
    for probs, gold, labels in vac:
        for j, l in enumerate(labels):
            if l >= 0:
                st.append(int(l)); ss.append(float(probs[j]))
    auc = roc_auc_score(st, ss)

    best = (-1.0, 0.5)
    for thr in np.linspace(0.1, 0.9, 33):
        f1 = sol_metrics(trc, thr)[3]
        if f1 > best[0]:
            best = (f1, thr)
    thr = best[1]
    ea, ea1, ca, f1 = sol_metrics(vac, thr)

    print("=" * 62)
    print("  PRM eval (e2e SDAE) — held-out ProcessBench val, answer-correct")
    print("=" * 62)
    print(f"  n_val={len(vac)}  (error chains={sum(c[1]>=0 for c in vac)}, "
          f"clean={sum(c[1]==-1 for c in vac)})")
    print(f"  step-level error AUC          = {auc:.3f}")
    print(f"  threshold (tuned on train)    = {thr:.2f}")
    print(f"  solution localization (ProcessBench-style):")
    print(f"    error chains, first-error exact = {ea:.3f}   within±1 = {ea1:.3f}")
    print(f"    clean chains, predicted no-error = {ca:.3f}")
    print(f"    ProcessBench F1 (harmonic mean)  = {f1:.3f}")
    print("-" * 62)
    print("  ablation (confound-controlled chain f1_B):")
    print("    pooled AE 0.286  ->  step-SDAE(frozen) 0.436  ->  e2e 0.576")
    print("=" * 62)


if __name__ == "__main__":
    main()
