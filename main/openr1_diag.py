"""
openr1_diag.py — WHY did PRM-rerank fail on OpenR1? Noise (shift) or anti-signal (self-correction)?

Streams the same mixed OpenR1 problems, scores every candidate with the SDAE-PRM (P_sound),
and reports over ALL candidates:
  * AUC(P_sound -> correct):  ~0.5 = noise (distribution shift);  <0.5 = anti-correlated
  * mean P_sound for correct vs incorrect candidates
  * AUC(#steps -> correct) and corr(P_sound, #steps): is length driving it (self-correction
    traces are long, and if long -> lower P_sound but MORE correct, that's the anti-signal).

    python main/openr1_diag.py --n_problems 250 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textutils as T
from sdae_prm import StepSDAE_PRM
from ridae import RiDAE
from openr1_rerank import reasoning_text, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--n_problems", type=int, default=250)
    ap.add_argument("--max_scan", type=int, default=6000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(dev); sdae.load_state_dict(ck["sdae"]); sdae.eval()

    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)

    P, Y, N = [], [], []      # P_sound, correct, num_steps
    n = scanned = 0
    for row in ds:
        scanned += 1
        if scanned > args.max_scan or n >= args.n_problems:
            break
        gens = row.get("generations") or []
        corr = row.get("correctness_math_verify") or []
        if len(gens) < 2 or len(corr) != len(gens):
            continue
        ncorr = sum(bool(x) for x in corr)
        if not (0 < ncorr < len(gens)):
            continue
        for g, c in zip(gens, corr):
            steps = T.segment_steps(reasoning_text(g))
            P.append(score(enc, sdae, steps, dev))
            Y.append(int(bool(c)))
            N.append(min(len(steps), 60))
        n += 1
        if n % 50 == 0:
            print(f"  ...{n} problems ({len(P)} candidates)", flush=True)

    P, Y, N = np.array(P), np.array(Y), np.array(N)
    auc_p = roc_auc_score(Y, P)
    auc_n = roc_auc_score(Y, N)
    print("=" * 58)
    print(f"  OpenR1 diagnostic — {n} problems, {len(P)} candidates")
    print("=" * 58)
    print(f"  AUC(P_sound -> correct)   = {auc_p:.3f}")
    print(f"    mean P_sound | correct   = {P[Y==1].mean():.3f}")
    print(f"    mean P_sound | incorrect = {P[Y==0].mean():.3f}")
    print(f"  AUC(#steps  -> correct)   = {auc_n:.3f}  (are correct traces longer?)")
    print(f"  corr(P_sound, #steps)     = {np.corrcoef(P, N)[0,1]:+.3f}")
    print("-" * 58)
    print("  read: AUC(P_sound)~0.5 -> noise/shift ; <0.5 -> anti-signal (self-correction)")
    print("        if correct traces are LONGER (AUC#steps>0.5) and P_sound is NEG-corr with")
    print("        #steps, the PRM demotes long self-correcting-but-correct traces.")


if __name__ == "__main__":
    main()
