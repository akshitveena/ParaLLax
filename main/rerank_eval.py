"""
rerank_eval.py — Phase-3 feasibility: does the SDAE-PRM validity score improve reasoning?

For each problem with >=2 candidate solutions, score each candidate's SOUNDNESS with the
e2e SDAE (segment -> embed steps -> chain head), pick the top, and measure final-answer
accuracy vs baselines: random pick, self-consistency (majority answer), oracle (any correct).

Honest note: the PRM scores VALIDITY, reranking wants CORRECTNESS. They align for the
majority but diverge on Type B (flawed-but-correct) — this measures whether validity is a
good enough proxy to beat the baselines.

    python main/rerank_eval.py --data data/raw/pilot_omni.jsonl \
        --checkpoint checkpoints_sdae_e2e/sdae_e2e_best.pt --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textutils as T
from sdae_prm import StepSDAE_PRM
from ridae import RiDAE


def score(enc, sdae, steps_text, dev, which):
    if not steps_text:
        return -1.0
    with torch.no_grad():
        emb = enc._encode_with_grad(steps_text)                 # (n, 384)
        X = emb.unsqueeze(0)
        pad = torch.zeros(1, emb.size(0), dtype=torch.bool, device=dev)
        _, prm, cl, _ = sdae(X, pad, None)
        if which == "p_sound":
            return 1.0 - torch.sigmoid(cl).item()               # P(A / sound)
        return -torch.sigmoid(prm)[0].max().item()              # -max step-error prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/pilot_omni.jsonl")
    ap.add_argument("--checkpoint", default="checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--which", default="p_sound", choices=["p_sound", "neg_max_err"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(dev); sdae.load_state_dict(ck["sdae"]); sdae.eval()

    recs = [json.loads(l) for l in open(args.data) if l.strip()]
    rand = sc = rr = orac = 0.0
    n = n_mixed = 0
    for r in recs:
        cands = [c for c in r.get("candidates", [])
                 if (c.get("response_text") or "").strip() and "answer_correct" in c]
        if len(cands) < 2:
            continue
        corrects = [bool(c["answer_correct"]) for c in cands]
        scored = []
        for c in cands:
            steps = T.segment_steps(c["response_text"])
            s = score(enc, sdae, steps, dev, args.which)
            ans = T.normalise_answer(T.extract_answer(c["response_text"])[0] or "")
            scored.append((s, bool(c["answer_correct"]), ans))
        n += 1
        if 0 < sum(corrects) < len(corrects):
            n_mixed += 1
        rand += np.mean(corrects)
        orac += float(any(corrects))
        rr += float(max(scored, key=lambda x: x[0])[1])
        ans = [a for _, _, a in scored if a]
        if ans:
            maj = Counter(ans).most_common(1)[0][0]
            sc += float(any(corr for _, corr, a in scored if a == maj and corr))

    print("=" * 58)
    print(f"  Phase-3 reranking feasibility  ({n} problems, {n_mixed} mixed)")
    print(f"  scorer = {args.which}   data = {Path(args.data).name}")
    print("=" * 58)
    print(f"  random pick (mean correct rate) = {rand/max(n,1):.3f}")
    print(f"  self-consistency (majority ans) = {sc/max(n,1):.3f}")
    print(f"  PRM-rerank                      = {rr/max(n,1):.3f}   <- ours")
    print(f"  oracle (any correct)            = {orac/max(n,1):.3f}")
    print("=" * 58)
    print("  reads: PRM-rerank > random  -> validity helps pick better candidates")
    print("         PRM-rerank vs self-consistency -> does it beat the standard baseline")
    print("         gap to oracle -> headroom")


if __name__ == "__main__":
    main()
