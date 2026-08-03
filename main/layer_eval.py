"""
layer_eval.py — validity-as-layer reranking on distribution-matched (Qwen) best-of-N.

Our SDAE-PRM scores each candidate's soundness; we rerank and compare to self-consistency /
random / oracle. The DECISIVE number is AUC(P_sound -> correct): on OOD R1 traces it was 0.478
(noise). If it is well above 0.5 here (in-distribution Qwen), the R1 failure was distribution
shift and validity-as-layer works.

    python main/layer_eval.py --data data/raw/qwen_bestofn.jsonl \
        --checkpoint checkpoints/checkpoints_sdae_e2e/sdae_e2e_best.pt --device cpu
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
from openr1_rerank import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/qwen_bestofn.jsonl")
    ap.add_argument("--checkpoint", default="checkpoints/checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(dev); sdae.load_state_dict(ck["sdae"]); sdae.eval()

    recs = [json.loads(l) for l in open(args.data) if l.strip()]
    rand = sc = rr = orac = 0.0
    n = n_mixed = 0
    allP, allY = [], []
    for r in recs:
        cs = [c for c in r["candidates"] if (c.get("response_text") or "").strip()]
        if len(cs) < 2:
            continue
        scored = []
        for c in cs:
            steps = T.segment_steps(c["response_text"])
            s = score(enc, sdae, steps, dev)
            scored.append((s, bool(c["correct"]), T.normalise_answer(c.get("answer") or "")))
            allP.append(s); allY.append(int(bool(c["correct"])))
        n += 1
        corrects = [c for _, c, _ in scored]
        if 0 < sum(corrects) < len(corrects):
            n_mixed += 1
        rand += float(np.mean(corrects))
        orac += float(any(corrects))
        rr += float(max(scored, key=lambda x: x[0])[1])
        ans = [a for _, _, a in scored if a]
        if ans:
            maj = Counter(ans).most_common(1)[0][0]
            sc += float(any(c for _, c, a in scored if a == maj and c))

    d = max(n, 1)
    auc = roc_auc_score(allY, allP) if len(set(allY)) > 1 else float("nan")
    print("=" * 60)
    print(f"  VALIDITY-AS-LAYER — {n} problems ({n_mixed} mixed), in-distribution Qwen")
    print("=" * 60)
    print(f"  AUC(P_sound -> correct)  = {auc:.3f}   (>0.5 => validity predicts correctness)")
    print(f"  random pick              = {rand/d:.3f}")
    print(f"  self-consistency         = {sc/d:.3f}")
    print(f"  PRM-rerank (ours)        = {rr/d:.3f}")
    print(f"  oracle (any correct)     = {orac/d:.3f}")
    print("-" * 60)
    print("  OpenR1 (OOD R1) had AUC 0.478 (noise). AUC >> 0.5 here => the R1 failure was")
    print("  distribution shift, and the validity layer works in-distribution.")


if __name__ == "__main__":
    main()
