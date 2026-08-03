"""
openr1_rerank.py — Phase-3 reranking on OpenR1-Math-220k (a real best-of-N test).

Streams problems that have a MIX of correct/incorrect DeepSeek-R1 generations, scores each
generation's reasoning soundness with the e2e SDAE-PRM, reranks (pick most-sound), and
compares final-answer accuracy to random / self-consistency / oracle.

We evaluate on MIXED problems only (>=1 correct AND >=1 wrong) — the subset where selection
actually matters, so oracle = 1.0 and the question is how close PRM-rerank gets, and whether
it beats random and self-consistency.

R1 traces carry the reasoning in <think>...</think>; we segment that (the analog of a
ProcessBench solution). Transfer from ProcessBench-trained PRM to R1 traces is the open risk.

    python main/openr1_rerank.py --n_problems 300 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textutils as T
from sdae_prm import StepSDAE_PRM
from ridae import RiDAE


def reasoning_text(gen: str) -> str:
    think, resp = T.split_think_response(gen)
    return think if think.strip() else (resp if resp.strip() else gen)


def score(enc, sdae, steps, dev, max_steps=60):
    steps = steps[:max_steps]
    if not steps:
        return -1.0
    with torch.no_grad():
        emb = enc._encode_with_grad(steps)
        X = emb.unsqueeze(0)
        pad = torch.zeros(1, emb.size(0), dtype=torch.bool, device=dev)
        _, prm, cl, _ = sdae(X, pad, None)
        return 1.0 - torch.sigmoid(cl).item()          # P(sound)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--n_problems", type=int, default=300)
    ap.add_argument("--max_scan", type=int, default=6000)
    ap.add_argument("--min_gens", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(dev); sdae.load_state_dict(ck["sdae"]); sdae.eval()

    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)

    rand = sc = rr = orac = 0.0
    n = scanned = 0
    for row in ds:
        scanned += 1
        if scanned > args.max_scan or n >= args.n_problems:
            break
        gens = row.get("generations") or []
        corr = row.get("correctness_math_verify") or []
        if len(gens) < args.min_gens or len(corr) != len(gens):
            continue
        ncorr = sum(bool(x) for x in corr)
        if not (0 < ncorr < len(gens)):           # need a genuine mix
            continue
        scored = []
        for g, c in zip(gens, corr):
            steps = T.segment_steps(reasoning_text(g))
            s = score(enc, sdae, steps, dev)
            a = T.normalise_answer(T.extract_answer(g)[0] or "")
            scored.append((s, bool(c), a))
        n += 1
        cs = [c for _, c, _ in scored]
        rand += float(np.mean(cs))
        orac += 1.0                                # mixed => a correct one always exists
        rr += float(max(scored, key=lambda x: x[0])[1])
        ans = [a for _, _, a in scored if a]
        if ans:
            maj = Counter(ans).most_common(1)[0][0]
            sc += float(any(c for _, c, a in scored if a == maj and c))
        if n % 50 == 0:
            print(f"  ...{n} mixed problems (scanned {scanned})", flush=True)

    d = max(n, 1)
    print("=" * 58)
    print(f"  OpenR1 reranking — {n} MIXED problems (scanned {scanned})")
    print("=" * 58)
    print(f"  random pick (mean correct rate) = {rand/d:.3f}")
    print(f"  self-consistency (majority ans) = {sc/d:.3f}")
    print(f"  PRM-rerank (ours)               = {rr/d:.3f}   <- validity selector")
    print(f"  oracle (any correct)            = {orac/d:.3f}")
    print("-" * 58)
    print("  PRM-rerank > random  -> validity helps select the correct solution")
    print("  PRM-rerank vs self-consistency -> beats the standard baseline?")


if __name__ == "__main__":
    main()
