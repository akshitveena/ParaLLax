"""
diagnose_sdae.py — apples-to-apples: does the SDAE's validity signal survive confound removal?

The pooled ceiling (0.286 f1_B) was measured AFTER residualizing length/latex/#steps/dataset.
The SDAE's 0.64 chain_f1_B was raw. This probes the SDAE's chain representation (attention-
pooled step-codes) with the SAME residualized protocol, so we compare like for like.

    python main/diagnose_sdae.py --cache data/step_cache.pt \
        --checkpoint checkpoints_sdae/sdae_best.pt --data_dir data/processed_pb --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdae_prm import StepSDAE_PRM
from train_sdae import StepDS, collate
from torch.utils.data import DataLoader
from data_pipeline import load_candidates


def probe(X, y, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return float(accuracy_score(yte, p)), float(f1_score(yte, p, pos_label="B"))


def residualize(X, C):
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--checkpoint", default="checkpoints_sdae/sdae_best.pt")
    ap.add_argument("--data_dir", default="data/processed_pb")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    recs = torch.load(args.cache, weights_only=False)
    ids = [r["id"] for r in recs]
    y = np.array([r["chain"] for r in recs])

    # SDAE pooled chain representation
    dev = torch.device(args.device)
    model = StepSDAE_PRM().to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.eval()
    pooled = []
    with torch.no_grad():
        for X, SL, pad, ch in DataLoader(StepDS(recs), batch_size=64, collate_fn=collate):
            _, _, _, pl = model(X.to(dev), pad.to(dev), None)
            pooled.append(pl.cpu().numpy())
    Z = np.concatenate(pooled, 0)

    # confounds by id, from the processed candidates
    cands = {c.record_id: c for c in load_candidates(Path(args.data_dir) / "candidates.jsonl")}
    length, latex, nsteps, ds = [], [], [], []
    for i in ids:
        c = cands.get(i)
        t = (c.response_text or c.full_text or "") if c else ""
        length.append(len(t.split()))
        latex.append((t.count("\\") + t.count("$")) / max(len(t.split()), 1))
        nsteps.append(c.num_steps if c else 0)
        ds.append(c.dataset if c else "unk")
    surf = StandardScaler().fit_transform(np.array([length, latex, nsteps], float).T)
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(np.array(ds).reshape(-1, 1))
    C = np.hstack([np.ones((len(ids), 1)), surf, oh])

    maj = float(max((y == "A").mean(), (y == "B").mean()))
    print("=" * 62)
    print("  SDAE — confound-controlled A/B (apples-to-apples vs pooled 0.286)")
    print("=" * 62)
    print(f"  n={len(y)}  majority={maj:.3f}")
    for name, X in [("sdae_pooled", Z), ("sdae_pooled_residualized", residualize(Z, C))]:
        acc, f1 = probe(X, y)
        print(f"  {name:28} acc={acc:.3f}  f1_B={f1:.3f}   Δ-vs-majority {acc - maj:+.3f}")
    print("-" * 62)
    print("  reference (pooled): raw_sbert_resid f1_B=0.286 | ridae_z_resid f1_B=0.267")
    print("  VERDICT: sdae_pooled_residualized f1_B >> 0.286  ->  step structure genuinely")
    print("           beat the pooling ceiling, not just length.")
    print("=" * 62)


if __name__ == "__main__":
    main()
