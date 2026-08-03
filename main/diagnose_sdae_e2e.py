"""
diagnose_sdae_e2e.py — leakage-free, confound-controlled A/B for the END-TO-END SDAE.

Encodes each candidate's CLEAN steps with the TRAINED encoder, attention-pools to a chain
code, residualizes length/latex/#steps/dataset, and probes A/B fit-on-train / eval-on-val
(same split as training). Compares to the frozen-SDAE (0.436) and pooled ceiling (0.286).

    python main/diagnose_sdae_e2e.py --cache data/step_cache.pt \
        --checkpoint checkpoints_sdae_e2e/sdae_e2e_best.pt --data_dir data/processed_pb --device cpu
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
from data_pipeline import load_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--checkpoint", default="checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--data_dir", default="data/processed_pb")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    recs = torch.load(args.cache, weights_only=False)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(args.device); sdae.load_state_dict(ck["sdae"]); sdae.eval()

    dev = torch.device(args.device)
    Z = []
    with torch.no_grad():
        for i in range(0, len(recs), 32):
            X, target, pad, SL, ch = encode_batch(enc, recs[i:i + 32], dev, rng=None)
            _, _, _, pooled = sdae(X, pad, None)
            Z.append(pooled.cpu().numpy())
    Z = np.concatenate(Z, 0)
    y = np.array([r["chain"] for r in recs])

    cd = {c.record_id: c for c in load_candidates(Path(args.data_dir) / "candidates.jsonl")}
    L, LA, NS, DS = [], [], [], []
    for r in recs:
        c = cd.get(r["id"]); t = (c.response_text or c.full_text or "") if c else ""
        L.append(len(t.split())); LA.append((t.count("\\") + t.count("$")) / max(len(t.split()), 1))
        NS.append(c.num_steps if c else 0); DS.append(c.dataset if c else "unk")
    surf = StandardScaler().fit_transform(np.array([L, LA, NS], float).T)
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(np.array(DS).reshape(-1, 1))
    C = np.hstack([np.ones((len(recs), 1)), surf, oh])

    # same train/val split as training; fit residualization + probe on TRAIN, eval on VAL
    rng = np.random.RandomState(42); idx = np.arange(len(recs)); rng.shuffle(idx)
    cut = int(0.8 * len(recs)); tri, vai = idx[:cut], idx[cut:]
    beta, *_ = np.linalg.lstsq(C[tri], Z[tri], rcond=None)
    Zr = Z - C @ beta

    def evl(X):
        clf = LogisticRegression(max_iter=3000).fit(X[tri], y[tri]); p = clf.predict(X[vai])
        return accuracy_score(y[vai], p), f1_score(y[vai], p, pos_label="B")

    a1, f1 = evl(Z); a2, f2 = evl(Zr)
    print("=" * 60)
    print("  E2E SDAE — leakage-free, confound-controlled (held-out val)")
    print("=" * 60)
    print(f"  e2e_pooled            acc={a1:.3f}  f1_B={f1:.3f}")
    print(f"  e2e_pooled_residual   acc={a2:.3f}  f1_B={f2:.3f}")
    print("  reference: frozen-SDAE resid f1_B=0.436 | pooled ceiling resid f1_B=0.286")
    print("=" * 60)


if __name__ == "__main__":
    main()
