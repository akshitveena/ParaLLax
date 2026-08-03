"""
domain_transfer_controlled.py — is validity DOMAIN-INDEPENDENT *beyond difficulty*?

The raw leave-one-subject-out A/B transfer (~0.95) is inflated by the difficulty gradient
(hard datasets -> more Type B), which transfers across subjects trivially. Here we residualize
[length, latex, #steps, dataset] out of the chain reps FIRST (residualizer fit on the training
subjects only), then run the same transfer. If A/B still separates on the held-out subject,
validity generalizes across domains beyond difficulty. Collapse toward 0.5 -> it was difficulty.

    python main/domain_transfer_controlled.py --cache data/step_cache.pt \
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
from data_pipeline import load_candidates
from analyse import infer_subject
from geometry_sdae import chain_reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--checkpoint", default="checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--data_dir", default="data/processed_pb")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    dev = torch.device(args.device)
    recs = torch.load(args.cache, weights_only=False)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(dev); sdae.load_state_dict(ck["sdae"]); sdae.eval()

    Z = chain_reps(enc, sdae, recs, dev)
    y = np.array([r["chain"] for r in recs])
    cd = {c.record_id: c for c in load_candidates(Path(args.data_dir) / "candidates.jsonl")}
    subj = np.array([infer_subject(cd[r["id"]].problem) if r["id"] in cd else "other" for r in recs])

    L, LA, NS, DS = [], [], [], []
    for r in recs:
        c = cd.get(r["id"]); t = (c.response_text or c.full_text or "") if c else ""
        L.append(len(t.split())); LA.append((t.count("\\") + t.count("$")) / max(len(t.split()), 1))
        NS.append(c.num_steps if c else 0); DS.append(r["split"])
    surf = StandardScaler().fit_transform(np.array([L, LA, NS], float).T)
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(np.array(DS).reshape(-1, 1))
    C = np.hstack([np.ones((len(recs), 1)), surf, oh])

    def probe(Xtr, ytr, Xte, yte):
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        bi = list(clf.classes_).index("B")
        return roc_auc_score((yte == "B").astype(int), clf.predict_proba(Xte)[:, bi])

    print("=" * 64)
    print("  DOMAIN TRANSFER — raw vs confound-controlled (difficulty removed)")
    print("=" * 64)
    print(f"  {'held-out subject':18} {'n':>5} {'raw AUC':>9} {'resid AUC':>10}")
    raws, ress = [], []
    for s in sorted(set(subj)):
        te = subj == s; tr = ~te
        if (y[te] == "B").sum() < 5 or (y[te] == "A").sum() < 5 or (y[tr] == "B").sum() < 5:
            continue
        beta, *_ = np.linalg.lstsq(C[tr], Z[tr], rcond=None)   # residualizer fit on TRAIN only
        Zr = Z - C @ beta
        a_raw = probe(Z[tr], y[tr], Z[te], y[te])
        a_res = probe(Zr[tr], y[tr], Zr[te], y[te])
        raws.append(a_raw); ress.append(a_res)
        print(f"  {s:18} {int(te.sum()):5} {a_raw:9.3f} {a_res:10.3f}")
    print("-" * 64)
    print(f"  {'MEAN':18} {'':5} {np.mean(raws):9.3f} {np.mean(ress):10.3f}")
    print("=" * 64)
    print("  resid AUC stays high across held-out subjects -> validity is DOMAIN-INDEPENDENT")
    print("  beyond difficulty. Collapse toward 0.5 -> the transfer was mostly difficulty.")


if __name__ == "__main__":
    main()
