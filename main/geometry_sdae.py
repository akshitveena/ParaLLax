"""
geometry_sdae.py — Phase 4: the geometric map of reasoning (step-structured model).

Chain representation = attention-pooled step-codes (256-d) from the e2e SDAE. We:
  * UMAP the space, colored by TYPE (A/B), SUBJECT, and DATASET  -> outputs_geometry/*.png
  * probe how decodable validity (A/B) vs topic (subject/dataset) are from z
  * DOMAIN TRANSFER: train the A/B probe on all-but-one subject, test on the held-out
    subject. If A/B still separates, validity is DOMAIN-INDEPENDENT (the geometric claim:
    z organizes by reasoning soundness, not by topic).

    python main/geometry_sdae.py --cache data/step_cache.pt \
        --checkpoint checkpoints_sdae_e2e/sdae_e2e_best.pt --out_dir outputs_geometry --device cpu
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
from train_sdae_e2e import encode_batch
from data_pipeline import load_candidates
from analyse import infer_subject


def chain_reps(enc, sdae, recs, dev, bs=32):
    Z = []
    with torch.no_grad():
        for i in range(0, len(recs), bs):
            X, t, pad, SL, ch = encode_batch(enc, recs[i:i + bs], dev, rng=None)
            _, _, _, pooled = sdae(X, pad, None)
            Z.append(pooled.cpu().numpy())
    return np.concatenate(Z, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--checkpoint", default="checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--data_dir", default="data/processed_pb")
    ap.add_argument("--out_dir", default="outputs_geometry")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import train_test_split

    dev = torch.device(args.device)
    recs = torch.load(args.cache, weights_only=False)
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(dev); sdae.load_state_dict(ck["sdae"]); sdae.eval()

    Z = chain_reps(enc, sdae, recs, dev)
    y = np.array([r["chain"] for r in recs])                      # 'A' / 'B'
    cd = {c.record_id: c for c in load_candidates(Path(args.data_dir) / "candidates.jsonl")}
    subj = np.array([infer_subject(cd[r["id"]].problem) if r["id"] in cd else "other" for r in recs])
    dset = np.array([r["split"] for r in recs])

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    import umap
    xy = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=42).fit_transform(Z)

    def plot(color, title, fname):
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in sorted(set(color)):
            m = color == c
            ax.scatter(xy[m, 0], xy[m, 1], s=7, alpha=0.5, label=str(c))
        ax.set_title(title); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(fontsize=7, markerscale=2); fig.tight_layout()
        fig.savefig(out / fname, dpi=140); plt.close(fig)
        print(f"[geometry] saved {out / fname}")

    plot(y, "step-code space by TYPE (A=sound, B=flawed)", "geom_by_type.png")
    plot(subj, "step-code space by SUBJECT", "geom_by_subject.png")
    plot(dset, "step-code space by DATASET", "geom_by_dataset.png")

    def probe_auc(X, yy):
        Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.25, random_state=42, stratify=yy)
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        bi = list(clf.classes_).index("B")
        return roc_auc_score((yte == "B").astype(int), clf.predict_proba(Xte)[:, bi])

    def probe_acc(X, yy):
        Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.25, random_state=42, stratify=yy)
        return accuracy_score(yte, LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict(Xte))

    print("\n[geometry] what does z encode? (validity vs topic)")
    print(f"  A/B (validity) AUC = {probe_auc(Z, y):.3f}")
    print(f"  subject accuracy   = {probe_acc(Z, subj):.3f}  (chance {1/len(set(subj)):.2f})")
    print(f"  dataset accuracy   = {probe_acc(Z, dset):.3f}  (chance {1/len(set(dset)):.2f})")

    print("\n[geometry] DOMAIN TRANSFER — train A/B on OTHER subjects, test on held-out:")
    for s in sorted(set(subj)):
        te = subj == s; tr = ~te
        if (y[te] == "B").sum() < 5 or (y[te] == "A").sum() < 5 or (y[tr] == "B").sum() < 5:
            continue
        clf = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
        bi = list(clf.classes_).index("B")
        auc = roc_auc_score((y[te] == "B").astype(int), clf.predict_proba(Z[te])[:, bi])
        print(f"    held-out {s:16} n={int(te.sum()):4}  A/B AUC = {auc:.3f}")

    print("\n  read: if A/B AUC stays high across held-out subjects while subject/dataset")
    print("  accuracy is modest -> z organizes by VALIDITY (domain-independent), not topic.")


if __name__ == "__main__":
    main()
