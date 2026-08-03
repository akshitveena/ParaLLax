"""
diagnose_ceiling.py — is there ANY wrong-approach-right-answer signal in pooled
MiniLM space, BEYOND surface confounds?

Residualizes embeddings against [length, latex-density, num_steps, dataset] (removes
the linearly-predictable confound component) and probes A/B on what's LEFT. Run on both
the RAW pretrained SBERT (is the signal even in the base encoder?) and RiDAE's z.

Decides the fork:
  * residualized acc ~= majority AND f1_B ~ 0  -> NO validity signal in pooled MiniLM
    space -> supervised fine-tuning on this encoder will hit the ceiling -> go to a
    validity-aware encoder (PRM) or a step-structured representation. Do NOT retrain here.
  * residualized acc stays above majority      -> the signal exists but is unexploited
    -> supervised, length-controlled contrastive fine-tuning is worth a retrain.

No training required — reads existing embeddings.
    python main/diagnose_ceiling.py --data_dir data/processed_pb \
        --checkpoint checkpoints_pb/ridae_best.pt --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import load_candidates
from ridae import RiDAE
import analyse as A


def probe(X, y, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return float(accuracy_score(yte, p)), float(f1_score(yte, p, pos_label="B"))


def residualize(X, C):
    """Remove the linear component predictable from confound matrix C (with intercept)."""
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_pb")
    ap.add_argument("--checkpoint", default="checkpoints_pb/ridae_best.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    cands = load_candidates(Path(args.data_dir) / "candidates.jsonl")
    labels = np.array([c.candidate_type for c in cands])
    m = np.isin(labels, ["A", "B"])
    y = labels[m]

    length, latex, nsteps, ds, txt = A._confound_features(cands)
    surf = StandardScaler().fit_transform(np.vstack([length, latex, nsteps]).T)
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(ds.reshape(-1, 1))
    C = np.hstack([np.ones((len(cands), 1)), surf, oh])[m]

    model = RiDAE.load(args.checkpoint, device=args.device); model.eval()
    z = model.encode([c.full_text for c in cands])[m]

    from sentence_transformers import SentenceTransformer
    base = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    sb = np.asarray(base.encode(list(txt), batch_size=64, show_progress_bar=False))[m]

    maj = float(max((y == "A").mean(), (y == "B").mean()))
    print("=" * 64)
    print("  CEILING DIAGNOSTIC — A/B signal beyond confounds?")
    print("=" * 64)
    print(f"  n(A/B)={len(y)}  majority baseline={maj:.3f}")
    print(f"  {'representation':26} {'acc':>6} {'f1_B':>6}  Δ-vs-majority")
    for name, X in [("raw_sbert", sb),
                    ("raw_sbert_residualized", residualize(sb, C)),
                    ("ridae_z", z),
                    ("ridae_z_residualized", residualize(z, C))]:
        acc, f1 = probe(X, y)
        print(f"  {name:26} {acc:6.3f} {f1:6.3f}     {acc - maj:+.3f}")
    print("-" * 64)
    print("  READ:")
    print("   residualized acc ~= majority & f1_B ~ 0  -> CEILING: no validity signal in")
    print("     pooled MiniLM -> go to PRM / step-structured (do NOT retrain this encoder).")
    print("   residualized acc clearly > majority       -> signal exists -> supervised,")
    print("     length-controlled contrastive fine-tuning is worth a retrain.")
    print("=" * 64)


if __name__ == "__main__":
    main()
