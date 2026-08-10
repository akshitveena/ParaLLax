"""
judge_confound_check.py — W3: does the LLM judge itself read difficulty?

The sharpest attack on the paper: on the hardest splits the Type-B labels come from an LLM judge
(kappa=0.60). If the judge is MORE likely to call a solution flawed simply because the problem is
hard, then the labels are confounded by difficulty and a detector "recovering" them is partly
recovering difficulty — circular.

Test it directly on the 100 problems where we have BOTH the judge's label AND the human label
(data/processbench_calib.jsonl). For each label source, measure how strongly the four confounds
[length, latex, #steps, dataset] predict a "B" verdict:

  * confound->label AUC   : how difficulty-driven each labeller is (higher = more difficulty-driven)
  * standardized LENGTH coefficient : length is the dominant confound (W13, r=+0.415)

If the JUDGE's difficulty-dependence materially exceeds the HUMANS', the judge labels are
contaminated and that must be reported. If they match, the judge's difficulty-dependence is just
the real phenomenon (Type-B genuinely rises with difficulty) and the labels are clean — say so.

Bootstrap the (judge - human) AUC gap so a small n=100 is not over-read.

    python experiments/judge_confound_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))
from multiseed_ablation import build_confounds


def confound_fit(Cfeat, yb, seed=0):
    """LogisticRegression(confounds -> B); return (AUC, standardized length coef)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if len(set(yb)) < 2:
        return float("nan"), float("nan")
    clf = LogisticRegression(max_iter=2000).fit(Cfeat, yb)
    auc = roc_auc_score(yb, clf.predict_proba(Cfeat)[:, 1])
    return auc, float(clf.coef_[0][0])          # col 0 of Cfeat = standardized length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(ROOT / "data/processbench_calib.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    calib = {json.loads(l)["id"]: json.loads(l)
             for l in Path(args.calib).read_text().splitlines() if l.strip()}
    base = torch.load(args.cache, weights_only=False)
    sub = [r for r in base if r["id"] in calib
           and calib[r["id"]].get("human_AB") in ("A", "B")
           and calib[r["id"]].get("ours_AB") in ("A", "B")]
    C = build_confounds(sub, args.data_dir)
    Cfeat = C[:, 1:]                              # drop intercept; col 0 = standardized length
    human = np.array([1 if calib[r["id"]]["human_AB"] == "B" else 0 for r in sub])
    judge = np.array([1 if calib[r["id"]]["ours_AB"] == "B" else 0 for r in sub])

    h_auc, h_len = confound_fit(Cfeat, human)
    j_auc, j_len = confound_fit(Cfeat, judge)
    agree = float((human == judge).mean())

    # bootstrap the judge-minus-human confound->label AUC gap
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(0); n = len(sub); diffs = []
    for _ in range(args.n_boot):
        idx = rng.randint(0, n, n)
        if len(set(human[idx])) < 2 or len(set(judge[idx])) < 2:
            continue
        ch = LogisticRegression(max_iter=1000).fit(Cfeat[idx], human[idx])
        cj = LogisticRegression(max_iter=1000).fit(Cfeat[idx], judge[idx])
        ah = roc_auc_score(human[idx], ch.predict_proba(Cfeat[idx])[:, 1])
        aj = roc_auc_score(judge[idx], cj.predict_proba(Cfeat[idx])[:, 1])
        diffs.append(aj - ah)
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    print("=" * 70)
    print(f"  W3 — DOES THE JUDGE READ DIFFICULTY?   (n={len(sub)} dual-labelled)")
    print("=" * 70)
    print(f"  human B rate {human.mean():.2f} | judge B rate {judge.mean():.2f} | "
          f"agreement {agree:.2f}")
    print(f"  {'labeller':<10}{'confound->B AUC':>18}{'std length coef':>18}")
    print("  " + "-" * 46)
    print(f"  {'human':<10}{h_auc:>18.3f}{h_len:>18.3f}")
    print(f"  {'judge':<10}{j_auc:>18.3f}{j_len:>18.3f}")
    print(f"\n  judge − human confound-AUC gap = {j_auc - h_auc:+.3f}"
          f"   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("=" * 70)
    if lo <= 0 <= hi:
        print("  READING: CI spans 0 — the judge is NO more difficulty-driven than humans.")
        print("  The judge's difficulty-dependence is the real phenomenon, not contamination.")
        print("  Labels are clean on this axis; report this as W3's answer.")
    elif lo > 0:
        print("  READING: gap > 0 (CI excludes 0) — the JUDGE is MORE difficulty-driven than")
        print("  humans. Label contamination on the difficulty axis; must be reported and the")
        print("  judge-labelled splits treated as a robustness check, not primary.")
    else:
        print("  READING: gap < 0 — judge is LESS difficulty-driven than humans (labels clean).")
    print("=" * 70)


if __name__ == "__main__":
    main()
