"""
r3_metrics_and_overcontrol.py
-----------------------------
Priority-1 reviewer item R3. Two analyses, both CPU-only, both seconds on an M3.
No GPU. Do NOT send this to Kaggle.

R3a. AUC + AUPRC alongside f1_B for every row of Table 1 and Figure 1, so a
     reviewer can separate genuine signal loss from threshold miscalibration.
     (The review notes AUC "deflates far less" -- 0.76->0.67 vs 0.43->0.08 --
     so reporting only f1_B overstates the collapse.)

R3b. Re-run the difficulty-only null AND the confound-controlled scores with
     `dataset` DROPPED from the confound set.
     Why this matters: your own App B says most of the null's 0.515 is carried
     by the coarse dataset variable, and the control residualizes that same
     variable out. If the collapse shrinks a lot without it, part of what you
     are calling "confound removal" is over-control -- removing a legitimate
     4-way source signal. You need to know this before a reviewer asks.

PRE-COMMIT (write your answer down before running R3b):
  - If controlled f1_B rises by < ~0.05 without the dataset dummy, the collapse
    is not an over-control artifact. Report it as a robustness row and move on.
  - If it rises substantially, the honest framing changes: difficulty-*within*-
    dataset explains less than you claim, and the dataset variable is doing
    heavy lifting. Say so, and lean harder on the within-dataset control
    (Fig 2b), which is immune to this critique by construction.
Either outcome is reportable. Report whichever occurs.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------- R3a -----
def all_metrics(y_true, scores, threshold=None):
    """f1_B at a threshold, plus threshold-free AUC and AUPRC.

    If threshold is None, uses the base-rate quantile -- state whichever rule
    you use in the paper (the review asks for this explicitly).
    """
    auc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    if threshold is None:
        threshold = np.quantile(scores, 1.0 - y_true.mean())
    f1 = f1_score(y_true, (scores >= threshold).astype(int))
    return dict(f1_B=f1, AUC=auc, AUPRC=auprc, base_rate=y_true.mean())


def table1_with_auc(rows):
    """rows: {name: (y_true, raw_scores, controlled_scores_or_None)}"""
    print(f"{'representation':<34}{'raw f1':>8}{'raw AUC':>9}{'raw AP':>8}"
          f"{'ctl f1':>8}{'ctl AUC':>9}{'ctl AP':>8}")
    for name, (y, raw, ctl) in rows.items():
        r = all_metrics(y, raw)
        line = f"{name:<34}{r['f1_B']:>8.3f}{r['AUC']:>9.3f}{r['AUPRC']:>8.3f}"
        if ctl is not None:
            c = all_metrics(y, ctl)
            line += f"{c['f1_B']:>8.3f}{c['AUC']:>9.3f}{c['AUPRC']:>8.3f}"
        else:
            line += f"{'---':>8}{'---':>9}{'---':>8}"
        print(line)


# ---------------------------------------------------------------- R3b -----
def residualize(X, C, train_idx):
    """Remove the linear component of C from every column of X.
    Fit on train only -- this is the protocol's leakage guard."""
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        m = Ridge(alpha=1.0).fit(C[train_idx], X[train_idx, j])
        out[:, j] = X[:, j] - m.predict(C)
    return out


def run_overcontrol_check(X, y, confounds, dataset_onehot, seed=0):
    """
    X               : (n, d) representation
    y               : (n,) 0/1 Type-B labels
    confounds       : (n, 3) log_length, latex_density, n_steps
    dataset_onehot  : (n, 4) one-hot source dataset
    """
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)

    results = {}
    for label, C in [("WITH dataset", np.hstack([confounds, dataset_onehot])),
                     ("WITHOUT dataset", confounds)]:
        # --- difficulty-only null on this confound set ---
        null = LogisticRegression(max_iter=2000).fit(C[tr], y[tr])
        s_null = null.predict_proba(C[te])[:, 1]

        # --- confound-controlled probe on the representation ---
        Xr = residualize(X, C, tr)
        probe = LogisticRegression(max_iter=2000).fit(Xr[tr], y[tr])
        s_ctl = probe.predict_proba(Xr[te])[:, 1]

        results[label] = dict(null=all_metrics(y[te], s_null),
                              controlled=all_metrics(y[te], s_ctl))

    print("\n=== R3b: does dropping the dataset dummy change the story? ===")
    for label, r in results.items():
        print(f"\n{label}")
        print(f"  difficulty-only null : f1={r['null']['f1_B']:.3f}  "
              f"AUC={r['null']['AUC']:.3f}  AP={r['null']['AUPRC']:.3f}")
        print(f"  controlled probe     : f1={r['controlled']['f1_B']:.3f}  "
              f"AUC={r['controlled']['AUC']:.3f}  AP={r['controlled']['AUPRC']:.3f}")

    d_null = (results["WITHOUT dataset"]["null"]["f1_B"]
              - results["WITH dataset"]["null"]["f1_B"])
    d_ctl = (results["WITHOUT dataset"]["controlled"]["f1_B"]
             - results["WITH dataset"]["controlled"]["f1_B"])
    print(f"\nDelta on dropping dataset -- null: {d_null:+.3f}   "
          f"controlled: {d_ctl:+.3f}")
    print("Controlled delta < ~0.05  -> collapse is not an over-control artifact.")
    print("Controlled delta large    -> report it; lean on the within-dataset control.")
    return results


if __name__ == "__main__":
    # Wire to your cached arrays, e.g.
    #   X  = np.load("cache/chain_vectors.npy")
    #   y  = np.load("cache/type_b_labels.npy")
    #   C  = np.load("cache/confounds_3.npy")        # length, latex, n_steps
    #   D  = np.load("cache/dataset_onehot.npy")
    #   run_overcontrol_check(X, y, C, D)
    #
    # And for R3a, pass the cached score vectors you already have for
    # Math-Shepherd, RLHFlow, pooled AE, Step-SDAE frozen / e2e.
    raise SystemExit("wire up your cached arrays, then run")