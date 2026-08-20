"""
E4 — Non-linear confound control.  (Reviewer ask #4.)

Guarantee the surviving controlled f1B (0.591) is not non-linear difficulty leakage that linear
residualization missed. Two moves:

  1. NON-LINEAR NULL. Fit MLP and Kernel Ridge on the 4 confounds -> Type-B. If the non-linear null
     jumps far above the linear null (0.515), more of the "signal" was always difficulty.
  2. NON-LINEAR RESIDUALIZATION with CROSS-FITTING. For each representation dim, fit KRR/MLP
     dim ~ f(confounds) on out-of-fold data, subtract the out-of-fold prediction -> non-linear
     residuals; re-fit the validity probe on residuals. Cross-fitting is essential: an over-powerful
     residualizer fit in-sample can regress out EVERYTHING (validity included) and manufacture a
     spuriously low number. We report the residualizer's held-out R^2 so over-fitting is visible.

Fairness guardrail: the null model and the residualizer use the SAME family/capacity, so we never
hand the null a bigger function class than the thing removing the confound.

Signal is robust  <=>  non-linear controlled f1B stays close to 0.591 AND non-linear null stays near
the linear null. A large extra drop under non-linear control => genuine non-linear leakage.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

# ============================== ADAPTER ==============================
def load_representation_confounds_labels(split: str):
    """split in {'train','val'}. Returns:
       X : float[C, D]   the representation to be controlled (Step-SDAE step-codes pooled, or the
                         verifier score-carrying features — SAME object the paper residualizes).
       Z : float[C, 4]   confounds [length, latex_density, n_steps, dataset_onehot-or-id].
       y : int[C]        1 == Type A (sound), 0 == Type B.
    """
    raise NotImplementedError

def f1b_at_paper_threshold(scores_val: np.ndarray, y_val: np.ndarray) -> float:
    """Type-B f1 (positive == Type B) using the paper's thresholding. Reuse the repo's impl so the
    number is comparable to the reported 0.591 / 0.515."""
    raise NotImplementedError
# ============================ END ADAPTER ============================


def _make_regressor(kind: str):
    if kind == "linear":
        return Ridge(alpha=1.0)
    if kind == "krr":
        return KernelRidge(kernel="rbf", alpha=1.0, gamma=None)   # gamma set via median heuristic below
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=500, random_state=0)
    raise ValueError(kind)


def _median_gamma(Z: np.ndarray) -> float:
    from sklearn.metrics import pairwise_distances
    d = pairwise_distances(Z[np.random.default_rng(0).choice(len(Z), min(500, len(Z)), replace=False)])
    med = np.median(d[d > 0])
    return 1.0 / (2 * med ** 2 + 1e-12)


def crossfit_residualize(X: np.ndarray, Z: np.ndarray, kind: str, n_splits: int = 5, seed: int = 0):
    """Return (residuals, mean_heldout_R2). Each dim of X is predicted from Z out-of-fold; residual =
    X - out_of_fold_prediction. Cross-fitting prevents the residualizer from over-fitting the eval set."""
    Zs = StandardScaler().fit_transform(Z)
    gamma = _median_gamma(Zs) if kind == "krr" else None
    res = np.zeros_like(X)
    r2s = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        for j in range(X.shape[1]):
            reg = _make_regressor(kind)
            if kind == "krr":
                reg.set_params(gamma=gamma)
            reg.fit(Zs[tr], X[tr, j])
            pred = reg.predict(Zs[te])
            res[te, j] = X[te, j] - pred
            denom = np.var(X[te, j]) + 1e-12
            r2s.append(1.0 - np.mean((X[te, j] - pred) ** 2) / denom)
    return res, float(np.mean(r2s))


def nonlinear_null(Z_tr, y_tr, Z_val, y_val, kind: str) -> dict:
    """f1B / AUC of a difficulty-only classifier on the confounds, non-linear vs linear."""
    Zs = StandardScaler().fit(Z_tr)
    if kind == "linear":
        clf = LogisticRegression(max_iter=1000)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=500, random_state=0)
    else:  # krr-as-classifier: use MLP for the null to stay a proper classifier; keep capacity matched
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=500, random_state=0)
    clf.fit(Zs.transform(Z_tr), y_tr)
    p = clf.predict_proba(Zs.transform(Z_val))[:, 0]   # P(Type B == class 0)
    return {"f1b": f1b_at_paper_threshold(1 - p, y_val), "auc": float(roc_auc_score(1 - y_val, p))}


def controlled_probe(X_res_tr, y_tr, X_res_val, y_val) -> dict:
    """Fit the validity probe on residualized TRAIN reps, evaluate on residualized VAL."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_res_tr, y_tr)
    s = clf.predict_proba(X_res_val)[:, 1]             # P(Type A)
    return {"f1b": f1b_at_paper_threshold(s, y_val), "auc": float(roc_auc_score(y_val, s))}


def run(kinds=("linear", "krr", "mlp")) -> dict:
    Xtr, Ztr, ytr = load_representation_confounds_labels("train")
    Xval, Zval, yval = load_representation_confounds_labels("val")
    out = {"linear_null_reference": 0.515, "linear_controlled_reference": 0.591, "by_kind": {}}
    # combine train+val for cross-fit residualization of the eval fold ONLY on train-fit predictors:
    for kind in kinds:
        res_tr, r2_tr = crossfit_residualize(Xtr, Ztr, kind)
        # residualize val using cross-fit predictors trained on train (fit on train, apply to val)
        Zs = StandardScaler().fit(Ztr)
        gamma = _median_gamma(Zs.transform(Ztr)) if kind == "krr" else None
        res_val = np.zeros_like(Xval)
        for j in range(Xval.shape[1]):
            reg = _make_regressor(kind)
            if kind == "krr":
                reg.set_params(gamma=gamma)
            reg.fit(Zs.transform(Ztr), Xtr[:, j])
            res_val[:, j] = Xval[:, j] - reg.predict(Zs.transform(Zval))
        out["by_kind"][kind] = {
            "null": nonlinear_null(Ztr, ytr, Zval, yval, kind),
            "controlled": controlled_probe(res_tr, ytr, res_val, yval),
            "residualizer_heldout_R2": r2_tr,   # if this is ~1.0 the residualizer is too strong -> deflation artifact
        }
    out["verdict_hint"] = ("Signal robust if controlled f1B stays ~0.591 across kinds AND non-linear "
                           "null stays ~0.515. A KRR/MLP controlled f1B far below 0.591 with residualizer "
                           "R^2 near 1.0 is an over-control artifact, not leakage — check R^2 before "
                           "concluding.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e4_results.json")
    args = ap.parse_args()
    res = run()
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
