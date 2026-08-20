"""
E5 — Make the causal result surgical.  (Reviewer ask #5 / §3.4 follow-up.)

Current ablation (top-k PLS difficulty subspace) reaches the control target (raw AUROC 0.761->0.688,
control target 0.673) but ALSO drops the step-label gate 0.735->0.637: it removes verification
competence along with difficulty. The ask: an erasure that removes difficulty while preserving the
validity signal AND the general step gate — turning "difficulty influences the verifier" into
"difficulty influences the verifier INDEPENDENTLY of validity".

Three operators on the SAME residual-stream activations H (layer ~12 of Math-Shepherd), one eval axis:
  baseline : top-k PLS/PCA difficulty subspace ablation (reproduce §3.4 — expected NOT surgical).
  leace    : least-squares concept erasure of the difficulty concept (minimal-collateral linear).
  oblique  : VALIDITY-PRESERVING erasure — erase difficulty within the null space of the validity
             subspace, i.e. remove difficulty variance then add back its projection onto the protected
             validity direction(s). Leaves the validity readout's input invariant along V by design.

"Surgical" == difficulty R^2 -> control target AND ΔAUROC(validity) ~ 0 AND Δ(step gate) ~ 0.

INTEGRITY FLAG: the paper found difficulty is redundantly written and self-repairing (Hydra). If
difficulty and validity are entangled here, NO linear operator can separate them and every operator
will still drop the gate. That negative is a STRONGER claim than the current one ("difficulty and
validity are entangled in this verifier") — this script is built so that outcome is reportable, not a
failure. Do not tune until the gate stops dropping; report what the operators actually do.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

# ============================== ADAPTER ==============================
def load_layer_activations(split: str):
    """split in {'train','val'}. Returns:
       H : float[C, d]  residual-stream activations at the ablation layer (paper uses ~L12), pooled per
                        candidate the same way §3.4 does.
       difficulty : float[C, 2]  [log length, n_steps] (the confound targets the paper decodes at R^2~0.92).
       y_valid : int[C]          1 == Type A / sound, 0 == Type B (for validity AUROC).
       step_gate_eval : callable(H_edited)->float  runs the verifier's per-step head on edited activations
                        and returns step-score AUROC vs human step labels (the 0.735 gate to preserve).
       validity_auroc : callable(H_edited)->float  verifier's Type-B AUROC on edited activations
                        (the 0.761 raw / 0.673 control-target number).
    """
    raise NotImplementedError
# ============================ END ADAPTER ============================


def _difficulty_directions(H: np.ndarray, D: np.ndarray, k: int) -> np.ndarray:
    """Top-k PLS directions predicting difficulty from H (paper's construction). Returns [d, k] ortho basis."""
    pls = PLSRegression(n_components=k).fit(H - H.mean(0), D - D.mean(0))
    Q, _ = np.linalg.qr(pls.x_weights_)   # orthonormalize the k weight vectors
    return Q[:, :k]


def _validity_directions(H: np.ndarray, y: np.ndarray, m: int = 4) -> np.ndarray:
    """Protected validity subspace: top-m PLS directions predicting the validity label from H."""
    pls = PLSRegression(n_components=m).fit(H - H.mean(0), (y - y.mean()).reshape(-1, 1))
    Q, _ = np.linalg.qr(pls.x_weights_)
    return Q[:, :m]


def ablate_subspace(H: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project H onto the orthogonal complement of `basis` (the paper's baseline ablation)."""
    P = basis @ basis.T
    return H - H @ P


def leace_erase(H: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Least-squares concept erasure (Belrose et al. 2023): remove the linear predictability of the
    difficulty concept D from H with minimal-norm edit. Implemented as whitened projection of the
    cross-covariance's column space out of (centered) H."""
    mu = H.mean(0)
    Hc = H - mu
    Sigma = np.cov(Hc, rowvar=False) + 1e-6 * np.eye(H.shape[1])
    W = np.linalg.cholesky(np.linalg.inv(Sigma))          # whitening transform (Sigma^{-1/2})
    Dc = D - D.mean(0)
    Xcov = (W @ Hc.T) @ Dc                                # cross-cov in whitened space
    U, s, _ = np.linalg.svd(Xcov, full_matrices=False)
    U = U[:, s > 1e-8]                                    # concept subspace in whitened space
    Wi = np.linalg.inv(W)
    P_white = U @ U.T
    return (Hc @ W.T @ (np.eye(P_white.shape[0]) - P_white) @ Wi.T) + mu


def oblique_validity_preserving(H: np.ndarray, D: np.ndarray, y: np.ndarray, k: int, m: int = 4) -> np.ndarray:
    """Erase the difficulty subspace but PROTECT the validity subspace: ablate difficulty, then restore
    the component the ablation removed that lies in the validity span. Result is invariant along V."""
    Bdiff = _difficulty_directions(H, D, k)
    V = _validity_directions(H, y, m)
    H_ab = ablate_subspace(H, Bdiff)                      # difficulty removed (may cost validity)
    removed = H - H_ab                                    # what the ablation took out
    Pv = V @ V.T
    return H_ab + removed @ Pv                            # add back only the validity-span part


def evaluate(name: str, H_edit: np.ndarray, loaders, k=None) -> dict:
    H, D, y, step_gate_eval, validity_auroc = loaders
    diff_r2 = _difficulty_decodability(H_edit, D)
    return {
        "operator": name, "k": k,
        "difficulty_R2_after": diff_r2,                   # target: down toward control level
        "validity_AUROC_after": float(validity_auroc(H_edit)),   # target: ~unchanged (0.761 -> stay high)
        "step_gate_after": float(step_gate_eval(H_edit)),        # target: ~0.735 (baseline drops to 0.637)
    }


def _difficulty_decodability(H: np.ndarray, D: np.ndarray) -> float:
    from sklearn.model_selection import cross_val_score
    r2 = []
    for j in range(D.shape[1]):
        r2.append(cross_val_score(Ridge(alpha=1.0), H, D[:, j], cv=5, scoring="r2").mean())
    return float(np.mean(r2))


def run(k: int = 16) -> dict:
    H, D, y, step_gate_eval, validity_auroc = load_layer_activations("val")
    loaders = (H, D, y, step_gate_eval, validity_auroc)
    baseline_refs = {"unablated_AUROC": 0.761, "control_target_AUROC": 0.673,
                     "step_gate_unablated": 0.735, "step_gate_after_baseline_ablation": 0.637,
                     "difficulty_R2_unablated": 0.92}
    Bdiff = _difficulty_directions(H, D, k)
    ops = {
        "baseline_subspace_ablation": ablate_subspace(H, Bdiff),
        "leace": leace_erase(H, D),
        "oblique_validity_preserving": oblique_validity_preserving(H, D, y, k),
    }
    results = {name: evaluate(name, He, loaders, k=k) for name, He in ops.items()}
    results["unablated"] = evaluate("unablated", H, loaders)
    verdict = ("SURGICAL if an operator drives difficulty_R2 toward the control level while keeping "
               "step_gate ~0.735 and validity_AUROC high. If EVERY operator still drops the step gate, "
               "difficulty and validity are entangled in this verifier — report that as the (stronger) "
               "result; do not tune to force surgery.")
    return {"k": k, "references": baseline_refs, "operators": results, "verdict_hint": verdict}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default="e5_results.json")
    args = ap.parse_args()
    res = run(k=args.k)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
