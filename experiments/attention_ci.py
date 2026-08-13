"""
attention_ci.py -- bootstrap CI for the error-step attention gap.

I could not compute this here: it needs the per-candidate attention weights,
which are not in RESULTS.md. Run this against your Phase-1a checkpoint and
send me the printed numbers; I will drop them into the paper.

The paper currently states the 0.168 vs 0.142 gap WITHOUT resting any claim
on it, so it is defensible as-is. This upgrades it from "mentioned" to
"tested".

Bootstrap over CANDIDATES, not over steps -- steps within a solution are not
independent, and resampling them would give a spuriously tight interval.
"""
import numpy as np

def attention_gap_ci(att_weights, error_masks, n_boot=10_000, seed=0):
    """
    att_weights : list of 1-D arrays, one per candidate.
                  att_weights[i][j] = chain-head attention on step j.
                  Each array should sum to ~1.
    error_masks : list of 1-D bool arrays, same shapes.
                  True where the step carries a human-labelled error.

    Returns the observed gap and a 95% percentile-bootstrap CI.
    """
    # keep only candidates that actually contain a labelled error --
    # a candidate with no error step contributes nothing to the contrast
    pairs = [(a, m) for a, m in zip(att_weights, error_masks) if m.any() and (~m).any()]
    n = len(pairs)
    if n == 0:
        raise ValueError("no candidates with both error and non-error steps")

    def gap(sample):
        err = np.concatenate([a[m] for a, m in sample])
        non = np.concatenate([a[~m] for a, m in sample])
        return err.mean() - non.mean()

    observed = gap(pairs)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = gap([pairs[i] for i in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # a uniform-attention reference: 1/N averaged over the same candidates
    uniform = np.mean([1.0 / len(a) for a, _ in pairs])

    print(f"candidates with a labelled error : {n}")
    print(f"uniform-attention reference      : {uniform:.4f}")
    print(f"mean attention, error steps      : "
          f"{np.concatenate([a[m] for a, m in pairs]).mean():.4f}")
    print(f"mean attention, non-error steps  : "
          f"{np.concatenate([a[~m] for a, m in pairs]).mean():.4f}")
    print(f"observed gap                     : {observed:+.4f}")
    print(f"95% bootstrap CI                 : [{lo:+.4f}, {hi:+.4f}]")
    print(f"excludes zero                    : {lo > 0 or hi < 0}")
    return observed, (lo, hi)


if __name__ == "__main__":
    # Wire this to however you dumped the attention weights in
    # experiments/mechinterp_m1.py, e.g.:
    #
    #   att   = np.load("results_mechinterp/chain_attention.npy", allow_pickle=True)
    #   masks = np.load("results_mechinterp/error_step_masks.npy", allow_pickle=True)
    #   attention_gap_ci(list(att), list(masks))
    #
    # If the attention weights were not saved, re-run the chain head over the
    # val split with output_attentions on the attention-pooling layer and
    # cache them; it is a forward pass on a 1.3M-param head and takes seconds.
    raise SystemExit("wire up the two arrays above, then run")