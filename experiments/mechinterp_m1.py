"""
mechinterp_m1.py — M1 (Mac/CPU): where does the difficulty confound enter OUR model?

Local counterpart of notebooks/mechinterp_M1_kaggle.ipynb — runs in ~1 min on CPU using the
frozen Phase-1a checkpoint already in the repo. No Kaggle needed (only M2's 7B needs a GPU).

Stages: per-step MiniLM embeddings H -> bottleneck step-codes Z -> attention-pooled chain c.
  M1b  probe each stage for each confound (ridge R² / logistic acc) — 'where difficulty lives'
  M1c  causal: project the length direction out of c, re-fit A/B, compare to residualization
  attn chain-head attention: does it weight long steps (surface) or error steps (signal)?

    python experiments/mechinterp_m1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main"))
from sdae_prm import StepSDAE_PRM

CKPT = ROOT / "experiments/results_multiseed/ckpts/frozen_seed0/sdae_best.pt"


def main():
    import json
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    recs = torch.load(ROOT / "data/step_cache.pt", weights_only=False)
    meta = {json.loads(l)["record_id"]: json.loads(l)
            for l in (ROOT / "data/processed_pb/candidates.jsonl").read_text().splitlines() if l.strip()}
    model = StepSDAE_PRM(); model.load_state_dict(torch.load(CKPT, map_location="cpu")); model.eval()
    y = np.array([r["chain"] for r in recs])

    L, NS, LA, DS = [], [], [], []
    for r in recs:
        m = meta.get(r["id"], {}); t = m.get("response_text") or m.get("full_text") or ""
        L.append(np.log1p(len(t.split()))); NS.append(len(r["steps_text"]))
        LA.append((t.count("\\") + t.count("$")) / max(len(t.split()), 1)); DS.append(r["split"])
    L = np.array(L); NS = np.array(NS, float); LA = np.array(LA); DS = np.array(DS)

    # M1a — extract the three stages
    H, Z, C = [], [], []
    with torch.no_grad():
        for r in recs:
            X = torch.from_numpy(r["steps_emb"]).float().unsqueeze(0)
            pad = torch.zeros(1, X.size(1), dtype=torch.bool)
            h = model.encode(X, pad); _, _, _, pooled = model(X, pad, None)
            H.append(r["steps_emb"].mean(0)); Z.append(h[0].mean(0).numpy()); C.append(pooled[0].numpy())
    H = np.array(H); Z = np.array(Z); C = np.array(C)

    # M1b — probe by stage
    print("=" * 70)
    print("  M1b — where each confound becomes linearly decodable (R²/acc, 5-fold)")
    print("=" * 70)
    stages = [("H (per-step MiniLM, pooled)", H), ("Z (bottleneck codes, pooled)", Z),
              ("c (attention-pooled chain)", C)]
    cont = [("log_length", L), ("n_steps", NS), ("latex_density", LA)]
    r2tab = {}
    print(f"  {'stage':<32}{'log_len':>10}{'n_steps':>10}{'latex':>9}{'dataset acc':>13}")
    for sn, S in stages:
        r2 = [cross_val_score(Ridge(1.0), S, v, cv=5, scoring="r2").mean() for _, v in cont]
        acc = cross_val_score(LogisticRegression(max_iter=2000), S, DS, cv=5, scoring="accuracy").mean()
        r2tab[sn] = r2
        print(f"  {sn:<32}{r2[0]:>10.3f}{r2[1]:>10.3f}{r2[2]:>9.3f}{acc:>13.3f}")
    print("  -> jump H->Z means AGGREGATION manufactures that confound (MiniLM can't see it per-step)")

    # M1c — causal length-ablation vs residualization
    def split(n, s=0):
        rng = np.random.RandomState(s); i = np.arange(n); rng.shuffle(i); c = int(.8 * n); return i[:c], i[c:]
    def pf(Zr, s=0):
        tr, va = split(len(Zr), s)
        clf = LogisticRegression(max_iter=2000).fit(Zr[tr], y[tr])
        return f1_score(y[va], clf.predict(Zr[va]), pos_label="B")
    surf = StandardScaler().fit_transform(np.c_[L, LA, NS])
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(DS.reshape(-1, 1))
    CF = np.hstack([np.ones((len(recs), 1)), surf, oh])
    tr, _ = split(len(C))
    beta = np.linalg.lstsq(CF[tr], C[tr], rcond=None)[0]
    w = Ridge(1.0).fit(C[tr], L[tr]).coef_; w = w / np.linalg.norm(w)
    print("\n" + "=" * 70)
    print("  M1c — causal: does removing the length direction reproduce residualization?")
    print("=" * 70)
    print(f"  A/B f1_B on c — raw {pf(C):.3f} | 4-confound residualized {pf(C - CF @ beta):.3f} "
          f"| length-direction ablated {pf(C - np.outer(C @ w, w)):.3f}")

    # attention analysis
    lr, ae, ao = [], [], []
    with torch.no_grad():
        for r in recs:
            T = len(r["steps_text"])
            if T < 2:
                continue
            X = torch.from_numpy(r["steps_emb"]).float().unsqueeze(0); pad = torch.zeros(1, T, dtype=torch.bool)
            h = model.encode(X, pad); a = model.attn(h).squeeze(-1); wts = torch.softmax(a, 1)[0].numpy()
            sl = np.array([len(s.split()) for s in r["steps_text"]], float)
            if sl.std() > 0:
                lr.append(np.corrcoef(wts, sl)[0, 1])
            lab = np.array(r["step_labels"]); ae += list(wts[lab == 1]); ao += list(wts[lab == 0])
    print("\n" + "=" * 70)
    print("  Attention analysis — chain head: surface (length) or signal (errors)?")
    print("=" * 70)
    print(f"  attn vs step-length corr {np.nanmean(lr):+.3f} | attn on ERROR {np.mean(ae):.4f} "
          f"vs non-error {np.mean(ao):.4f}")

    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        labels = [s.split(" ")[0] for s, _ in stages]
        plt.figure(figsize=(7, 4))
        for i, (nm, _) in enumerate(cont):
            plt.plot(labels, [r2tab[s][i] for s, _ in stages], marker="o", label=nm)
        plt.ylabel("probe R² (5-fold)"); plt.xlabel("pipeline stage")
        plt.title("M1: where the difficulty confound becomes linear"); plt.legend(); plt.grid(alpha=.3)
        out = HERE / "results_mechinterp"; out.mkdir(exist_ok=True)
        plt.tight_layout(); plt.savefig(out / "m1_where_confound.png", dpi=140)
        print(f"\n  saved {out/'m1_where_confound.png'}")
    except Exception as e:
        print(f"  plot skipped: {e}")


if __name__ == "__main__":
    main()
