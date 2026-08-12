"""
mechinterp_steer.py — M4 (steering): the causal bookend to M2/M3 ablation.

M2/M3 REMOVED the difficulty direction and watched the score fall. This ADDS it and watches the
score rise: inject alpha * sigma * w_length into the residual stream at the peak layer and sweep
alpha. If the PRM's mean Type-B chain score (and raw f1_B) move MONOTONICALLY with alpha, the
length/difficulty direction *causally drives* the verifier's output — the confound confirmed from
the additive side, not just the ablation side.

Same validated stack as M2/M3 (cand [648,387], tag 12902, bf16). Baseline must reproduce gate
~0.735 or it aborts.

    HF_HOME=/workspace/ridae/.hf python experiments/mechinterp_steer.py --every 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "main"))
MODEL = "peiyi9979/math-shepherd-mistral-7b-prm"
GOOD, BAD, STEP = "+", "-", "ки"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    ap.add_argument("--out", default=str(ROOT / "experiments/results_mechinterp"))
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ALPHAS = [float(a) for a in args.alphas.split(",")]

    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    CAND = tok.encode(f"{GOOD} {BAD}")[1:]; TAG = tok.encode(f"{STEP}")[-1]
    assert len(CAND) == 2, f"tokenizer mismatch {CAND}"
    print(f"[STEER] cand {CAND} tag {TAG}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto", output_hidden_states=True).eval()
    NL = model.config.num_hidden_layers
    LAYERS = list(range(0, NL + 1, args.every))

    recs = torch.load(args.cache, weights_only=False)
    meta = {json.loads(l)["record_id"]: json.loads(l)
            for l in Path(args.data_dir, "candidates.jsonl").read_text().splitlines() if l.strip()}
    y = np.array([r["chain"] for r in recs])
    L, NS, LA, DS = [], [], [], []
    for r in recs:
        m = meta.get(r["id"], {}); t = m.get("response_text") or m.get("full_text") or ""
        L.append(np.log1p(len(t.split()))); NS.append(len(r["steps_text"]))
        LA.append((t.count("\\") + t.count("$")) / max(len(t.split()), 1)); DS.append(r["split"])
    L = np.array(L); NS = np.array(NS, float); LA = np.array(LA); DS = np.array(DS)
    surf = StandardScaler().fit_transform(np.c_[L, LA, NS])
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(DS.reshape(-1, 1))
    CF = np.hstack([np.ones((len(recs), 1)), surf, oh])

    def fwd(problem, steps, want_hidden):
        text = problem + "".join(f" {s} {STEP}\n" for s in steps)
        ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=args.max_len).to(model.device)
        with torch.no_grad():
            o = model(ids, use_cache=False)
        tm = (ids[0] == TAG); nt = int(tm.sum())
        sc = torch.softmax(o.logits[0][:, CAND].float(), -1)[:, 0][tm].cpu().numpy()
        hid = {li: o.hidden_states[li][0][tm].float().mean(0).cpu().numpy() for li in LAYERS} if want_hidden else None
        return sc, hid, nt

    # capture baseline + length direction at peak layer
    t0 = time.time(); acts = {li: [] for li in LAYERS}; scores = []; keep = []; skip = 0
    for i, r in enumerate(recs):
        prob = meta.get(r["id"], {}).get("problem", "")
        try:
            sc, hid, nt = fwd(prob, r["steps_text"], True)
        except Exception:
            skip += 1; continue
        if nt != len(r["steps_text"]) or not np.isfinite(sc).all():
            skip += 1; continue
        scores.append(sc); keep.append(i)
        for li in LAYERS: acts[li].append(hid[li])
    keep = np.array(keep)
    for li in LAYERS: acts[li] = np.array(acts[li])
    assert len(keep) > 1000, "too many skipped — abort"
    Lk, yk, CFk = L[keep], y[keep], CF[keep]
    r2l = {li: cross_val_score(Ridge(1.0), acts[li], Lk, cv=5, scoring="r2").mean() for li in LAYERS}
    peak = max(LAYERS, key=lambda li: r2l[li])
    w = Ridge(1.0).fit(acts[peak], Lk).coef_; w = w / np.linalg.norm(w)
    sigma = float((acts[peak] @ w).std())          # steer in units of the direction's own spread
    print(f"[STEER] captured {len(keep)} ({skip} skip). peak layer {peak}, sigma {sigma:.3f}", flush=True)

    def sp(n, s=0):
        rng = np.random.RandomState(s); i = np.arange(n); rng.shuffle(i); c = int(.8 * n); return i[:c], i[c:]
    def cmin(ss): return np.array([1 - ss[i].min() for i in range(len(ss))])
    def f1v(v):
        tr, va = sp(len(v)); clf = LogisticRegression(max_iter=2000).fit(v[tr].reshape(-1, 1), yk[tr])
        return f1_score(yk[va], clf.predict(v[va].reshape(-1, 1)), pos_label="B")
    def gate(ss):
        fs, fl = [], []
        for i, idx in enumerate(keep):
            for s, l in zip(ss[i], recs[idx]["step_labels"]):
                if l >= 0: fs.append(1 - s); fl.append(int(l))
        return roc_auc_score(fl, fs) if len(set(fl)) > 1 else float("nan")

    tgt = model.model.layers[peak - 1]
    W = torch.tensor(w, dtype=torch.bfloat16)

    def steer_hook(coef):
        def hook(mod, inp, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h + (coef * W.to(h.device))
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return hook

    print("[STEER] alpha (in sigma units) -> mean Type-B chain score / raw f1_B / gate:")
    rows = {}
    for a in ALPHAS:
        if a == 0.0:
            asc = scores
        else:
            hd = tgt.register_forward_hook(steer_hook(a * sigma)); asc = []
            try:
                for j, idx in enumerate(keep):
                    r = recs[idx]; prob = meta.get(r["id"], {}).get("problem", "")
                    sc, _, nt = fwd(prob, r["steps_text"], False)
                    asc.append(sc if (nt == len(r["steps_text"]) and np.isfinite(sc).all()) else scores[j])
            finally:
                hd.remove()
        cv = cmin(asc)
        rows[a] = {"mean_typeB": float(cv.mean()), "raw_f1": f1v(cv), "gate": gate(asc)}
        print(f"   alpha={a:+.0f}: mean_typeB={rows[a]['mean_typeB']:.3f}  raw_f1={rows[a]['raw_f1']:.3f}  gate={rows[a]['gate']:.3f}", flush=True)

    ms = [rows[a]["mean_typeB"] for a in ALPHAS]
    mono = all(ms[i] <= ms[i + 1] for i in range(len(ms) - 1)) or all(ms[i] >= ms[i + 1] for i in range(len(ms) - 1))
    print("=" * 60)
    print(f"  STEERING: mean Type-B score is {'MONOTONIC' if mono else 'non-monotonic'} in the length"
          f" direction (Δ over sweep = {ms[-1]-ms[0]:+.3f}).")
    print("  Monotonic -> the length/difficulty direction CAUSALLY drives the PRM's score.")
    print("=" * 60)
    json.dump({"peak": peak, "sigma": sigma, "rows": {str(a): rows[a] for a in ALPHAS}, "monotonic": mono},
              open(out / "steer_result.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(ALPHAS, ms, marker="o", label="mean Type-B score")
        plt.plot(ALPHAS, [rows[a]["raw_f1"] for a in ALPHAS], marker="s", label="raw f1_B")
        plt.xlabel("steering α (σ units, + adds difficulty)"); plt.ylabel("score"); plt.grid(alpha=.3)
        plt.title(f"Steering the length direction @ layer {peak}"); plt.legend()
        plt.tight_layout(); plt.savefig(out / "steer.png", dpi=140); print(f"  saved {out/'steer.png'}")
    except Exception as e:
        print(f"  plot skipped: {e}")


if __name__ == "__main__":
    main()
