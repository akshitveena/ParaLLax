"""
mechinterp_m2.py — M2 on the A100 box: where is difficulty represented in the 7B PRM, and can it
be removed? (Kaggle T4x2 failed here: its transformers tokenizes the 'ки' tag incompatibly ->
96% skipped, gate at chance; and device_map OOMs on the ablation pass. The A100 has the validated
stack from 2c — tag 12902, gate 0.735 — and 40GB, so run it here.)

Uses the SAME tokenizer handling as experiments/prm_external.py (which scored gate 0.735) and
bf16, so the baseline matches the paper's 2c numbers.

  1. Capture residual-stream activations (every 4th layer) at each step's 'ки' tag, over 1,700
     solutions; the same forward gives the PRM step scores (baseline).
  2. Probe each layer -> log length / n_steps. R²-by-layer = where difficulty lives (figure).
  3. Mean-ablate the length direction at the peak layer; re-score with the hook live.
  4. Re-measure raw/ctl f1_B + step-label gate. Report the decisive comparison (all 3 outcomes).

    HF_HOME=/workspace/ridae/.hf python experiments/mechinterp_m2.py --every 4
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
    ap.add_argument("--out", default=str(ROOT / "experiments/results_mechinterp"))
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # --- tokenizer/model: exactly the prm_external.py handling that scored gate 0.735 ---
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    CAND = tok.encode(f"{GOOD} {BAD}")[1:]
    TAG = tok.encode(f"{STEP}")[-1]
    print(f"[M2] cand {CAND} tag {TAG}  (expect [648,387] / 12902 on the validated stack)", flush=True)
    assert len(CAND) == 2, f"tokenizer mismatch: {CAND} — wrong transformers version"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto", output_hidden_states=True).eval()
    NL = model.config.num_hidden_layers
    LAYERS = list(range(0, NL + 1, args.every))
    print(f"[M2] layers={NL}, caching {LAYERS}", flush=True)

    # --- data + confounds ---
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
        hid = None
        if want_hidden:
            hid = {li: o.hidden_states[li][0][tm].float().mean(0).cpu().numpy() for li in LAYERS}
        return sc, hid, nt

    # --- Stage A: capture ---
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
        if (i + 1) % 200 == 0:
            print(f"  captured {i+1}/{len(recs)} ({(time.time()-t0)/60:.1f}m, skip {skip})", flush=True)
    keep = np.array(keep)
    for li in LAYERS: acts[li] = np.array(acts[li])
    print(f"[M2] captured {len(keep)} ({skip} skipped) in {(time.time()-t0)/60:.1f}m", flush=True)
    assert len(keep) > 1000, "too many skipped — tokenizer/tag problem, do not trust results"

    # --- Stage B: probe by layer ---
    Lk, NSk, yk, CFk = L[keep], NS[keep], y[keep], CF[keep]
    r2l, r2n = [], []
    print("[M2] layer probes:")
    for li in LAYERS:
        r2l.append(cross_val_score(Ridge(1.0), acts[li], Lk, cv=5, scoring="r2").mean())
        r2n.append(cross_val_score(Ridge(1.0), acts[li], NSk, cv=5, scoring="r2").mean())
        print(f"   layer {li:2d}: log_len R2={r2l[-1]:.3f}  n_steps R2={r2n[-1]:.3f}")
    peak = LAYERS[int(np.argmax(r2l))]
    print(f"[M2] peak length layer = {peak} (R2={max(r2l):.3f})")

    # --- baseline raw/ctl/gate (bf16, matches 2c) ---
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
    bv = cmin(scores); tr, _ = sp(len(bv)); beta = np.linalg.lstsq(CFk[tr], bv[tr], rcond=None)[0]
    BASE = {"raw": f1v(bv), "ctl": f1v(bv - CFk @ beta), "gate": gate(scores)}
    print(f"[M2] BASELINE (bf16): raw {BASE['raw']:.3f} ctl {BASE['ctl']:.3f} gate {BASE['gate']:.3f}")

    # --- Stage C: mean-ablate the length direction at the peak layer, re-score ---
    w = Ridge(1.0).fit(acts[peak], Lk).coef_; w = w / np.linalg.norm(w)
    mu = float((acts[peak] @ w).mean())
    assert peak >= 1
    tgt = model.model.layers[peak - 1]
    W = torch.tensor(w, dtype=torch.bfloat16)
    def hook(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        wv = W.to(h.device); proj = (h.float() @ wv.float())
        h = h - ((proj - mu).to(h.dtype)).unsqueeze(-1) * wv
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    hd = tgt.register_forward_hook(hook)
    try:
        asc = []
        for j, idx in enumerate(keep):
            r = recs[idx]; prob = meta.get(r["id"], {}).get("problem", "")
            sc, _, nt = fwd(prob, r["steps_text"], False)
            asc.append(sc if (nt == len(r["steps_text"]) and np.isfinite(sc).all()) else scores[j])
            if (j + 1) % 200 == 0: print(f"  ablated {j+1}/{len(keep)}", flush=True)
    finally:
        hd.remove()
    av = cmin(asc); tr, _ = sp(len(av)); beta2 = np.linalg.lstsq(CFk[tr], av[tr], rcond=None)[0]
    ABL = {"raw": f1v(av), "ctl": f1v(av - CFk @ beta2), "gate": gate(asc)}

    print("\n" + "=" * 64)
    print(f"  M2 — length-direction mean-ablation at layer {peak} (bf16, A100)")
    print("=" * 64)
    print(f"  {'':<10}{'raw f1_B':>10}{'ctl f1_B':>10}{'step gate':>11}")
    print(f"  {'baseline':<10}{BASE['raw']:>10.3f}{BASE['ctl']:>10.3f}{BASE['gate']:>11.3f}")
    print(f"  {'ablated':<10}{ABL['raw']:>10.3f}{ABL['ctl']:>10.3f}{ABL['gate']:>11.3f}")
    print(f"  {'Δ':<10}{ABL['raw']-BASE['raw']:>+10.3f}{ABL['ctl']-BASE['ctl']:>+10.3f}{ABL['gate']-BASE['gate']:>+11.3f}")
    print("=" * 64)
    dr, dg = BASE["raw"] - ABL["raw"], BASE["gate"] - ABL["gate"]
    if dr > 0.10 and dg < 0.05:
        print("  BEST CASE: raw f1_B falls toward controlled, step gate preserved -> difficulty is a")
        print("  localizable, removable direction. Causal + statistical control agree.")
    elif dr > 0.10:
        print("  raw falls BUT gate falls too -> direction entangled with competence (removed capability).")
    else:
        print("  raw barely moves -> difficulty is DISTRIBUTED, not one direction (real negative).")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4)); plt.plot(LAYERS, r2l, marker="o", label="log_length")
        plt.plot(LAYERS, r2n, marker="s", label="n_steps"); plt.axvline(peak, ls="--", c="gray")
        plt.xlabel("residual-stream layer"); plt.ylabel("probe R² (5-fold)")
        plt.title("M2: where difficulty lives in Math-Shepherd-7B"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(out / "m2_where.png", dpi=140); print(f"  saved {out/'m2_where.png'}")
    except Exception as e:
        print(f"  plot skipped: {e}")
    json.dump({"peak": peak, "baseline": BASE, "ablated": ABL,
               "r2_len": dict(zip(map(str, LAYERS), r2l)), "n_kept": int(len(keep)), "n_skip": skip},
              open(out / "m2_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
