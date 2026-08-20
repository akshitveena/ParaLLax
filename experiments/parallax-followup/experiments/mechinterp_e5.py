"""
E5 — Surgical erasure (reviewer M5): remove difficulty while PRESERVING validity AUROC and the step
gate. M3's subspace ablation hit the difficulty target but dropped the gate 0.735->0.637 (non-surgical).
This tests operators designed to be surgical. A100 box, bf16, validated Math-Shepherd stack.

Three operators on layer-`peak` residual activations, one re-scoring eval:
  baseline : mean-ablate the top-k PLS difficulty subspace (reproduce M3 — expected NON-surgical).
  oblique  : erase the difficulty subspace but ADD BACK its projection onto the validity direction w_v
             (protect the validity readout's input along w_v). The validity-preserving operator.
  leace    : LEACE concept erasure of difficulty (if `concept-erasure` importable; else skipped).

Eval per operator (re-run the 7B with an erasure hook at layer peak):
  difficulty R^2 on the ERASED activations  (should fall toward chance -> difficulty removed)
  validity AUROC (chain-level Type-B)        (should stay ~ baseline  -> validity preserved)
  step-label gate                            (should stay ~ baseline  -> competence preserved)
SURGICAL == difficulty removed AND ΔAUROC(validity) ~ 0 AND Δgate ~ 0.
INTEGRITY: if NO operator is surgical (all drop the gate), that is the STRONGER 'entangled' claim —
report it, do not tune. (Consistent with the paper's Hydra finding.)

    HF_HOME=/workspace/ridae/.hf python experiments/.../mechinterp_e5.py --every 4 --k 16
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "main"))
MODEL = "peiyi9979/math-shepherd-mistral-7b-prm"
GOOD, BAD, STEP = "+", "-", "ки"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--out", default=str(ROOT / "experiments/results_mechinterp"))
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    try: tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception: tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    CAND = tok.encode(f"{GOOD} {BAD}")[1:]; TAG = tok.encode(f"{STEP}")[-1]
    assert len(CAND) == 2, f"tokenizer mismatch {CAND}"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto", output_hidden_states=True).eval()
    NL = model.config.num_hidden_layers; LAYERS = list(range(0, NL + 1, args.every))

    recs = torch.load(args.cache, weights_only=False)
    meta = {json.loads(l)["record_id"]: json.loads(l)
            for l in Path(args.data_dir, "candidates.jsonl").read_text().splitlines() if l.strip()}
    y = np.array([r["chain"] for r in recs]); ybin = (y == "B").astype(int)
    L, NS = [], []
    for r in recs:
        t = meta.get(r["id"], {}).get("response_text") or meta.get(r["id"], {}).get("full_text") or ""
        L.append(np.log1p(len(t.split()))); NS.append(len(r["steps_text"]))
    L = np.array(L); NS = np.array(NS, float)

    def fwd(problem, steps, want_hidden):
        text = problem + "".join(f" {s} {STEP}\n" for s in steps)
        ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=args.max_len).to(model.device)
        with torch.no_grad(): o = model(ids, use_cache=False)
        tm = (ids[0] == TAG); nt = int(tm.sum())
        sc = torch.softmax(o.logits[0][:, CAND].float(), -1)[:, 0][tm].cpu().numpy()
        hid = {li: o.hidden_states[li][0][tm].float().mean(0).cpu().numpy() for li in LAYERS} if want_hidden else None
        return sc, hid, nt

    # capture baseline
    t0 = time.time(); acts = {li: [] for li in LAYERS}; scores = []; keep = []; skip = 0
    for i, r in enumerate(recs):
        try: sc, hid, nt = fwd(meta.get(r["id"], {}).get("problem", ""), r["steps_text"], True)
        except Exception: skip += 1; continue
        if nt != len(r["steps_text"]) or not np.isfinite(sc).all(): skip += 1; continue
        scores.append(sc); keep.append(i)
        for li in LAYERS: acts[li].append(hid[li])
        if (i + 1) % 400 == 0: print(f"  captured {i+1} (skip {skip}, {(time.time()-t0)/60:.1f}m)", flush=True)
    keep = np.array(keep)
    for li in LAYERS: acts[li] = np.array(acts[li])
    assert len(keep) > 1000
    Lk, NSk, yk, ybk = L[keep], NS[keep], y[keep], ybin[keep]
    r2 = {li: cross_val_score(Ridge(1.0), acts[li], Lk, cv=5, scoring="r2").mean() for li in LAYERS}
    peak = max(LAYERS, key=lambda li: r2[li])
    H = acts[peak]

    def sp(n, s=0):
        rng = np.random.RandomState(s); i = np.arange(n); rng.shuffle(i); c = int(.8*n); return i[:c], i[c:]
    def cmin(ss): return np.array([1 - ss[i].min() for i in range(len(ss))])
    def val_auc(ss):
        v = cmin(ss); _, va = sp(len(v)); return roc_auc_score(ybk[va], v[va])
    def gate(ss):
        fs, fl = [], []
        for i, idx in enumerate(keep):
            for s2, l in zip(ss[i], recs[idx]["step_labels"]):
                if l >= 0: fs.append(1 - s2); fl.append(int(l))
        return roc_auc_score(fl, fs) if len(set(fl)) > 1 else float("nan")
    BASE = {"val_auc": val_auc(scores), "gate": gate(scores), "diff_r2": r2[peak]}
    print(f"[E5] peak {peak} | BASELINE val_AUC {BASE['val_auc']:.3f} gate {BASE['gate']:.3f} diff_R2 {BASE['diff_r2']:.3f}", flush=True)
    assert BASE["gate"] > 0.65

    # operators: difficulty subspace W_d, validity direction w_v
    pls = PLSRegression(n_components=args.k).fit(H, np.c_[Lk, NSk]); Wd = np.linalg.qr(pls.x_weights_)[0][:, :args.k]
    wv = Ridge(1.0).fit(H, ybk.astype(float)).coef_; wv = wv / np.linalg.norm(wv)  # validity direction
    mu = (H @ Wd).mean(0)
    tgt = model.model.layers[peak - 1]

    ops = {}
    ops["baseline"] = (Wd, mu, None)                 # mean-ablate Wd
    ops["oblique"]  = (Wd, mu, wv)                    # mean-ablate Wd but protect wv

    try:
        from concept_erasure import LeaceEraser
        er = LeaceEraser.fit(torch.tensor(H), torch.tensor(np.c_[Lk, NSk]))
        ops["leace"] = ("leace", er, None)
    except Exception as e:
        print(f"[E5] LEACE skipped ({type(e).__name__}) — install concept-erasure to include it", flush=True)

    def hook_for(spec):
        kind = spec[0]
        if isinstance(kind, str) and kind == "leace":
            er = spec[1]
            def hk(mod, inp, o):
                h = o[0] if isinstance(o, tuple) else o
                he = er(h.reshape(-1, h.shape[-1]).float()).to(h.dtype).reshape(h.shape)
                return (he,)+tuple(o[1:]) if isinstance(o, tuple) else he
            return hk
        B, m_, protect = spec
        Bt = torch.tensor(B, dtype=torch.bfloat16); mt = torch.tensor(m_, dtype=torch.float32)
        vt = None if protect is None else torch.tensor(protect, dtype=torch.bfloat16)
        def hk(mod, inp, o):
            h = o[0] if isinstance(o, tuple) else o
            Bd = Bt.to(h.device); coord = h.float() @ Bd.float()
            delta = ((coord - mt.to(h.device)) @ Bd.float().T)              # difficulty component
            if vt is not None:                                             # oblique: keep wv-aligned part
                vv = vt.to(h.device).float(); delta = delta - (delta @ vv).unsqueeze(-1) * vv
            h = h - delta.to(h.dtype)
            return (h,)+tuple(o[1:]) if isinstance(o, tuple) else h
        return hk

    print(f"[E5] {'op':<10}{'val_AUC':>9}{'Δval':>8}{'gate':>8}{'Δgate':>8}{'diff_R2':>9}  verdict")
    print(f"     {'baseline*':<10}{BASE['val_auc']:>9.3f}{0.0:>8.3f}{BASE['gate']:>8.3f}{0.0:>8.3f}{BASE['diff_r2']:>9.3f}  (unablated ref)")
    results = {"peak": peak, "baseline": BASE, "ops": {}}
    for name, spec in ops.items():
        hd = tgt.register_forward_hook(hook_for(spec)); asc = []; hcap = []
        try:
            for j, idx in enumerate(keep):
                r = recs[idx]
                # also recapture erased H at peak to measure residual difficulty R2
                sc, hid, nt = fwd(meta.get(r["id"], {}).get("problem", ""), r["steps_text"], True)
                asc.append(sc if (nt == len(r["steps_text"]) and np.isfinite(sc).all()) else scores[j])
                hcap.append(hid[peak])
        finally:
            hd.remove()
        He = np.array(hcap)
        dr2 = cross_val_score(Ridge(1.0), He, Lk, cv=5, scoring="r2").mean()
        va, ga = val_auc(asc), gate(asc)
        surg = (dr2 < BASE["diff_r2"] - 0.15) and (abs(va-BASE["val_auc"]) < 0.03) and (abs(ga-BASE["gate"]) < 0.03)
        results["ops"][name] = {"val_auc": va, "gate": ga, "diff_r2": dr2, "surgical": bool(surg)}
        print(f"     {name:<10}{va:>9.3f}{va-BASE['val_auc']:>+8.3f}{ga:>8.3f}{ga-BASE['gate']:>+8.3f}{dr2:>9.3f}  "
              f"{'SURGICAL' if surg else 'not surgical'}", flush=True)
    print("\n[E5] SURGICAL = difficulty R2 falls >0.15 AND |Δval_AUC|<0.03 AND |Δgate|<0.03.")
    print("     If none surgical (all drop the gate) -> report the STRONGER 'difficulty entangled with")
    print("     validity/competence' claim (consistent with M3 + Hydra). Do not tune.")
    json.dump(results, open(out / "e5_result.json", "w"), indent=2)
    print(f"[E5] wrote {out/'e5_result.json'}")


if __name__ == "__main__":
    main()
