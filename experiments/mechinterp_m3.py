"""
mechinterp_m3.py — M3 on the A100 box: resolve M2's single-direction null.

M2 found: difficulty is richly linearly represented in Math-Shepherd's residual stream (R²≈0.92)
but mean-ablating ONE length direction at the peak layer barely moved the score (distributed, not
one direction). M3 asks the sharper follow-ups:

  M3a  SUBSPACE ablation sweep — mean-ablate the top-k difficulty directions (PLS-derived,
       nested) at the peak layer, k = 1,2,4,8,16,32,64. Does raw f1_B fall toward its controlled
       value as k grows? -> quantifies HOW distributed / at what rank it becomes removable.
  M3b  SELF-REPAIR (Hydra effect) — with the ablation LIVE at the peak layer, re-probe difficulty
       at downstream layers. If R² recovers, the network reconstructs difficulty downstream ->
       a mechanistic reason single-point ablation fails.
  M3c  ATTRIBUTION — where difficulty is written: (i) per-LAYER, the residual delta projected onto
       the length direction; (ii) per-HEAD at the peak layer, each attention head's write onto the
       length direction (scoped head attribution, not full path-patching).

Reuses M2's validated tokenizer/precision (cand [648,387], tag 12902, bf16). Baseline must
reproduce gate ~0.735 or the run aborts.

    HF_HOME=/workspace/ridae/.hf python experiments/mechinterp_m3.py --every 4
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
    ap.add_argument("--ks", default="1,2,4,8,16,32,64")
    ap.add_argument("--out", default=str(ROOT / "experiments/results_mechinterp"))
    args = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    KS = [int(k) for k in args.ks.split(",")]

    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    CAND = tok.encode(f"{GOOD} {BAD}")[1:]; TAG = tok.encode(f"{STEP}")[-1]
    assert len(CAND) == 2, f"tokenizer mismatch {CAND}"
    print(f"[M3] cand {CAND} tag {TAG}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto", output_hidden_states=True).eval()
    NL = model.config.num_hidden_layers
    NH = model.config.num_attention_heads
    HD = model.config.hidden_size // NH
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

    # ---- capture baseline ----
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
        if (i + 1) % 400 == 0: print(f"  captured {i+1} (skip {skip}, {(time.time()-t0)/60:.1f}m)", flush=True)
    keep = np.array(keep)
    for li in LAYERS: acts[li] = np.array(acts[li])
    assert len(keep) > 1000, "too many skipped — abort"
    Lk, NSk, yk, CFk = L[keep], NS[keep], y[keep], CF[keep]
    r2l = {li: cross_val_score(Ridge(1.0), acts[li], Lk, cv=5, scoring="r2").mean() for li in LAYERS}
    peak = max(LAYERS, key=lambda li: r2l[li])
    print(f"[M3] captured {len(keep)} ({skip} skip). peak length layer {peak} R2={r2l[peak]:.3f}")

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
    ybin = (yk == "B").astype(int)
    def auc_of(v):
        _, va = sp(len(v)); return roc_auc_score(ybin[va], v[va])          # threshold-free
    def ap_of(v):
        _, va = sp(len(v)); return average_precision_score(ybin[va], v[va])
    bv = cmin(scores); tr, _ = sp(len(bv)); beta = np.linalg.lstsq(CFk[tr], bv[tr], rcond=None)[0]
    bvc = bv - CFk @ beta
    BASE = {"raw": f1v(bv), "ctl": f1v(bvc), "gate": gate(scores),
            "raw_auc": auc_of(bv), "ctl_auc": auc_of(bvc),
            "raw_ap": ap_of(bv), "ctl_ap": ap_of(bvc)}
    print(f"[M3] BASELINE  raw f1 {BASE['raw']:.3f} / AUC {BASE['raw_auc']:.3f} / AP {BASE['raw_ap']:.3f}")
    print(f"[M3] CONTROLLED ref  f1 {BASE['ctl']:.3f} / AUC {BASE['ctl_auc']:.3f} / AP {BASE['ctl_ap']:.3f}   gate {BASE['gate']:.3f}")
    print(f"[M3] HONEST convergence target = CONTROLLED AUC {BASE['ctl_auc']:.3f}  "
          f"(the f1 {BASE['ctl']:.3f} is the degenerate-threshold value R3a flagged — do NOT converge to it)")
    assert BASE["gate"] > 0.65, "baseline gate too low — stack problem, abort"

    # ---- difficulty subspace basis (nested top-k), PLS on [log_length, n_steps] ----
    pls = PLSRegression(n_components=max(KS)).fit(acts[peak], np.c_[Lk, NSk])
    Wd = np.linalg.qr(pls.x_weights_)[0]                 # (d, kmax) orthonormal, nested columns
    tgt = model.model.layers[peak - 1]

    def subspace_hook(B, mu):
        Bt = torch.tensor(B, dtype=torch.bfloat16)
        mut = torch.tensor(mu, dtype=torch.float32)
        def hook(mod, inp, o):
            h = o[0] if isinstance(o, tuple) else o
            Bd = Bt.to(h.device); coord = h.float() @ Bd.float()          # (B,L,k)
            delta = ((coord - mut.to(h.device)) @ Bd.float().T).to(h.dtype)
            h = h - delta
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return hook

    def rescore(hook, want_hidden=False):
        hd = tgt.register_forward_hook(hook); out_sc = []; dh = {li: [] for li in LAYERS if li > peak}
        try:
            for j, idx in enumerate(keep):
                r = recs[idx]; prob = meta.get(r["id"], {}).get("problem", "")
                sc, hid, nt = fwd(prob, r["steps_text"], want_hidden)
                out_sc.append(sc if (nt == len(r["steps_text"]) and np.isfinite(sc).all()) else scores[j])
                if want_hidden and hid is not None:
                    for li in dh: dh[li].append(hid[li])
        finally:
            hd.remove()
        return out_sc, {li: np.array(v) for li, v in dh.items()}

    # ---- M3a subspace ablation sweep ----
    print("[M3a] subspace ablation sweep — THRESHOLD-FREE (raw f1 / raw AUC / raw AP / gate vs k):")
    print(f"      convergence target = controlled AUC {BASE['ctl_auc']:.3f}  |  chance AUC 0.500")
    sweep = {}
    for k in KS:
        B = Wd[:, :k]; mu = (acts[peak] @ B).mean(0)
        asc, _ = rescore(subspace_hook(B, mu))
        av = cmin(asc)
        sweep[k] = {"raw": f1v(av), "raw_auc": auc_of(av), "raw_ap": ap_of(av), "gate": gate(asc)}
        print(f"   k={k:>3}: raw_f1={sweep[k]['raw']:.3f}  raw_AUC={sweep[k]['raw_auc']:.3f}  "
              f"raw_AP={sweep[k]['raw_ap']:.3f}  gate={sweep[k]['gate']:.3f}")
    print(f"   READING: does raw_AUC fall to the controlled AUC ({BASE['ctl_auc']:.3f}) WHILE the gate holds?")
    print("     yes -> convergence to control is real, threshold-free (claim survives R3a).")
    print("     raw_AUC blows PAST it toward 0.50 as the gate collapses -> the f1 'convergence' was a")
    print("     degenerate-threshold artifact; the ablation over-suppresses (removes competence too).")

    # ---- M3b self-repair: with the largest-k ablation live, re-probe downstream R² ----
    kmax = max(KS); B = Wd[:, :kmax]; mu = (acts[peak] @ B).mean(0)
    _, dh = rescore(subspace_hook(B, mu), want_hidden=True)
    print(f"[M3b] self-repair — downstream length R² (baseline vs ablated k={kmax}):")
    repair = {}
    for li in [x for x in LAYERS if x > peak]:
        base_r2 = r2l[li]
        abl_r2 = cross_val_score(Ridge(1.0), dh[li], Lk, cv=5, scoring="r2").mean()
        repair[li] = (base_r2, abl_r2)
        tag = "RECONSTRUCTED" if abl_r2 > base_r2 - 0.15 else "suppressed"
        print(f"   layer {li:2d}: baseline {base_r2:.3f} -> ablated {abl_r2:.3f}   {tag}")

    # ---- M3c attribution: per-layer write onto length dir; per-head write at peak layer ----
    w = Ridge(1.0).fit(acts[peak], Lk).coef_; w = w / np.linalg.norm(w)
    layer_proj = {li: float(np.abs(acts[li] @ w).mean()) for li in LAYERS}
    # per-head write at peak: hook the peak layer's o_proj INPUT (concat of head outputs)
    headcap = {}
    def ohook(mod, inp, o):
        headcap["x"] = inp[0].detach()            # (B, L, NH*HD) pre-o_proj
    hp = model.model.layers[peak - 1].self_attn.o_proj.register_forward_hook(ohook)
    Wo = model.model.layers[peak - 1].self_attn.o_proj.weight.detach()   # (d, NH*HD)
    head_write = np.zeros(NH)
    wt = torch.tensor(w, dtype=Wo.dtype, device=Wo.device)
    try:
        n_used = 0
        for idx in keep[:400]:                     # 400 solutions is plenty for a stable mean
            r = recs[idx]; prob = meta.get(r["id"], {}).get("problem", "")
            try:
                fwd(prob, r["steps_text"], False)
            except Exception:
                continue
            x = headcap["x"][0]                    # (L, NH*HD)
            for hh in range(NH):
                sl = slice(hh * HD, (hh + 1) * HD)
                contrib = x[:, sl].to(Wo.dtype) @ Wo[:, sl].T     # (L, d) this head's residual write
                head_write[hh] += float((contrib.float() @ wt.float()).abs().mean())
            n_used += 1
        head_write /= max(n_used, 1)
    finally:
        hp.remove()
    top_heads = np.argsort(-head_write)[:8]
    print(f"[M3c] per-head write onto length dir @ layer {peak} — top heads: "
          f"{[(int(h), round(float(head_write[h]),3)) for h in top_heads]}")

    # ---- outputs ----
    json.dump({"peak": peak, "baseline": BASE, "sweep": {str(k): v for k, v in sweep.items()},
               "self_repair": {str(li): list(v) for li, v in repair.items()},
               "layer_proj": {str(li): v for li, v in layer_proj.items()},
               "head_write": head_write.tolist(), "n_kept": int(len(keep))},
              open(out / "m3_result.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(list(sweep), [v["raw"] for v in sweep.values()], marker="o", label="raw f1_B")
        ax[0].plot(list(sweep), [v["gate"] for v in sweep.values()], marker="s", label="step gate")
        ax[0].axhline(BASE["raw"], ls="--", c="gray", label="baseline raw")
        ax[0].axhline(BASE["ctl"], ls=":", c="red", label="controlled")
        ax[0].set_xscale("log", base=2); ax[0].set_xlabel("k ablated directions"); ax[0].set_ylabel("raw f1_B")
        ax[0].set_title("M3a: subspace ablation sweep"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].bar(range(NH), head_write); ax[1].set_xlabel("head"); ax[1].set_ylabel("|write onto length dir|")
        ax[1].set_title(f"M3c: per-head difficulty write @ layer {peak}")
        fig.tight_layout(); fig.savefig(out / "m3_analysis.png", dpi=140)
        print(f"  saved {out/'m3_analysis.png'} and m3_result.json")
    except Exception as e:
        print(f"  plot skipped: {e}")


if __name__ == "__main__":
    main()
