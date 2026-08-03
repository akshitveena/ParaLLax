"""
multiseed_ablation.py — error bars on the core ablation (#1).

Trains the FROZEN step-structured SDAE across N seeds and computes each seed's
confound-controlled, leakage-free f1_B on that seed's OWN 80/20 split; also the raw-SBERT
pooled baseline across the same seeds (training-free). Emits mean±std + box plot + CSV.

Each seed varies init + data order + corruption + split together (real run-to-run variance).
Leakage-free: the model trains on the seed's train split; the probe is fit on train and
evaluated on the held-out val split (unseen by model and probe).

E2e multi-seed (the 0.576 number) is a separate ~3 hr/seed background job — see --emit_e2e_cmds.

    python experiments/multiseed_ablation.py --seeds 0,1,2,3,4 --device cpu
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "main"))
from sdae_prm import StepSDAE_PRM
from train_sdae import StepDS, collate
from torch.utils.data import DataLoader
from data_pipeline import load_candidates


def pooled_reps(model, recs, dev="cpu", bs=64):
    Z = []
    with torch.no_grad():
        for X, SL, pad, ch in DataLoader(StepDS(recs), batch_size=bs, collate_fn=collate, shuffle=False):
            _, _, _, pl = model(X.to(dev), pad.to(dev), None)
            Z.append(pl.cpu().numpy())
    return np.concatenate(Z, 0)


def build_confounds(recs, data_dir):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    cd = {c.record_id: c for c in load_candidates(Path(data_dir) / "candidates.jsonl")}
    L, LA, NS, DS = [], [], [], []
    for r in recs:
        c = cd.get(r["id"]); t = (c.response_text or c.full_text or "") if c else ""
        L.append(len(t.split())); LA.append((t.count("\\") + t.count("$")) / max(len(t.split()), 1))
        NS.append(c.num_steps if c else 0); DS.append(r["split"])
    surf = StandardScaler().fit_transform(np.array([L, LA, NS], float).T)
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(np.array(DS).reshape(-1, 1))
    return np.hstack([np.ones((len(recs), 1)), surf, oh])


def probe_f1(Z, C, y, seed):
    """Confound-controlled, leakage-free f1_B on the seed's own 80/20 split (matches train_sdae)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    rng = np.random.RandomState(seed); idx = np.arange(len(Z)); rng.shuffle(idx)
    cut = int(0.8 * len(Z)); tri, vai = idx[:cut], idx[cut:]
    beta, *_ = np.linalg.lstsq(C[tri], Z[tri], rcond=None)      # residualizer fit on train only
    Zr = Z - C @ beta
    clf = LogisticRegression(max_iter=2000).fit(Zr[tri], y[tri])
    return float(f1_score(y[vai], clf.predict(Zr[vai]), pos_label="B"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results_multiseed"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out); (out / "ckpts").mkdir(parents=True, exist_ok=True)
    recs = torch.load(args.cache, weights_only=False)
    y = np.array([r["chain"] for r in recs])
    C = build_confounds(recs, args.data_dir)
    sbert = np.array([r["steps_emb"].mean(0) for r in recs])    # raw-SBERT pooled = mean over steps

    rows = []
    for s in seeds:
        f1_sbert = probe_f1(sbert, C, y, s)
        ckpt = out / "ckpts" / f"frozen_seed{s}"
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        subprocess.run([sys.executable, str(ROOT / "main/train_sdae.py"),
                        "--cache", args.cache, "--ckpt_dir", str(ckpt), "--epochs", "30",
                        "--seed", str(s), "--device", args.device],
                       check=True, env=env, stdout=subprocess.DEVNULL)
        model = StepSDAE_PRM(); model.load_state_dict(torch.load(ckpt / "sdae_best.pt", map_location="cpu"))
        model.eval()
        Z = pooled_reps(model, recs, args.device)
        f1_frozen = probe_f1(Z, C, y, s)
        rows.append((s, f1_sbert, f1_frozen))
        print(f"  seed {s}:  raw_sbert f1_B={f1_sbert:.3f}   frozen_step_SDAE f1_B={f1_frozen:.3f}", flush=True)

    sb = np.array([r[1] for r in rows]); fr = np.array([r[2] for r in rows])
    import csv
    with (out / "metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["seed", "raw_sbert_f1B", "frozen_step_sdae_f1B"]); w.writerows(rows)

    print("\n" + "=" * 58)
    print(f"  CORE ABLATION — multi-seed (n={len(seeds)} seeds), confound-controlled f1_B")
    print("=" * 58)
    print(f"  raw-SBERT (pooled) : {sb.mean():.3f} ± {sb.std():.3f}")
    print(f"  frozen step-SDAE   : {fr.mean():.3f} ± {fr.std():.3f}")
    print(f"  (e2e step-SDAE     : run separately — ~3 hr/seed, background)")
    print("=" * 58)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([sb, fr], labels=["raw-SBERT\n(pooled)", "step-SDAE\n(frozen)"], showmeans=True)
        for i, d in enumerate([sb, fr], 1):
            ax.scatter([i] * len(d), d, color="black", alpha=0.5, zorder=3)
        ax.set_ylabel("confound-controlled f1_B"); ax.set_title(f"Core ablation ({len(seeds)} seeds)")
        fig.tight_layout(); fig.savefig(out / "ablation_boxplot.png", dpi=140)
        print(f"  saved {out/'ablation_boxplot.png'} and metrics.csv")
    except Exception as e:
        print(f"  plot skipped: {e}")


if __name__ == "__main__":
    main()
