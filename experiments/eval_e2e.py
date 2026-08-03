"""
eval_e2e.py — evaluation harness for the e2e checkpoints (Phase 2a + 2b).

One checkpoint in, every headline number out, under the SAME protocols used in Phase 1a-1d so
the results drop straight into the existing tables:

  * PROBE (2a for F-e2e / 2b-1d for R-e2e): clean-encode every candidate with the TRAINED
    encoder, attention-pool to a chain code, residualize length/latex/#steps/dataset (fit on
    train only), logistic probe A/B, evaluate on the held-out split -> f1_B, AUC, accuracy.
  * RETRIEVAL (2b-1c): r_i = decode(encode(text-corrupted_i)); target t_i = the FROZEN cached
    MiniLM embedding of the clean steps (never trained -> co-adaptation collapse impossible).
    Candidate-level mean-pool -> Recall@1/@10, median rank, MRR. Corruption uses a fixed rng so
    R-e2e and F-e2e see identical damage.
  * SPECIFICITY (2b-1c): mean pairwise cosine among r_i, with targets as the ceiling reference.
  * Val L_denoise, wall-clock and hardware (Appendix A.4).

Per-seed by design so it maps onto a SLURM array; --aggregate then folds the JSONs into a table.

    python experiments/eval_e2e.py --ckpt ckpt/seed0/sdae_e2e_best.pt --seed 0 --tag F_e2e
    python experiments/eval_e2e.py --aggregate --out results_e2e
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))
from sdae_prm import StepSDAE_PRM
from ridae import RiDAE
from train_sdae_e2e import encode_batch, split_recs
from multiseed_ablation import build_confounds
from content_probe import retrieval_metrics, mean_pairwise_cos
from phase1d_validity_probe import probe_full


def load_e2e(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    enc = RiDAE(device=device)
    enc.st.load_state_dict(ck["enc"])
    sdae = StepSDAE_PRM().to(device)
    sdae.load_state_dict(ck["sdae"])
    enc.eval(); sdae.eval()
    return enc, sdae


@torch.no_grad()
def clean_pooled(enc, sdae, recs, dev, bs=32):
    Z = []
    for i in range(0, len(recs), bs):
        X, _, pad, _, _ = encode_batch(enc, recs[i:i + bs], dev, rng=None)
        _, _, _, pooled = sdae(X, pad, None)
        Z.append(pooled.float().cpu().numpy())
    return np.concatenate(Z, 0)


@torch.no_grad()
def recon_and_target(enc, sdae, recs, dev, del_ratio, bs=32, corrupt_seed=1234):
    """Mean-pooled reconstruction from TEXT-corrupted input, and the frozen clean target."""
    rng = random.Random(corrupt_seed)
    R, T = [], []
    for i in range(0, len(recs), bs):
        X, target, pad, _, _ = encode_batch(enc, recs[i:i + bs], dev, rng=rng)
        recon, _, _, _ = sdae(X, pad, None)
        valid = (~pad).float().unsqueeze(-1)
        den = valid.sum(1).clamp(min=1)
        R.append(((recon * valid).sum(1) / den).float().cpu().numpy())
        T.append(((target * valid).sum(1) / den).float().cpu().numpy())
    return np.concatenate(R, 0), np.concatenate(T, 0)


@torch.no_grad()
def val_denoise(enc, sdae, va, dev, del_ratio, bs=32):
    import torch.nn.functional as F
    rng = random.Random(0); tot = 0.0; n = 0
    for i in range(0, len(va), bs):
        X, target, pad, _, _ = encode_batch(enc, va[i:i + bs], dev, rng=rng)
        recon, _, _, _ = sdae(X, pad, None)
        valid = ~pad
        tot += float((1.0 - F.cosine_similarity(recon, target, dim=-1))[valid].sum())
        n += int(valid.sum())
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="F_e2e", help="F_e2e (2a) or R_e2e (2b)")
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--del_ratio", type=float, default=0.25)
    ap.add_argument("--out", default=str(HERE / "results_e2e"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        return aggregate(out)

    t0 = time.time()
    recs = torch.load(args.cache, weights_only=False)
    y = np.array([r["chain"] for r in recs]); C = build_confounds(recs, args.data_dir)
    _, va = split_recs(recs, args.seed)
    enc, sdae = load_e2e(args.ckpt, args.device)

    # probe: over the full corpus — probe_full does its own fit-on-train / eval-on-val split.
    Z = clean_pooled(enc, sdae, recs, args.device)
    f1, auc, acc = probe_full(Z, C, y, args.seed)
    # retrieval: HELD-OUT ONLY, so N and the chance baseline match Phase 1c (N~340, chance
    # R@1 ~ 1/340). Running it over all 1,700 would mix train candidates in and silently
    # change the chance floor, making the e2e and frozen numbers incomparable.
    Rp, Tp = recon_and_target(enc, sdae, va, args.device, args.del_ratio)
    ret = retrieval_metrics(Rp, Tp)
    res = dict(tag=args.tag, seed=args.seed, f1_B=f1, AUC=auc, accuracy=acc,
               decode_spec=mean_pairwise_cos(Rp), target_spec=mean_pairwise_cos(Tp),
               r_std=float(Rp.std(0).mean()),
               val_denoise=val_denoise(enc, sdae, va, args.device, args.del_ratio),
               **{k: v for k, v in ret.items()},
               device=args.device, gpu=(torch.cuda.get_device_name(0)
                                       if torch.cuda.is_available() else platform.processor()),
               minutes=round((time.time() - t0) / 60, 2))
    (out / f"{args.tag}_seed{args.seed}.json").write_text(json.dumps(res, indent=2))
    print(f"[{args.tag} s{args.seed}] f1_B={f1:.3f} AUC={auc:.3f} acc={acc:.3f} | "
          f"R@1={ret['recall1']:.3f} medrank={ret['median_rank']:.0f} "
          f"spec={res['decode_spec']:.3f} | L_den={res['val_denoise']:.3f} "
          f"({res['minutes']:.1f}m on {res['gpu']})", flush=True)
    # sentence-transformers leaves worker threads alive, so the interpreter can hang for many
    # minutes after the result is written — on SLURM that burns the whole wall-clock limit and
    # holds the GPU. The JSON is already on disk here, so exit hard.
    sys.stdout.flush(); os._exit(0)


def aggregate(out: Path):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*_seed*.json"))]
    if not rows:
        print(f"no result JSONs in {out}"); return
    tags = sorted({r["tag"] for r in rows})
    keys = ["f1_B", "AUC", "accuracy", "recall1", "recall10", "median_rank", "mrr",
            "decode_spec", "val_denoise"]
    print("=" * 78)
    print("  PHASE 2a/2b — e2e results (mean ± std over seeds)")
    print("=" * 78)
    print(f"  {'metric':<14}" + "".join(f"{t:<22}" for t in tags))
    print("  " + "-" * 74)
    for k in keys:
        line = f"  {k:<14}"
        for t in tags:
            v = np.array([r[k] for r in rows if r["tag"] == t])
            line += f"{v.mean():.3f} ± {v.std():.3f}      " if len(v) else "n/a".ljust(22)
        print(line)
    print("=" * 78)
    for t in tags:
        sub = [r for r in rows if r["tag"] == t]
        print(f"  {t}: n={len(sub)} seeds | {sum(r['minutes'] for r in sub):.0f} min total "
              f"on {sub[0]['gpu']} (Appendix A.4)")
    import csv
    with (out / "e2e_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["tag", "seed"] + keys)
        for r in rows:
            w.writerow([r["tag"], r["seed"]] + [round(r[k], 4) for k in keys])
    print(f"  wrote {out/'e2e_summary.csv'}")


if __name__ == "__main__":
    main()
