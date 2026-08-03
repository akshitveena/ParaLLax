"""
train_sdae.py — train the Step-structured Denoising AE + PRM on cached step embeddings.

Denoising: mask a fraction of step-vectors, reconstruct the clean step sequence (keeps the
autoencoder). PRM: per-step error head on ProcessBench step labels. Chain: A/B head on the
attention-pooled step-codes. The headline number is val chain f1_B — compare it to the
POOLED-vector ceiling (~0.29 f1_B). If it clears that, the step-structured representation
captured the validity signal pooling threw away.

    python main/train_sdae.py --cache data/step_cache.pt --ckpt_dir checkpoints_sdae \
        --epochs 30 --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdae_prm import StepSDAE_PRM, losses


class StepDS(Dataset):
    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        return (torch.from_numpy(r["steps_emb"]).float(),
                torch.from_numpy(r["step_labels"]).long(),
                0 if r["chain"] == "A" else 1)


def collate(batch):
    embs, labs, chains = zip(*batch)
    T = max(e.size(0) for e in embs); D = embs[0].size(1); B = len(embs)
    X = torch.zeros(B, T, D)
    SL = torch.full((B, T), -1, dtype=torch.long)
    pad = torch.ones(B, T, dtype=torch.bool)
    for i, (e, l) in enumerate(zip(embs, labs)):
        n = e.size(0)
        X[i, :n] = e; SL[i, :n] = l; pad[i, :n] = False
    return X, SL, pad, torch.tensor(chains)


def make_corrupt(pad, frac, gen):
    return (torch.rand(pad.shape, generator=gen) < frac) & (~pad)


def evaluate(model, loader, device):
    from sklearn.metrics import f1_score, roc_auc_score
    model.eval()
    ct, cp, st, ss = [], [], [], []
    with torch.no_grad():
        for X, SL, pad, ch in loader:
            recon, prm, cl, _ = model(X.to(device), pad.to(device), None)
            cp += (torch.sigmoid(cl).cpu().numpy() > 0.5).astype(int).tolist()
            ct += ch.numpy().tolist()
            m = (SL >= 0) & (~pad)
            st += SL[m].numpy().tolist()
            ss += torch.sigmoid(prm).cpu()[m].numpy().tolist()
    f1 = f1_score(ct, cp, pos_label=1) if len(set(ct)) > 1 else 0.0
    try:
        auc = roc_auc_score(st, ss) if len(set(st)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return f1, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--ckpt_dir", default="checkpoints_sdae")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--corrupt_frac", type=float, default=0.25)
    ap.add_argument("--lam_prm", type=float, default=1.0)
    ap.add_argument("--lam_ab", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    recs = torch.load(args.cache, weights_only=False)
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(recs)); rng.shuffle(idx)
    cut = int(0.8 * len(recs))
    tr = [recs[i] for i in idx[:cut]]; va = [recs[i] for i in idx[cut:]]
    dev = torch.device(args.device)

    trl = DataLoader(StepDS(tr), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val = DataLoader(StepDS(va), batch_size=64, collate_fn=collate)
    model = StepSDAE_PRM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    print(f"[sdae] train={len(tr)} val={len(va)} device={dev} params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    best, no_imp, t0 = -1.0, 0, time.time()
    for ep in range(args.epochs):
        model.train(); agg = [0.0, 0.0, 0.0]
        for X, SL, pad, ch in trl:
            X, SL, pad, ch = X.to(dev), SL.to(dev), pad.to(dev), ch.to(dev)
            cm = make_corrupt(pad.cpu(), args.corrupt_frac, gen).to(dev)
            recon, prm, cl, _ = model(X, pad, cm)
            tot, ld, lp, la = losses(recon, X, prm, SL, cl, ch, pad, cm,
                                     args.lam_prm, args.lam_ab)
            opt.zero_grad(); tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            agg[0] += float(ld); agg[1] += float(lp); agg[2] += float(la)
        f1, auc = evaluate(model, val, dev)
        n = max(len(trl), 1)
        print(f"[sdae] ep{ep:2d}  denoise={agg[0]/n:.3f} prm={agg[1]/n:.3f} ab={agg[2]/n:.3f}"
              f"  |  val chain_f1_B={f1:.3f}  step_AUC={auc:.3f}")
        if f1 > best + 1e-4:
            best, no_imp = f1, 0
            torch.save(model.state_dict(), Path(args.ckpt_dir) / "sdae_best.pt")
            print("        ^ improved — saved")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"[sdae] early stop at epoch {ep}"); break

    print("=" * 60)
    print(f"[sdae] done in {(time.time()-t0)/60:.1f} min | best val chain_f1_B={best:.3f}")
    print(f"[sdae] POOLED-vector ceiling was ~0.29 f1_B. If best clears it, the step-")
    print(f"[sdae] structured representation captured what pooling averaged away.")


if __name__ == "__main__":
    main()
