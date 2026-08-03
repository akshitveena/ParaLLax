"""
train_sdae_e2e.py — Step-structured Denoising AE + PRM, END-TO-END (encoder UNFROZEN).

Upgrades over train_sdae.py:
  (1) The step encoder (MiniLM) is TRAINABLE — it can reshape step representations toward
      validity instead of using frozen features.
  (2) TEXT-level corruption: we delete words from each step (TSDAE-style noise) and the
      trainable encoder must encode the damaged text.
  Collapse-safety: the reconstruction TARGET is the FIXED frozen MiniLM embedding of the
  CLEAN step (precomputed in the cache), so the encoder cannot cheat by collapsing.

Not included (deliberately): a from-scratch token decoder (full TSDAE text generation) —
heavy and low-ROI on 1.7k candidates + CPU; measure the unfreeze lift first.

    python main/train_sdae_e2e.py --cache data/step_cache.pt \
        --ckpt_dir checkpoints_sdae_e2e --epochs 12 --device cpu
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdae_prm import StepSDAE_PRM
from ridae import RiDAE


def corrupt_text(s, rng, del_ratio=0.25):
    w = s.split()
    if len(w) < 4:
        return s
    keep = [x for x in w if rng.random() > del_ratio]
    return " ".join(keep) if keep else " ".join(w[:1])


def split_recs(recs, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(recs)); rng.shuffle(idx)
    cut = int(0.8 * len(recs))
    return [recs[i] for i in idx[:cut]], [recs[i] for i in idx[cut:]]


def encode_batch(enc, recs, dev, rng=None):
    """Encode each candidate's steps with the TRAINABLE encoder. rng!=None -> corrupt text.
    Returns X (B,T,384 with grad), target (B,T,384 frozen), pad (B,T), SL (B,T), chains (B,)."""
    flat, offs = [], []
    for r in recs:
        steps = r["steps_text"]
        if rng is not None:
            steps = [corrupt_text(t, rng) for t in steps]
        offs.append((len(flat), len(flat) + len(steps)))
        flat.extend(steps)
    emb = enc._encode_with_grad(flat)                          # (total, 384), grad-tracked
    Xs = [emb[a:b] for (a, b) in offs]
    X = pad_sequence(Xs, batch_first=True)                     # (B, T, 384)
    target = pad_sequence([torch.from_numpy(r["steps_emb"]).float() for r in recs],
                          batch_first=True).to(dev)
    SL = pad_sequence([torch.from_numpy(r["step_labels"]).long() for r in recs],
                      batch_first=True, padding_value=-1).to(dev)
    lengths = [b - a for (a, b) in offs]
    pad = torch.ones(len(recs), X.size(1), dtype=torch.bool, device=dev)
    for i, n in enumerate(lengths):
        pad[i, :n] = False
    chains = torch.tensor([0 if r["chain"] == "A" else 1 for r in recs], device=dev)
    return X, target, pad, SL, chains


def evaluate(enc, sdae, va, dev, bs=32):
    from sklearn.metrics import f1_score, roc_auc_score
    enc.eval(); sdae.eval()
    ct, cp, st, ss = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(va), bs):
            batch = va[i:i + bs]
            X, target, pad, SL, ch = encode_batch(enc, batch, dev, rng=None)   # clean at eval
            _, prm, cl, _ = sdae(X, pad, None)
            cp += (torch.sigmoid(cl).cpu().numpy() > 0.5).astype(int).tolist()
            ct += ch.cpu().numpy().tolist()
            m = (SL >= 0) & (~pad)
            st += SL[m].cpu().numpy().tolist(); ss += torch.sigmoid(prm)[m].cpu().numpy().tolist()
    f1 = f1_score(ct, cp, pos_label=1) if len(set(ct)) > 1 else 0.0
    try:
        auc = roc_auc_score(st, ss) if len(set(st)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return f1, auc


def evaluate_denoise(enc, sdae, va, dev, del_ratio, bs=32):
    """Mean val L_denoise under a FIXED corruption rng (reproducible across epochs/seeds).

    Model selection for Variant R-e2e (--heads none): that variant has no chain loss, so
    selecting on val chain_f1 would be selecting on noise. Selection must track the only
    objective it actually optimises."""
    enc.eval(); sdae.eval()
    rng = random.Random(0); tot = 0.0; n = 0
    with torch.no_grad():
        for i in range(0, len(va), bs):
            X, target, pad, SL, ch = encode_batch(enc, va[i:i + bs], dev, rng=rng)
            recon, _, _, _ = sdae(X, pad, None)
            valid = ~pad
            cos = F.cosine_similarity(recon, target, dim=-1)
            tot += float((1.0 - cos)[valid].sum()); n += int(valid.sum())
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--ckpt_dir", default="checkpoints_sdae_e2e")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--enc_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--del_ratio", type=float, default=0.25)
    ap.add_argument("--lam_prm", type=float, default=1.0)
    ap.add_argument("--lam_ab", type=float, default=1.0)
    ap.add_argument("--heads", default="prm_chain",
                    choices=["prm_chain", "prm", "chain", "none"],
                    help="active supervised heads; 'none' = Variant R-e2e (recon-only, Phase 2b)")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    recs = torch.load(args.cache, weights_only=False)
    if "steps_text" not in recs[0]:
        print("ERROR: cache has no steps_text — rebuild with the updated build_step_embeddings.py")
        sys.exit(1)
    tr, va = split_recs(recs, args.seed)
    dev = torch.device(args.device)

    enc = RiDAE(device=args.device)          # used only as the trainable step-text encoder
    sdae = StepSDAE_PRM().to(dev)
    opt = torch.optim.AdamW([
        {"params": enc.st.parameters(), "lr": args.enc_lr},
        {"params": sdae.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    print(f"[e2e] train={len(tr)} val={len(va)} device={dev} | encoder UNFROZEN, text-corrupt "
          f"del_ratio={args.del_ratio}")

    best, no_imp, t0 = -1.0, 0, time.time()
    for ep in range(args.epochs):
        enc.train(); sdae.train()
        order = list(range(len(tr))); rng.shuffle(order)
        agg = [0.0, 0.0, 0.0]; nb = 0
        for i in range(0, len(order), args.batch_size):
            batch = [tr[j] for j in order[i:i + args.batch_size]]
            X, target, pad, SL, ch = encode_batch(enc, batch, dev, rng=rng)
            recon, prm, cl, _ = sdae(X, pad, None)
            valid = ~pad
            l_den = (1.0 - F.cosine_similarity(recon, target, dim=-1))[valid].mean()
            pm = (SL >= 0) & valid
            l_prm = (F.binary_cross_entropy_with_logits(prm[pm], SL[pm].float())
                     if pm.any() else torch.zeros((), device=dev))
            l_ab = F.binary_cross_entropy_with_logits(cl, ch.float())
            loss = l_den
            if "prm" in args.heads:
                loss = loss + args.lam_prm * l_prm
            if "chain" in args.heads:
                loss = loss + args.lam_ab * l_ab
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.st.parameters()) + list(sdae.parameters()), 1.0)
            opt.step()
            agg[0] += float(l_den); agg[1] += float(l_prm); agg[2] += float(l_ab); nb += 1
        f1, auc = evaluate(enc, sdae, va, dev)
        # Selection metric follows the objective: recon-only (R-e2e) has no chain loss, so it
        # selects on val L_denoise (lower better); anything with a chain head selects on f1_B.
        if args.heads == "none":
            vd = evaluate_denoise(enc, sdae, va, dev, args.del_ratio)
            score, shown = -vd, f"val L_denoise={vd:.4f}"
        else:
            score, shown = f1, f"val chain_f1_B={f1:.3f} step_AUC={auc:.3f}"
        print(f"[e2e] ep{ep:2d}  denoise={agg[0]/nb:.3f} prm={agg[1]/nb:.3f} ab={agg[2]/nb:.3f}"
              f"  |  {shown}  ({(time.time()-t0)/60:.1f}m)")
        if score > best + 1e-4:
            best, no_imp = score, 0
            torch.save({"sdae": sdae.state_dict(), "enc": enc.st.state_dict()},
                       Path(args.ckpt_dir) / "sdae_e2e_best.pt")
            print("        ^ improved — saved")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"[e2e] early stop at epoch {ep}"); break

    print("=" * 60)
    print(f"[e2e] done in {(time.time()-t0)/60:.1f} min | best val chain_f1_B={best:.3f}")
    print(f"[e2e] compare to frozen-encoder SDAE (0.640 raw / 0.436 confound-controlled).")


if __name__ == "__main__":
    main()
