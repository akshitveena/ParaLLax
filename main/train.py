"""
train.py — RiDAE training loop.

    L_total = L_reconstruct + L_MNR + lambda * L_triplet     (lambda = 0.3)

Split is BY PROBLEM (contrastive_group): all candidates / corruptions from one
problem stay in one split. Corruptions are built on training candidates only.

Hard-negative mining (schema Block 8): once a first run has populated
hardness_score, training uses a WeightedRandomSampler that oversamples corruptions
of high-hardness Type B candidates (the ones the encoder still confuses) and
undersamples easy ones. On the first run (all hardness_score == None) sampling is
uniform, so this is a no-op until compute_hardness.py has run.

Usage:
    python main/train.py --data_dir data/processed --output_dir outputs
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_pipeline import load_candidates, load_contrastive_pairs
from corruption import build_corruption_dataset
from ridae import RiDAE

_PRIORITY_W = {"high": 1.5, "medium": 1.0, "low": 0.7}


class CorruptionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return s.corrupted_text, s.original_text


def collate(batch):
    corrupted, original = zip(*batch)
    return list(corrupted), list(original)


def split_by_group(candidates, train_frac=0.8):
    groups = sorted({c.contrastive_group for c in candidates})
    cut = int(len(groups) * train_frac)
    train_ids = set(groups[:cut])
    train = [c for c in candidates if c.contrastive_group in train_ids]
    val = [c for c in candidates if c.contrastive_group not in train_ids]
    return train, val, train_ids


def sample_weights(samples) -> list[float]:
    """priority weight x hardness weight. Uniform until hardness is populated."""
    w = []
    for s in samples:
        pw = _PRIORITY_W.get(s.corruption_priority, 1.0)
        hw = (0.5 + float(s.hardness_score)) if s.hardness_score is not None else 1.0
        w.append(pw * hw)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda_contrastive", type=float, default=0.3)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(data_dir / "candidates.jsonl")
    pairs_path = data_dir / "contrastive_pairs.json"
    pairs = load_contrastive_pairs(pairs_path) if pairs_path.exists() else []
    print(f"[train] {len(candidates)} candidates, {len(pairs)} contrastive pairs")

    train_cands, val_cands, train_ids = split_by_group(candidates)
    train_samples = build_corruption_dataset(train_cands, seed=args.seed)
    val_samples = build_corruption_dataset(val_cands, seed=args.seed + 1)
    print(f"[train] train groups={len(train_ids)} | train corruptions={len(train_samples)} "
          f"| val corruptions={len(val_samples)}")

    # Cross-model pairs (QwQ-B vs Claude-A) are the hardest negatives — oversample
    # them so the encoder must learn approach geometry, not stylistic signatures (item 6).
    train_pairs = [p for p in pairs if p.contrastive_group in train_ids]
    a_pool, b_pool = [], []
    for p in train_pairs:
        reps = 3 if p.cross_model else 1
        a_pool += [p.type_a.full_text] * reps
        b_pool += [p.type_b.full_text] * reps
    n_cross = sum(p.cross_model for p in train_pairs)
    print(f"[train] contrastive pool: {len(a_pool)} A / {len(b_pool)} B "
          f"({n_cross} cross-model pairs, oversampled 3x)")

    has_hardness = any(s.hardness_score is not None for s in train_samples)
    if has_hardness:
        weights = sample_weights(train_samples)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(CorruptionDataset(train_samples), batch_size=args.batch_size,
                            sampler=sampler, collate_fn=collate)
        print("[train] hard-negative mining ACTIVE (hardness_score populated)")
    else:
        loader = DataLoader(CorruptionDataset(train_samples), batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate)
        print("[train] uniform sampling (no hardness_score yet — run compute_hardness.py "
              "after this run to enable mining)")

    model = RiDAE(encoder_name=args.encoder, device=args.device)
    print(f"[train] encoder={args.encoder} device={model.device}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step):
        return args.lr * (step + 1) / args.warmup_steps if step < args.warmup_steps else args.lr

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(["epoch", "step", "L_reconstruct", "L_MNR", "L_triplet", "L_total", "lr"])

    rng = random.Random(args.seed)
    best_val, best_epoch, no_improve, gstep = float("inf"), -1, 0, 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for corrupted, original in loader:
            for g in optim.param_groups:
                g["lr"] = lr_at(gstep)

            l_rec, _ = model.reconstruction_loss(corrupted, original)
            l_mnr = model.mnr_loss(corrupted, original)
            l_tri = torch.zeros((), device=model.device)
            if len(a_pool) >= 2 and len(b_pool) >= 2:
                k = min(args.batch_size, len(a_pool), len(b_pool))
                ai = rng.sample(range(len(a_pool)), k); bi = rng.sample(range(len(b_pool)), k)
                l_tri = model.contrastive_loss_from_pairs([a_pool[i] for i in ai],
                                                          [b_pool[i] for i in bi])
            l_total = l_rec + l_mnr + args.lambda_contrastive * l_tri

            optim.zero_grad(); l_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            if gstep % 50 == 0:
                with log_path.open("a", newline="") as fh:
                    csv.writer(fh).writerow([epoch, gstep, f"{l_rec.item():.5f}",
                                             f"{l_mnr.item():.5f}", f"{float(l_tri.detach()):.5f}",
                                             f"{l_total.item():.5f}", f"{lr_at(gstep):.2e}"])
                print(f"  e{epoch} s{gstep}  rec={l_rec.item():.4f} mnr={l_mnr.item():.4f} "
                      f"tri={float(l_tri.detach()):.4f} total={l_total.item():.4f}")
            gstep += 1

        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for corrupted, original in DataLoader(CorruptionDataset(val_samples),
                                                  batch_size=args.batch_size, collate_fn=collate):
                l_rec, _ = model.reconstruction_loss(corrupted, original)
                vloss += l_rec.item() * len(corrupted); n += len(corrupted)
        vloss /= max(n, 1)
        print(f"[train] epoch {epoch}: val L_reconstruct = {vloss:.5f}")
        if vloss < best_val - 1e-5:
            best_val, best_epoch, no_improve = vloss, epoch, 0
            model.save(ckpt_dir / "ridae_best.pt"); print("        ^ improved — saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[train] early stopping at epoch {epoch}"); break

    print("=" * 60)
    print(f"[train] done in {(time.time()-t0)/60:.1f} min | best val={best_val:.5f} @epoch {best_epoch}")
    print(f"[train] checkpoint: {ckpt_dir/'ridae_best.pt'} | log: {log_path}")


if __name__ == "__main__":
    main()
