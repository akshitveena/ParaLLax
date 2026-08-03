"""
train_supervised.py — make VALIDITY the objective (the fix for the length-confound negative).

The reconstruction-only run organised z around length, not wrong-approach-right-answer.
This trains z with a SUPERVISED contrastive objective (SupCon) on the A/B labels, so the
encoder is pushed to separate sound from flawed reasoning — the property we care about.

Two design points that matter:
  * LENGTH-MATCHED batches. B correlates with length; if batches mix all lengths the model
    just relearns length. We sort by length and draw each batch from a narrow length WINDOW,
    so within a batch A and B have similar length -> length is useless -> the only way to
    satisfy SupCon is to encode validity.
  * Reconstruction OFF by default (--lambda_recon 0): the ceiling diagnostic showed it
    degraded the signal. Supervised contrastive is the primary (and only) objective.

Success test (run diagnose_ceiling.py after): z_residualized f1_B must clearly beat 0.286
(raw SBERT's residual). If not, the pooled-MiniLM ceiling is confirmed -> PRM/step-structured.

    python main/train_supervised.py --data_dir data/processed_pb \
        --checkpoint_dir checkpoints_pb_sup --epochs 12 --device cpu
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_pipeline import load_candidates
from ridae import RiDAE


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al.) on 64-d codes. Positives = same class."""
    z = F.normalize(z, dim=1)
    B = z.size(0)
    sim = (z @ z.t()) / tau
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()          # stability
    self_mask = 1.0 - torch.eye(B, device=z.device)
    exp = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    lab = labels.view(-1, 1)
    pos = (lab == lab.t()).float() * self_mask
    pos_count = pos.sum(1)
    valid = pos_count > 0
    if not valid.any():
        return torch.zeros((), device=z.device, requires_grad=True)
    loss = -(pos * log_prob).sum(1)[valid] / pos_count[valid]
    return loss.mean()


def length_window_batches(lengths, batch_size, window_size, rng):
    """Batches drawn from narrow length windows -> within-batch length is homogeneous."""
    order = list(np.argsort(lengths))
    batches = []
    for i in range(0, len(order), window_size):
        w = order[i:i + window_size]
        rng.shuffle(w)
        for j in range(0, len(w), batch_size):
            b = w[j:j + batch_size]
            if len(b) >= 4:
                batches.append(b)
    rng.shuffle(batches)
    return batches


def _probe(ztr, ytr, zte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    clf = LogisticRegression(max_iter=2000).fit(ztr, ytr)
    p = clf.predict(zte)
    return accuracy_score(yte, p), f1_score(yte, p, pos_label="B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_pb")
    ap.add_argument("--checkpoint_dir", default="checkpoints_pb_sup")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window_size", type=int, default=192, help="length-window width for batching")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda_recon", type=float, default=0.0, help="aux reconstruction weight (0=off)")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    cands = load_candidates(Path(args.data_dir) / "candidates.jsonl")
    cands = [c for c in cands if c.candidate_type in ("A", "B") and c.include_in_training and c.full_text]
    # split by problem group (no leakage)
    groups = sorted({c.contrastive_group for c in cands})
    cut = int(len(groups) * 0.8)
    train_ids = set(groups[:cut])
    tr = [c for c in cands if c.contrastive_group in train_ids]
    va = [c for c in cands if c.contrastive_group not in train_ids]
    print(f"[sup] train={len(tr)} (A={sum(c.candidate_type=='A' for c in tr)}, "
          f"B={sum(c.candidate_type=='B' for c in tr)})  val={len(va)}")

    tr_text = [c.full_text for c in tr]
    tr_lab = np.array([0 if c.candidate_type == "A" else 1 for c in tr])
    tr_len = np.array([len((c.response_text or c.full_text).split()) for c in tr])
    va_text = [c.full_text for c in va]
    va_lab = np.array([c.candidate_type for c in va])

    model = RiDAE(encoder_name=args.encoder, device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"[sup] device={model.device}  objective=SupCon(len-matched)  lambda_recon={args.lambda_recon}")

    best_f1, best_ep, no_imp = -1.0, -1, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        batches = length_window_batches(tr_len, args.batch_size, args.window_size, rng)
        running = 0.0
        for bi, idx in enumerate(batches):
            texts = [tr_text[i] for i in idx]
            labels = torch.tensor(tr_lab[idx], device=model.device)
            z = model.bottleneck(model._encode_with_grad(texts))
            loss = supcon_loss(z, labels, args.tau)
            if args.lambda_recon > 0:
                rec, _ = model.reconstruction_loss(texts, texts)
                loss = loss + args.lambda_recon * rec
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss)
        # val: A/B probe on z (the thing we actually want)
        model.eval()
        ztr = model.encode(tr_text); zva = model.encode(va_text)
        acc, f1 = _probe(ztr, tr_lab_str(tr_lab), zva, va_lab)
        print(f"[sup] epoch {ep}: supcon={running/max(len(batches),1):.4f}  "
              f"val A/B probe acc={acc:.3f} f1_B={f1:.3f}")
        if f1 > best_f1 + 1e-4:
            best_f1, best_ep, no_imp = f1, ep, 0
            model.save(ckpt_dir / "ridae_best.pt"); print("        ^ improved — saved")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"[sup] early stop at epoch {ep}"); break

    print("=" * 60)
    print(f"[sup] done in {(time.time()-t0)/60:.1f} min | best val f1_B={best_f1:.3f} @epoch {best_ep}")
    print(f"[sup] checkpoint: {ckpt_dir/'ridae_best.pt'}")
    print(f"[sup] NEXT: python main/diagnose_ceiling.py --data_dir {args.data_dir} "
          f"--checkpoint {ckpt_dir/'ridae_best.pt'} --device {args.device}")
    print("[sup] SUCCESS if z_residualized f1_B clearly beats 0.286; else -> PRM/step-structured.")


def tr_lab_str(arr):
    return np.array(["A" if v == 0 else "B" for v in arr])


if __name__ == "__main__":
    main()
