"""
encoder_ladder.py — E2: does the pooled→floor collapse and the step-structure lift survive as
the encoder scales up? Kills the "22M-on-a-laptop" objection.

Core ablation (pooled vs step-structured, 5 seeds, confound-controlled) at three encoder scales:
  22M   sentence-transformers/all-MiniLM-L6-v2   (384-d) — the paper's encoder
  110M  sentence-transformers/all-mpnet-base-v2  (768-d)
  335M  thenlper/gte-large                       (1024-d)

Both outcomes are publishable and MUST be reported (pre-commitment):
  * pooled stays at the floor at every scale  -> the confound critique is scale-invariant.
  * pooled climbs off the floor at scale      -> the ceiling was a small-encoder artifact; say so.
    That weakens one claim and strengthens the paper's credibility.

Re-embeds step TEXT (already in step_cache) with each encoder, caches per encoder, then:
  pooled          = mean over step embeddings -> confound-controlled probe
  step-structured = frozen StepSDAE_PRM(in_dim=D) trained 5 seeds -> attn-pooled -> probe
Same confound protocol / split arithmetic as every other phase.

    python experiments/encoder_ladder.py --encoder sentence-transformers/all-mpnet-base-v2 \
        --seeds 0,1,2,3,4 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))
from sdae_prm import StepSDAE_PRM, losses
from train_sdae import collate, make_corrupt
from multiseed_ablation import build_confounds
from phase1d_validity_probe import probe_full


def embed_steps(encoder, recs, device, cache_path, bs=256):
    """Per-record step embeddings from step TEXT, cached per encoder (re-embedding is the cost)."""
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(encoder, device=device)
    flat, offs = [], []
    for r in recs:
        steps = r["steps_text"]
        offs.append((len(flat), len(flat) + len(steps))); flat.extend(steps)
    emb = st.encode(flat, batch_size=bs, convert_to_numpy=True, show_progress_bar=True,
                    normalize_embeddings=False)
    out = []
    for r, (a, b) in zip(recs, offs):
        out.append({**{k: r[k] for k in ("id", "split", "chain", "step_labels")},
                    "steps_emb": emb[a:b].astype("float32")})
    torch.save(out, cache_path)
    return out


class DS(Dataset):
    def __init__(self, recs): self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return (torch.from_numpy(r["steps_emb"]).float(),
                torch.from_numpy(np.asarray(r["step_labels"])).long(),
                0 if r["chain"] == "A" else 1)


def split(n, seed):
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    cut = int(0.8 * n); return idx[:cut], idx[cut:]


def train_frozen_sdae(tr, in_dim, seed, dev, epochs=30, patience=6, frac=0.25):
    """Frozen step-SDAE (denoise+PRM+chain), select on val chain_f1 — matches Phase 1a."""
    from sklearn.metrics import f1_score
    torch.manual_seed(seed)
    model = StepSDAE_PRM(in_dim=in_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    tri, vai = split(len(tr), seed)
    trd = [tr[i] for i in tri]; vad = [tr[i] for i in vai]
    trl = DataLoader(DS(trd), batch_size=32, shuffle=True, collate_fn=collate,
                     generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(DS(vad), batch_size=64, collate_fn=collate)
    best, best_state, no_imp = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for X, SL, pad, ch in trl:
            X, SL, pad, ch = X.to(dev), SL.to(dev), pad.to(dev), ch.to(dev)
            cm = make_corrupt(pad.cpu(), frac, gen).to(dev)
            recon, prm, cl, _ = model(X, pad, cm)
            tot, *_ = losses(recon, X, prm, SL, cl, ch, pad, cm)
            opt.zero_grad(); tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); ct, cp = [], []
        with torch.no_grad():
            for X, SL, pad, ch in vl:
                _, _, cl, _ = model(X.to(dev), pad.to(dev), None)
                cp += (torch.sigmoid(cl).cpu().numpy() > 0.5).astype(int).tolist()
                ct += ch.numpy().tolist()
        f1 = f1_score(ct, cp, pos_label=1) if len(set(ct)) > 1 else 0.0
        if f1 > best + 1e-4:
            best, no_imp = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def sdae_pooled(model, recs, dev, bs=64):
    Z = []
    with torch.no_grad():
        for X, SL, pad, ch in DataLoader(DS(recs), batch_size=bs, collate_fn=collate, shuffle=False):
            _, _, _, pl = model(X.to(dev), pad.to(dev), None)
            Z.append(pl.cpu().numpy())
    return np.concatenate(Z, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=str(HERE / "results_ladder"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out); (out / "emb").mkdir(parents=True, exist_ok=True)
    base = torch.load(args.cache, weights_only=False)
    y = np.array([r["chain"] for r in base]); C = build_confounds(base, args.data_dir)
    tag = args.encoder.split("/")[-1]

    recs = embed_steps(args.encoder, base, args.device, out / "emb" / f"{tag}.pt")
    D = recs[0]["steps_emb"].shape[1]
    print(f"[E2] {tag}: dim={D}, embedded {len(recs)} solutions", flush=True)

    pooled = np.array([r["steps_emb"].mean(0) for r in recs])
    pooled_rows, step_rows = [], []
    for s in seeds:
        pf, pa, pc = probe_full(pooled, C, y, s)
        pooled_rows.append(pf)
        m = train_frozen_sdae(recs, D, s, args.device)
        sf, sa, sc = probe_full(sdae_pooled(m, recs, args.device), C, y, s)
        step_rows.append(sf)
        print(f"  seed {s}: pooled ctl f1_B={pf:.3f}   step-SDAE ctl f1_B={sf:.3f}", flush=True)

    pm, ps = np.mean(pooled_rows), np.std(pooled_rows)
    sm, ss = np.mean(step_rows), np.std(step_rows)
    with (out / f"ladder_{tag}.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["encoder", "dim", "kind", "seed", "ctl_f1B"])
        for s, v in zip(seeds, pooled_rows): w.writerow([tag, D, "pooled", s, round(v, 4)])
        for s, v in zip(seeds, step_rows): w.writerow([tag, D, "step", s, round(v, 4)])

    print("\n" + "=" * 60)
    print(f"  E2 — ENCODER LADDER  {tag}  (dim {D})  confound-controlled f1_B")
    print("=" * 60)
    print(f"  pooled     : {pm:.3f} ± {ps:.3f}")
    print(f"  step-SDAE  : {sm:.3f} ± {ss:.3f}   (lift {sm-pm:+.3f})")
    print(f"  ({(time.time()-t0)/60:.1f} min on {args.device}; wrote ladder_{tag}.csv)")
    print("  Compare pooled across scales: flat at floor => confound critique is scale-invariant.")


if __name__ == "__main__":
    main()
