"""
dose_response.py — the "toxicology" of reasoning corruption.

For a TRAINED RiDAE, sweep each corruption operator across severity DOSES and measure
reconstruction error (1 - cos between decode(z of corrupted) and encode(original)) on
held-out candidates. Each curve's knee is that damage type's BREAKDOWN POINT: the
maximum dose at which the original is still recoverable. This answers, empirically:
HOW MUCH and WHAT KIND of damage can a reasoning chain sustain and still be reconstructed.

Reading it:
  * low, flat curve            -> the model shrugs off this damage (robust; weak signal)
  * sharp knee at dose d*      -> d* is the usable corruption dose for that operator
  * high even at small dose    -> this damage destroys identifiability (don't train on it)

Run AFTER train.py:
    python main/dose_response.py --data_dir data/processed_pb \
        --checkpoint checkpoints_pb/ridae_best.pt --output_dir outputs_pb --device cpu
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch

from data_pipeline import load_candidates
from corruption import DOSE_OPERATORS
from ridae import RiDAE


def _recon_err(model: RiDAE, corrupted: list[str], originals: list[str], batch: int = 32) -> float:
    """Mean reconstruction error (1 - cos) over the set, batched to bound memory."""
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(corrupted), batch):
            c = corrupted[i:i + batch]; o = originals[i:i + batch]
            loss, _ = model.reconstruction_loss(c, o)
            total += float(loss) * len(c); n += len(c)
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_pb")
    ap.add_argument("--checkpoint", default="checkpoints_pb/ridae_best.pt")
    ap.add_argument("--output_dir", default="outputs_pb")
    ap.add_argument("--doses", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--n", type=int, default=200, help="held-out candidates sampled for the sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    doses = [float(x) for x in args.doses.split(",")]
    cands = load_candidates(Path(args.data_dir) / "candidates.jsonl")
    cands = [c for c in cands if c.include_in_training and c.full_text]
    rng = random.Random(args.seed); rng.shuffle(cands)
    cands = cands[:args.n]
    originals = [c.full_text for c in cands]
    print(f"[dose] {len(cands)} held-out candidates | operators={list(DOSE_OPERATORS)} | doses={doses}")

    model = RiDAE.load(args.checkpoint, device=args.device)
    model.eval()

    rows = []
    for op_name, op in DOSE_OPERATORS.items():
        knee_prev = None
        for s in doses:
            r = random.Random(1234)                         # fixed so doses are comparable
            corrupted = [op(c.full_text, r, s) for c in cands]
            err = _recon_err(model, corrupted, originals)
            rows.append((op_name, s, err))
            print(f"    {op_name:16} dose={s:.1f}  recon_err={err:.4f}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "dose_response.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["operator", "dose", "recon_err"]); w.writerows(rows)
    print(f"[dose] wrote {out/'dose_response.csv'}")

    # breakdown point = smallest dose whose error exceeds halfway between the operator's
    # min and max error (a simple, honest knee estimate).
    print("\n[dose] BREAKDOWN POINTS (first dose past the half-rise):")
    for op_name in DOSE_OPERATORS:
        pts = [(s, e) for (o, s, e) in rows if o == op_name]
        errs = [e for _, e in pts]
        lo, hi = min(errs), max(errs)
        thresh = lo + 0.5 * (hi - lo)
        knee = next((s for s, e in pts if e >= thresh), None)
        print(f"    {op_name:16} breakdown≈{knee}   (err {lo:.3f} -> {hi:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for op_name in DOSE_OPERATORS:
            xs = [s for (o, s, e) in rows if o == op_name]
            ys = [e for (o, s, e) in rows if o == op_name]
            ax.plot(xs, ys, marker="o", label=op_name)
        ax.set_xlabel("dose (severity)"); ax.set_ylabel("reconstruction error (1 − cos)")
        ax.set_title("Reasoning corruption dose–response")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(out / "dose_response.png", dpi=140)
        print(f"[dose] wrote {out/'dose_response.png'}")
    except Exception as e:
        print(f"[dose] plot skipped: {e}")


if __name__ == "__main__":
    main()
