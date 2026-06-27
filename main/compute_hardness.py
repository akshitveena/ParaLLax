"""
compute_hardness.py — populate Block-8 `hardness_score` after a training run.

For every contrastive pair, encode the Type A and Type B candidate with the
trained RiDAE and measure how *close* the Type B sits to its paired Type A in
z-space. Closeness in [0,1] (cosine mapped to [0,1]) is written back onto the
Type B candidate's `hardness_score` field in candidates.jsonl.

    ~1 -> encoder barely separates them (true hard negative) -> oversample
    ~0 -> already separated                                  -> undersample

Run AFTER main/train.py and BEFORE re-running main/train.py:
    python main/compute_hardness.py --checkpoint checkpoints/ridae_best.pt
The next train.py run will then activate hard-negative mining automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_pipeline import load_candidates, load_contrastive_pairs, save_candidates
from ridae import RiDAE


def _cos01(u: np.ndarray, v: np.ndarray) -> float:
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) or 1e-9
    cos = float(np.dot(u, v) / denom)
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))   # [-1,1] -> [0,1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ridae_best.pt")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    candidates = load_candidates(data_dir / "candidates.jsonl")
    pairs = load_contrastive_pairs(data_dir / "contrastive_pairs.json")
    if not pairs:
        print("[hardness] no contrastive pairs — nothing to score.")
        return

    model = RiDAE.load(args.checkpoint, device=args.device)

    # Encode each pair's A and B, compute hardness per group.
    a_texts = [p.type_a.full_text for p in pairs]
    b_texts = [p.type_b.full_text for p in pairs]
    z_a = model.encode(a_texts)
    z_b = model.encode(b_texts)
    hardness_by_group = {p.contrastive_group: _cos01(z_b[i], z_a[i])
                         for i, p in enumerate(pairs)}

    # Write hardness onto the Type B candidates of those groups.
    updated = 0
    for c in candidates:
        if c.candidate_type == "B" and c.contrastive_group in hardness_by_group:
            c.hardness_score = round(hardness_by_group[c.contrastive_group], 4)
            updated += 1
        # (Type A / others keep hardness_score = None.)

    save_candidates(candidates, data_dir / "candidates.jsonl")
    vals = np.array(list(hardness_by_group.values()))
    print(f"[hardness] scored {updated} Type-B candidates across {len(pairs)} pairs")
    print(f"[hardness] hardness: mean={vals.mean():.3f} min={vals.min():.3f} "
          f"max={vals.max():.3f}  (high = hard negative = oversampled next run)")
    print(f"[hardness] wrote {data_dir/'candidates.jsonl'} — re-run main/train.py to use it.")


if __name__ == "__main__":
    main()
