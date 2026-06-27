"""
baseline_umap.py — the control experiment. RUN THIS FIRST, before any training.

Encodes every candidate with the PRE-TRAINED sentence-transformer (no bottleneck,
no fine-tuning), then UMAPs and linear-probes the raw 384-d embeddings. This tells
you what signal the generic encoder already carries, so you know exactly what
RiDAE's reconstruction objective contributes on top.

    If A and B already separate: signal is in the surface text.
    If A and B overlap:          signal is structural — RiDAE's whole reason to exist.
Both outcomes are scientifically valid. Report baseline + trained together.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Make main/ importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
from data_pipeline import load_candidates           # noqa: E402
from analyse import run_umap, plot_umap, linear_probe, log_ablation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    candidates = load_candidates(Path(args.data_dir) / "candidates.jsonl")
    print(f"[baseline] {len(candidates)} candidates")

    st = SentenceTransformer(args.encoder)
    texts = [c.full_text for c in candidates]
    labels = np.array([c.candidate_type for c in candidates])
    emb = st.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    emb2d = run_umap(emb)
    plot_umap(emb2d, labels, "Baseline encoder z-space (no fine-tuning)",
              out_dir / "baseline_umap.png")

    lp = linear_probe(emb, labels)
    if "error" in lp:
        print(f"[baseline] probe skipped: {lp}")
    else:
        print(f"BASELINE — Linear probe accuracy: {100*lp['accuracy']:.1f}%. "
              f"F1 Type B: {100*lp['f1_B']:.1f}%")
        log_ablation(out_dir, "baseline", lp["accuracy"], lp.get("f1_B"))  # Fig 8 anchor
    print("Run this again after training (main/analyse.py) to see RiDAE improvement.")


if __name__ == "__main__":
    main()
