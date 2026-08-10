"""
prm800k_build.py — E3: build the answer-correct-with-error slice from PRM800K (Birchlabs flattened
stepwise-critic mirror), as a second corpus. Output matches step_cache.pt exactly, so
difficulty_baseline.py / train_sdae.py / prm_external.py all run on it unchanged.

SCHEMA (Birchlabs/openai-prm800k-stepwise-critic): one row per candidate next-step —
  instruction, responses(prior steps taken), next_response(this candidate step),
  rating ∈ {-1,0,1}, is_preferred_response, is_solution, is_human_response, answer.

EXPLICIT LABEL MAPPING (a reviewer who knows PRM800K will check this):
  * Reconstruct the FOLLOWED solution per problem = the preferred-response chain
    (is_preferred_response==True), ordered by len(responses); next_response is the step taken.
  * steps_text  = [next_response along the preferred chain]
  * step_labels = [1 if rating == -1 (human-labelled ERROR) else 0]   (1=error, matches ProcessBench)
  * Include only chains that REACH a solution (is_solution==True on the terminal step) and have
    >=2 steps. Type B = chain reaches a solution AND contains a labelled error; Type A = no error.
  * CAVEAT (documented, not hidden): PRM800K does not expose a clean final-answer-correctness flag
    here (answer is usually None), so "reached is_solution on the human-preferred path" is our
    answer-correct proxy — the labeler was building a correct solution. This is the one mapping
    assumption; stated in the appendix.

    python experiments/prm800k_build.py --inspect --source Birchlabs/openai-prm800k-stepwise-critic
    python experiments/prm800k_build.py --build   --source Birchlabs/openai-prm800k-stepwise-critic
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_dataset_rows(source, split="train"):
    p = Path(source)
    if p.exists() and p.suffix in (".jsonl", ".json"):
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    from datasets import load_dataset
    return list(load_dataset(source, split=split))


def reconstruct(rows):
    """Group flattened rows by problem, rebuild each preferred solution chain.

    Returns list of (problem, steps_text, step_labels, reached_solution)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["instruction"]].append(r)
    out = []
    for problem, rs in groups.items():
        pref = [r for r in rs if r.get("is_preferred_response")]
        if not pref:
            continue
        pref.sort(key=lambda r: len(r.get("responses") or []))
        seen_depth, chain = set(), []
        for r in pref:                                   # one step per depth, in order
            d = len(r.get("responses") or [])
            if d in seen_depth:
                continue
            seen_depth.add(d); chain.append(r)
        steps_text = [r["next_response"] for r in chain]
        step_labels = [1 if r.get("rating") == -1 else 0 for r in chain]
        reached = any(r.get("is_solution") for r in chain)
        out.append((problem, steps_text, step_labels, reached))
    return out


def inspect(source):
    rows = load_dataset_rows(source)
    print(f"[E3] {len(rows)} flattened rows")
    chains = reconstruct(rows)
    reached = [c for c in chains if c[3] and len(c[1]) >= 2]
    A = sum(1 for c in reached if not any(c[2]))
    B = sum(1 for c in reached if any(c[2]))
    print(f"[E3] {len(chains)} problems | {len(reached)} solution-reaching chains (>=2 steps)")
    print(f"[E3] Type A (no error) = {A}   Type B (has error) = {B}   "
          f"(B rate {B/max(A+B,1):.3f})")
    steplens = [len(c[1]) for c in reached]
    print(f"[E3] steps/chain: min {min(steplens)} med {int(np.median(steplens))} max {max(steplens)}")
    ex = next((c for c in reached if any(c[2])), None)
    if ex:
        print("\n[E3] example Type-B chain (error steps marked *):")
        for t, l in zip(ex[1], ex[2]):
            print(f"   {'*' if l else ' '} {t[:90]}")
    print("\nIf Type-B count is usable (>~150), run --build. If tiny, tell me and we adjust the")
    print("inclusion rule (e.g. include self-corrected non-preferred error steps).")


def build(source, out, data_dir):
    rows = load_dataset_rows(source)
    chains = [c for c in reconstruct(rows) if c[3] and len(c[1]) >= 2]
    recs = []
    for i, (problem, steps_text, step_labels, _) in enumerate(chains):
        recs.append({"id": f"prm800k-{i}", "split": "prm800k",
                     "chain": "B" if any(step_labels) else "A",
                     "steps_text": steps_text,
                     "step_labels": np.array(step_labels, dtype=np.int64)})
    A = sum(r["chain"] == "A" for r in recs); B = sum(r["chain"] == "B" for r in recs)
    print(f"[E3] built {len(recs)} chains  (A={A} B={B}, B rate {B/max(len(recs),1):.3f})")
    if B < 50:
        print(f"[E3] WARNING: only {B} Type-B — too few for a stable ablation. Consider adjusting "
              f"the inclusion rule before trusting downstream numbers.")

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    flat, offs = [], []
    for r in recs:
        offs.append((len(flat), len(flat) + len(r["steps_text"]))); flat.extend(r["steps_text"])
    emb = st.encode(flat, batch_size=256, convert_to_numpy=True, show_progress_bar=True)
    for r, (a, b) in zip(recs, offs):
        r["steps_emb"] = emb[a:b].astype("float32")

    import torch
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(recs, out)
    cand = Path(data_dir); cand.mkdir(parents=True, exist_ok=True)
    with (cand / "candidates.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps({"record_id": r["id"],
                                 "response_text": "\n".join(r["steps_text"]),
                                 "num_steps": len(r["steps_text"])}) + "\n")
    print(f"[E3] wrote {out} and {cand/'candidates.jsonl'}")
    print(f"[E3] NEXT: python experiments/difficulty_baseline.py --cache {out} "
          f"--data_dir {cand}  (does pooled->floor collapse REPLICATE on PRM800K?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Birchlabs/openai-prm800k-stepwise-critic")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/step_cache_prm800k.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_prm800k"))
    args = ap.parse_args()
    if args.inspect:
        inspect(args.source)
    elif args.build:
        build(args.source, args.out, args.data_dir)
    else:
        print("pass --inspect (first) or --build")


if __name__ == "__main__":
    main()
