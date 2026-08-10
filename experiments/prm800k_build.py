"""
prm800k_build.py — E3: build the answer-correct-with-error slice from PRM800K, as a second corpus.

Kills "one benchmark" and partially kills the judge objection (PRM800K phase-2 has HUMAN step
labels). Output matches step_cache.pt's schema exactly, so difficulty_baseline.py, train_sdae.py
and prm_external.py all run on it unchanged.

PRM800K is stepwise-critique data, NOT linear solutions — a reviewer who knows it will check the
mapping, so it is made explicit here and must be validated against the real data before trusting:

  A "solution" = the path the labeler actually followed = the chosen_completion at each step
                 (human_completion when chosen_completion is null).
  steps_text   = [text of the followed completion at each step]
  step_labels  = [1 if that completion's rating == -1 (human-labelled ERROR) else 0]
                 (1=error matches ProcessBench's convention used throughout this project)
  Type B (chain='B') = final answer correct AND >=1 step labelled error (rating -1)
  Type A (chain='A') = final answer correct AND no error
  Solutions that never reach a correct final answer are EXCLUDED (we study wrong-approach-
  RIGHT-answer, same inclusion rule as the ProcessBench slice).

Because the exact HF id / field names drift, run --inspect FIRST and confirm the structure, then
--build. --inspect makes no assumptions; --build asserts them and fails loudly on mismatch.

    # on the box (has internet):
    python experiments/prm800k_build.py --inspect --source <hf_id_or_local.jsonl>
    python experiments/prm800k_build.py --build   --source <hf_id_or_local.jsonl> \
        --out data/step_cache_prm800k.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_rows(source, limit=0):
    """Accept a local .jsonl path OR an HF dataset id. Yields raw dict rows."""
    p = Path(source)
    if p.exists() and p.suffix in (".jsonl", ".json"):
        for i, line in enumerate(p.read_text().splitlines()):
            if line.strip():
                yield json.loads(line)
            if limit and i + 1 >= limit:
                return
    else:
        from datasets import load_dataset
        ds = load_dataset(source, split="train")
        for i, r in enumerate(ds):
            yield r
            if limit and i + 1 >= limit:
                return


def inspect(source):
    rows = list(load_rows(source, limit=3))
    print(f"[E3] loaded {len(rows)} sample rows from {source}\n")
    for i, r in enumerate(rows):
        print(f"--- row {i} top-level keys: {list(r.keys())}")
        q = r.get("question", r)
        if isinstance(q, dict):
            print(f"    question keys: {list(q.keys())}")
        lab = r.get("label", {})
        if isinstance(lab, dict):
            print(f"    label keys: {list(lab.keys())}  finish_reason={lab.get('finish_reason')}")
            steps = lab.get("steps", [])
            print(f"    n steps: {len(steps)}")
            if steps:
                s0 = steps[0]
                print(f"    step[0] keys: {list(s0.keys())}")
                comps = s0.get("completions") or []
                if comps:
                    print(f"    step[0].completions[0] keys: {list(comps[0].keys())}")
                    print(f"    step[0].completions[0] rating: {comps[0].get('rating')}")
                print(f"    step[0].chosen_completion: {s0.get('chosen_completion')}")
        print()
    print("Confirm: chosen_completion index selects the followed step; rating -1 = error;")
    print("finish_reason=='solution' means a final answer was reached. If any differ, tell me.")


def followed_step(step):
    """Return (text, rating) for the completion actually followed at this step, or None."""
    ci = step.get("chosen_completion")
    comps = step.get("completions") or []
    if ci is not None and 0 <= ci < len(comps):
        c = comps[ci]; return c.get("text", ""), c.get("rating")
    hc = step.get("human_completion")
    if hc:                                      # human-written continuation = correct by construction
        txt = hc.get("text", "") if isinstance(hc, dict) else str(hc)
        return txt, 1
    return None


def build(source, out, data_dir):
    rows = list(load_rows(source))
    print(f"[E3] {len(rows)} raw rows", flush=True)
    recs, n_excl_noanswer, n_excl_short = [], 0, 0
    ca = cb = 0
    for idx, r in enumerate(rows):
        q = r.get("question", {})
        gold = str(q.get("ground_truth_answer", "")).strip()
        lab = r.get("label", {}) or {}
        if lab.get("finish_reason") != "solution":     # must reach a final answer
            n_excl_noanswer += 1; continue
        steps_text, step_labels = [], []
        for st in lab.get("steps", []):
            fs = followed_step(st)
            if fs is None:
                continue
            txt, rating = fs
            steps_text.append(txt)
            step_labels.append(1 if rating == -1 else 0)   # 1 = human-labelled error
        if len(steps_text) < 2:
            n_excl_short += 1; continue
        chain = "B" if any(step_labels) else "A"
        ca += chain == "A"; cb += chain == "B"
        recs.append({"id": f"prm800k-{idx}", "split": "prm800k", "chain": chain,
                     "steps_text": steps_text,
                     "step_labels": np.array(step_labels, dtype=np.int64)})

    print(f"[E3] built {len(recs)}  (A={ca} B={cb})  "
          f"excluded: no-final-answer={n_excl_noanswer}, <2 steps={n_excl_short}")
    if not recs:
        print("[E3] ZERO records — the schema assumptions are wrong. Run --inspect and send output.")
        sys.exit(1)

    # embed step text with the SAME encoder as ProcessBench (MiniLM) for a like-for-like corpus
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
    # also emit a candidates.jsonl-shaped file so build_confounds works (needs response text)
    cand_dir = Path(data_dir); cand_dir.mkdir(parents=True, exist_ok=True)
    with (cand_dir / "candidates.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps({"record_id": r["id"],
                                 "response_text": "\n".join(r["steps_text"]),
                                 "num_steps": len(r["steps_text"])}) + "\n")
    print(f"[E3] wrote {out} and {cand_dir/'candidates.jsonl'} ({len(recs)} records)")
    print(f"[E3] NEXT: difficulty_baseline.py --cache {out} --data_dir {cand_dir}  (does pooled->floor")
    print(f"          collapse REPLICATE on PRM800K?); then train_sdae.py / prm_external.py on it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="HF dataset id OR local phase2 .jsonl path")
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
        print("pass --inspect (do this first) or --build")


if __name__ == "__main__":
    main()
