"""
phase2d_judge_baseline.py — Phase 2d: the zero-shot LLM judge as an EXTERNAL baseline.

The comparison table currently holds only internal ablations. This adds the already-validated
mechanism judge (kappa=0.60 vs ProcessBench humans) as a real external baseline row, scored on
the SAME held-out split, under the same reporting discipline.

Leakage: the judge is zero-shot and never sees train labels, so there is nothing to fit — we run
it on the HELD-OUT split only (~340 records), never the train split.

Confound control on a HARD LABEL: residualization needs a continuous score, and the judge emits a
categorical mechanism. Per the spec we therefore report RAW f1_B and mark the control not
applicable — rather than inventing a pseudo-confidence. In its place we run an honest secondary
diagnostic: predict judge CORRECTNESS from the four confounds. If length/latex/#steps/dataset
predict when the judge is wrong, its errors are confound-structured, which is the informative
part of the control we cannot run directly.

    python experiments/phase2d_judge_baseline.py submit  --seed 0     # costs money, asks first
    python experiments/phase2d_judge_baseline.py analyze --seed 0     # free, from cached labels
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "api_generation"))
import mechanism_judge as MJ                     # JUDGE_SYSTEM, build_prompt, parse_label, MODEL

B_MECHS = {"flawed_lucky", "unfaithful", "spurious"}      # -> B ; sound_* -> A
# Batch API Sonnet: $1.50/M in, $7.50/M out at 50% off. ~2.5k in + 2.5k out per judged chain.
COST_PER_CALL = (2500 * 1.50 + 2500 * 7.50) / 1e6


def heldout(seed, cache):
    """The seed's held-out 20% — identical split arithmetic to every other phase."""
    recs = torch.load(cache, weights_only=False)
    rng = np.random.RandomState(seed); idx = np.arange(len(recs)); rng.shuffle(idx)
    return recs, [recs[i] for i in idx[int(0.8 * len(recs)):]]


def load_meta(data_dir):
    out = {}
    for l in Path(data_dir, "candidates.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            out[r["record_id"]] = (r.get("problem", ""), r.get("gold_solution", "") or "",
                                   r.get("response_text") or r.get("full_text") or "")
    return out


def cmd_submit(args):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY (it lives in .env — never commit or print it)."); sys.exit(1)
    _, va = heldout(args.seed, args.cache)
    meta = load_meta(args.data_dir)

    reqs, index = [], {}
    for r in va:
        problem, gold, resp = meta.get(r["id"], ("", "", ""))
        if not resp:
            continue
        cid = f"j-{r['id']}".replace("/", "-")[:64]
        index[cid] = r["id"]
        reqs.append({"custom_id": cid, "params": {
            "model": MJ.MODEL, "max_tokens": args.max_tokens, "system": MJ.JUDGE_SYSTEM,
            "messages": [{"role": "user",
                          "content": MJ.build_prompt(problem, gold, resp)}]}})

    est = len(reqs) * COST_PER_CALL
    print(f"[2d] held-out records to judge: {len(reqs)} (seed {args.seed})")
    print(f"[2d] estimated Batch API cost: ${est:.2f}  (model {MJ.MODEL})")
    if not args.yes and input("[2d] proceed? [y/N] ").strip().lower() != "y":
        print("[2d] aborted — nothing submitted, nothing spent."); return

    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=reqs)
    print(f"[2d] batch {batch.id} submitted — polling every {args.poll}s", flush=True)
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(args.poll)

    rows = []
    for res in client.messages.batches.results(batch.id):
        rid = index[res.custom_id]; mech = None
        if res.result.type == "succeeded":
            txt = "".join(bl.text for bl in res.result.message.content
                          if getattr(bl, "type", None) == "text")
            mech = MJ.parse_label(txt)
        rows.append({"id": rid, "mechanism": mech})
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"seed": args.seed, "model": MJ.MODEL, "rows": rows}, indent=2))
    print(f"[2d] wrote {len(rows)} judgements -> {p}")


def cmd_analyze(args):
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from multiseed_ablation import build_confounds

    blob = json.loads(Path(args.labels).read_text())
    pred_of = {r["id"]: r["mechanism"] for r in blob["rows"]}
    recs, va = heldout(blob["seed"], args.cache)

    keep = [r for r in va if pred_of.get(r["id"])]
    y = np.array([r["chain"] for r in keep])
    pred = np.array(["B" if pred_of[r["id"]] in B_MECHS else "A" for r in keep])
    n_unparsed = len(va) - len(keep)

    f1 = f1_score(y, pred, pos_label="B"); acc = accuracy_score(y, pred)
    print("=" * 72)
    print("  PHASE 2d — ZERO-SHOT LLM JUDGE (external baseline, held-out only)")
    print("=" * 72)
    print(f"  model {blob['model']} | seed {blob['seed']} | n={len(keep)}"
          f"{f' ({n_unparsed} unparsed, excluded)' if n_unparsed else ''}")
    print(f"  raw f1_B = {f1:.3f}   accuracy = {acc:.3f}")
    print("  confound-controlled: N/A — the judge emits a hard categorical label, and")
    print("  residualization requires a continuous score. Reported raw, per spec.")

    # honest stand-in: are the judge's ERRORS predictable from the confounds?
    C = build_confounds(keep, args.data_dir)
    correct = (pred == y).astype(int)
    if len(set(correct)) > 1:
        auc = cross_val_score(LogisticRegression(max_iter=2000), C, correct,
                              cv=5, scoring="roc_auc").mean()
        print(f"\n  secondary diagnostic — predicting judge CORRECTNESS from the four confounds:")
        print(f"    5-fold AUC = {auc:.3f} "
              f"({'errors are confound-structured' if auc > 0.60 else 'errors look confound-independent'})")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.set_defaults(fn=cmd_submit)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--max_tokens", type=int, default=2500)
    s.add_argument("--poll", type=int, default=30); s.add_argument("--yes", action="store_true")
    s.add_argument("--out", default=str(HERE / "results_judge/judge_heldout.json"))
    a = sub.add_parser("analyze"); a.set_defaults(fn=cmd_analyze)
    a.add_argument("--labels", default=str(HERE / "results_judge/judge_heldout.json"))
    for p in (s, a):
        p.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
        p.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
