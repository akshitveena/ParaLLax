"""
prm_external.py — Phase 2c: an established open PRM under OUR confound protocol.

Purpose: turn "our small model inflates under naive evaluation" into evidence about how the
FIELD evaluates verifiers. Inference only, one model, no sweep.

Model: peiyi9979/math-shepherd-mistral-7b-prm. We deliberately do NOT fall back to
Qwen/Qwen2.5-Math-PRM-7B: ProcessBench is Qwen's own benchmark, so a Qwen math-PRM plausibly
saw ProcessBench-adjacent data. That contamination would inflate its RAW score for reasons
unrelated to confounds — corrupting exactly the raw-vs-controlled contrast this experiment
exists to measure. Math-Shepherd is a different lab and predates ProcessBench.

Two modes, on purpose:
  score    (GPU) forward-pass all 1,700 solutions once, dump per-step scores to JSON.
  analyze  (CPU) all statistics from that JSON — so the protocol can be re-derived, audited and
           re-run locally for free without touching the GPU again.

Scoring format is Math-Shepherd's documented one: steps are terminated with the step tag 'ки',
and the step score is softmax over the '+'/'-' candidate tokens at each tag position.

SANITY GATE (analyze): per-step scores are checked against ProcessBench's HUMAN step labels.
If step-level AUC is at chance the scoring format is wrong and every downstream number is
garbage — the gate fails loudly instead of silently reporting a confident wrong result.

Aggregation to a chain-level Type-B score: 1 - min(step_scores) (standard weakest-step),
with 1 - mean(step_scores) logged as a secondary readout.

    python experiments/prm_external.py score   --out results_prm/scores.json   # GPU
    python experiments/prm_external.py analyze --scores results_prm/scores.json # CPU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))

MODEL = "peiyi9979/math-shepherd-mistral-7b-prm"
GOOD, BAD, STEP_TAG = "+", "-", "ки"


def cmd_score(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time()
    recs = torch.load(args.cache, weights_only=False)
    probs = {json.loads(l)["record_id"]: json.loads(l)["problem"]
             for l in Path(args.data_dir, "candidates.jsonl").read_text().splitlines() if l.strip()}

    tok = AutoTokenizer.from_pretrained(args.model)
    cand_ids = tok.encode(f"{GOOD} {BAD}")[1:]              # [good_id, bad_id]
    tag_id = tok.encode(f"{STEP_TAG}")[-1]
    print(f"[2c] candidate token ids={cand_ids}  step_tag id={tag_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        device_map=args.device).eval()

    recs = recs[:args.limit] if args.limit else recs
    out, skipped = [], 0
    for i, r in enumerate(recs):
        steps = r["steps_text"]
        body = "".join(f" {s} {STEP_TAG}\n" for s in steps)
        text = f"{probs.get(r['id'], '')}{body}"
        ids = tok.encode(text, return_tensors="pt", truncation=True,
                         max_length=args.max_len).to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[:, :, cand_ids]
            scores = logits.softmax(dim=-1)[:, :, 0]        # P(good) at every position
            step_scores = scores[ids == tag_id].float().cpu().numpy().tolist()
        # alignment guard: truncation can drop trailing steps -> record and skip, never pad
        if len(step_scores) != len(steps):
            skipped += 1
            if skipped <= 3:
                print(f"  [warn] {r['id']}: {len(step_scores)} scores vs {len(steps)} steps "
                      f"(truncated) — skipped", flush=True)
            continue
        out.append({"id": r["id"], "chain": r["chain"], "split": r["split"],
                    "step_scores": step_scores,
                    "step_labels": r["step_labels"].tolist()})
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(recs)} ({(time.time()-t0)/60:.1f}m)", flush=True)

    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    p.write_text(json.dumps({"model": args.model, "gpu": gpu,
                             "minutes": round((time.time() - t0) / 60, 2),
                             "n": len(out), "skipped": skipped, "rows": out}, indent=2))
    print(f"[2c] wrote {len(out)} scored solutions ({skipped} skipped) -> {p}")
    print(f"[2c] {(time.time()-t0)/60:.1f} min on {gpu} (Appendix A.4)")


def cmd_analyze(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from multiseed_ablation import build_confounds

    blob = json.loads(Path(args.scores).read_text())
    rows = blob["rows"]
    print(f"[2c] {blob['model']} | n={len(rows)} | scored on {blob.get('gpu')} "
          f"in {blob.get('minutes')} min")

    # ---- SANITY GATE: do per-step scores predict HUMAN step labels? ----
    fs, fl = [], []
    for r in rows:
        for s, l in zip(r["step_scores"], r["step_labels"]):
            if l >= 0:
                fs.append(1.0 - s); fl.append(int(l))      # 1-P(good) should predict error=1
    step_auc = roc_auc_score(fl, fs) if len(set(fl)) > 1 else float("nan")
    print(f"[2c] SANITY step-level AUC vs human labels = {step_auc:.3f} (n={len(fl)} steps)")
    if not (step_auc > 0.55):
        print("[2c] *** GATE FAILED *** step scores are at/near chance against human labels.")
        print("      The scoring format is almost certainly wrong (tag id, prompt layout, or")
        print("      token alignment). Downstream numbers would be meaningless — fix before use.")
        if not args.force:
            sys.exit(1)

    ids = [r["id"] for r in rows]
    y = np.array([r["chain"] for r in rows])
    agg = {"min (weakest-step, primary)": np.array([1.0 - min(r["step_scores"]) for r in rows]),
           "mean (secondary)":            np.array([1.0 - float(np.mean(r["step_scores"])) for r in rows])}

    # confounds, restricted to the scored subset and in its order
    full = torch.load(args.cache, weights_only=False)
    keep = {i: k for k, i in enumerate(ids)}
    sub = [r for r in full if r["id"] in keep]
    sub.sort(key=lambda r: keep[r["id"]])
    C = build_confounds(sub, args.data_dir)

    rng = np.random.RandomState(args.seed); idx = np.arange(len(rows)); rng.shuffle(idx)
    cut = int(0.8 * len(rows)); tri, vai = idx[:cut], idx[cut:]
    yb_val = (y[vai] == "B").astype(int)

    print("\n" + "=" * 74)
    print("  PHASE 2c — EXTERNAL PRM UNDER OUR CONFOUND PROTOCOL")
    print(f"  {blob['model']}   (threshold fit on TRAIN only; held-out n={len(vai)})")
    print("=" * 74)
    print(f"  {'aggregation':<30}{'raw f1_B':<12}{'ctrl f1_B':<12}{'raw AUC':<11}{'ctrl AUC'}")
    print("  " + "-" * 70)
    for name, s in agg.items():
        def fit(v):
            clf = LogisticRegression(max_iter=2000).fit(v[tri].reshape(-1, 1), y[tri])
            pred = clf.predict(v[vai].reshape(-1, 1))
            return (f1_score(y[vai], pred, pos_label="B"), roc_auc_score(yb_val, v[vai]))
        raw_f1, raw_auc = fit(s)
        beta, *_ = np.linalg.lstsq(C[tri], s[tri], rcond=None)   # residualize, fit on train
        sr = s - C @ beta
        ctl_f1, ctl_auc = fit(sr)
        print(f"  {name:<30}{raw_f1:<12.3f}{ctl_f1:<12.3f}{raw_auc:<11.3f}{ctl_auc:.3f}"
              f"   (Δf1 {ctl_f1-raw_f1:+.3f}, ΔAUC {ctl_auc-raw_auc:+.3f})")
    print("=" * 74)
    print("  READING: substantial deflation raw->controlled means the difficulty-artifact")
    print("  critique generalises beyond our model. No deflation means strong PRMs already")
    print("  capture confound-independent signal and our critique is specific to lightweight/")
    print("  pooled approaches. Both outcomes are reportable; neither is a failure.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--model", default=MODEL)
    s.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    s.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    s.add_argument("--out", default=str(HERE / "results_prm/scores.json"))
    s.add_argument("--device", default="auto"); s.add_argument("--dtype", default="bf16")
    s.add_argument("--max_len", type=int, default=2048); s.add_argument("--limit", type=int, default=0)
    a = sub.add_parser("analyze"); a.set_defaults(fn=cmd_analyze)
    a.add_argument("--scores", default=str(HERE / "results_prm/scores.json"))
    a.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    a.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--force", action="store_true", help="continue even if the sanity gate fails")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
