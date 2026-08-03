"""
processbench_calibrate.py — calibrate OUR Type-B definition against ProcessBench's
HUMAN labels, on the same solutions.

ProcessBench (Qwen/ProcessBench) gives, per solution: problem, step-segmented
`steps`, `final_answer_correct`, and `label` = the index of the first human-flagged
erroneous step (-1 if the whole chain is clean). Among ANSWER-CORRECT solutions:

    human version:  label >= 0  -> Type B (wrong-path-right-answer)   label == -1 -> Type A
    our version:    mechanism_judge -> flawed_lucky/unfaithful/spurious = B ; sound_* = A

We run OUR mechanism judge (imported verbatim) on the same chains and cross-tabulate.
The 2x2 confusion matrix + Cohen's kappa + the disagreement cells tell us where the
two definitions diverge — the whole point. Expected signal: the Human-B / Ours-A cell
should concentrate on SELF-CORRECTED errors (human flags the first error step, but the
model recovers, so the final path is sound), because our `flawed_lucky` requires the
error to persist causally into the answer. That would show our definition is a stricter,
causally-grounded SUBSET of ProcessBench's "any first-error step".

Sampling: balanced by default (equal clean/error per split) so every matrix cell fills;
we therefore report metrics CONDITIONED on the human label (robust to the mix) as the
headline, and raw agreement/kappa as secondary (they depend on the chosen balance).

Run:
    python api_generation/processbench_calibrate.py --splits omnimath,olympiadbench \
        --per_split 50 --out data/processbench_calib.jsonl
Requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # for mechanism_judge
import mechanism_judge as MJ  # noqa: E402  (JUDGE_SYSTEM, build_prompt, parse_label, LABELS, MODEL)

FLAWED = {"flawed_lucky", "unfaithful", "spurious"}


def sample_split(ds, per_split: int, balanced: bool, seed: int):
    corr = [r for r in ds if r["final_answer_correct"]]
    clean = [r for r in corr if r["label"] == -1]
    err = [r for r in corr if r["label"] >= 0]
    rng = random.Random(seed)
    rng.shuffle(clean); rng.shuffle(err); rng.shuffle(corr)
    if balanced:
        k = per_split // 2
        picked = clean[:k] + err[:k]
    else:
        picked = corr[:per_split]
    rng.shuffle(picked)
    return picked


def kappa(matrix: dict) -> float:
    """Cohen's kappa from a 2x2 dict keyed ('A'|'B','A'|'B') = (human, ours)."""
    n = sum(matrix.values())
    if n == 0:
        return float("nan")
    po = (matrix[("A", "A")] + matrix[("B", "B")]) / n
    h_a = (matrix[("A", "A")] + matrix[("A", "B")]) / n           # human marginal A
    o_a = (matrix[("A", "A")] + matrix[("B", "A")]) / n           # ours marginal A
    pe = h_a * o_a + (1 - h_a) * (1 - o_a)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="omnimath,olympiadbench,math,gsm8k")
    ap.add_argument("--per_split", type=int, default=50)
    ap.add_argument("--balanced", action="store_true", default=True,
                    help="equal clean/error per split so all cells fill (default on)")
    ap.add_argument("--natural", dest="balanced", action="store_false",
                    help="sample answer-correct at the true base rate instead")
    ap.add_argument("--max_tokens", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/processbench_calib.jsonl")
    ap.add_argument("--poll_seconds", type=int, default=30)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment."); sys.exit(1)

    from datasets import load_dataset

    items = []          # (split, human_B, human_label_idx, problem, steps)
    requests, index = [], {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ds = load_dataset("Qwen/ProcessBench", split=split)
        picked = sample_split(ds, args.per_split, args.balanced, args.seed)
        for r in picked:
            i = len(items)
            steps = list(r["steps"])
            response = "\n".join(steps)
            human_B = r["label"] >= 0
            items.append({"split": split, "id": r["id"], "human_B": human_B,
                          "human_label_idx": r["label"], "problem": r["problem"],
                          "steps": steps})
            cid = f"pb-{i}"
            index[cid] = i
            requests.append({"custom_id": cid, "params": {
                "model": MJ.MODEL, "max_tokens": args.max_tokens, "system": MJ.JUDGE_SYSTEM,
                "messages": [{"role": "user",
                              "content": MJ.build_prompt(r["problem"], "", response)}]}})

    print(f"[calib] sampled {len(items)} answer-correct solutions "
          f"({'balanced' if args.balanced else 'natural'}) -> judging with mechanism_judge")
    if not requests:
        print("[calib] nothing to judge."); return

    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"[calib] batch {batch.id} submitted — polling every {args.poll_seconds}s")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(args.poll_seconds)

    # collect our verdicts
    n_unparsed = 0
    for result in client.messages.batches.results(batch.id):
        i = index[result.custom_id]
        if result.result.type != "succeeded":
            items[i]["our_mechanism"] = None; items[i]["judge_text"] = f"[{result.result.type}]"
            n_unparsed += 1
            continue
        text = "".join(blk.text for blk in result.result.message.content
                       if getattr(blk, "type", None) == "text").strip()
        lab = MJ.parse_label(text)
        items[i]["our_mechanism"] = lab
        items[i]["judge_text"] = text
        if lab is None:
            n_unparsed += 1

    # confusion matrix over parsed items
    matrix = {("A", "A"): 0, ("A", "B"): 0, ("B", "A"): 0, ("B", "B"): 0}
    mech_dist = Counter()
    for it in items:
        lab = it.get("our_mechanism")
        if lab is None:
            continue
        mech_dist[lab] += 1
        ours = "B" if lab in FLAWED else "A"
        human = "B" if it["human_B"] else "A"
        it["ours_AB"] = ours; it["human_AB"] = human
        matrix[(human, ours)] += 1

    # persist per-item for inspection
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for it in items:
            slim = {k: it[k] for k in ("split", "id", "human_B", "human_label_idx",
                                        "our_mechanism", "ours_AB", "human_AB")
                    if k in it}
            slim["problem"] = it["problem"][:200]
            idx = it.get("human_label_idx", -1)
            steps = it.get("steps", [])
            slim["flagged_step"] = steps[idx][:500] if 0 <= idx < len(steps) else ""
            slim["judge_text"] = (it.get("judge_text") or "")[-1200:]   # reasoning for adjudication
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")

    # ---- report ----
    n = sum(matrix.values())
    print("\n" + "=" * 66)
    print("  PROCESSBENCH CALIBRATION — human (ProcessBench) vs ours (mechanism)")
    print("=" * 66)
    print(f"  judged {n}  (unparsed/failed {n_unparsed})")
    print("\n  mechanism distribution (ours):")
    for lab in ("sound_canonical", "sound_alternative", "flawed_lucky", "unfaithful", "spurious"):
        print(f"     {lab:<18} {mech_dist.get(lab,0)}")
    print("\n  CONFUSION MATRIX (rows = human, cols = ours)")
    print(f"                 ours:A    ours:B")
    print(f"     human:A     {matrix[('A','A')]:>6}    {matrix[('A','B')]:>6}   (clean chains)")
    print(f"     human:B     {matrix[('B','A')]:>6}    {matrix[('B','B')]:>6}   (human found error)")

    human_B = matrix[("B", "A")] + matrix[("B", "B")]
    human_A = matrix[("A", "A")] + matrix[("A", "B")]
    recall_B = matrix[("B", "B")] / human_B if human_B else float("nan")   # we catch human errors
    spec_A = matrix[("A", "A")] / human_A if human_A else float("nan")     # we agree on clean
    agree = (matrix[("A", "A")] + matrix[("B", "B")]) / n if n else float("nan")
    print("\n  CONDITIONAL RATES (robust to sampling mix):")
    print(f"     P(ours=B | human found error) = {recall_B:.2f}   "
          f"<- how often we CATCH a human-flagged error")
    print(f"     P(ours=A | human says clean)  = {spec_A:.2f}   "
          f"<- how often we AGREE it's clean")
    print(f"\n  raw agreement = {agree:.2f}   Cohen's kappa = {kappa(matrix):.2f}   "
          f"(both depend on the {('balanced' if args.balanced else 'natural')} mix)")

    # interpretation hint
    under = matrix[("B", "A")]      # human error, we said sound -> under-count (self-correction?)
    over = matrix[("A", "B")]       # human clean, we invented a flaw -> over-count
    print("\n  DISAGREEMENT READ:")
    print(f"     under-count (human:B / ours:A) = {under}   "
          f"(expected: self-corrected errors — our 'persists-to-answer' bar is stricter)")
    print(f"     over-count  (human:A / ours:B) = {over}   "
          f"(if large: judge is hallucinating flaws — loosen it)")

    # show a few under-count cases to eyeball the self-correction hypothesis
    unders = [it for it in items if it.get("human_AB") == "B" and it.get("ours_AB") == "A"][:3]
    if unders:
        print("\n  SAMPLE under-count cases (human flagged step, we called sound):")
        for it in unders:
            print(f"     - {it['id']} (human error @ step {it['human_label_idx']}, "
                  f"ours={it['our_mechanism']}): {it['problem'][:90]}...")
    print("=" * 66)
    print(f"[calib] per-item verdicts written to {args.out}")


if __name__ == "__main__":
    main()
