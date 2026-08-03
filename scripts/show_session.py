"""
show_session.py — one screen showing everything built this session in action.

Reads the RAW file (after verify + approach_judge + mechanism_judge) and the
PROCESSED candidates, and prints:
  1. the pipeline signals present per stage (verify / similarity / mechanism)
  2. the SIMILARITY-vs-VALIDITY contrast — the over-count we engineered out
     (candidates similarity called "different -> Type B" that the mechanism
      judge rescued to sound_alternative -> Type A)
  3. the final validity-based A/B mix, mechanism distribution, contrastive pairs
  4. one worked example of a rescued candidate

Usage:
  python scripts/show_session.py --raw data/raw/pilot_omni.jsonl \
                                 --proc /tmp/pilot_proc/candidates.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

A_MECH = {"sound_canonical", "sound_alternative"}
B_MECH = {"flawed_lucky", "unfaithful", "spurious"}


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/pilot_omni.jsonl")
    ap.add_argument("--proc", default="/tmp/pilot_proc/candidates.jsonl")
    args = ap.parse_args()

    recs = load(args.raw)
    cands = [(r, c) for r in recs for c in r.get("candidates", [])]
    correct = [(r, c) for r, c in cands if c.get("answer_correct")]

    print("=" * 70)
    print("  RiDAE SESSION — EVERYTHING IN ACTION")
    print("=" * 70)
    print(f"  problems={len(recs)}  candidates={len(cands)}  correct={len(correct)}")

    # ---- 1. which signals each stage produced ----
    n_verify = sum("answer_correct" in c for _, c in cands)
    n_sim    = sum(c.get("approach_matches_gold") is not None for _, c in correct)
    n_mech   = sum(bool(c.get("type_b_mechanism")) for _, c in correct)
    print("\n  STAGE SIGNALS (on correct candidates)")
    print(f"    verify_answers  -> answer_correct set on {n_verify}/{len(cands)}")
    print(f"    approach_judge  -> similarity signal on  {n_sim}/{len(correct)}")
    print(f"    mechanism_judge -> validity label on     {n_mech}/{len(correct)}")

    # ---- 2. similarity (canonicality) vs mechanism (validity) ----
    print("\n  SIMILARITY (old proxy)  vs  VALIDITY (this session)")
    sim_b = [(r, c) for r, c in correct if c.get("approach_judge") == "different"]
    print(f"    similarity would call Type B (approach=different): {len(sim_b)}")
    rescued = [(r, c) for r, c in sim_b if c.get("type_b_mechanism") in A_MECH]
    confirmed = [(r, c) for r, c in sim_b if c.get("type_b_mechanism") in B_MECH]
    print(f"       -> RESCUED to Type A by mechanism (sound_alternative): {len(rescued)}")
    print(f"       -> CONFIRMED Type B by mechanism (real flaw):          {len(confirmed)}")
    over = len(sim_b) - len(confirmed)
    print(f"    over-count removed: {over} candidate(s) that 'different' "
          f"wrongly flagged as wrong")

    # ---- 3. mechanism distribution + final A/B ----
    dist = Counter(c.get("type_b_mechanism") for _, c in correct if c.get("type_b_mechanism"))
    print("\n  MECHANISM DISTRIBUTION (validity axis)")
    for lab in ("sound_canonical", "sound_alternative", "flawed_lucky", "unfaithful", "spurious"):
        tag = "A" if lab in A_MECH else "B"
        print(f"    {lab:<18} -> Type {tag}: {dist.get(lab, 0)}")
    tb = sum(dist[m] for m in B_MECH)
    print(f"    TRUE Type-B rate among judged correct: "
          f"{100*tb/max(sum(dist.values()),1):.0f}%")

    # ---- processed view (final classification + pairs) ----
    if Path(args.proc).exists():
        proc = load(args.proc)
        bytype = Counter(c["candidate_type"] for c in proc)
        pairs = 0
        pp = Path(args.proc).parent / "contrastive_pairs.json"
        if pp.exists():
            pairs = len(json.loads(pp.read_text()))
        print("\n  FINAL (data_pipeline output)")
        print(f"    A={bytype.get('A',0)}  B={bytype.get('B',0)}  "
              f"C={bytype.get('C',0)}  D={bytype.get('D',0)}")
        print(f"    contrastive pairs={pairs}  "
              f"include_in_training={sum(c['include_in_training'] for c in proc)}")
    else:
        print(f"\n  (run data_pipeline to produce {args.proc} for the final view)")

    # ---- 4. one worked rescued example ----
    if rescued:
        r, c = rescued[0]
        print("\n  WORKED EXAMPLE — a candidate 'different' rescued to Type A")
        print(f"    {c['candidate_id']}  (effort={c.get('reasoning_effort')})")
        print(f"    problem: {r['problem'][:120]}...")
        print(f"    similarity said : different  (old proxy -> Type B)")
        print(f"    mechanism said  : {c['type_b_mechanism']}  -> Type A  (valid, "
              f"just a different route)")
    print("=" * 70)


if __name__ == "__main__":
    main()
