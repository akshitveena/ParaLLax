"""
mechanism_judge.py — Claude VALIDITY judge (the real A vs B authority).

The similarity judge (approach_judge.py) only answers "same or different from the
reference?" — that is CANONICALITY, and different != wrong: a genuinely different
but valid route (sound_alternative) is still Type A. So similarity over-counts
Type B by sweeping in valid alternatives.

This judge answers the question that actually defines Type B: *is the reasoning
SOUND?* For each CORRECT candidate it classifies HOW the correct answer was
reached, on the validity axis, into one of five mechanisms:

    sound_canonical   -> Type A   (valid, ~ the reference approach)
    sound_alternative -> Type A   (valid, a different route — broad OR narrow)
    flawed_lucky      -> Type B   (a real invalid step, yet the answer is right)
    unfaithful        -> Type B   (stated steps don't entail the answer)
    spurious          -> Type B   (guessed/recalled, reasoning back-filled)

It writes `type_b_mechanism` (the 5-way label) back onto each judged candidate;
data_pipeline treats it as AUTHORITATIVE for A/B (overriding the similarity/gap
proxies). It does NOT touch `answer_correct` or `approach_matches_gold` — it
composes with files already run through verify_answers.py / approach_judge.py
without re-spending on those steps.

Only CORRECT candidates are judged (A/B only exists among correct answers); a
gold_solution is used as the reference when present (it sharpens canonical vs
alternative) but is optional — validity can be judged from the student work alone.

Run AFTER verify_answers.py, alongside/after approach_judge.py, BEFORE data_pipeline.py:
    python api_generation/mechanism_judge.py --in data/raw/candidates_raw.jsonl \
                                             --out data/raw/candidates_raw.jsonl
Requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import textutils as T  # noqa: E402

MODEL = "claude-sonnet-4-6"
LABELS = {"sound_canonical", "sound_alternative", "flawed_lucky", "unfaithful", "spurious"}


def parse_label(text: str) -> str | None:
    """Extract the verdict from a free-form reply that ends in a LABEL: line.

    The judge now reasons before deciding, so labels can appear mid-deliberation.
    Take the one after the explicit 'label:' marker; else the LAST label mentioned
    (the conclusion), never the first."""
    t = text.lower()
    if "label:" in t:
        tail = t.rsplit("label:", 1)[1]
        hit = [(tail.find(lab), lab) for lab in LABELS if lab in tail]
        if hit:
            return min(hit)[1]                       # first label after the marker
    last = [(t.rfind(lab), lab) for lab in LABELS if lab in t]
    return max(last)[1] if last else None            # else the last one mentioned

# Finalized prompt. The case-1/case-2 framing NAMES the phenomenon (right answer via
# unsound reasoning) descriptively — never imperatively ("find"/"prefer") — so it
# calibrates without biasing the judge toward over-calling flaws. sound_alternative
# explicitly admits the brittle case; flawed_lucky keeps the invalid-step GATE so the
# "beauty" (a correct answer the logic had no right to reach) doesn't lower the bar.
JUDGE_SYSTEM = (
    "You analyze how a student reached the CORRECT final answer to a competition math "
    "problem. Your job is to judge the underlying LOGIC — is the reasoning actually "
    "sound? — not the wording, notation, elegance, or whether the method is general.\n\n"
    "A correct final answer can come about in two fundamentally different ways:\n"
    "  (1) the reasoning is SOUND and genuinely earns the answer, or\n"
    "  (2) the answer is right even though the reasoning is NOT sound — the interesting "
    "case where a solution lands correctly that, by its own logic, had no right to.\n"
    "Classify into exactly one of five labels (the first two are case 1; the last three "
    "are the three ways case 2 happens):\n\n"
    "sound_canonical — valid reasoning, essentially the reference approach.\n\n"
    "sound_alternative — valid reasoning via a genuinely different route from the "
    "reference. Every step is logically justified and the conclusion truly follows. "
    "Classify here whether the method is broad OR narrow/problem-specific: a correct but "
    "non-generalizing approach is still SOUND, not wrong.\n\n"
    "flawed_lucky — the reasoning contains a real conceptual error or a logically unsound "
    "step, yet still arrives at the correct answer: a compensating error (two mistakes "
    "cancel), an answer insensitive to the flaw, or an unsound step the student treats as "
    "valid that happens to hold here. The gate is a GENUINELY invalid step — not mere "
    "difference or narrowness.\n\n"
    "unfaithful — the stated steps do not actually produce the answer because the REAL "
    "cause is elsewhere: a different/hidden computation, or a post-hoc rationalization "
    "pasted onto an already-known answer. The stated reasoning is a facade.\n\n"
    "spurious — the answer looks guessed, recalled, or pattern-matched, with the reasoning "
    "back-filled rather than deriving it.\n\n"
    "IMPORTANT — incompleteness is NOT a flaw. A derivation that is valid as far as it goes "
    "but does not fully prove the result (e.g. checks small cases and asserts the rest, or "
    "omits a completeness argument) is SOUND-but-partial, not flawed_lucky and not "
    "unfaithful. Only classify a flaw when there is a genuinely INVALID step or a hidden/"
    "post-hoc cause — never merely because a proof is unfinished or informal.\n\n"
    "Distinguishing sound_alternative from flawed_lucky often requires actually checking "
    "the questionable steps. So FIRST check the load-bearing steps for a genuine error. "
    "But be efficient and DECISIVE: a flaw label (flawed_lucky/unfaithful/spurious) "
    "requires you to point to a SPECIFIC invalid step. If you cannot identify one after "
    "checking the main steps, the reasoning is sound — do NOT exhaustively recompute every "
    "line. Always finish, on a separate final line, with exactly:\n"
    "LABEL: <one of sound_canonical, sound_alternative, flawed_lucky, unfaithful, spurious>"
)


def _clip(text: str, cap: int) -> str:
    """Keep the whole solution when possible; if it overflows, keep HEAD + TAIL so the
    CONCLUSION (where the answer/final steps live) survives. A naive text[:cap] drops the
    ending and makes complete solutions look 'cut off' -> false unfaithful/incomplete."""
    text = text or ""
    if len(text) <= cap:
        return text
    head = int(cap * 0.6); tail = cap - head
    return text[:head] + "\n...[middle elided]...\n" + text[-tail:]


def build_prompt(problem: str, gold_solution: str, response: str,
                 sol_cap: int = 12000, ref_cap: int = 4000) -> str:
    ref = (f"REFERENCE SOLUTION:\n{_clip(gold_solution, ref_cap)}\n\n" if gold_solution.strip()
           else "REFERENCE SOLUTION: (none provided)\n\n")
    return (f"PROBLEM:\n{problem}\n\n{ref}"
            f"STUDENT SOLUTION (reached the correct answer):\n{_clip(response, sol_cap)}\n\n"
            "Verify the reasoning step by step, then end with the LABEL line.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--out", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--max_tokens", type=int, default=2500)  # room to VERIFY (coordinate-bash needs it)
    ap.add_argument("--poll_seconds", type=int, default=30)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in your environment."); sys.exit(1)

    records = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]

    requests, index = [], {}
    n_skip_wrong = 0
    for ri, rec in enumerate(records):
        gold_solution = str(rec.get("gold_solution", "")).strip()
        for ci, cand in enumerate(rec.get("candidates", [])):
            # Only correct candidates matter for A vs B; needs answer_correct from verify.
            if not cand.get("answer_correct"):
                n_skip_wrong += 1
                continue
            resp = cand.get("response_text") or T.split_think_response(cand.get("full_text", ""))[1]
            if not resp.strip():
                continue
            cand.pop("type_b_mechanism", None)   # clear any stale label so a re-judge is clean
            cid = f"mech-{ri}-{ci}"
            index[cid] = (ri, ci)
            requests.append({"custom_id": cid, "params": {
                "model": MODEL, "max_tokens": args.max_tokens, "system": JUDGE_SYSTEM,
                "messages": [{"role": "user",
                              "content": build_prompt(rec.get("problem", ""), gold_solution, resp)}]}})

    print(f"[mechanism] correct candidates to judge: {len(requests)}  (skipped {n_skip_wrong} wrong)")
    if not requests:
        print("[mechanism] nothing to judge. (Did you run verify_answers first? "
              "It sets answer_correct.)")
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                                  encoding="utf-8")
        return

    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"[mechanism] batch {batch.id} submitted — polling every {args.poll_seconds}s")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(args.poll_seconds)

    counts: Counter = Counter()
    n_unparsed = 0
    unparsed_samples: list[tuple[str, str]] = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        ri, ci = index[result.custom_id]
        text = "".join(blk.text for blk in result.result.message.content
                       if getattr(blk, "type", None) == "text").strip()
        label = parse_label(text)
        if label is None:
            n_unparsed += 1
            if len(unparsed_samples) < 8:
                unparsed_samples.append((result.custom_id, text[:80]))
            continue
        records[ri]["candidates"][ci]["type_b_mechanism"] = label
        counts[label] += 1

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                              encoding="utf-8")

    judged = sum(counts.values())
    type_b = sum(counts[m] for m in ("flawed_lucky", "unfaithful", "spurious"))
    print(f"[mechanism] judged {judged}  (unparsed {n_unparsed})")
    if n_unparsed:
        print(f"[mechanism] WARNING: {n_unparsed} reply(ies) had no LABEL line — likely "
              f"truncated mid-verification. Raise --max_tokens and re-run; rates below "
              f"exclude them.")
    for cid, txt in unparsed_samples:
        print(f"    [unparsed] {cid}: {txt!r}")
    for lab in ("sound_canonical", "sound_alternative", "flawed_lucky", "unfaithful", "spurious"):
        tag = "A" if lab.startswith("sound") else "B"
        print(f"    {lab:<18} -> Type {tag}: {counts.get(lab, 0)}")
    print(f"[mechanism] Type-B (validity) rate among judged correct: "
          f"{100*type_b/max(judged,1):.0f}%")
    print(f"[mechanism] wrote {args.out}")
    print(f"[mechanism] NEXT: python main/data_pipeline.py --raw {args.out} "
          f"--out_dir /tmp/pilot_proc --allow_small")


if __name__ == "__main__":
    main()
