"""
data_pipeline.py — turn RAW per-problem generation records into PROCESSED,
fully-classified Candidates following the complete RiDAE Dataset Schema.

Per candidate it computes Blocks 3-9:
  3 answer extraction/verification   6 reasoning structure       9 quality control
  4 two-pass type classification     7 corruption-readiness
  5 approach analysis (free)         8 training labels (+ contrastive wiring)

Then builds contrastive pairs and runs the Section 7 validation checks.

Pipeline:  claude_generate.py -> [score_candidates.py] -> data_pipeline.py
Outputs:   data/processed/candidates.jsonl   (flat, one Candidate per line)
           data/processed/contrastive_pairs.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from schema import Candidate, ProblemRecord, ContrastivePair, TYPE_TO_LABEL
import textutils as T
from approach_analysis import (classify_approach, detect_thinking_switch,
                               stated_approach_sentence, approaches_match,
                               settled_thinking_approach)

_CONF_RANK = {"high": 3, "medium": 2, "low": 1}

# VALIDITY axis (mechanism judge). sound_* = the reasoning is valid -> Type A;
# the other three = right answer via UNSOUND reasoning -> Type B. "Wrong approach"
# is a soundness failure, not difference or brittleness.
_SOUND_MECH = {"sound_canonical", "sound_alternative"}
_FLAWED_MECH = {"flawed_lucky", "unfaithful", "spurious"}


# --------------------------------------------------------------------------- #
# Per-candidate processing (Blocks 3-9)
# --------------------------------------------------------------------------- #
def process_candidate(raw: dict, pf: dict) -> Candidate:
    """Build a fully-populated Candidate from a raw candidate dict + problem fields."""
    c = Candidate(**pf)
    # ---- Block 1 (carry through what generation provided) ----
    c.candidate_id = raw.get("candidate_id", "")
    c.model = raw.get("model", "")
    c.temperature = float(raw.get("temperature", 0.0))
    c.reasoning_effort = raw.get("reasoning_effort")
    c.generation_seed = int(raw.get("generation_seed", 0))
    c.input_tokens = int(raw.get("input_tokens", 0))
    c.thinking_tokens = int(raw.get("thinking_tokens", 0))
    c.output_tokens = int(raw.get("output_tokens", 0))
    c.total_cost_usd = float(raw.get("total_cost_usd", 0.0))
    c.generation_timestamp = raw.get("generation_timestamp", "")

    # ---- Block 2 (raw text) ----
    full = raw.get("full_text", "")
    c.thinking_text = raw.get("thinking_text", "")
    c.response_text = raw.get("response_text", "")
    if not c.response_text and full:
        c.thinking_text, c.response_text = T.split_think_response(full)
    c.full_text = full or T.assemble_full_text(c.thinking_text, c.response_text)
    c.has_think_tags = T.has_think_tags(full)
    # thinking_source: trust the generator, else infer. Claude ET == architecturally
    # separate thinking; QwQ <think> == inline (same autoregressive pass); else none.
    c.thinking_source = raw.get("thinking_source") or (
        "none" if not c.thinking_text
        else "inline_think_tags" if c.has_think_tags
        else "claude_api_block" if c.has_extended_thinking
        else "parsed_reasoning")     # Groq reasoning models (Qwen3 / gpt-oss) via parsed mode

    # ---- Block 3 (answers) ----  Always extract from response_text only (item 2).
    ans, method = T.extract_answer(c.response_text)
    c.answer_extracted = ans
    match, mmethod = T.answers_match(ans, c.gold_answer)
    # Respect an upstream verdict (verify_answers.py LLM judge) when present.
    c.answer_correct = bool(raw["answer_correct"]) if "answer_correct" in raw else match
    c.answer_match_method = raw.get("answer_match_method", mmethod)
    c.normalised_extracted = T.normalise_answer(ans)
    c.normalised_gold = T.normalise_answer(c.gold_answer)
    if c.thinking_text:
        c.answer_in_thinking = T.extract_answer(c.thinking_text)[0]

    # ---- Block 6 (structure) ----
    c.reasoning_steps = T.segment_steps(c.response_text)
    c.num_steps = len(c.reasoning_steps)
    c.step_types = [T.classify_step_type(s, i, c.num_steps)
                    for i, s in enumerate(c.reasoning_steps)]
    if c.thinking_text:
        c.think_step_count = len(T.segment_steps(c.thinking_text))
    elif c.thinking_source == "none":
        c.think_step_count = 0

    # ---- Block 5 (approach) ----
    c.approach_in_response, c.approach_keywords = classify_approach(c.response_text)
    if raw.get("approach_in_response"):                  # Claude disambiguation override
        c.approach_in_response = raw["approach_in_response"]
    c.stated_approach_sentence = stated_approach_sentence(c.response_text)
    gold_approach, _ = classify_approach(c.gold_solution) if c.gold_solution else ("unknown", [])
    c.approach_matches_gold = approaches_match(c.approach_in_response, gold_approach)
    if raw.get("approach_matches_gold") is not None:     # similarity judge: canonicality signal (NOT A/B)
        c.approach_matches_gold = bool(raw["approach_matches_gold"])
    if raw.get("type_b_mechanism"):                       # mechanism judge: VALIDITY -> authoritative A/B
        c.type_b_mechanism = str(raw["type_b_mechanism"])
    if c.thinking_text:
        # Use the SETTLED approach (after the last self-correction), not the first
        # attempt — a self-corrected-but-faithful chain must NOT count as a gap.
        c.approach_in_thinking, _ = settled_thinking_approach(c.thinking_text)
        if raw.get("approach_in_thinking"):             # Claude disambiguation override
            c.approach_in_thinking = raw["approach_in_thinking"]
        gap = approaches_match(c.approach_in_thinking, c.approach_in_response)
        c.thinking_response_gap = (gap is False)
        if c.thinking_response_gap:
            switch = detect_thinking_switch(c.thinking_text)
            c.gap_description = (
                f"Settled thinking approach ({c.approach_in_thinking}) differs from "
                f"response approach ({c.approach_in_response}). "
                + (switch or ""))

    # ---- Block 4 (two-pass classification) ----
    c.candidate_type, c.type_confidence, c.type_source = _classify(c)

    # ---- Block 5 error fields (Type B only) ----
    if c.candidate_type == "B":
        if c.type_b_mechanism in _FLAWED_MECH:        # mechanism judge: the validity failure
            c.error_type = c.type_b_mechanism
            c.error_location = "reasoning_validity"
            c.error_description = (
                f"Mechanism judge: {c.type_b_mechanism} — the reasoning is unsound yet "
                f"the answer is correct.")
        else:                                          # legacy proxy path (no mechanism verdict)
            c.error_location = "step_2_approach"
            c.error_type = "wrong_framing" if c.thinking_response_gap else "wrong_concept"
            c.error_description = c.gap_description or (
                f"Response approach ({c.approach_in_response}) diverges from the "
                f"gold approach ({gold_approach}).")

    # ---- Block 7 (corruption readiness) ----
    c.can_approach_corrupt = (c.approach_in_response != "unknown"
                              and bool(c.stated_approach_sentence) and c.num_steps >= 2)
    c.can_step_corrupt = c.num_steps >= 3
    c.can_conclusion_corrupt = c.answer_extracted is not None and T.is_numeric_answer(c.answer_extracted)
    if c.candidate_type == "B" and c.error_location:
        c.corruption_priority = "high"
    elif c.candidate_type in ("A", "B") and c.approach_in_response != "unknown":
        c.corruption_priority = "medium"
    else:
        c.corruption_priority = "low"

    # ---- Block 8 (training labels; contrastive wiring filled after grouping) ----
    c.training_label = TYPE_TO_LABEL.get(c.candidate_type, 2)
    c.contrastive_group = c.record_id
    c.hardness_score = raw.get("hardness_score")  # preserved across re-runs if present

    # ---- Block 9 (quality control) ----
    c.generation_error = raw.get("generation_error")
    c.was_retried = bool(raw.get("was_retried", False))
    c.blackhole_detected = bool(raw.get("blackhole_detected", False))
    c.quality_flags = _quality_flags(c)
    if c.blackhole_detected and "reasoning_blackhole" not in c.quality_flags:
        c.quality_flags.append("reasoning_blackhole")
    if method != "boxed" and c.answer_extracted is not None:
        c.extraction_warning = f"answer extracted via fallback method: {method}"
    # Garbage (truncation / temp-1.1 loops / gpt-oss reasoning blackholes) is excluded.
    bad = {"incomplete_response", "repeated_content", "reasoning_blackhole"} & set(c.quality_flags)
    c.include_in_training = (c.answer_extracted is not None and c.num_steps > 0
                             and bool(c.full_text) and c.generation_error is None
                             and not c.blackhole_detected and not bad)
    return c


def _classify(c: Candidate) -> tuple[str, str, str]:
    """Block 4 rules -> (candidate_type, type_confidence, type_source).

    A thinking-response gap only earns HIGH confidence when the thinking is
    architecturally separate (Claude ET, thinking_source == 'claude_api_block').
    For QwQ inline think tags the gap is textual, not architecturally proven, so
    it earns only MEDIUM confidence (schema: "The faithfulness problem").
    """
    if not c.answer_correct:
        if c.approach_matches_gold is True:           # wrong answer, sound path (rare)
            return "C", "low", "answer+approach"
        return "D", "high", "answer_only"
    # answer is correct.
    # VALIDITY (mechanism judge) is authoritative: it judges whether the reasoning is
    # actually sound, which is what A vs B means. It overrides the similarity/gap
    # proxies, which only see canonicality (different != wrong).
    if c.type_b_mechanism in _SOUND_MECH:
        return "A", "high", "answer+mechanism"
    if c.type_b_mechanism in _FLAWED_MECH:
        return "B", "high", "answer+mechanism"
    if c.thinking_response_gap:
        conf = "high" if c.thinking_source == "claude_api_block" else "medium"
        return "B", conf, "answer+thinking_gap"
    if c.approach_matches_gold is False:
        return "B", "medium", "answer+approach"
    if c.approach_matches_gold is True:
        return "A", "high", "answer+approach"
    return "A", "low", "answer_only"                  # unknown -> conservative A (Claude may flip)


def _quality_flags(c: Candidate) -> list[str]:
    flags = []
    resp = c.response_text or ""
    if len(resp.split()) < 80:                              # ~<100 tokens
        flags.append("very_short")
    if c.num_steps == 0:
        flags.append("no_steps_found")
    if c.answer_extracted is None:
        flags.append("answer_not_extracted")
    if c.approach_in_response == "unknown":
        flags.append("approach_unknown")
    if T.has_repeated_ngram(resp, n=20, max_repeat=3):      # QwQ temp-1.1 loops (item 4)
        flags.append("repeated_content")
    if T.is_incomplete_response(resp):                      # truncated mid-reasoning (item 4)
        flags.append("incomplete_response")
    if T.has_cjk(c.thinking_text or ""):                    # R1 language-mixing (point 5):
        flags.append("language_mixing")                     # valid but flagged, NOT excluded
    return flags


# --------------------------------------------------------------------------- #
# Loading + contrastive wiring
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> list[Candidate]:
    """Load RAW per-problem records and process every candidate (Blocks 3-9)."""
    out: list[Candidate] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "candidates" in rec:                         # nested ProblemRecord
                pr = ProblemRecord.from_dict(rec)
                pf = pr.problem_fields()
                for raw_c in pr.candidates:
                    out.append(process_candidate(raw_c, pf))
            else:                                           # already-flat candidate
                out.append(Candidate.from_dict(rec))
    _wire_contrastive(out)
    return out


def _wire_contrastive(candidates: list[Candidate]) -> None:
    """Set is_contrastive_anchor / _negative / pair_id (Block 8) in place."""
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[c.contrastive_group].append(c)
    for grp, members in groups.items():
        a_models = {m.model for m in members if m.candidate_type == "A"}
        b_models = {m.model for m in members if m.candidate_type == "B"}
        has_a, has_b = bool(a_models), bool(b_models)
        # Cross-model pair: an A and a B from DIFFERENT models on the same problem.
        cross = has_a and has_b and bool(a_models ^ b_models) and (a_models != b_models)
        for m in members:
            m.is_contrastive_anchor = (m.candidate_type == "A" and has_b)
            m.is_contrastive_negative = (m.candidate_type == "B" and has_a)
            m.contrastive_pair_id = (f"{grp}_pair"
                                     if (m.is_contrastive_anchor or m.is_contrastive_negative)
                                     else None)
            if (m.is_contrastive_anchor or m.is_contrastive_negative) and cross:
                m.cross_model_pair = True


def build_contrastive_pairs(candidates: list[Candidate]) -> list[ContrastivePair]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[c.contrastive_group].append(c)
    pairs = []
    for grp, members in groups.items():
        a = [m for m in members if m.candidate_type == "A"]
        b = [m for m in members if m.candidate_type == "B"]
        if not a or not b:
            continue
        # Prefer a cross-model A/B combo (item 6: the hardest, highest-value pairs),
        # then break ties by combined classification confidence.
        combos = [(x, y) for x in a for y in b]
        combos.sort(key=lambda xy: (
            xy[0].model != xy[1].model,
            _CONF_RANK.get(xy[0].type_confidence, 0) + _CONF_RANK.get(xy[1].type_confidence, 0)),
            reverse=True)
        best_a, best_b = combos[0]
        pairs.append(ContrastivePair(grp, best_a.problem, best_a.gold_answer,
                                     best_a.dataset, best_a, best_b,
                                     cross_model=best_a.model != best_b.model))
    return pairs


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_candidates(candidates: list[Candidate], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_candidates(path: str | Path) -> list[Candidate]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Candidate.from_dict(json.loads(line)))
    return out


def save_contrastive_pairs(pairs: list[ContrastivePair], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([p.to_dict() for p in pairs], fh, ensure_ascii=False, indent=2)


def load_contrastive_pairs(path: str | Path) -> list[ContrastivePair]:
    with Path(path).open(encoding="utf-8") as fh:
        return [ContrastivePair.from_dict(d) for d in json.load(fh)]


# --------------------------------------------------------------------------- #
# Statistics + Section 7 validation
# --------------------------------------------------------------------------- #
def print_statistics(candidates: list[Candidate], pairs: list[ContrastivePair]) -> None:
    n = len(candidates)
    by_type = Counter(c.candidate_type for c in candidates)
    correct = sum(c.answer_correct for c in candidates)
    print("=" * 64)
    print("CANDIDATE STATISTICS")
    print("=" * 64)
    print(f"  Total candidates : {n}   |   unique problems: "
          f"{len({c.contrastive_group for c in candidates})}")
    print(f"  Answer-correct   : {correct} ({100*correct/max(n,1):.1f}%)")
    for t in ("A", "B", "C", "D"):
        print(f"      Type {t}: {by_type.get(t,0):>5}")
    print(f"  Type B rate (of correct): {100*by_type.get('B',0)/max(correct,1):.1f}%")
    print(f"  Contrastive pairs        : {len(pairs)}")
    print(f"  include_in_training=True : "
          f"{sum(c.include_in_training for c in candidates)}")
    print("=" * 64)


def validation_checks(candidates: list[Candidate], pairs: list[ContrastivePair],
                      allow_small: bool = False) -> bool:
    """Section 7 pre-training checks with the schema's real thresholds.

    Returns True iff every check passes. `allow_small` skips the absolute-count
    checks (total candidates, contrastive-pair minimum) so the offline synthetic
    corpus can be exercised; all rate-based checks still apply. The CALLER hard-
    stops the pipeline on a False return, per "any failure stops the pipeline".
    """
    def rate(ds, pred):
        sub = [c for c in candidates if c.dataset == ds]
        return (sum(pred(c) for c in sub) / len(sub)) if sub else None

    checks = []  # (name, passed, shown, is_count_check)
    n = len(candidates)
    checks.append(("Total candidates (2700 +/- 50)", 2650 <= n <= 2750, f"{n}", True))

    bands = {"OmniMath": (0.40, 0.60), "OlympiadBench": (0.40, 0.60), "MATH": (0.55, 0.75)}
    for ds, (lo, hi) in bands.items():
        r = rate(ds, lambda c: c.answer_correct)
        if r is not None:
            checks.append((f"{ds} answer-correct rate ({int(lo*100)}-{int(hi*100)}%)",
                           lo <= r <= hi, f"{100*r:.0f}%", False))

    omni_correct = [c for c in candidates if c.dataset == "OmniMath" and c.answer_correct]
    if omni_correct:
        brate = sum(c.candidate_type == "B" for c in omni_correct) / len(omni_correct)
        checks.append(("OmniMath Type-B rate of correct (35-55%)",
                       0.35 <= brate <= 0.55, f"{100*brate:.0f}%", False))

    checks.append(("Contrastive pairs (>=250)", len(pairs) >= 250, f"{len(pairs)}", True))

    et = [c for c in candidates if c.has_extended_thinking]
    if et:
        gap = sum(bool(c.thinking_response_gap) for c in et) / len(et)
        checks.append(("ET thinking-response-gap rate (>=15%)", gap >= 0.15, f"{100*gap:.0f}%", False))

    inc = sum(c.include_in_training for c in candidates) / max(n, 1)
    checks.append(("include_in_training rate (>=90%)", inc >= 0.90, f"{100*inc:.0f}%", False))
    clean = sum(len(c.quality_flags) == 0 for c in candidates) / max(n, 1)
    checks.append(("quality_flags-empty rate (>=80%)", clean >= 0.80, f"{100*clean:.0f}%", False))
    cost = sum(c.total_cost_usd for c in candidates)
    checks.append(("total cost (< $25)", cost < 25.0, f"${cost:.2f}", False))

    print("VALIDATION CHECKS (Section 7)")
    print("-" * 64)
    ok = True
    for name, passed, shown, is_count in checks:
        skipped = allow_small and is_count and not passed
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        if not skipped:
            ok &= passed
        print(f"  [{status}] {name:<42} {shown}")
    print("-" * 64)
    if allow_small:
        print("  (--allow_small: absolute-count checks are advisory for offline testing)")
    return ok


# --------------------------------------------------------------------------- #
def main() -> None:
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/candidates_raw.jsonl")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--allow_small", action="store_true",
                    help="advisory absolute-count checks (offline synthetic testing)")
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        print(f"[data_pipeline] raw file not found: {raw}")
        print("  Generate it with api_generation/claude_generate.py,")
        print("  or scripts/make_synthetic_corpus.py for an offline test corpus.")
        return

    candidates = load_raw(raw)
    pairs = build_contrastive_pairs(candidates)
    out_dir = Path(args.out_dir)
    save_candidates(candidates, out_dir / "candidates.jsonl")
    save_contrastive_pairs(pairs, out_dir / "contrastive_pairs.json")
    print_statistics(candidates, pairs)
    ok = validation_checks(candidates, pairs, allow_small=args.allow_small)
    print(f"[data_pipeline] wrote {out_dir/'candidates.jsonl'} and "
          f"{out_dir/'contrastive_pairs.json'}")
    if not ok and not args.allow_small:
        print("[data_pipeline] VALIDATION FAILED — pipeline stops (Section 7). "
              "Fix generation before training, or pass --allow_small for offline tests.")
        sys.exit(1)
    if not ok and args.allow_small:
        print("[data_pipeline] validation has failures, but --allow_small is set "
              "(offline test) — proceeding anyway. These WILL hard-stop on a real run.")


if __name__ == "__main__":
    main()
