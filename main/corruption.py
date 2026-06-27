"""
corruption.py — the three corruption strategies that drive RiDAE learning.

  1. Approach corruption   — replace the stated framing, keep the math.
  2. Step corruption       — delete / shuffle / insert an interior step.
  3. Conclusion corruption — perturb the final \\boxed{} answer.

Only the *response* block is corrupted; a thinking block (if present) is left
intact. Each corruption respects the Block-7 readiness flags computed upstream,
and every emitted sample carries the source candidate's corruption_priority and
hardness_score so the training sampler can weight it.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, asdict
from typing import Optional

from schema import Candidate
from textutils import (split_think_response, segment_steps, assemble_full_text,
                       is_numeric_answer)


@dataclass
class CorruptedSample:
    original_text: str
    corrupted_text: str
    corruption_type: str            # "approach" | "step" | "conclusion"
    candidate_type: str             # "A" | "B" | "C" | "D"
    contrastive_group: str
    dataset: str
    corruption_priority: str = "low"
    hardness_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CorruptedSample":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _reassemble(think: str, response: str) -> str:
    return assemble_full_text(think, response)


# --------------------------------------------------------------------------- #
# Approach corruption
# --------------------------------------------------------------------------- #
APPROACH_SIGNALS: dict[str, re.Pattern] = {
    "additive":       re.compile(r"\b(add|sum|total|plus|combine|altogether)\b", re.I),
    "multiplicative": re.compile(r"\b(multiply|product|times|factor|each|per)\b", re.I),
    "proportional":   re.compile(r"\b(ratio|proportion|percent|percentage|fraction|rate)\b", re.I),
    "algebraic":      re.compile(r"\b(let\s+[a-z]|equation|variable|solve for|substitute|isolate)\b", re.I),
    "sequential":     re.compile(r"\b(first|then|next|step by step|sequence|iterate|recursion)\b", re.I),
}

APPROACH_REPLACEMENTS: dict[str, list[str]] = {
    "additive": [
        "Since we need the total, I will add all the components together.",
        "The natural move here is to sum the relevant quantities.",
        "I will combine the parts by addition to reach the result.",
    ],
    "multiplicative": [
        "This requires finding the multiplicative relationship between the quantities.",
        "I will solve this by taking the product of the relevant factors.",
        "The key is to scale one quantity by another through multiplication.",
    ],
    "proportional": [
        "I will set up a proportion relating the quantities as a ratio.",
        "This is best handled by reasoning about the percentage relationship.",
        "I will express the quantities as a fraction and reason proportionally.",
    ],
    "algebraic": [
        "Let x denote the unknown and form an equation to solve for it.",
        "I will introduce a variable and isolate it algebraically.",
        "Setting up an equation and solving for the unknown is the way forward.",
    ],
    "sequential": [
        "I will work through this step by step in sequence.",
        "Proceeding iteratively, I handle one stage at a time.",
        "The approach is a recursive breakdown into ordered steps.",
    ],
}
_APPROACH_KEYS = list(APPROACH_REPLACEMENTS.keys())


def _detect_approach(step: str) -> Optional[str]:
    for name, pat in APPROACH_SIGNALS.items():
        if pat.search(step):
            return name
    return None


def approach_corruption(full_text: str, rng: random.Random) -> str:
    think, response = split_think_response(full_text)
    steps = segment_steps(response)
    if not steps:
        return full_text
    detected = _detect_approach(steps[0])
    new_cat = rng.choice([k for k in _APPROACH_KEYS if k != detected])
    steps[0] = rng.choice(APPROACH_REPLACEMENTS[new_cat])
    return _reassemble(think, "\n".join(steps))


# --------------------------------------------------------------------------- #
# Step corruption
# --------------------------------------------------------------------------- #
_INSERT_STEPS = [
    "I should also consider the inverse of this relationship.",
    "It may help to take the square of both sides at this point.",
    "Note that this quantity must also be divided by the total count.",
    "We can additionally factor out the common term here.",
]


def _sep(response: str) -> str:
    return "\n\n" if "\n\n" in response else ("\n" if "\n" in response else " ")


def step_corruption(full_text: str, rng: random.Random, mode: str = "auto") -> str:
    think, response = split_think_response(full_text)
    steps = segment_steps(response)
    if len(steps) < 3:
        return full_text
    sep = _sep(response)
    if mode == "auto":
        mode = "delete" if rng.random() < 0.7 else "shuffle"
    interior = list(range(1, len(steps) - 1))
    if mode == "delete":
        steps.pop(rng.choice(interior))
    elif mode == "shuffle":
        if len(interior) >= 2:
            i = rng.choice(interior[:-1]); steps[i], steps[i + 1] = steps[i + 1], steps[i]
        else:
            i = interior[0]; steps[i], steps[i - 1] = steps[i - 1], steps[i]
    elif mode == "insert":
        steps.insert(rng.choice(interior), rng.choice(_INSERT_STEPS))
    return _reassemble(think, sep.join(steps))


# --------------------------------------------------------------------------- #
# Conclusion corruption
# --------------------------------------------------------------------------- #
def _perturb_number(value: float, rng: random.Random) -> str:
    strat = rng.choice(["add_small", "sub_small", "mul2", "div2", "add10", "sub10"])
    out = {"add_small": value + rng.randint(1, 10), "sub_small": value - rng.randint(1, 10),
           "mul2": value * 2, "div2": value / 2,
           "add10": value + 10, "sub10": value - 10}[strat]
    return str(int(out)) if float(out).is_integer() else repr(round(out, 4))


def conclusion_corruption(full_text: str, gold: str, rng: random.Random) -> str:
    think, response = split_think_response(full_text)
    boxed = list(re.finditer(r"\\boxed\{", response))
    if boxed:
        m = boxed[-1]; i, depth = m.end(), 1
        while i < len(response) and depth > 0:
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        inner = response[m.end():i]
        try:
            wrong = _perturb_number(float(inner.strip().replace(",", "")), rng)
        except ValueError:
            try:
                wrong = _perturb_number(float(str(gold).strip().replace(",", "")), rng)
            except (ValueError, TypeError):
                wrong = (inner.strip() + "0") or "0"
        return _reassemble(think, response[:m.end()] + wrong + response[i:])
    try:
        wrong = _perturb_number(float(str(gold).strip().replace(",", "")), rng)
    except (ValueError, TypeError):
        wrong = "0"
    return _reassemble(think, response.rstrip() + f"\n\nTherefore the final answer is \\boxed{{{wrong}}}.")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def corrupt_candidate(candidate: Candidate, seed: int) -> list[CorruptedSample]:
    """Apply the corruptions this candidate is *flagged ready for* (Block 7)."""
    rng = random.Random(seed + (candidate.source_idx + 1) * 1009
                        + hash(candidate.candidate_id or candidate.full_text) % 100003)
    original = candidate.full_text
    out: list[CorruptedSample] = []

    builders = []
    if candidate.can_approach_corrupt:
        builders.append(("approach", lambda: approach_corruption(original, rng)))
    if candidate.can_step_corrupt:
        builders.append(("step", lambda: step_corruption(original, rng)))
    if candidate.can_conclusion_corrupt:
        builders.append(("conclusion", lambda: conclusion_corruption(original, candidate.gold_answer, rng)))

    for ctype, fn in builders:
        corrupted = fn()
        if corrupted and corrupted != original:
            out.append(CorruptedSample(
                original_text=original, corrupted_text=corrupted, corruption_type=ctype,
                candidate_type=candidate.candidate_type,
                contrastive_group=candidate.contrastive_group, dataset=candidate.dataset,
                corruption_priority=candidate.corruption_priority,
                hardness_score=candidate.hardness_score))
    return out


def build_corruption_dataset(candidates: list[Candidate], seed: int = 42) -> list[CorruptedSample]:
    samples: list[CorruptedSample] = []
    for c in candidates:
        if c.include_in_training:
            samples.extend(corrupt_candidate(c, seed))
    return samples
