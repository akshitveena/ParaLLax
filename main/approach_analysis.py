"""
approach_analysis.py — Block 5, "the interpretability core", done for free.

Makes the implicit reasoning approach explicit as structured data using the
keyword classifier from RiDAE_DatasetSchema.docx Section 6 (no API cost). Claude is
only needed to disambiguate the residual 'unknown' cases (see score_candidates.py).

Outputs feed: type classification (Block 4), corruption (approach_keywords,
stated_approach_sentence), and the latent-space analysis.
"""
from __future__ import annotations

import re
from typing import Optional

from textutils import segment_steps

# Section 6.1 keyword sets (probabilistic split out from combinatorial per Block 5).
APPROACH_KEYWORDS: dict[str, list[str]] = {
    # General math vocabulary + QwQ-characteristic phrasings ("let's denote", "set ...",
    # "notice that") so QwQ candidates don't fall through to 'unknown' (Section 6.1, item 5).
    "algebraic": ["let x", "let n", "let's denote", "denote", "equation", "solve for",
                  "substitute", "variable", "expression", "polynomial", "factor",
                  "expand", "set up the equation"],
    "geometric": ["area", "perimeter", "angle", "triangle", "circle", "coordinate",
                  "distance", "parallel", "perpendicular", "vector"],
    "number_theoretic": ["divisible", "modulo", "mod", "prime", "gcd", "lcm",
                         "remainder", "congruent", "integer", "factor pairs"],
    "combinatorial": ["choose", "permutation", "combination", "count", "ways",
                      "arrange", "select", "cases"],
    "analytic": ["derivative", "integral", "limit", "converge", "function",
                 "continuous", "differentiate", "series"],
    "probabilistic": ["probability", "expected value", "random", "distribution", "odds"],
    "pattern_matching": ["pattern", "sequence", "observe", "notice that", "notice",
                         "conjecture", "first few", "list out", "try"],
}

# Self-correction / abandonment markers (QwQ self-corrects constantly). The approach
# that follows the LAST marker is the *settled* approach — what the model committed to.
ABANDON_MARKERS = [
    re.compile(r"wait,?\s+(that|this).{0,40}(wrong|doesn'?t work|isn'?t right)", re.I),
    re.compile(r"\bactually\b", re.I),
    re.compile(r"let me reconsider", re.I),
    re.compile(r"that (doesn'?t|does not) work", re.I),
    re.compile(r"let me try .{0,40}\binstead\b", re.I),
    re.compile(r"scratch that", re.I),
    re.compile(r"\bhold on\b", re.I),
    re.compile(r"\bno,\s", re.I),
]

# Section 6.2 — thinking-block approach-switch markers (highest-confidence Type B).
_SWITCH_MARKERS = [
    re.compile(r"actually,?\s+let me use", re.I),
    re.compile(r"let me try .*? instead", re.I),
    re.compile(r"wait,?\s+(that|this).{0,30}(doesn'?t work|wrong|isn'?t right)", re.I),
    re.compile(r"that approach is wrong", re.I),
]


def classify_approach(text: str) -> tuple[str, list[str]]:
    """Return (approach_label, matched_keywords[:5]).

    >=2 distinct categories match -> 'mixed'; none -> 'unknown'.
    """
    if not text:
        return "unknown", []
    low = text.lower()
    hits: dict[str, list[str]] = {}
    for label, kws in APPROACH_KEYWORDS.items():
        found = [kw for kw in kws if kw in low]
        if found:
            hits[label] = found
    if not hits:
        return "unknown", []
    if len(hits) >= 2:
        # 'mixed', but still surface up to 5 keywords for corruption vocabulary.
        flat = [kw for found in hits.values() for kw in found]
        return "mixed", flat[:5]
    label = next(iter(hits))
    return label, hits[label][:5]


def detect_thinking_switch(thinking_text: str) -> Optional[str]:
    """If the thinking block visibly abandons one approach for another, return a
    short description; else None. NOTE: a switch is NOT itself a gap — a model that
    self-corrects and then writes a response using its *settled* approach is
    faithful. Switches are used to locate the settled approach, not to declare a gap."""
    if not thinking_text:
        return None
    for pat in _SWITCH_MARKERS:
        m = pat.search(thinking_text)
        if m:
            return f"Thinking contains a self-correction marker: '{m.group(0).strip()}'."
    return None


def settled_thinking_region(thinking_text: str) -> str:
    """Return the part of the think block that reflects the FINAL settled approach.

    Text after the LAST abandonment marker; if there are none, the last quarter of
    the block (where the model has converged). This prevents misclassifying a
    self-corrected-but-faithful candidate as a thinking-response gap (schema:
    "The Think Block Quality Problem").
    """
    t = (thinking_text or "").strip()
    if not t:
        return ""
    last_end = -1
    for pat in ABANDON_MARKERS:
        for m in pat.finditer(t):
            last_end = max(last_end, m.end())
    if last_end >= 0:
        tail = t[last_end:].strip()
        if len(tail.split()) >= 5:          # enough text after the marker to classify
            return tail
    # No usable marker -> last quarter of the block.
    words = t.split()
    return " ".join(words[max(0, len(words) - max(20, len(words) // 4)):])


def settled_thinking_approach(thinking_text: str) -> tuple[str, list[str]]:
    """Approach label of the settled region of the think block."""
    return classify_approach(settled_thinking_region(thinking_text))


def stated_approach_sentence(response_text: str) -> str:
    """First sentence/clause of the response that states the approach. Verbatim."""
    steps = segment_steps(response_text)
    if not steps:
        return ""
    first = steps[0]
    # Trim a leading numbered prefix like "1. " for a clean stated sentence.
    first = re.sub(r"^\s*(?:step\s*)?\d+[\.\):]\s*", "", first, flags=re.I)
    # Keep just the first sentence.
    parts = re.split(r"(?<=[.!?])\s", first, maxsplit=1)
    return parts[0].strip()


def approaches_match(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    """Compare two approach labels. None if either is unknown/None (can't tell)."""
    if not a or not b or a == "unknown" or b == "unknown":
        return None
    return a == b
