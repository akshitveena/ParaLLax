"""
damage.py — RiDAE v3.1 component 3: the Controlled Reasoning Perturbation Engine.

WHY THIS EXISTS (empirical motivation, not architecture astrology):
Phase 1b found that every corruption we had — mask / shuffle / noise / all — produced
statistically indistinguishable representations (f1_B 0.500-0.541, all CIs overlapping), and
that "none" scored the same as "mask". We concluded the denoising objective was inert. But
look at WHAT we were corrupting: masked embedding vectors, permuted positions, Gaussian noise,
deleted words. Every one of those is a SURFACE operation. None of them damages the *reasoning*.
Of course they were interchangeable — they were all the same kind of nothing.

The v3.1 taxonomy is semantic: damage that breaks an argument rather than a string. That gives
Phase 1b a sharp, falsifiable follow-up: typed semantic damage should NOT be interchangeable,
and should not be inert. If it is, the architecture's premise is wrong and we report that.

Every damage returns structured PROVENANCE (type, operation, which steps, what changed). That
is not bookkeeping — components 7 (transformation space), 8 (recoverability) and 9 (operator
discovery) all need (clean, damaged, damage_type) triples to learn from. Building the engine
without provenance would strand every downstream module.

Six families, per the v3.1 damage taxonomy:
  structural   missing / swapped / duplicated steps            [programmatic]
  dependency   broken logical links, missing references        [programmatic]
  constraint   removed or weakened conditions and bounds       [programmatic]
  other        notation drift, unit change, numeric noise      [programmatic]
  conceptual   wrong idea / formula / theorem                  [LLM-backed — see damage_llm.py]
  abstraction  abstraction shift, representation mix           [LLM-backed]

The four programmatic families are honest but shallow proxies for their semantic categories;
conceptual and abstraction genuinely require a generator and are exposed as a seam rather than
faked with regexes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field

PROGRAMMATIC = ("structural", "dependency", "constraint", "other")
LLM_BACKED = ("conceptual", "abstraction")
ALL_TYPES = PROGRAMMATIC + LLM_BACKED


@dataclass
class DamageRecord:
    """Provenance for one applied damage. Consumed by components 7/8/9."""
    damage_type: str
    operation: str
    step_indices: list[int] = field(default_factory=list)
    detail: str = ""

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- structural
def _drop_step(steps, rng):
    if len(steps) < 3:
        return None
    i = rng.randrange(1, len(steps) - 1)          # keep first/last: setup and conclusion
    out = steps[:i] + steps[i + 1:]
    return out, DamageRecord("structural", "drop_step", [i], steps[i][:80])


def _swap_steps(steps, rng):
    if len(steps) < 3:
        return None
    i = rng.randrange(0, len(steps) - 1)
    out = list(steps); out[i], out[i + 1] = out[i + 1], out[i]
    return out, DamageRecord("structural", "swap_steps", [i, i + 1], "adjacent order inverted")


def _duplicate_step(steps, rng):
    if len(steps) < 2:
        return None
    i = rng.randrange(0, len(steps))
    out = steps[:i + 1] + [steps[i]] + steps[i + 1:]
    return out, DamageRecord("structural", "duplicate_step", [i], steps[i][:80])


# --------------------------------------------------------------------------- dependency
_REF_PATTERNS = [
    r"\b(?:from|in|by|using|per|via)\s+(?:step|equation|eq\.?|line|part)\s*\(?\d+\)?",
    r"\bstep\s*\(?\d+\)?",
    r"\b(?:the above|the previous(?:\s+step)?|previously|as shown above|from before|"
    r"as established|earlier)\b",
    r"\bsubstitut\w+\s+(?:this|that|it|the above)\b",
]


def _break_reference(steps, rng):
    cands = [(i, m) for i, s in enumerate(steps)
             for p in _REF_PATTERNS for m in re.finditer(p, s, re.I)]
    if not cands:
        return None
    i, m = cands[rng.randrange(len(cands))]
    ref = m.group(0)
    out = list(steps)
    num = re.search(r"\d+", ref)
    if num and rng.random() < 0.5:
        # misdirect the pointer: same shape, wrong target — the subtler failure
        wrong = str(max(1, int(num.group(0)) + rng.choice([-2, -1, 1, 2])))
        new_ref = ref[:num.start()] + wrong + ref[num.end():]
        op, detail = "misdirect_reference", f"{ref!r} -> {new_ref!r}"
        out[i] = out[i][:m.start()] + new_ref + out[i][m.end():]
    else:
        # sever the link entirely: the step no longer says what it builds on
        op, detail = "drop_reference", f"removed {ref!r}"
        out[i] = re.sub(r"\s{2,}", " ", out[i][:m.start()] + out[i][m.end():]).strip()
    return out, DamageRecord("dependency", op, [i], detail)


# --------------------------------------------------------------------------- constraint
_CONSTRAINT_PATTERNS = [
    r"\b(?:for all|for any|for every|for each)\b[^,.;]{2,60}",
    r"\b(?:assuming|given that|provided that|suppose that|so long as)\b[^,.;]{2,60}",
    r"\b(?:where|with)\s+[a-zA-Z]\s*(?:[<>≤≥≠=]|\\neq|\\leq|\\geq)[^,.;]{1,40}",
    r"\bif\s+[a-zA-Z][^,.;]{2,50}",
    r"\b(?:since|because)\s+[^,.;]{2,60}",
]


def _remove_constraint(steps, rng):
    cands = [(i, m) for i, s in enumerate(steps)
             for p in _CONSTRAINT_PATTERNS for m in re.finditer(p, s, re.I)]
    if not cands:
        return None
    i, m = cands[rng.randrange(len(cands))]
    out = list(steps)
    out[i] = re.sub(r"\s{2,}", " ", (out[i][:m.start()] + out[i][m.end():])).strip(" ,;")
    if len(out[i].split()) < 2:                    # don't blank a step; that's structural damage
        return None
    return out, DamageRecord("constraint", "remove_condition", [i],
                             f"dropped {m.group(0)[:70]!r}")


# --------------------------------------------------------------------------- other
def _notation_drift(steps, rng):
    """Rename a variable in exactly ONE step -> a local inconsistency, not a global rename."""
    counts = {}
    for i, s in enumerate(steps):
        for v in set(re.findall(r"(?<![A-Za-z\\])([a-zA-Z])(?![A-Za-z])", s)):
            counts.setdefault(v, []).append(i)
    multi = {v: ix for v, ix in counts.items() if len(ix) >= 2 and v not in "aAI"}
    if not multi:
        return None
    v = sorted(multi)[rng.randrange(len(multi))]
    i = multi[v][rng.randrange(len(multi[v]))]
    alt = next((c for c in "qwzkm" if c not in counts), "q")
    out = list(steps)
    out[i] = re.sub(rf"(?<![A-Za-z\\]){re.escape(v)}(?![A-Za-z])", alt, out[i])
    return out, DamageRecord("other", "notation_drift", [i], f"{v} -> {alt} in step {i} only")


def _numeric_perturb(steps, rng):
    cands = [(i, m) for i, s in enumerate(steps) for m in re.finditer(r"(?<![\w.])\d+(?![\w.])", s)]
    if not cands:
        return None
    i, m = cands[rng.randrange(len(cands))]
    old = int(m.group(0))
    new = old + rng.choice([-2, -1, 1, 2]) if old > 2 else old + rng.choice([1, 2])
    out = list(steps)
    out[i] = out[i][:m.start()] + str(new) + out[i][m.end():]
    return out, DamageRecord("other", "numeric_perturb", [i], f"{old} -> {new}")


OPS = {
    "structural": [_drop_step, _swap_steps, _duplicate_step],
    "dependency": [_break_reference],
    "constraint": [_remove_constraint],
    "other": [_notation_drift, _numeric_perturb],
}


def apply_damage(steps: list[str], damage_type: str, rng, max_tries: int = 6):
    """Apply one damage of `damage_type`. Returns (damaged_steps, DamageRecord) or None.

    None means "not applicable to this candidate" (e.g. no constraint clause present) — the
    caller must record that rather than silently substituting a different damage, otherwise the
    per-type comparison is contaminated by whatever the fallback was.
    """
    if damage_type in LLM_BACKED:
        raise NotImplementedError(
            f"'{damage_type}' is LLM-backed — see damage_llm.py. Faking it with regexes would "
            f"make it another surface operation, which is exactly what Phase 1b showed is inert.")
    ops = OPS[damage_type]
    for _ in range(max_tries):
        got = ops[rng.randrange(len(ops))](steps, rng)
        if got is not None:
            return got
    return None


def applicability(recs, rng, types=PROGRAMMATIC):
    """What fraction of the corpus each damage type can actually be applied to.

    Reported before any training run: a type applicable to only a small slice cannot be compared
    on equal footing with one that always applies, and pretending otherwise would repeat the
    Phase-1b mistake of comparing things that aren't comparable.
    """
    out = {}
    for t in types:
        ok = sum(apply_damage(r["steps_text"], t, rng) is not None for r in recs)
        out[t] = ok / max(len(recs), 1)
    return out
