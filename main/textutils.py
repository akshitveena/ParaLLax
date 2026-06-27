"""
textutils.py — shared text parsing used across the pipeline.

Kept dependency-free (stdlib only) so schema.py, data_pipeline.py, corruption.py
and approach_analysis.py can all import it without circular imports.

Covers: think/response splitting, answer extraction + normalisation/matching,
step segmentation, and per-step type labelling (Block 6).
"""
from __future__ import annotations

import re
from typing import Optional

# Separators (schema Block 2): full_text = thinking + "[RESPONSE]" + response,
# with the thinking block wrapped in [THINKING]...[/THINKING].
THINK_OPEN, THINK_CLOSE, RESPONSE_MARK = "[THINKING]", "[/THINKING]", "[RESPONSE]"


# --------------------------------------------------------------------------- #
# Think / response split
# --------------------------------------------------------------------------- #
_THINK_PATTERNS = [
    re.compile(r"(.*?)</think>(.*)", re.DOTALL),                         # DeepSeek-R1
    re.compile(r"\[THINKING\](.*?)\[/THINKING\](.*)", re.DOTALL),        # Claude export
]


def split_think_response(text: str) -> tuple[str, str]:
    """Return (think_text, response_text). No delimiter -> all response."""
    if not text:
        return "", ""
    for pat in _THINK_PATTERNS:
        m = pat.search(text)
        if m:
            # Strip the opening marker for either convention ([THINKING] or <think>).
            think = m.group(1).replace(THINK_OPEN, "").replace("<think>", "").strip()
            resp = m.group(2)
            if resp.startswith(RESPONSE_MARK):
                resp = resp[len(RESPONSE_MARK):]
            return think, resp.strip()
    # Unclosed <think> (e.g. truncated at max_tokens before </think>): everything
    # after the opening tag is thinking, and there is no response yet.
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[1].strip(), ""
    resp = text
    if resp.startswith(RESPONSE_MARK):
        resp = resp[len(RESPONSE_MARK):]
    return "", resp.strip()


def assemble_full_text(think_text: str, response_text: str) -> str:
    """Build the canonical full_text used as encoder input."""
    if think_text:
        return f"{THINK_OPEN}{think_text}{THINK_CLOSE}{RESPONSE_MARK}{response_text}"
    return response_text


def has_think_tags(raw_text: str) -> bool:
    return "<think>" in (raw_text or "")


# --------------------------------------------------------------------------- #
# Answer extraction / matching (Block 3)
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{")
_ANSWER_IS = re.compile(r"answer\s*(?:is|=|:)\s*\$?\.?\s*([-+]?[0-9][0-9,]*\.?[0-9]*|[^\n.]{1,40})",
                        re.IGNORECASE)
_NUM = re.compile(r"[-+]?[0-9][0-9,]*\.?[0-9]*")


def extract_last_boxed(text: str) -> Optional[str]:
    """Content of the LAST \\boxed{...}, respecting nested braces."""
    last = None
    for m in _BOXED.finditer(text or ""):
        i, depth, out = m.end(), 1, []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(c); i += 1
        last = "".join(out).strip()
    return last


def extract_answer(text: str) -> tuple[Optional[str], str]:
    """Return (answer, method). method in {boxed, answer_is, last_number, none}."""
    if not text:
        return None, "none"
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed, "boxed"
    m = _ANSWER_IS.search(text)
    if m:
        return m.group(1).strip(), "answer_is"
    tail = "\n".join(text.strip().splitlines()[-3:])
    nums = _NUM.findall(tail)
    if nums:
        return nums[-1].strip(), "last_number"
    return None, "none"


def normalise_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    s = str(ans).strip().replace("$", "").replace(",", "")
    s = s.replace("\\%", "").replace("%", "").replace("\\!", "").replace("\\,", "")
    s = s.strip("$ .")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else repr(f)
    except (ValueError, TypeError):
        return s


def _rhs(s: str) -> str:
    """Right-hand side of the last '=' — turns 'C = 506' / 'x = 5' into '506' / '5'.

    Helps numeric competition answers stated as an assignment. (Symbolic/multi-form
    answers still need a math-equivalence checker or LLM judge — handled upstream.)
    """
    s = str(s or "")
    return s.rsplit("=", 1)[-1].strip() if "=" in s else s


def answers_match(extracted: Optional[str], gold: str) -> tuple[bool, str]:
    """Return (match, method). method in {exact, float, normalised, none}.

    Tries the raw strings and the post-'=' RHS of each, so 'C = 506' matches '506'.
    """
    e_raw, g_raw = str(extracted or "").strip(), str(gold or "").strip()
    if e_raw == "" and g_raw == "":
        return False, "none"
    if e_raw == g_raw and e_raw != "":
        return True, "exact"
    e_vars = {normalise_answer(extracted), normalise_answer(_rhs(e_raw))}
    g_vars = {normalise_answer(gold), normalise_answer(_rhs(g_raw))}
    if e_vars & g_vars - {""}:
        return True, "normalised"
    for e in e_vars:
        for g in g_vars:
            try:
                if e and g and abs(float(e) - float(g)) < 1e-3:
                    return True, "float"
            except (ValueError, TypeError):
                continue
    return False, "none"


def is_numeric_answer(ans: Optional[str]) -> bool:
    try:
        float(normalise_answer(ans))
        return True
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Step segmentation + typing (Block 6)
# --------------------------------------------------------------------------- #
_NUMBERED = re.compile(r"(?m)^\s*(?:step\s*)?(\d+)[\.\):]\s+", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")


def segment_steps(text: str) -> list[str]:
    """Numbered steps -> paragraphs -> lines -> sentences."""
    text = (text or "").strip()
    if not text:
        return []
    if len(_NUMBERED.findall(text)) >= 2:
        marks = list(_NUMBERED.finditer(text))
        parts = []
        if marks[0].start() > 0:
            pre = text[:marks[0].start()].strip()
            if pre:
                parts.append(pre)
        for j, m in enumerate(marks):
            end = marks[j + 1].start() if j + 1 < len(marks) else len(text)
            seg = text[m.start():end].strip()
            if seg:
                parts.append(seg)
        if len(parts) >= 2:
            return parts
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 2:
        return paras
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2:
        return lines
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents if sents else [text]


_VERIFY = re.compile(r"\b(verify|check|confirm|sanity|double[- ]check)\b", re.I)
_CONCLUDE = re.compile(r"\b(therefore|thus|hence|answer is|final answer|\\boxed)\b", re.I)
_APPROACH = re.compile(r"\b(i will|let me|i'?ll|approach|method|strategy|plan to|let'?s)\b", re.I)
_SETUP = re.compile(r"\b(given|we have|suppose|let\s+[a-z]|denote|define)\b", re.I)


def has_repeated_ngram(text: str, n: int = 20, max_repeat: int = 3) -> bool:
    """True if any n-token window appears more than `max_repeat` times.

    Catches QwQ's temperature-1.1 degenerate loops (repetitive / circular output).
    """
    toks = (text or "").split()
    if len(toks) < n * 2:
        return False
    from collections import Counter
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return any(v > max_repeat for v in grams.values())


def is_incomplete_response(text: str) -> bool:
    """True if the response has no \\boxed{} AND does not end on a complete thought.

    Catches generations that ran out of tokens mid-reasoning (truncated Type D).
    """
    t = (text or "").strip()
    if not t:
        return True
    if "\\boxed{" in t:
        return False
    last = t.splitlines()[-1].strip()
    if not last.endswith((".", "!", "?", "}", "$")):
        return True
    return len(last.split()) < 4


_CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def has_cjk(text: str) -> bool:
    """True if the text contains Chinese characters. R1 distils sometimes language-mix
    in the CoT — such candidates are valid but the English approach-classifier is
    unreliable on them, so they get flagged (not excluded)."""
    return bool(_CJK.search(text or ""))


def classify_step_type(step: str, idx: int, n: int) -> str:
    """One of setup|approach_statement|computation|verification|conclusion."""
    if idx == n - 1 or _CONCLUDE.search(step):
        return "conclusion"
    if _VERIFY.search(step):
        return "verification"
    if idx == 0 and _APPROACH.search(step):
        return "approach_statement"
    if _APPROACH.search(step):
        return "approach_statement"
    if _SETUP.search(step) or idx == 0:
        return "setup"
    return "computation"
