"""
schema.py — the complete RiDAE Dataset Schema as dataclasses.

Mirrors RiDAE_DatasetSchema.docx exactly: a per-problem ProblemRecord wrapping a
list of Candidate objects, and a Candidate carrying all 9 blocks of fields.

Two on-disk shapes:
  * RAW  (data/raw/*.jsonl)       — one ProblemRecord per line (nested candidates).
                                     Written by claude_generate.py / synthetic.
  * PROCESSED (candidates.jsonl)  — one Candidate per line, flat, with the
                                     problem-level fields denormalised onto it and
                                     Blocks 3-9 computed. Written by data_pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Any


# Approach value set (Block 5)
APPROACHES = ("algebraic", "geometric", "combinatorial", "number_theoretic",
              "analytic", "probabilistic", "pattern_matching", "mixed", "unknown")

# training_label mapping (Block 8): 0=A, 1=B, 2=D (C folded into 2 for now)
TYPE_TO_LABEL = {"A": 0, "B": 1, "C": 2, "D": 2}


@dataclass
class Candidate:
    # ---- denormalised problem-level context (needed downstream) ----
    record_id: str = ""
    problem: str = ""
    gold_answer: str = ""
    gold_solution: str = ""
    dataset: str = ""
    dataset_split: str = "test"
    difficulty: str = ""
    subject: str = "other"
    source_idx: int = -1
    has_extended_thinking: bool = False

    # ---- Block 1: identity & generation metadata ----
    candidate_id: str = ""
    model: str = ""
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None     # gpt-oss diversity axis: low|medium|high
    generation_seed: int = 0
    input_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    generation_timestamp: str = ""

    # ---- Block 2: raw text blocks ----
    thinking_text: str = ""
    response_text: str = ""
    full_text: str = ""
    # Where the thinking came from. Keeps QwQ and Claude data cleanly distinguished:
    #   'claude_api_block' — architecturally separate thinking (strongest claims)
    #   'inline_think_tags'— QwQ <think>...</think>, same autoregressive pass (weaker)
    #   'none'             — no observable thinking block
    thinking_source: str = "none"

    # ---- Block 3: answer extraction & verification ----
    answer_extracted: Optional[str] = None
    answer_correct: bool = False
    answer_in_thinking: Optional[str] = None
    answer_match_method: str = "none"
    normalised_extracted: str = ""
    normalised_gold: str = ""

    # ---- Block 4: type classification ----
    candidate_type: str = "D"
    type_confidence: str = "low"
    type_source: str = "answer_only"

    # ---- Block 5: approach analysis ----
    approach_in_thinking: Optional[str] = None
    approach_in_response: str = "unknown"
    approach_matches_gold: Optional[bool] = None
    thinking_response_gap: Optional[bool] = None
    gap_description: Optional[str] = None
    stated_approach_sentence: str = ""
    approach_keywords: list[str] = field(default_factory=list)
    error_location: Optional[str] = None
    error_type: Optional[str] = None
    error_description: Optional[str] = None

    # ---- Block 6: reasoning structure ----
    reasoning_steps: list[str] = field(default_factory=list)
    num_steps: int = 0
    step_types: list[str] = field(default_factory=list)
    has_think_tags: bool = False
    think_step_count: Optional[int] = None

    # ---- Block 7: corruption-readiness flags ----
    can_approach_corrupt: bool = False
    can_step_corrupt: bool = False
    can_conclusion_corrupt: bool = False
    corruption_priority: str = "low"

    # ---- Block 8: training labels ----
    training_label: int = 2
    contrastive_group: str = ""
    is_contrastive_anchor: bool = False
    is_contrastive_negative: bool = False
    contrastive_pair_id: Optional[str] = None
    include_in_training: bool = False
    hardness_score: Optional[float] = None      # null until first training run
    # True when this candidate's contrastive partner came from a DIFFERENT model
    # (e.g. QwQ Type-B vs Claude Type-A on the same problem). The hardest, most
    # valuable pairs: the encoder cannot cheat on stylistic signatures.
    cross_model_pair: bool = False

    # ---- Block 9: quality control ----
    generation_error: Optional[str] = None
    extraction_warning: Optional[str] = None
    quality_flags: list[str] = field(default_factory=list)
    manually_reviewed: bool = False
    # True if the generation was retried (e.g. Groq rate-limit). Retries land on a
    # different server instance, so generation_seed no longer guarantees repro.
    was_retried: bool = False
    # True if a reasoning-loop ("blackhole") was detected in the CoT — gpt-oss-20B
    # sometimes repeats forever. Such candidates are excluded from training.
    blackhole_detected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ProblemRecord:
    """Top-level per-problem record (RAW shape)."""
    record_id: str
    problem: str
    gold_answer: str
    gold_solution: str
    dataset: str
    dataset_split: str
    difficulty: str
    subject: str
    source_idx: int
    has_extended_thinking: bool
    generation_date: str
    candidates: list[dict] = field(default_factory=list)   # raw candidate dicts

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def problem_fields(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "problem": self.problem,
            "gold_answer": self.gold_answer, "gold_solution": self.gold_solution,
            "dataset": self.dataset, "dataset_split": self.dataset_split,
            "difficulty": self.difficulty, "subject": self.subject,
            "source_idx": self.source_idx,
            "has_extended_thinking": self.has_extended_thinking,
        }


@dataclass
class ContrastivePair:
    contrastive_group: str
    problem: str
    gold_answer: str
    dataset: str
    type_a: Candidate
    type_b: Candidate
    cross_model: bool = False        # A and B came from different models (hardest pairs)

    def to_dict(self) -> dict:
        return {
            "contrastive_group": self.contrastive_group,
            "problem": self.problem,
            "gold_answer": self.gold_answer,
            "dataset": self.dataset,
            "type_a": self.type_a.to_dict(),
            "type_b": self.type_b.to_dict(),
            "cross_model": self.cross_model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContrastivePair":
        return cls(
            contrastive_group=d["contrastive_group"],
            problem=d["problem"],
            gold_answer=d["gold_answer"],
            dataset=d["dataset"],
            type_a=Candidate.from_dict(d["type_a"]),
            type_b=Candidate.from_dict(d["type_b"]),
            cross_model=d.get("cross_model", False),
        )
