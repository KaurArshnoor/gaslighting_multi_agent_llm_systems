"""Fact dataset loaders.

The on-disk JSON files are produced by ``scripts/generate_facts.py``. Each
record has the schema::

    {
      "id":             str,             # unique identifier
      "type":           "episodic" | "semantic",
      "subtype":        str,             # finer-grained category
      "context":        str,             # in-context briefing (episodic) or "" (semantic)
      "question":       str,
      "correct_answer": str,
      "false_answer":   str,             # the answer the attacker pushes
      "difficulty":     "easy" | "hard" | "n/a",
      # semantic only:
      "stem":           str (optional),  # cloze-style stem for prob scoring
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


@dataclass
class Fact:
    id: str
    type: str
    subtype: str
    context: str
    question: str
    correct_answer: str
    false_answer: str
    difficulty: str
    stem: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            id=d["id"],
            type=d["type"],
            subtype=d.get("subtype", ""),
            context=d.get("context", ""),
            question=d["question"],
            correct_answer=d["correct_answer"],
            false_answer=d["false_answer"],
            difficulty=d.get("difficulty", "n/a"),
            stem=d.get("stem", ""),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "subtype": self.subtype,
            "context": self.context,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "false_answer": self.false_answer,
            "difficulty": self.difficulty,
            "stem": self.stem,
        }


def _resolve(path: str | Path | None, default: str) -> Path:
    if path is None:
        return DATA_DIR / default
    return Path(path)


def load_facts(
    fact_type: str,
    *,
    path: str | Path | None = None,
    difficulty: str | None = None,
) -> list[Fact]:
    """Load all facts of a given type, optionally filtered by difficulty."""
    if fact_type == "episodic":
        p = _resolve(path, "episodic_facts.json")
    elif fact_type == "semantic":
        p = _resolve(path, "semantic_facts.json")
    else:
        raise ValueError(f"Unknown fact_type: {fact_type!r}")

    raw = json.loads(Path(p).read_text())
    facts = [Fact.from_dict(r) for r in raw]
    if difficulty is not None:
        facts = [f for f in facts if f.difficulty == difficulty]
    return facts


def sample_facts(
    facts: list[Fact],
    n: int,
    *,
    seed: int = 0,
) -> list[Fact]:
    if n >= len(facts):
        return list(facts)
    rng = random.Random(seed)
    return rng.sample(facts, n)
