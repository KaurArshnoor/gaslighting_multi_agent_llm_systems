"""LLM-as-a-judge capitulation classifier.

Uses GPT-4o (configurable) to read a transcript and emit structured JSON:

  {"capitulated": bool,
   "type": "paraphrase"|"independent"|"concede"|"n/a",
   "rationale": str}

The judge is intentionally separated from the probabilistic
P(false)>threshold rule so we have two independent capitulation signals:

  * Probabilistic (token-level): from the victim's own logits.
  * Behavioral / linguistic: from the judge's reading.

We validate the judge against human labels on a 50-instance pilot and
report agreement (Cohen's kappa, F1).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.prompts import judge_system_prompt, judge_user_prompt


@dataclass
class JudgeConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_output_tokens: int = 300
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class JudgeResult:
    fact_id: str
    capitulated: bool
    type: str
    rationale: str
    raw: str


class LLMJudge:
    def __init__(self, cfg: JudgeConfig | None = None):
        self.cfg = cfg or JudgeConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get(self.cfg.api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing {self.cfg.api_key_env}")
            self._client = OpenAI(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def judge_one(self, exchange: dict) -> JudgeResult:
        client = self._get_client()
        transcript = []
        for t in exchange["turns"]:
            transcript.append({"role": "victim", "content": t["victim_text"]})
            if t["attacker_text"]:
                transcript.append({"role": "attacker", "content": t["attacker_text"]})

        resp = client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_output_tokens,
            messages=[
                {"role": "system", "content": judge_system_prompt()},
                {"role": "user", "content": judge_user_prompt(exchange["fact"], transcript)},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = self._parse(raw)
        return JudgeResult(
            fact_id=exchange["fact_id"],
            capitulated=bool(parsed.get("capitulated", False)),
            type=str(parsed.get("type", "n/a")),
            rationale=str(parsed.get("rationale", "")),
            raw=raw,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Last-ditch: extract a {...} substring
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {"capitulated": False, "type": "n/a", "rationale": "PARSE_ERROR"}


# ---------------------------------------------------------------------------
# Validation helpers (used by scripts/run_llm_judge.py)
# ---------------------------------------------------------------------------
def cohens_kappa(y_a: list[int], y_b: list[int]) -> float:
    """Cohen's kappa between two raters on binary labels."""
    import numpy as np
    a = np.array(y_a)
    b = np.array(y_b)
    po = float((a == b).mean())
    classes = np.unique(np.concatenate([a, b]))
    pe = 0.0
    n = len(a)
    for c in classes:
        pe += (a == c).mean() * (b == c).mean()
    return (po - pe) / (1 - pe + 1e-12)


def f1_against_human(judge_labels: list[int], human_labels: list[int]) -> dict:
    from sklearn.metrics import f1_score, accuracy_score
    return {
        "accuracy": accuracy_score(human_labels, judge_labels),
        "f1": f1_score(human_labels, judge_labels, zero_division=0),
        "kappa": cohens_kappa(human_labels, judge_labels),
        "n": len(human_labels),
    }
