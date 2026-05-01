"""OpenAI-based Attacker agent.

The attacker maintains a small message buffer for one exchange. Each call to
``reply`` takes the latest victim response, appends it, and returns the next
attacker message. Two attack conditions are supported:

* ``"bare"``  -- denial without supporting reasoning.
* ``"cot"``   -- fabricated multi-step chain-of-thought that arrives at the
                 false answer. Specificity is controllable: "detailed" or
                 "vague".

The system prompt that controls strategy lives in ``src.utils.prompts``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.prompts import (
    attacker_first_message,
    attacker_system_prompt,
)


@dataclass
class AttackerConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_output_tokens: int = 400
    api_key_env: str = "OPENAI_API_KEY"
    organization: Optional[str] = None


@dataclass
class OpenAIAttacker:
    """Stateful attacker for one exchange.

    Construct one instance per (fact, condition) exchange. The system prompt
    is baked at construction time, and ``messages`` is mutated as the
    dialogue proceeds.
    """
    fact: dict
    condition: str               # "bare" | "cot"
    specificity: str = "detailed"
    config: AttackerConfig = field(default_factory=AttackerConfig)
    messages: list[dict] = field(default_factory=list)
    _client: object = None       # lazy

    def __post_init__(self) -> None:
        self.messages = [
            {
                "role": "system",
                "content": attacker_system_prompt(
                    fact=self.fact,
                    condition=self.condition,
                    specificity=self.specificity,
                ),
            }
        ]

    # ------------------------------------------------------------------
    # Lazy client (so importing this module never requires openai+key)
    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing OPENAI API key. Set ${self.config.api_key_env} in .env"
                )
            self._client = OpenAI(
                api_key=api_key,
                organization=self.config.organization,
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def opening(self) -> str:
        """The attacker's first message after seeing the victim's first answer."""
        msg = attacker_first_message(self.fact)
        self.messages.append({"role": "assistant", "content": msg})
        return msg

    def reply(self, victim_message: str) -> str:
        """Send the victim's latest message; return the attacker's next reply."""
        # Append victim's message in the role the attacker thinks of as "user"
        self.messages.append({"role": "user", "content": victim_message})
        out = self._call_api()
        self.messages.append({"role": "assistant", "content": out})
        return out

    # ------------------------------------------------------------------
    # Internal: API call with retry
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _call_api(self) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.config.model,
            messages=self.messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        return resp.choices[0].message.content.strip()
