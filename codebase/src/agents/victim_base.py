"""Abstract Victim agent.

A Victim is a local open-source model that:
  * Maintains a multi-turn message buffer (system + user/assistant turns).
  * Can generate a free-form reply to the latest user message.
  * Can produce per-turn diagnostics for analysis:
      - last-token hidden-state activation at every transformer layer,
      - P(correct_answer) and P(false_answer) under the model's logits,
      - entropy of the answer-token distribution.

Concrete subclasses (`LlamaVictim`, `PythiaVictim`, `MistralVictim`) implement
two model-specific things:

  1. ``_format_dialogue(messages, suffix=...)`` -- builds the prompt string.
  2. ``_load()`` -- loads the underlying model + tokenizer, and registers any
     activation-capture machinery (TransformerLens hooks for Llama; native
     forward hooks for Pythia / Mistral).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..utils.prompts import capitulation_probe_question, victim_system_prompt


@dataclass
class VictimOutput:
    """Result of a single ``victim.respond(...)`` call."""
    text: str                            # free-form reply
    p_correct: float                     # P(correct_answer | prompt)
    p_false: float                       # P(false_answer   | prompt)
    entropy: float                       # H of next-token distribution at answer pos
    activations: np.ndarray              # shape (n_layers, hidden_size) fp16
    raw_logprob_correct: float           # log P
    raw_logprob_false: float
    answer_prompt: str                   # the actual probe prompt used


@dataclass
class TurnActivations:
    """Just the activations for a turn, when we don't need the rest."""
    activations: np.ndarray
    layer_idx: list[int]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class VictimBase(ABC):
    """Subclass and override ``_load`` and ``_format_dialogue``."""

    def __init__(
        self,
        hf_id: str,
        *,
        device: str | None = None,
        dtype: str = "float16",
        n_layers: Optional[int] = None,
        gen: Optional[GenerationConfig] = None,
        debug: bool = False,
    ) -> None:
        self.hf_id = hf_id
        self.device = device or self._default_device()
        self.dtype = dtype
        self.n_layers_hint = n_layers
        self.gen = gen or GenerationConfig()
        self.debug = debug

        # Filled in by _load
        self.model = None
        self.tokenizer = None
        self.n_layers: int = n_layers or 0
        self.hidden_size: int = 0

        # Stateful chat buffer for the current exchange.
        self.messages: list[dict] = []

        self._load()

    # ------------------------------------------------------------------
    @staticmethod
    def _default_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Stateful chat API
    # ------------------------------------------------------------------
    def reset(self, fact: dict) -> None:
        """Begin a new exchange: install system prompt + initial briefing."""
        from ..utils.prompts import initial_briefing_message
        self.messages = [
            {"role": "system", "content": victim_system_prompt()},
            {"role": "user", "content": initial_briefing_message(fact)},
        ]
        self._current_fact = fact

    def add_attacker(self, message: str) -> None:
        """Append a new attacker message as a user-role turn."""
        self.messages.append({"role": "user", "content": message})

    # ------------------------------------------------------------------
    # Core: respond(...) does everything we need for one victim turn
    # ------------------------------------------------------------------
    def respond(self) -> VictimOutput:
        """Generate the next victim reply AND extract per-turn diagnostics.

        Order of operations:

          1. Probe forward pass on (history + "Restate your current best
             answer:" + "My answer is "). We capture last-token activations,
             compute P(correct), P(false), and entropy.
          2. Free-form generation forward pass on the same chat history.
             The reply is appended to ``self.messages`` as an assistant turn.
        """
        fact = self._current_fact

        # --- (1) probe pass -----------------------------------------------
        probe_messages = self.messages + [
            {"role": "user", "content": capitulation_probe_question(fact)}
        ]
        probe_prompt = self._format_dialogue(
            probe_messages, suffix=self._answer_prefix()
        )
        probe = self._probe_forward(probe_prompt, fact=fact)

        # --- (2) generation pass ------------------------------------------
        gen_prompt = self._format_dialogue(self.messages, suffix=self._answer_prefix())
        reply = self._generate(gen_prompt)
        self.messages.append({"role": "assistant", "content": reply})

        return VictimOutput(
            text=reply,
            p_correct=probe["p_correct"],
            p_false=probe["p_false"],
            entropy=probe["entropy"],
            activations=probe["activations"],
            raw_logprob_correct=probe["logprob_correct"],
            raw_logprob_false=probe["logprob_false"],
            answer_prompt=probe_prompt,
        )

    # ------------------------------------------------------------------
    # Probe pass -- shared implementation in terms of model-specific helpers
    # ------------------------------------------------------------------
    def _probe_forward(self, probe_prompt: str, fact: dict) -> dict:
        """Forward pass that returns activations + answer probabilities.

        Concrete subclasses provide:
          * ``_run_with_hooks(prompt)``  -> (logits, hidden_states_per_layer)
          * ``_score_continuation(prompt, continuation)`` -> sum log-prob
        """
        import torch

        logits, hidden_states = self._run_with_hooks(probe_prompt)

        # Next-token entropy from the final position.
        last_logits = logits[0, -1, :]
        log_probs = torch.log_softmax(last_logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = float(-(probs * log_probs).sum().item())

        # Score the two candidate answers as continuations.
        lp_correct = self._score_continuation(probe_prompt, fact["correct_answer"])
        lp_false = self._score_continuation(probe_prompt, fact["false_answer"])

        # Convert to (calibrated) probabilities by softmax over the two.
        # NOTE: this is P(correct) / (P(correct)+P(false)), conditional on
        # the answer being one of the two -- i.e. a *forced-choice* probability.
        z = np.logaddexp(lp_correct, lp_false)
        p_correct = float(np.exp(lp_correct - z))
        p_false = 1.0 - p_correct

        # hidden_states: list of (1, seq, hidden) -- we want the last position
        # at every layer. Stack and take :, -1, :.
        hs = torch.stack(hidden_states, dim=0)        # (n_layers+?, 1, seq, h)
        last = hs[:, 0, -1, :].to(torch.float16).cpu().numpy()
        return {
            "p_correct": p_correct,
            "p_false": p_false,
            "entropy": entropy,
            "logprob_correct": float(lp_correct),
            "logprob_false": float(lp_false),
            "activations": last,
        }

    # ------------------------------------------------------------------
    # Convenience: most chat formats end with something that primes an answer.
    # Subclasses override if they need a different prefix.
    # ------------------------------------------------------------------
    def _answer_prefix(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # Methods every concrete victim must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def _load(self) -> None:
        """Load model + tokenizer, set self.model, self.tokenizer,
        self.n_layers, self.hidden_size."""

    @abstractmethod
    def _format_dialogue(self, messages: list[dict], *, suffix: str = "") -> str:
        """Convert messages into the model's expected prompt string."""

    @abstractmethod
    def _run_with_hooks(self, prompt: str):
        """Forward pass capturing per-layer hidden states.

        Returns ``(logits, hidden_states_list)`` where ``logits`` has shape
        (1, seq, vocab) and ``hidden_states_list`` is a list of length
        ``n_layers`` of tensors with shape (1, seq, hidden_size).
        """

    @abstractmethod
    def _score_continuation(self, prompt: str, continuation: str) -> float:
        """Sum log-prob of `continuation` tokens given `prompt`."""

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Free-form generation. Returns just the new tokens, decoded."""
