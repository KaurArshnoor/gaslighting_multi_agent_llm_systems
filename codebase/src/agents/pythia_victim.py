"""Pythia-6.9B victim agent (EleutherAI).

Pythia is a *base* (non-instruction-tuned) GPT-NeoX model. It does not have a
chat template, so we use a transcript-style prompt format that resembles
multi-agent debate. The same format is used uniformly by all agents in
exchanges so all probe positions are aligned.
"""

from __future__ import annotations

from .hf_victim import HFVictimBase


def _plain_chat_format(messages: list[dict], *, suffix: str = "") -> str:
    """Plain-text transcript suitable for base GPT-style models."""
    parts: list[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        if role == "system":
            parts.append(f"Instructions:\n{content}\n")
        elif role == "user":
            # Both initial briefings AND attacker pushbacks come in as "user".
            parts.append(f"Critic:\n{content}\n")
        elif role == "assistant":
            parts.append(f"Defender:\n{content}\n")
    parts.append("Defender:\n")
    if suffix:
        parts.append(suffix)
    return "\n".join(parts)


class PythiaVictim(HFVictimBase):
    LAYERS_ATTR = "gpt_neox.layers"

    def _format_dialogue(self, messages: list[dict], *, suffix: str = "") -> str:
        return _plain_chat_format(messages, suffix=suffix)

    def _answer_prefix(self) -> str:
        return "My current answer is: "

    def _trim_after_generation(self, text: str) -> str:
        # Cut off if the model starts hallucinating the next "Critic:" turn.
        for marker in ("\nCritic:", "\nInstructions:", "\nDefender:"):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text
