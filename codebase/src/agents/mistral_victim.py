"""Mistral-7B-v0.1 victim agent (MistralAI).

Mistral-7B-v0.1 is a *base* model (not the Instruct variant). It does not
have a chat template, but it is known to respond reasonably to a [INST]...
[/INST] format because that matches the Mistral instruct family it was the
seed for. We use a Llama-2-style INST format here.
"""

from __future__ import annotations

from .hf_victim import HFVictimBase


def _mistral_v01_format(messages: list[dict], *, suffix: str = "") -> str:
    """Approximate Mistral [INST]...[/INST] format for the v0.1 base model."""
    parts: list[str] = []
    sys = ""
    for m in messages:
        if m["role"] == "system":
            sys = m["content"].strip()
            break

    pending_user: list[str] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            pending_user.append(m["content"].strip())
        elif m["role"] == "assistant":
            user_block = "\n".join(pending_user)
            if sys and not parts:
                parts.append(f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user_block} [/INST] {m['content'].strip()} </s>")
            else:
                parts.append(f"<s>[INST] {user_block} [/INST] {m['content'].strip()} </s>")
            pending_user = []
    # Open the final assistant turn.
    if pending_user:
        user_block = "\n".join(pending_user)
        if sys and not parts:
            parts.append(f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user_block} [/INST] ")
        else:
            parts.append(f"<s>[INST] {user_block} [/INST] ")
    if suffix:
        parts.append(suffix)
    return "".join(parts)


class MistralVictim(HFVictimBase):
    LAYERS_ATTR = "model.layers"

    def _format_dialogue(self, messages: list[dict], *, suffix: str = "") -> str:
        return _mistral_v01_format(messages, suffix=suffix)

    def _answer_prefix(self) -> str:
        return "My current answer is: "

    def _trim_after_generation(self, text: str) -> str:
        # Mistral base will sometimes start a new [INST] block.
        for marker in ("[INST]", "</s>", "<s>"):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text
