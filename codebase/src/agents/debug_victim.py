"""A tiny stand-in victim for local debugging without a GPU or large weights.

Loads ``sshleifer/tiny-gpt2`` (a 4-layer toy model) so the full pipeline --
generation, hooks, probe forward pass, log-prob scoring -- can be exercised
end-to-end on CPU in seconds. Activations from this model are obviously not
meaningful; this exists so smoke tests can run.

Activate via env-var ``VICTIM_DEBUG=1`` in :func:`build_victim`.
"""

from __future__ import annotations

from .hf_victim import HFVictimBase


class DebugVictim(HFVictimBase):
    LAYERS_ATTR = "transformer.h"   # GPT-2 layout

    def __init__(self, **kwargs):
        kwargs.setdefault("hf_id", "sshleifer/tiny-gpt2")
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)

    def _format_dialogue(self, messages: list[dict], *, suffix: str = "") -> str:
        parts = []
        for m in messages:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        parts.append("ASSISTANT:")
        if suffix:
            parts.append(suffix)
        return "\n".join(parts)

    def _answer_prefix(self) -> str:
        return " My answer is "
