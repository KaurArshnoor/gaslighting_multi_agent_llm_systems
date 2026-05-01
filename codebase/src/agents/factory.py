"""Factory: build the right Victim based on a name string + config dict."""

from __future__ import annotations

import os
from typing import Optional

from .victim_base import GenerationConfig, VictimBase


def build_victim(
    name: str,
    cfg: dict,
    *,
    debug: Optional[bool] = None,
    device: Optional[str] = None,
) -> VictimBase:
    """Build a Victim by name. ``cfg`` is the ``victims:<name>`` block.

    If ``VICTIM_DEBUG`` is set in env (or ``debug=True``) we substitute a
    tiny GPT-2 stand-in so the pipeline can be exercised on CPU.
    """
    if debug is None:
        debug = os.environ.get("VICTIM_DEBUG", "0") == "1"

    gen = GenerationConfig()  # populated downstream from config[generation]

    if debug:
        from .debug_victim import DebugVictim
        return DebugVictim(device=device or "cpu", gen=gen, debug=True)

    name = name.lower()
    if name == "llama":
        from .llama_victim import LlamaVictim
        return LlamaVictim(
            hf_id=cfg["hf_id"],
            device=device,
            dtype=cfg.get("dtype", "float16"),
            n_layers=cfg.get("n_layers"),
            gen=gen,
        )
    if name == "pythia":
        from .pythia_victim import PythiaVictim
        return PythiaVictim(
            hf_id=cfg["hf_id"],
            device=device,
            dtype=cfg.get("dtype", "float16"),
            n_layers=cfg.get("n_layers"),
            gen=gen,
        )
    if name == "mistral":
        from .mistral_victim import MistralVictim
        return MistralVictim(
            hf_id=cfg["hf_id"],
            device=device,
            dtype=cfg.get("dtype", "float16"),
            n_layers=cfg.get("n_layers"),
            gen=gen,
        )
    raise ValueError(f"Unknown victim name: {name!r}")
