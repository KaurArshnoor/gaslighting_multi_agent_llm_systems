"""Layer-wise activation shift analysis (RQ2).

For each exchange we have an activations tensor of shape
``(n_turns, n_layers, hidden_size)``. This module computes:

  * cosine_distance(a_t, a_{t-1}) per layer per turn
  * mean shift profile per condition (averaged across exchanges)
  * shift_t = max_layer cosine_distance(a_t, a_1) -- per-turn divergence
    from the initial state of the exchange
  * a contrast measure between CoT-backed and bare-denial conditions
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """1 - cosine similarity, computed in fp32 for stability."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    num = (a * b).sum(axis=axis)
    den = (np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)) + 1e-8
    return 1.0 - num / den


def per_exchange_shift(
    activations: np.ndarray,
    *,
    relative_to: str = "previous",
) -> np.ndarray:
    """Layer-wise cosine distance per turn.

    activations: (n_turns, n_layers, hidden)
    Returns: (n_turns - 1, n_layers) of cosine-distance values.
    """
    if activations.ndim != 3:
        raise ValueError(f"expected 3D activations, got {activations.shape}")
    n_turns, n_layers, _ = activations.shape
    out = np.zeros((n_turns - 1, n_layers), dtype=np.float32)
    for t in range(1, n_turns):
        ref = activations[t - 1] if relative_to == "previous" else activations[0]
        out[t - 1] = cosine_distance(activations[t], ref, axis=-1)
    return out


# ---------------------------------------------------------------------------
def load_exchange_activations(activations_dir: str | Path, rel_path: str) -> np.ndarray:
    return np.load(Path(activations_dir).parent / rel_path)


def shift_table(
    exchanges: Iterable[dict],
    activations_dir: str | Path,
    *,
    relative_to: str = "previous",
) -> pd.DataFrame:
    """Long-format table of layer-wise shift values across all exchanges.

    Columns: fact_id, victim, condition, fact_type, turn, layer, cosine_shift.
    """
    activations_dir = Path(activations_dir)
    rows = []
    for ex in exchanges:
        if ex.get("activations_path") is None:
            continue
        act = np.load(activations_dir.parent / ex["activations_path"])
        shift = per_exchange_shift(act, relative_to=relative_to)
        n_steps, n_layers = shift.shape
        for t in range(n_steps):
            for layer in range(n_layers):
                rows.append({
                    "fact_id": ex["fact_id"],
                    "victim": ex["victim_name"],
                    "condition": ex["condition"],
                    "fact_type": ex["fact"]["type"],
                    "turn": t + 2,                # shift[0] compares turns 1->2
                    "layer": layer,
                    "cosine_shift": float(shift[t, layer]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def heatmap_matrix(
    df: pd.DataFrame,
    *,
    victim: str,
    condition: str,
    fact_type: str | None = None,
) -> pd.DataFrame:
    """Build a (turn x layer) mean shift matrix for a given cell."""
    sub = df[(df["victim"] == victim) & (df["condition"] == condition)]
    if fact_type is not None:
        sub = sub[sub["fact_type"] == fact_type]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["turn", "layer"], as_index=False)["cosine_shift"]
           .mean()
           .pivot(index="turn", columns="layer", values="cosine_shift")
    )


def condition_contrast(
    df: pd.DataFrame,
    *,
    victim: str,
    fact_type: str | None = None,
) -> pd.DataFrame:
    """Mean cosine shift in CoT minus mean in bare, per (turn, layer).

    Positive values = CoT attack induces a larger representational shift than
    bare denial at that (turn, layer) cell.
    """
    cot = heatmap_matrix(df, victim=victim, condition="cot", fact_type=fact_type)
    bare = heatmap_matrix(df, victim=victim, condition="bare", fact_type=fact_type)
    if cot.empty or bare.empty:
        return pd.DataFrame()
    return cot.subtract(bare, fill_value=0.0)
