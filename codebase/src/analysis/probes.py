"""Linear (and small MLP) probes for capitulation prediction (RQ3).

We train one probe per (victim, layer) on the victim's last-token activation
vector at turn ``t`` and predict whether the victim WILL capitulate by turn
``T_final``. The label is per-exchange (capitulated or not), and we replicate
that label across all turn-level activation vectors to give the probe more
data; we also report a strict version that uses only the *first* turn's
activation as the predictor, which is the cleanest test of "the decision
exists before the model's expressed CoT".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .stats import permutation_test


# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    layer: int
    victim: str
    condition: str
    fact_type: str
    accuracy: float
    f1: float
    auc: float
    accuracy_majority_baseline: float
    permutation_p: float
    n: int
    n_pos: int


# ---------------------------------------------------------------------------
def assemble_probe_data(
    exchanges: Iterable[dict],
    activations_dir: Path,
    *,
    use_turn: int | str = "first",     # "first" | "all" | int (1-indexed)
    capit_threshold: float = 0.5,
) -> dict:
    """Returns ``{layer: (X, y, meta)}`` keyed by layer index.

    Each ``(X, y)`` pair is suitable for sklearn-style training; ``meta`` is
    a DataFrame of per-row metadata (fact_id, victim, condition, ...).
    """
    layers_data: dict[int, dict] = {}
    activations_dir = Path(activations_dir)
    meta_rows = []
    feature_rows: dict[int, list[np.ndarray]] = {}
    label_rows: list[int] = []

    for ex in exchanges:
        if ex.get("activations_path") is None:
            continue
        # Determine label: did the victim capitulate by the final turn?
        last = ex["turns"][-1]
        capit = int(last["p_false"] > capit_threshold)

        act = np.load(activations_dir.parent / ex["activations_path"])
        # act: (n_turns, n_layers, hidden)
        if isinstance(use_turn, int):
            indices = [use_turn - 1]
        elif use_turn == "first":
            indices = [0]
        elif use_turn == "all":
            indices = list(range(act.shape[0]))
        else:
            raise ValueError(f"bad use_turn: {use_turn!r}")

        for idx in indices:
            for layer in range(act.shape[1]):
                feature_rows.setdefault(layer, []).append(act[idx, layer])
            label_rows.append(capit)
            meta_rows.append({
                "fact_id": ex["fact_id"],
                "victim": ex["victim_name"],
                "condition": ex["condition"],
                "fact_type": ex["fact"]["type"],
                "turn": idx + 1,
            })

    meta = pd.DataFrame(meta_rows)
    out: dict[int, dict] = {}
    for layer, feats in feature_rows.items():
        X = np.stack(feats, axis=0).astype(np.float32)
        y = np.array(label_rows[: len(feats)], dtype=np.int64)
        # We re-slice y by layer in case feature_rows has different lengths --
        # in practice they should all be the same.
        y = np.array(label_rows[:X.shape[0]], dtype=np.int64)
        out[layer] = {"X": X, "y": y, "meta": meta.iloc[: X.shape[0]].reset_index(drop=True)}
    return out


# ---------------------------------------------------------------------------
def train_one_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    permutation_n: int = 1000,
    probe_type: str = "logistic",
    seed: int = 0,
) -> dict:
    """Cross-validated probe with permutation test."""
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(seed)
    if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return {"accuracy": float("nan"), "f1": float("nan"),
                "auc": float("nan"), "majority": float(np.mean(y == np.bincount(y).argmax())),
                "permutation_p": float("nan")}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s, aucs = [], [], []
    y_pred_oof = np.zeros_like(y)
    for train, test in skf.split(X, y):
        if probe_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed)
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=1)
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])
        proba = clf.predict_proba(X[test])[:, 1]
        y_pred_oof[test] = pred
        accs.append(accuracy_score(y[test], pred))
        f1s.append(f1_score(y[test], pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y[test], proba))
        except ValueError:
            aucs.append(float("nan"))

    perm = permutation_test(y, y_pred_oof, n_shuffles=permutation_n, rng=rng)
    return {
        "accuracy": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "auc": float(np.nanmean(aucs)),
        "majority": float(np.mean(y == np.bincount(y).argmax())),
        "permutation_p": perm["p"],
    }


# ---------------------------------------------------------------------------
def probe_per_layer(
    layers_data: dict,
    *,
    layers: Iterable[int] | None = None,
    probe_type: str = "logistic",
    permutation_n: int = 1000,
) -> pd.DataFrame:
    """Train one probe per layer; returns a DataFrame of results."""
    rows = []
    iter_layers = layers if layers is not None else sorted(layers_data.keys())
    for layer in iter_layers:
        if layer not in layers_data:
            continue
        d = layers_data[layer]
        res = train_one_probe(
            d["X"], d["y"],
            permutation_n=permutation_n,
            probe_type=probe_type,
        )
        rows.append({"layer": layer, **res, "n": len(d["y"]),
                     "n_pos": int(d["y"].sum())})
    return pd.DataFrame(rows)
