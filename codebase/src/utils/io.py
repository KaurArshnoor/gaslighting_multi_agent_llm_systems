"""Lightweight I/O helpers used across the codebase."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# JSONL helpers (transcripts, judge outputs, metrics rows)
# ---------------------------------------------------------------------------
def dump_jsonl(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Run directory layout
# ---------------------------------------------------------------------------
def ensure_run_dir(out_dir: str | Path) -> dict[str, Path]:
    """Create the standard run-directory layout and return its paths.

    Layout::

        out_dir/
          transcripts.jsonl     # one record per turn
          exchanges.jsonl       # one record per (fact, condition) exchange
          activations/          # per-exchange .npy files
          judge.jsonl           # LLM judge outputs (one row per exchange)
          metrics.json          # aggregate metrics
          config.yaml           # snapshot of config used for this run
    """
    out_dir = Path(out_dir)
    (out_dir / "activations").mkdir(parents=True, exist_ok=True)
    return {
        "root": out_dir,
        "transcripts": out_dir / "transcripts.jsonl",
        "exchanges": out_dir / "exchanges.jsonl",
        "activations": out_dir / "activations",
        "judge": out_dir / "judge.jsonl",
        "metrics": out_dir / "metrics.json",
        "config": out_dir / "config.yaml",
    }
