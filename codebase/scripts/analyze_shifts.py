"""Compute layer-wise activation shift heatmaps across runs.

Writes:
  - <run>/shift_table.parquet     : long-format (fact_id, victim, condition,
                                    fact_type, turn, layer, cosine_shift)
  - <run>/shift_heatmap_<victim>_<condition>.png  (per cell)
  - <run>/shift_contrast_<victim>.png             (cot - bare per (turn, layer))
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.shift import (
    condition_contrast,
    heatmap_matrix,
    shift_table,
)
from src.utils.io import load_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _save_heatmap(mat, title: str, path: Path) -> None:
    if mat.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.heatmap(mat, cmap="viridis", cbar_kws={"label": "cosine shift"})
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Turn")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out) if args.out else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exchanges_files = sorted(runs_dir.rglob("exchanges.jsonl"))
    if not exchanges_files:
        raise SystemExit(f"No exchanges.jsonl found under {runs_dir}")

    print(f"Building shift table from {len(exchanges_files)} run(s)...")
    all_rows = []
    for ef in exchanges_files:
        ex = load_jsonl(ef)
        df = shift_table(ex, ef)
        all_rows.append(df)
    import pandas as pd
    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if df.empty:
        raise SystemExit("Shift table empty.")

    parquet_path = out_dir / "shift_table.parquet"
    df.to_parquet(parquet_path)
    print(f"Wrote {parquet_path}: {len(df)} rows")

    # Per-cell heatmaps
    for (victim, condition), _ in df.groupby(["victim", "condition"]):
        mat = heatmap_matrix(df, victim=victim, condition=condition)
        _save_heatmap(mat, f"{victim} / {condition}",
                      out_dir / f"shift_heatmap_{victim}_{condition}.png")

    # CoT - bare contrast per victim
    for victim in df["victim"].unique():
        c = condition_contrast(df, victim=victim)
        if c.empty:
            continue
        plt.figure(figsize=(10, 5))
        sns.heatmap(c, cmap="RdBu_r", center=0,
                    cbar_kws={"label": "cosine shift (cot - bare)"})
        plt.title(f"{victim}: CoT - Bare shift contrast")
        plt.xlabel("Layer")
        plt.ylabel("Turn")
        plt.tight_layout()
        plt.savefig(out_dir / f"shift_contrast_{victim}.png", dpi=120)
        plt.close()
    print(f"Wrote heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
