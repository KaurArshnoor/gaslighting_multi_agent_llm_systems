"""Train per-layer capitulation probes and write a results CSV.

Two settings are reported:
  * ``use_turn=first``  -- predicts capitulation from the activation at turn 1
                           (the strict "before any expressed CoT" test).
  * ``use_turn=all``    -- pools all turns; gives more data but conflates
                           "the answer is now the false answer" with "the
                           model is *deciding* to capitulate".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.probes import assemble_probe_data, probe_per_layer
from src.utils.io import load_jsonl, load_yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--use-turn", default="first", choices=["first", "all"])
    p.add_argument("--probe-type", default="logistic", choices=["logistic", "mlp"])
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out) if args.out else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)

    exchange_files = sorted(runs_dir.rglob("exchanges.jsonl"))
    all_rows = []
    for ef in exchange_files:
        exchanges = load_jsonl(ef)
        if not exchanges:
            continue
        layers_data = assemble_probe_data(
            exchanges, ef, use_turn=args.use_turn,
        )
        # Train on early layers and middle layers separately
        early = cfg["probes"]["early_layers"]
        mid = cfg["probes"]["middle_layers"]

        for tag, layers in [("early", early), ("middle", mid)]:
            res = probe_per_layer(
                layers_data, layers=layers,
                probe_type=args.probe_type,
                permutation_n=cfg["probes"].get("permutation_n", 1000),
            )
            res["tag"] = tag
            res["run_dir"] = str(ef.parent)
            all_rows.append(res)

    if not all_rows:
        raise SystemExit("No probe data assembled.")
    out = pd.concat(all_rows, ignore_index=True)
    out_path = out_dir / f"probes_{args.use_turn}_{args.probe_type}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.groupby("tag")[["accuracy", "f1", "auc", "majority", "permutation_p"]]
              .mean().to_string())


if __name__ == "__main__":
    main()
