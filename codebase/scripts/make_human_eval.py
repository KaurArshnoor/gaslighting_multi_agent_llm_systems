"""Sample exchanges and emit CSV/HTML annotation forms for human eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.human_eval import (
    export_csv,
    export_html,
    inter_rater_agreement,
    stratified_sample,
)
from src.utils.io import load_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--n-per-condition", type=int, default=25)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score", nargs=2, metavar=("CSV_A", "CSV_B"),
                   help="instead of sampling, score agreement between two filled CSVs")
    return p.parse_args()


def main():
    args = parse_args()
    if args.score:
        out = inter_rater_agreement(*args.score)
        print(out)
        return

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out) if args.out else runs_dir / "human_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_exchanges = []
    for ef in sorted(runs_dir.rglob("exchanges.jsonl")):
        all_exchanges.extend(load_jsonl(ef))
    if not all_exchanges:
        raise SystemExit("No exchanges found.")

    sampled = stratified_sample(all_exchanges, n_per_condition=args.n_per_condition, seed=args.seed)
    print(f"Sampled {len(sampled)} exchanges across {len(set(e['condition'] for e in sampled))} conditions.")

    csv_path = export_csv(sampled, out_dir / "human_eval.csv")
    html_path = export_html(sampled, out_dir / "human_eval.html")
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    print("Have two annotators independently fill `annotations_<name>.csv`,")
    print("then run: python scripts/make_human_eval.py --score a.csv b.csv")


if __name__ == "__main__":
    main()
