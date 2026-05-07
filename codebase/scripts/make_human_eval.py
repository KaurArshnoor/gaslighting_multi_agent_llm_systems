"""Sample exchanges and emit CSV/HTML annotation forms for human eval (v2).

The v2 form collects two binary capitulation labels (any-turn, final-turn)
plus the type. Invalid exchanges (per ``runs/exchange_validity.csv``) are
filtered out at sample time. If prior annotations from an earlier round
exist (``runs/human_eval/annotations_*.csv``), the form pre-populates the
any-turn and type columns from them so annotators only add the new
final-turn judgment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

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
    p.add_argument("--valid-only", action="store_true", default=True,
                   help="restrict sample to exchanges flagged valid (default)")
    p.add_argument("--include-invalid", dest="valid_only", action="store_false",
                   help="don't filter; sample from all exchanges")
    p.add_argument("--score", nargs=2, metavar=("CSV_A", "CSV_B"),
                   help="instead of sampling, score agreement between two filled CSVs")
    return p.parse_args()


def _load_valid_keys(runs_dir: Path) -> set[tuple]:
    """Return set of (fact_id, victim, condition) tuples flagged valid."""
    p = runs_dir / "exchange_validity.csv"
    if not p.exists():
        raise SystemExit(
            f"{p} not found. Run scripts/flag_exchange_validity.py first."
        )
    df = pd.read_csv(p)
    df = df[df["valid"] == True]
    return {(r["fact_id"], r["victim"], r["condition"]) for _, r in df.iterrows()}


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

    valid_keys = _load_valid_keys(runs_dir) if args.valid_only else None
    if valid_keys is not None:
        print(f"Restricting to {len(valid_keys)} valid exchanges "
              f"(of {len(all_exchanges)} total).")

    sampled = stratified_sample(
        all_exchanges,
        n_per_condition=args.n_per_condition,
        seed=args.seed,
        valid_keys=valid_keys,
    )
    print(f"Sampled {len(sampled)} exchanges across "
          f"{len(set(e['condition'] for e in sampled))} conditions.")

    # Find prior annotation CSVs to pre-populate from
    prior_csvs = sorted(out_dir.glob("annotations_*.csv"))
    # Don't include any *_v2.csv files (those are output of this round)
    prior_csvs = [p for p in prior_csvs if "_v2" not in p.stem]
    if prior_csvs:
        print(f"Pre-populating from prior annotations: "
              f"{', '.join(p.name for p in prior_csvs)}")

    csv_path = export_csv(sampled, out_dir / "human_eval_v2.csv",
                           prior_csv_paths=prior_csvs)
    html_path = export_html(sampled, out_dir / "human_eval_v2.html",
                             prior_csv_paths=prior_csvs)
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    print()
    print("Open the HTML in a browser, fill in any-turn + final-turn + type for")
    print("each exchange, and click Export CSV. Save as annotations_<name>_v2.csv.")
    print("Then score agreement with:")
    print("  python scripts/make_human_eval.py --score a_v2.csv b_v2.csv")


if __name__ == "__main__":
    main()
