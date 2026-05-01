"""Run the LLM-as-a-judge classifier across all exchanges in a runs dir.

Writes ``judge.jsonl`` next to each ``exchanges.jsonl``. If a
``human_labels.csv`` is present in the run dir, also reports F1 / kappa
agreement with the human labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.llm_judge import JudgeConfig, LLMJudge, f1_against_human
from src.utils.io import dump_jsonl, load_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, only judge the first N exchanges per run")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    judge = LLMJudge(JudgeConfig(model=args.model))

    for ef in sorted(runs_dir.rglob("exchanges.jsonl")):
        exchanges = load_jsonl(ef)
        if args.limit:
            exchanges = exchanges[:args.limit]
        out_path = ef.parent / "judge.jsonl"
        results = []
        for i, ex in enumerate(exchanges):
            print(f"[{ef.parent.name}] {i+1}/{len(exchanges)} {ex['fact_id']}", flush=True)
            try:
                r = judge.judge_one(ex)
            except Exception as e:
                print(f"  ! judge error: {e}")
                continue
            results.append({
                "fact_id": r.fact_id,
                "victim": ex["victim_name"],
                "condition": ex["condition"],
                "fact_type": ex["fact"]["type"],
                "judge_capitulated": r.capitulated,
                "judge_type": r.type,
                "judge_rationale": r.rationale,
            })
        dump_jsonl(results, out_path)
        print(f"  wrote {out_path}")

        # If a human-label CSV is present, compute agreement on the overlap.
        human_csv = ef.parent / "human_labels.csv"
        if human_csv.exists():
            human = pd.read_csv(human_csv)
            jdf = pd.DataFrame(results)
            merged = jdf.merge(human, on="fact_id", how="inner")
            if len(merged):
                agg = f1_against_human(
                    merged["judge_capitulated"].astype(int).tolist(),
                    merged["capitulated"].fillna(0).astype(int).tolist(),
                )
                (ef.parent / "judge_validation.json").write_text(json.dumps(agg, indent=2))
                print(f"  validation: {agg}")


if __name__ == "__main__":
    main()
