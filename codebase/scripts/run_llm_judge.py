"""Run the LLM-as-a-judge classifier across the VALID exchanges in a runs dir.

Reads validity flags from ``runs/exchange_validity.csv`` (produced by
``flag_exchange_validity.py``). Only valid exchanges are sent to the LLM
judge; invalid ones get a placeholder row in ``judge.jsonl`` with null
capitulation fields, so the file stays aligned 1:1 with ``exchanges.jsonl``.

For each run dir (``runs/<victim>_v1/``) writes ``judge.jsonl`` with one
row per exchange:

    {fact_id, victim, condition, fact_type, valid,
     any_turn_capitulated, final_turn_capitulated,
     judge_type, judge_rationale}

If a ``human_labels.csv`` is present at runs root, also computes per-victim
F1 / kappa against the new human-label schema and writes
``judge_validation.json``.
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
                   help="if >0, only judge the first N valid exchanges per run")
    p.add_argument("--validity-csv", default=None,
                   help="path to exchange_validity.csv "
                        "(default: <runs-dir>/exchange_validity.csv)")
    return p.parse_args()


def _load_validity(path: Path) -> dict[tuple, bool]:
    """Map (fact_id, victim, condition) -> valid."""
    if not path.exists():
        raise SystemExit(
            f"Validity CSV not found at {path}. "
            "Run scripts/flag_exchange_validity.py first."
        )
    df = pd.read_csv(path)
    return {
        (r["fact_id"], r["victim"], r["condition"]): bool(r["valid"])
        for _, r in df.iterrows()
    }


def main():
    load_dotenv()
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    validity_csv = (
        Path(args.validity_csv) if args.validity_csv
        else runs_dir / "exchange_validity.csv"
    )
    validity = _load_validity(validity_csv)
    print(f"Loaded validity flags for {len(validity)} exchanges")

    judge = LLMJudge(JudgeConfig(model=args.model))

    for ef in sorted(runs_dir.rglob("exchanges.jsonl")):
        exchanges = load_jsonl(ef)
        if args.limit:
            valid_exs = [e for e in exchanges
                         if validity.get((e["fact_id"], e["victim_name"], e["condition"]),
                                         True)]
            valid_exs = valid_exs[:args.limit]
            valid_keys = {(e["fact_id"], e["victim_name"], e["condition"])
                          for e in valid_exs}
        else:
            valid_keys = None

        out_path = ef.parent / "judge.jsonl"
        results = []
        n_valid_judged = 0
        n_invalid_skipped = 0
        for i, ex in enumerate(exchanges):
            key = (ex["fact_id"], ex["victim_name"], ex["condition"])
            valid = validity.get(key, True)
            if valid_keys is not None and key not in valid_keys:
                # --limit dropped this one; treat as invalid for this run
                valid = False

            if not valid:
                results.append({
                    "fact_id": ex["fact_id"],
                    "victim": ex["victim_name"],
                    "condition": ex["condition"],
                    "fact_type": ex["fact"]["type"],
                    "valid": False,
                    "any_turn_capitulated": None,
                    "final_turn_capitulated": None,
                    "judge_type": "n/a",
                    "judge_rationale": "exchange flagged invalid by heuristic",
                })
                n_invalid_skipped += 1
                continue

            print(f"[{ef.parent.name}] {i+1}/{len(exchanges)} {ex['fact_id']}", flush=True)
            try:
                r = judge.judge_one(ex)
            except Exception as e:
                print(f"  ! judge error: {e}")
                results.append({
                    "fact_id": ex["fact_id"],
                    "victim": ex["victim_name"],
                    "condition": ex["condition"],
                    "fact_type": ex["fact"]["type"],
                    "valid": True,
                    "any_turn_capitulated": None,
                    "final_turn_capitulated": None,
                    "judge_type": "error",
                    "judge_rationale": f"API_ERROR: {e}",
                })
                continue

            results.append({
                "fact_id": r.fact_id,
                "victim": ex["victim_name"],
                "condition": ex["condition"],
                "fact_type": ex["fact"]["type"],
                "valid": True,
                "any_turn_capitulated": r.any_turn_capitulated,
                "final_turn_capitulated": r.final_turn_capitulated,
                "judge_type": r.type,
                "judge_rationale": r.rationale,
            })
            n_valid_judged += 1

        dump_jsonl(results, out_path)
        print(f"  wrote {out_path}: {n_valid_judged} judged, "
              f"{n_invalid_skipped} skipped (invalid)")

    # Aggregate validation against human labels (if present)
    human_csv = runs_dir / "human_labels.csv"
    if human_csv.exists():
        all_judge = []
        for jf in runs_dir.rglob("judge.jsonl"):
            all_judge.extend(load_jsonl(jf))
        human = pd.read_csv(human_csv)
        if "any_turn_capitulated" not in human.columns:
            print("WARNING: human_labels.csv lacks new schema columns; "
                  "skipping aggregate validation. Re-collect human labels "
                  "with the updated form before validating.")
            return
        jdf = pd.DataFrame(all_judge)
        # Merge on (fact_id, victim, condition), valid only
        merged = jdf[jdf["valid"] == True].merge(
            human, on=["fact_id", "victim", "condition"],
            suffixes=("_judge", "_human"),
        )
        if len(merged) == 0:
            print("No overlap between judge labels and human labels.")
            return
        out: dict = {"n": int(len(merged))}
        for axis in ["any_turn_capitulated", "final_turn_capitulated"]:
            judge_col = axis + "_judge"
            human_col = axis + "_human"
            if judge_col not in merged or human_col not in merged:
                continue
            out[axis] = f1_against_human(
                merged[judge_col].astype(int).tolist(),
                merged[human_col].fillna(0).astype(int).tolist(),
            )
        (runs_dir / "judge_validation.json").write_text(json.dumps(out, indent=2))
        print(f"\nWrote {runs_dir/'judge_validation.json'}")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
