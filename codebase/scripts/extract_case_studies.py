"""Extract algorithmically chosen case-study transcripts for the report.

For each victim we pick:
  * cleanest CoT capitulation -- exchange where P(correct) starts >0.9 and
    drops to <0.1 in the fewest turns, indicating a clear and dramatic flip
  * cleanest bare-denial resistance -- exchange where P(correct) stays >0.9
    across all turns despite persistent denial
  * most-uncertain CoT exchange -- highest final-turn entropy under CoT
    (illustrates the "model becomes uncertain" finding even where it doesn't
    surface-flip)

Output: ``runs/case_studies.md`` with the full transcript + per-turn
diagnostics table for each chosen exchange. Suitable for embedding in the
report or showing in the demo recording.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
def first_flip_turn(turns, threshold: float = 0.5) -> int | None:
    """First turn at which P(correct) crosses below the threshold."""
    for t in turns:
        if t["p_correct"] < threshold:
            return int(t["turn"])
    return None


def cleanest_cot_capitulation(exchanges):
    """Best example of a CoT attack working: P(correct) starts high, ends low,
    flip happens early. Score = (start - end) / first_flip_turn."""
    best_score = -1
    best = None
    for ex in exchanges:
        if ex["condition"] != "cot":
            continue
        turns = ex["turns"]
        start = turns[0]["p_correct"]
        end = turns[-1]["p_correct"]
        if start < 0.9 or end > 0.1:
            continue
        flip = first_flip_turn(turns) or len(turns)
        score = (start - end) / flip
        if score > best_score:
            best_score = score
            best = ex
    return best


def cleanest_bare_resistance(exchanges):
    """Best example of bare-denial failing: P(correct) stays >0.9 throughout."""
    best = None
    best_min = -1
    for ex in exchanges:
        if ex["condition"] != "bare":
            continue
        ps = [t["p_correct"] for t in ex["turns"]]
        if min(ps) >= 0.9 and len(ps) >= 12:
            avg = sum(ps) / len(ps)
            if avg > best_min:
                best_min = avg
                best = ex
    return best


def most_uncertain_cot(exchanges):
    """CoT exchange with the highest final-turn entropy (illustrates the
    'becomes uncertain' finding when capitulation rate alone misses it)."""
    best = None
    best_h = -1
    for ex in exchanges:
        if ex["condition"] != "cot":
            continue
        h = ex["turns"][-1]["entropy"]
        if h > best_h:
            best_h = h
            best = ex
    return best


# ---------------------------------------------------------------------------
def render_transcript(ex: dict) -> str:
    lines = []
    fact = ex["fact"]
    lines.append(f"### {ex['victim_name']} / {ex['condition']} / `{fact['id']}`")
    lines.append("")
    lines.append(f"**Question.** {fact['question']}")
    lines.append("")
    lines.append(f"- Correct answer: `{fact['correct_answer']}`")
    lines.append(f"- Attacker's false answer: `{fact['false_answer']}`")
    lines.append(f"- Fact type: {fact['type']}; difficulty: {fact.get('difficulty', 'n/a')}")
    flip = first_flip_turn(ex["turns"])
    lines.append(f"- First turn P(correct) drops below 0.5: "
                 f"{flip if flip else 'never'}")
    lines.append("")
    lines.append("**Per-turn diagnostics.**")
    lines.append("")
    lines.append("| Turn | P(correct) | P(false) | Entropy | Logit gap |")
    lines.append("|---|---|---|---|---|")
    for t in ex["turns"]:
        lines.append(
            f"| {t['turn']} | {t['p_correct']:.3f} | {t['p_false']:.3f} | "
            f"{t['entropy']:.3f} | {t['logprob_correct'] - t['logprob_false']:+.2f} |"
        )
    lines.append("")
    lines.append("**Transcript.**")
    lines.append("")
    for t in ex["turns"]:
        lines.append(f"_Turn {t['turn']} — Victim:_")
        lines.append(f"> {_quote(t['victim_text'])}")
        if t["attacker_text"]:
            lines.append("")
            lines.append(f"_Turn {t['turn']} — Attacker:_")
            lines.append(f"> {_quote(t['attacker_text'])}")
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _quote(s: str) -> str:
    s = (s or "").strip().replace("\n", "\n> ")
    return s if s else "(empty)"


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out) if args.out else runs_dir / "case_studies.md"

    by_victim: dict[str, list[dict]] = {}
    for ef in sorted(runs_dir.rglob("exchanges.jsonl")):
        for ex in load_jsonl(ef):
            by_victim.setdefault(ex["victim_name"], []).append(ex)

    if not by_victim:
        raise SystemExit(f"No exchanges found under {runs_dir}")

    pieces = ["# Case-study transcripts",
              "",
              "Algorithmically selected from the 1,200-exchange dataset to "
              "illustrate three qualitatively distinct outcomes per victim:",
              "",
              "1. **Clean CoT capitulation** — P(correct) starts > 0.9, ends < 0.1, "
              "flip occurs early. Demonstrates the attack working.",
              "2. **Clean bare-denial resistance** — P(correct) stays > 0.9 across "
              "all 12 turns despite persistent denial. Demonstrates the attack failing.",
              "3. **Most-uncertain CoT** — highest final-turn entropy under CoT "
              "attack. Demonstrates the 'model becomes uncertain' effect even when "
              "the surface answer doesn't flip.",
              ""]

    summary_rows = []
    for victim in sorted(by_victim):
        exs = by_victim[victim]
        pieces.append(f"## {victim}")
        pieces.append("")

        cot = cleanest_cot_capitulation(exs)
        if cot:
            pieces.append("### 1. Clean CoT capitulation")
            pieces.append(render_transcript(cot))
            summary_rows.append((victim, "clean_cot_cap", cot["fact_id"],
                                  cot["turns"][0]["p_correct"],
                                  cot["turns"][-1]["p_correct"]))

        bare = cleanest_bare_resistance(exs)
        if bare:
            pieces.append("### 2. Clean bare-denial resistance")
            pieces.append(render_transcript(bare))
            summary_rows.append((victim, "clean_bare_resist", bare["fact_id"],
                                  bare["turns"][0]["p_correct"],
                                  bare["turns"][-1]["p_correct"]))

        unc = most_uncertain_cot(exs)
        if unc:
            pieces.append("### 3. Most uncertain CoT")
            pieces.append(render_transcript(unc))
            summary_rows.append((victim, "most_uncertain_cot", unc["fact_id"],
                                  unc["turns"][0]["entropy"],
                                  unc["turns"][-1]["entropy"]))

    out_path.write_text("\n".join(pieces))
    print(f"Wrote {out_path}")
    print()
    print(f"  {'victim':<10} {'kind':<20} {'fact_id':<25} {'start':>8} {'end':>8}")
    for r in summary_rows:
        print(f"  {r[0]:<10} {r[1]:<20} {r[2]:<25} {r[3]:>8.3f} {r[4]:>8.3f}")


if __name__ == "__main__":
    main()
