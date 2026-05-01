"""End-to-end experiment driver.

Runs the 2x2 factorial against a single victim model. For all three victims,
invoke this script three times (one per --victim).

Example::

    python scripts/run_experiment.py \\
        --victim llama \\
        --conditions bare cot \\
        --fact-types episodic semantic \\
        --n-exchanges 100 \\
        --turns 12 \\
        --out runs/llama_v1
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.agents import AttackerConfig, build_victim, GenerationConfig
from src.data import load_facts, sample_facts
from src.dialogue import DialogueRunner
from src.dialogue.runner import NoAttackRunner
from src.utils.io import ensure_run_dir, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--victim", required=True, choices=["llama", "pythia", "mistral"])
    p.add_argument("--conditions", nargs="+", default=["bare", "cot"],
                   choices=["bare", "cot"])
    p.add_argument("--fact-types", nargs="+", default=["episodic", "semantic"],
                   choices=["episodic", "semantic"])
    p.add_argument("--n-exchanges", type=int, default=100,
                   help="exchanges per (condition, fact_type) cell")
    p.add_argument("--turns", type=int, default=12, help="turns per exchange (10-15)")
    p.add_argument("--out", required=True, help="output run directory")
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--specificity", default="detailed", choices=["detailed", "vague"])
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--include-control", action="store_true",
                   help="also run a no-attack control cell (for shift baseline)")
    p.add_argument("--debug", action="store_true",
                   help="use the tiny CPU debug victim instead of the real one")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    cfg = load_yaml(args.config)
    random.seed(args.seed)

    paths = ensure_run_dir(args.out)
    # snapshot config
    paths["config"].write_text(Path(args.config).read_text())

    # ---- build victim ------------------------------------------------------
    victim_cfg = cfg["victims"][args.victim]
    print(f"Loading victim: {args.victim} ({victim_cfg['hf_id']})", flush=True)
    victim = build_victim(args.victim, victim_cfg, debug=args.debug)
    # apply generation config
    gcfg = cfg.get("generation", {})
    victim.gen = GenerationConfig(
        max_new_tokens=gcfg.get("max_new_tokens", 200),
        temperature=gcfg.get("temperature", 0.7),
        top_p=gcfg.get("top_p", 0.95),
        do_sample=gcfg.get("do_sample", True),
    )

    # ---- attacker config ---------------------------------------------------
    acfg = cfg["attacker"]
    attacker_cfg = AttackerConfig(
        model=acfg.get("model", "gpt-4o-mini"),
        temperature=acfg.get("temperature", 0.7),
        max_output_tokens=acfg.get("max_output_tokens", 400),
    )

    runner = DialogueRunner(
        victim=victim,
        victim_name=args.victim,
        attacker_cfg=attacker_cfg,
        n_turns=args.turns,
        run_dir=args.out,
    )

    # ---- iterate factorial cells -------------------------------------------
    for fact_type in args.fact_types:
        all_facts = load_facts(fact_type)
        sampled = sample_facts(all_facts, args.n_exchanges, seed=args.seed)
        print(f"\n=== {fact_type}: {len(sampled)} facts ===", flush=True)
        for cond in args.conditions:
            print(f"\n--- condition={cond} ---", flush=True)
            for i, fact in enumerate(sampled):
                fact_dict = fact.to_dict()
                print(f"[{i+1}/{len(sampled)}] {fact_dict['id']}", flush=True)
                runner.run_exchange(
                    fact_dict, condition=cond, specificity=args.specificity,
                )

    # ---- optional no-attack control ---------------------------------------
    if args.include_control:
        ctl = NoAttackRunner(
            victim=victim, victim_name=args.victim,
            attacker_cfg=attacker_cfg, n_turns=args.turns, run_dir=args.out,
        )
        for fact_type in args.fact_types:
            all_facts = load_facts(fact_type)
            sampled = sample_facts(all_facts, args.n_exchanges // 2, seed=args.seed + 1)
            print(f"\n=== control / {fact_type}: {len(sampled)} facts ===", flush=True)
            for i, fact in enumerate(sampled):
                ctl.run_exchange(fact.to_dict())

    print(f"\nDone. Outputs in {paths['root']}", flush=True)


if __name__ == "__main__":
    main()
