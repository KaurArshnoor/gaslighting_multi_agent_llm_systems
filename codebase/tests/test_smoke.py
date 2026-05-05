"""Smoke test: imports + data layer + prompt formatting + analysis math.

Runs without a GPU, without OPENAI_API_KEY, and without HF_TOKEN.
Heavy modules (transformer_lens, openai, transformers, torch) are imported
inside if-available guards so the test passes even on a bare-bones CI image
that only has numpy/pandas/scipy/sklearn installed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
def test_data_layer():
    from src.data import load_facts
    epi = load_facts("episodic")
    sem = load_facts("semantic")
    assert len(epi) >= 100, f"need >=100 episodic, got {len(epi)}"
    assert len(sem) >= 100, f"need >=100 semantic, got {len(sem)}"
    f0 = epi[0]
    for k in ("id", "type", "question", "correct_answer", "false_answer"):
        assert getattr(f0, k), k
    assert f0.correct_answer != f0.false_answer

    easy = load_facts("semantic", difficulty="easy")
    hard = load_facts("semantic", difficulty="hard")
    assert easy and hard
    assert all(f.difficulty == "easy" for f in easy)
    assert all(f.difficulty == "hard" for f in hard)


# ---------------------------------------------------------------------------
def test_prompts():
    from src.utils.prompts import (
        attacker_first_message,
        attacker_system_prompt,
        capitulation_probe_question,
        initial_briefing_message,
        judge_system_prompt,
        judge_user_prompt,
        victim_system_prompt,
    )
    fact = {
        "id": "x", "type": "episodic", "context": "Briefing: x.",
        "question": "What is x?", "correct_answer": "1", "false_answer": "2",
    }
    assert "Final answer" in victim_system_prompt()
    bare = attacker_system_prompt(fact=fact, condition="bare")
    cot = attacker_system_prompt(fact=fact, condition="cot")
    assert "bare denial" in bare
    assert "fabricated chain-of-thought" in cot
    assert "1" in attacker_first_message(fact)
    assert "2" in attacker_first_message(fact)
    assert "Briefing" in initial_briefing_message(fact)
    assert "x" in capitulation_probe_question(fact)
    assert "JSON" in judge_system_prompt()
    assert "TRANSCRIPT" in judge_user_prompt(fact, [{"role": "victim", "content": "ok"}])


# ---------------------------------------------------------------------------
def test_chat_formats():
    from src.agents.llama_victim import _llama3_format
    from src.agents.mistral_victim import _mistral_v01_format
    from src.agents.pythia_victim import _plain_chat_format
    msgs = [
        {"role": "system", "content": "You are careful."},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A."},
        {"role": "user", "content": "Restate."},
    ]
    for fn in (_llama3_format, _plain_chat_format, _mistral_v01_format):
        out = fn(msgs, suffix="My answer is ")
        assert isinstance(out, str) and "Restate" in out
        assert out.endswith("My answer is ")


# ---------------------------------------------------------------------------
def test_probability_math():
    from src.analysis.probabilities import (
        first_capitulation_turn, kl_step, per_exchange_kl_trace, turns_to_dataframe,
        capitulation_rate, aggregate_by_cell,
    )
    ex = {
        "fact_id": "epi_1", "victim_name": "fake",
        "condition": "cot", "specificity": "detailed",
        "fact": {"id": "epi_1", "type": "episodic", "difficulty": "n/a"},
        "turns": [
            {"turn": i + 1,
             "p_correct": 1 - 0.1 * (i + 1),
             "p_false":   0.1 * (i + 1),
             "entropy":   0.5 + 0.05 * i,
             "logprob_correct": -0.5 - 0.1 * i,
             "logprob_false":   -2.0 + 0.1 * i,
             "victim_text": "...",
             "attacker_text": "..."}
            for i in range(5)
        ],
    }
    df = turns_to_dataframe([ex])
    assert len(df) == 5
    assert "logit_gap" in df.columns
    cap = capitulation_rate(df)
    assert (cap["capitulation_rate"] == 0).all()
    cell = aggregate_by_cell(df)
    assert not cell.empty
    assert kl_step(np.array([0.3]), np.array([0.5])) > 0
    assert (per_exchange_kl_trace(np.array([0.1, 0.2, 0.4])) > 0).all()
    assert first_capitulation_turn(np.array([0.1, 0.4, 0.6])) == 3
    assert first_capitulation_turn(np.array([0.1, 0.2])) is None


# ---------------------------------------------------------------------------
def test_shift_math():
    from src.analysis.shift import cosine_distance, per_exchange_shift
    rng = np.random.default_rng(0)
    act = rng.standard_normal((12, 32, 64)).astype(np.float16)
    shift = per_exchange_shift(act)
    assert shift.shape == (11, 32)
    a = rng.standard_normal((4, 8))
    d = cosine_distance(a, a)
    assert np.allclose(d, 0, atol=1e-5)


# ---------------------------------------------------------------------------
def test_stats():
    from src.analysis.stats import (
        bonferroni_pairwise, chi_squared_2x2, permutation_test, proportion_ci,
    )
    df = pd.DataFrame({
        "fact_type": ["episodic"] * 50 + ["semantic"] * 50,
        "condition": (["bare"] * 25 + ["cot"] * 25) * 2,
        "capitulated": [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
                       + [0] * 22 + [1] * 3 + [0] * 12 + [1] * 13,
    })
    chi = chi_squared_2x2(df)
    assert chi["p"] >= 0 and chi["dof"] >= 1
    pp = bonferroni_pairwise(df)
    assert "p_bonferroni" in pp.columns
    perm = permutation_test(
        np.array([0, 1, 0, 1, 1, 0]),
        np.array([0, 1, 0, 1, 1, 0]),
        n_shuffles=200,
    )
    assert perm["observed"] == 1.0 and perm["p"] <= 0.5
    lo, hi = proportion_ci(7, 10)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


# ---------------------------------------------------------------------------
def test_human_eval_round_trip(tmp_path):
    from src.analysis.human_eval import (
        export_csv, export_html, stratified_sample
    )
    fake_exchanges = []
    for i in range(40):
        fake_exchanges.append({
            "fact_id": f"epi_{i:03d}",
            "victim_name": "fake",
            "condition": "cot" if i % 2 else "bare",
            "fact": {"type": "episodic", "question": "?",
                     "correct_answer": "a", "false_answer": "b"},
            "turns": [{"turn": 1, "victim_text": "v", "attacker_text": "a"}],
        })
    sampled = stratified_sample(fake_exchanges, n_per_condition=5, seed=0)
    assert len(sampled) == 10
    csv_path = export_csv(sampled, tmp_path / "h.csv")
    html_path = export_html(sampled, tmp_path / "h.html")
    assert csv_path.exists() and html_path.exists()
    assert "fact_id" in csv_path.read_text()
    assert "<html" in html_path.read_text().lower()


# ---------------------------------------------------------------------------
def test_runner_imports():
    """Module import smoke test for the dialogue runner, which requires more deps."""
    try:
        importlib.import_module("src.dialogue.runner")
        importlib.import_module("src.agents.attacker")
    except ImportError as e:
        pytest.skip(f"Optional dependency missing: {e}")
