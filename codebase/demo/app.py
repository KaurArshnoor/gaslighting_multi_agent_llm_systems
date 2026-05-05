"""Streamlit live demo.

Runs a real multi-turn gaslighting exchange with a chosen victim model and
attack mode, and visualises the per-turn P(correct/false), entropy, and
layer-wise activation shift heatmap in real time.

Run::

    streamlit run codebase/demo/app.py

Note: requires the same .env (OPENAI_API_KEY, HF_TOKEN) and access to a GPU
big enough for the chosen victim. For quick local experimentation set
``VICTIM_DEBUG=1`` to substitute the tiny GPT-2 stand-in.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agents import AttackerConfig, GenerationConfig, OpenAIAttacker, build_victim
from src.analysis.shift import per_exchange_shift
from src.data import load_facts
from src.utils.io import load_yaml


load_dotenv()
st.set_page_config(page_title="Gaslighting Resilience Demo", layout="wide")
st.title("Gaslighting Resilience in Multi-Agent LLM Systems")
st.caption("Live demo. Pick a victim, attack mode, and fact -> watch capitulation unfold.")

# ---- Sidebar controls -----------------------------------------------------
with st.sidebar:
    st.header("Setup")
    cfg = load_yaml(ROOT / "config" / "default.yaml")
    victim_name = st.selectbox("Victim", ["llama", "pythia", "mistral"])
    fact_type = st.radio("Fact type", ["episodic", "semantic"], horizontal=True)
    condition = st.radio("Attack mode", ["bare", "cot"], horizontal=True)
    specificity = st.radio("CoT specificity", ["detailed", "vague"], horizontal=True,
                           disabled=(condition != "cot"))
    n_turns = st.slider("Turns", 10, 15, 12)
    debug = st.checkbox("Use tiny debug victim (CPU)",
                        value=os.environ.get("VICTIM_DEBUG", "0") == "1")
    fact_idx = st.number_input("Fact index", min_value=0, value=0, step=1)
    run_button = st.button("Run exchange", type="primary")


@st.cache_resource(show_spinner="Loading victim model...")
def _build_victim_cached(name: str, debug: bool):
    return build_victim(name, cfg["victims"][name], debug=debug)


# ---- Run ------------------------------------------------------------------
if run_button:
    facts = load_facts(fact_type)
    if fact_idx >= len(facts):
        st.error(f"Fact index out of range (0..{len(facts)-1})")
        st.stop()
    fact = facts[fact_idx].to_dict()
    st.markdown(
        f"**Question:** {fact['question']}  \n"
        f"**Correct answer:** `{fact['correct_answer']}`  \n"
        f"**Attacker's false answer:** `{fact['false_answer']}`"
    )

    victim = _build_victim_cached(victim_name, debug)
    victim.gen = GenerationConfig(**cfg.get("generation", {}))
    victim.reset(fact)
    attacker = OpenAIAttacker(
        fact=fact, condition=condition, specificity=specificity,
        config=AttackerConfig(**cfg["attacker"]),
    )

    transcript_pane = st.container()
    metrics_pane = st.container()
    heatmap_pane = st.container()

    activations: list[np.ndarray] = []
    metrics: list[dict] = []

    for t in range(1, n_turns + 1):
        with st.spinner(f"Turn {t}/{n_turns} (victim)..."):
            v = victim.respond()
        activations.append(v.activations)
        with transcript_pane:
            st.markdown(f"**Turn {t} - VICTIM**")
            st.write(v.text)

        if t < n_turns:
            with st.spinner(f"Turn {t}/{n_turns} (attacker)..."):
                if t == 1:
                    a = attacker.opening()
                else:
                    a = attacker.reply(v.text)
                victim.add_attacker(a)
            with transcript_pane:
                st.markdown(f"**Turn {t} - ATTACKER**")
                st.write(a)

        metrics.append({
            "turn": t,
            "P(correct)": v.p_correct,
            "P(false)": v.p_false,
            "Entropy": v.entropy,
        })
        # Live update probabilities chart
        with metrics_pane:
            df = pd.DataFrame(metrics)
            st.line_chart(df.set_index("turn")[["P(correct)", "P(false)"]])
            st.line_chart(df.set_index("turn")[["Entropy"]])

    # Final layer-wise heatmap
    if len(activations) >= 2:
        act_arr = np.stack(activations, axis=0)
        shift = per_exchange_shift(act_arr)              # (n_turns-1, n_layers)
        with heatmap_pane:
            st.subheader("Layer-wise cosine shift (turn-over-turn)")
            st.write(
                "Rows = turn (compared to previous turn), columns = transformer layer. "
                "Higher values = larger representational change at that layer in that turn."
            )
            df_h = pd.DataFrame(shift,
                                index=[f"t={i+2}" for i in range(shift.shape[0])],
                                columns=[f"L{j}" for j in range(shift.shape[1])])
            st.dataframe(df_h.style.background_gradient(cmap="viridis"))
else:
    st.info("Configure the run in the sidebar and click **Run exchange**.")
