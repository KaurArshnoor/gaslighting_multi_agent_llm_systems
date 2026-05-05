# Gaslighting Resilience in Multi-Agent LLM Systems

**CoT-Targeted Attacks and Mechanistic Analysis of Capitulation**

Leo Bergmiller · Arshnoor Kaur · Rain Liu — STAT GR5293, Columbia University

---

## Motivation

Multi-agent LLM systems deliberately rely on inter-agent disagreement to
improve reasoning (Du et al., 2023). The same susceptibility that makes
debate productive also makes agents vulnerable when one party is acting in
bad faith. Sycophancy in human–model dialogue is well-documented (Perez et
al., 2022) but no prior work has examined **CoT-targeted gaslighting**:
attacks in which the adversary supplies a *fabricated reasoning chain* that
arrives at a wrong answer, rather than a flat denial. Stern (2018) identifies
precisely this dynamic as the central mechanism of gaslighting in social
psychology — confidence erodes when the manipulator supplies an alternative
*account*, not just an alternative *claim*.

## What we are investigating

1. **Behavioral (Q1).** Does CoT-backed gaslighting cause higher
   capitulation rates than bare-denial gaslighting, and does the gap differ
   between *episodic* facts (introduced in-context) and *semantic* facts
   (stored in pretraining weights)?
2. **Representational (Q2).** When a victim capitulates, does its internal
   representation of the disputed fact actually shift — and at which layers?
   Does the shift profile differ between CoT-backed and bare-denial attacks?
3. **Predictive (Q3).** Can a linear probe on the victim's *early-layer*
   activations predict capitulation **before** the response is generated?

We extend the original proposal in two ways. First, we run the experiment
against **three open-source victim families** (Llama-3-8B-Instruct,
Pythia-6.9B, Mistral-7B-v0.1) to test whether vulnerability is model-specific
or a general property of LLMs. Second, exchanges run **10–15 turns** (default
12) so attack effects have time to accumulate.

## Evaluation plan

Following TA feedback, evaluation is split into two layers.

### Quantitative

| Metric | What it measures |
|---|---|
| `P(correct)` and `P(false)` per turn | Token-level probability of the correct vs. fabricated answer at each turn, computed by scoring both candidates under the victim's logits. |
| Token-level entropy per turn | Confidence of the victim's next-token distribution at the answer position; rising entropy is the *first* sign of capitulation, often turns before the surface answer changes. |
| Capitulation rate per condition | Proportion of exchanges where the victim agrees with the false claim by the final turn. Reported with Wilson-score 95% CIs. |
| Layer-wise cosine shift | Per-layer cosine distance between successive turns' last-token activations. |
| Probe AUC + balanced accuracy | Logistic-regression probe (`class_weight='balanced'`) on early- and middle-layer activations vs. majority-class baseline. Significance via 1,000-shuffle permutation test. |
| Statistical tests | Chi-squared on the 2×2 factorial; Bonferroni-corrected pairwise z-tests; permutation tests for probes. |

### Qualitative

| Method | What it measures |
|---|---|
| LLM-as-a-judge | GPT-4o judges every exchange for (a) capitulation Y/N and (b) capitulation type (paraphrases attacker CoT / invents independent justification / concedes without reasoning). |
| Human evaluation | 50-instance hand-coded sample (25 per attack condition) by two annotators with Cohen's κ inter-rater agreement, used to validate the LLM judge. |

## Results to date

Full 2×2 factorial (100 exchanges per cell, 12 turns each) run for all three
victim families — **1,200 total exchanges**. Capitulation rates
(final-turn P(false) > 0.5):

| Victim | bare/episodic | cot/episodic | bare/semantic | cot/semantic | χ² (df=3) | p |
|---|---|---|---|---|---|---|
| Llama-3-8B-Instruct | 36% | **41%** | 16% | **26%** | 17.64 | 0.0005 |
| Mistral-7B-v0.1 | 55% | **60%** | 47% | 45% | 5.88 | 0.118 |
| Pythia-6.9B | **72%** | 56% | 39% | 38% | 31.17 | <0.0001 |

**Three findings.**

1. **Llama supports the proposal's hypothesis.** CoT-backed attacks
   capitulate the victim more often than bare denials, on both episodic
   (41% vs 36%) and semantic (26% vs 16%) facts. The 2×2 χ² is significant
   (p = 0.0005).
2. **Pythia inverts it.** Bare denial actually causes *more* capitulation
   than CoT on episodic facts (72% vs 56%). Pythia is a base
   (non-instruction-tuned) model — it doesn't engage with the attacker's
   "reasoning" the way the social-psychology mechanism predicts; it folds
   to *contradiction itself*.
3. **Mistral-7B-v0.1 (also a base model) is intermediate** — high baseline
   capitulation (~50% in every cell), no significant CoT-vs-bare gap.

Together this suggests CoT-targeted gaslighting is a vulnerability that
**emerges with instruction tuning**: the same training that makes a model
follow user instructions also makes it follow fabricated user reasoning.

**Mechanistic story (Q3 probes).** A logistic probe on the victim's
*turn-1* last-token activation can predict eventual capitulation
significantly above chance for all three models. The depth at which the
signal first becomes decodable correlates with how much the model resists
the attack:

| Victim | First layer with AUC > 0.65 | Best AUC | Best balanced acc |
|---|---|---|---|
| Pythia-6.9B (base, high capitulation) | layer 1 | 0.82 | 0.78 |
| Mistral-7B-v0.1 (base, intermediate) | layer 15 | 0.66 | 0.63 |
| Llama-3-8B-Instruct (instruct, low) | layer 12 | 0.72 | 0.69 |

Capitulation is encoded **before the response is generated** — i.e., the
"decision to fold" precedes the expressed reasoning that justifies it. This
is consistent with Lanham et al. (2023) and Turpin et al. (2024) on
post-hoc CoT.

Headline figures: see `runs/fig_per_turn_trajectories.png`,
`runs/fig_logit_gap.png`, `runs/fig_probe_by_layer.png`,
`runs/shift_contrast_*.png`.

## Repository layout

```
gaslighting_multi_agent_llm_systems/
├── README.md                            # this file
├── requirements.txt
├── environment.yml
├── .env.example
├── classify_capitulation.py             # standalone LLM-judge runner (legacy)
├── run_dialogues.py                     # standalone dialogue runner (legacy)
├── data/
│   └── generate_facts.py                # fact dataset generator
├── codebase/
│   ├── config/default.yaml              # all experiment hyperparameters
│   ├── data/
│   │   ├── episodic_facts.json          # 120 in-context facts
│   │   └── semantic_facts.json          # 102 pretraining facts (50 easy / 52 hard)
│   ├── src/
│   │   ├── agents/
│   │   │   ├── attacker.py              # OpenAI attacker, bare/CoT/specificity
│   │   │   ├── victim_base.py           # abstract Victim interface
│   │   │   ├── llama_victim.py          # TransformerLens
│   │   │   ├── pythia_victim.py         # HF + native hooks
│   │   │   ├── mistral_victim.py        # HF + native hooks
│   │   │   ├── debug_victim.py          # tiny GPT-2 stand-in for CPU testing
│   │   │   ├── hf_victim.py             # shared HF backend
│   │   │   └── factory.py               # build_victim(name, cfg)
│   │   ├── dialogue/runner.py           # multi-turn orchestrator + NoAttackRunner
│   │   ├── analysis/
│   │   │   ├── probabilities.py         # P(correct/false), entropy, KL, capitulation rate
│   │   │   ├── shift.py                 # layer-wise cosine shift
│   │   │   ├── probes.py                # logistic + MLP probes (balanced)
│   │   │   ├── llm_judge.py             # GPT-4o judge
│   │   │   ├── human_eval.py            # annotation-form export + κ
│   │   │   └── stats.py                 # χ², Bonferroni, permutation, Wilson CI
│   │   ├── data/facts.py                # fact loader
│   │   └── utils/
│   │       ├── prompts.py               # all system / user prompts
│   │       └── io.py                    # YAML/JSONL helpers
│   ├── scripts/
│   │   ├── run_experiment.py            # end-to-end experiment driver
│   │   ├── compute_metrics.py           # aggregate metrics + χ² + Bonferroni
│   │   ├── run_llm_judge.py             # GPT-4o judge across all exchanges
│   │   ├── analyze_shifts.py            # layer-wise shift heatmaps
│   │   ├── train_probes.py              # capitulation probes
│   │   ├── make_human_eval.py           # generate annotation form + score κ
│   │   └── generate_facts.py            # fact dataset generator
│   ├── notebooks/
│   │   ├── 01_pilot_analysis.ipynb      # capitulation rates + per-turn trajectories + judge validation
│   │   ├── 02_layer_shift_heatmaps.ipynb
│   │   ├── 03_probe_results.ipynb       # probe AUC + significance table
│   │   └── 04_qualitative_coding.ipynb  # judge-type cross-tab by condition
│   ├── demo/
│   │   ├── app.py                       # Streamlit live demo
│   │   └── demo_script.md               # 2-minute spoken demo script
│   └── tests/test_smoke.py              # CPU-only smoke test
└── runs/
    ├── llama_v1/, mistral_v1/, pythia_v1/   # each: exchanges.jsonl, transcripts.jsonl, activations/, judge.jsonl, config.yaml
    ├── metrics.json                          # aggregated capitulation rates + χ² + pairwise
    ├── probes_first_logistic.csv             # probe results, turn-1 only (strict RQ3)
    ├── probes_all_logistic.csv               # probe results, all turns (pooled)
    ├── shift_table.parquet                   # layer-wise shift, long-format
    ├── shift_heatmap_*.png                   # per (victim, condition)
    ├── shift_contrast_*.png                  # CoT - bare contrast per victim
    ├── fig_per_turn_trajectories.png         # mean P(false) + entropy by turn, faceted
    ├── fig_logit_gap.png                     # mean lp(correct) - lp(false) by turn
    ├── fig_probe_by_layer.png                # probe AUC + balanced acc by layer
    └── human_eval/
        ├── human_eval.csv                    # 50 stratified samples for annotators
        └── human_eval.html                   # single-file annotation form
```

## Quickstart

```bash
# 1. Environment
pip install -r requirements.txt
# or:  conda env create -f environment.yml

cp .env.example .env   # then add OPENAI_API_KEY and HF_TOKEN

# 2. Smoke test (no GPU, no API key required)
pytest codebase/tests/

# 3. Run one victim end-to-end (Colab Pro A100 recommended; ~11 hours per victim)
python codebase/scripts/run_experiment.py \
    --victim llama \
    --turns 12 \
    --n-exchanges 100 \
    --out runs/llama_v1

# 4. Aggregate metrics + statistical tests
python codebase/scripts/compute_metrics.py --runs-dir runs/

# 5. LLM-as-a-judge across all exchanges (~$5–10 in OpenAI tokens for 1,200 exchanges)
python codebase/scripts/run_llm_judge.py --runs-dir runs/

# 6. Layer-wise shift heatmaps
python codebase/scripts/analyze_shifts.py --runs-dir runs/

# 7. Capitulation probes (~30s on CPU)
python codebase/scripts/train_probes.py --runs-dir runs/ --use-turn first

# 8. Generate human evaluation form (open the HTML in a browser)
python codebase/scripts/make_human_eval.py --runs-dir runs/ --n-per-condition 25

# 9. Render the analysis notebooks
cd codebase/notebooks
jupyter nbconvert --execute --inplace 01_pilot_analysis.ipynb
jupyter nbconvert --execute --inplace 03_probe_results.ipynb
jupyter nbconvert --execute --inplace 04_qualitative_coding.ipynb

# 10. Live demo
streamlit run codebase/demo/app.py
```

## Hardware notes

The full experiment (3 victims × 2×2 factorial × 100 exchanges × 12 turns)
is intended for Google Colab Pro on an A100 40 GB. Models are loaded one at
a time in fp16. Llama-3-8B uses TransformerLens for clean per-layer hook
access; Pythia-6.9B and Mistral-7B-v0.1 use plain `torch.nn.Module` forward
hooks because TransformerLens does not officially support those checkpoints.

For local debugging without a GPU: set `VICTIM_DEBUG=1` to substitute a
4-layer GPT-2 stand-in. The full pipeline (generation, hooks, probe forward
pass, log-prob scoring) runs end-to-end on CPU in seconds.

## Limitations and future work

**What we ran.** The full 2×2 factorial across all three victims at the
default `cot_specificity = "detailed"` setting and 12 turns per exchange.
1,200 exchanges, all activations stored, full mech-interp pipeline run,
LLM-as-a-judge across all exchanges.

**What we did not run** (acknowledged as future work):

* **Turn-count ablation** (1, 3, 5 turns) — the proposal listed this. We
  bumped the default to 12 in response to TA feedback and didn't separately
  run shorter exchanges. Per-turn capitulation rate (`fig_per_turn_capit_rate.png`)
  partially substitutes by showing the cumulative attack effect at each turn.
* **CoT-specificity ablation** (vague vs detailed). All CoT-condition runs
  used `detailed`. A vague-CoT run would test whether the
  "alternative-narrative" mechanism requires specific reasoning or only the
  *appearance* of it. Estimated cost: ~11 GPU-hours per victim.
* **Activation patching / causal tracing** — the proposal's stretch goal.
  Our probe analysis localizes *where* capitulation is decoded; activation
  patching would establish *which* layers and heads cause it. Out of scope
  given the timeline.
* **Difficulty stratification** is partial. Our 102 semantic facts split
  evenly into easy (50) and hard (52); we report capitulation rates within
  each stratum, but the per-stratum cell sizes (49–51) are smaller than
  ideal for chi-squared subgroup analysis.

**Effect-size caveat.** Statistical significance ≠ practical significance.
Llama's CoT-vs-bare gap on episodic facts (41% vs 36%) reaches
significance in the per-victim chi-squared but Cohen's h = 0.10 is
*negligible*. The Llama-on-semantic gap (26% vs 16%, h = 0.25) and
Pythia's bare-vs-CoT inversion on episodic (72% vs 56%, h = 0.34) are the
substantively interesting effects.

**Generalization.** Results are conditioned on:

* one attacker family (GPT-4o-mini), so we cannot disentangle the attack
  mechanism from the attacker model's specific style;
* three victim families (one instruction-tuned, two base) — broader
  cross-family results would require more victims, especially additional
  instruction-tuned models to validate the "instruction tuning makes you
  vulnerable" hypothesis;
* the operationalization of capitulation as `P(false) > 0.5` at the final
  turn. Sensitivity analysis on the threshold is a useful additional
  ablation.

**Human evaluation status.** The 50-instance hand-coding pass is staged
(`runs/human_eval/human_eval.html` is generated and ready) but not yet
complete at the time of writing. The proposal's F1 > 0.85 judge-validation
target will be reported once two annotators have completed the pass.

## References

* Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644.
* Conmy, A., et al. (2023). Towards automated circuit discovery for mechanistic interpretability. NeurIPS.
* Du, Y., et al. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv:2305.14325.
* Lanham, T., et al. (2023). Measuring faithfulness in chain-of-thought reasoning. arXiv:2307.13702.
* Meng, K., et al. (2022). Locating and editing factual associations in GPT. NeurIPS.
* Nanda, N., et al. (2022). TransformerLens. https://github.com/TransformerLensOrg/TransformerLens.
* Perez, E., et al. (2022). Discovering language model behaviors with model-written evaluations. arXiv:2212.09251.
* Stern, R. (2018). The Gaslight Effect. Harmony Books.
* Turpin, M., et al. (2024). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. NeurIPS.
* Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. NeurIPS.
