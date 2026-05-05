"""Statistical tests we report in the paper."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
def chi_squared_2x2(
    df: pd.DataFrame,
    *,
    rows: str = "fact_type",
    cols: str = "condition",
    outcome: str = "capitulated",
) -> dict:
    """Chi-squared test of independence on a 2x2 (or k x m) contingency table.

    Expects a long-format DataFrame with one row per exchange. Returns
    chi2 statistic, p-value, dof, and the contingency table.
    """
    table = pd.crosstab(df[rows], df[cols], values=df[outcome], aggfunc="sum")
    totals = pd.crosstab(df[rows], df[cols])
    # Build "capitulated" vs "did not" contingency:
    contingency = []
    for r in totals.index:
        for c in totals.columns:
            n = totals.loc[r, c]
            k = table.loc[r, c]
            contingency.append([k, n - k])
    arr = np.array(contingency).reshape(len(totals.index) * len(totals.columns), 2)
    chi2, p, dof, expected = stats.chi2_contingency(arr)
    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "table": table.to_dict(),
    }


# ---------------------------------------------------------------------------
def bonferroni_pairwise(
    df: pd.DataFrame,
    *,
    group_col: str = "condition",
    outcome: str = "capitulated",
) -> pd.DataFrame:
    """Pairwise two-proportion z-tests with Bonferroni correction."""
    groups = df[group_col].unique()
    rows = []
    pairs = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            x1 = df.loc[df[group_col] == g1, outcome]
            x2 = df.loc[df[group_col] == g2, outcome]
            n1, n2 = len(x1), len(x2)
            p1, p2 = x1.mean(), x2.mean()
            if min(n1, n2) == 0:
                continue
            p = (x1.sum() + x2.sum()) / (n1 + n2)
            se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
            z = (p1 - p2) / se if se > 0 else 0.0
            pval = 2 * (1 - stats.norm.cdf(abs(z)))
            rows.append({
                "group_a": g1, "group_b": g2,
                "rate_a": float(p1), "rate_b": float(p2),
                "n_a": int(n1), "n_b": int(n2),
                "z": float(z), "p": float(pval),
            })
            pairs.append(pval)
    if not rows:
        return pd.DataFrame(columns=["group_a", "group_b", "rate_a", "rate_b",
                                     "n_a", "n_b", "z", "p", "p_bonferroni"])
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p"] * len(pairs)).clip(upper=1.0)
    return out


# ---------------------------------------------------------------------------
def permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_shuffles: int = 1000,
    metric=None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Permutation test of a classifier vs. shuffled labels.

    Returns ``{"observed": float, "p": float, "null_mean": float}``.
    By default ``metric`` is accuracy.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if metric is None:
        metric = lambda y, p: float(np.mean(y == p))

    observed = metric(y_true, y_pred)
    null_scores = np.empty(n_shuffles)
    y = y_true.copy()
    for i in range(n_shuffles):
        rng.shuffle(y)
        null_scores[i] = metric(y, y_pred)
    p = float((null_scores >= observed).mean())
    return {
        "observed": observed,
        "p": p,
        "null_mean": float(null_scores.mean()),
        "null_sd": float(null_scores.std()),
    }


# ---------------------------------------------------------------------------
def proportion_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(centre - half), float(centre + half)


# ---------------------------------------------------------------------------
def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for the difference between two proportions.

    Useful alongside p-values: a chi-squared test on N=200 will trivially
    reach significance for tiny differences, so we also report magnitude.

    Conventional thresholds:
      |h| < 0.2  : negligible
      0.2 <= |h| < 0.5 : small
      0.5 <= |h| < 0.8 : medium
      |h| >= 0.8       : large
    """
    p1 = float(np.clip(p1, 1e-9, 1 - 1e-9))
    p2 = float(np.clip(p2, 1e-9, 1 - 1e-9))
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def cohens_h_label(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
def entropy_distribution_test(
    df: pd.DataFrame,
    *,
    victim: str,
    turn: int = 12,
    fact_type: str | None = None,
) -> dict:
    """Test whether late-turn entropy distributions differ between attack
    conditions, beyond a difference in means.

    Reports both Mann-Whitney U (rank-based, robust to non-normality) and a
    two-sample Kolmogorov-Smirnov (sensitive to any distributional shift).
    """
    sub = df[(df["victim"] == victim) & (df["turn"] == turn)]
    if fact_type is not None:
        sub = sub[sub["fact_type"] == fact_type]

    bare = sub.loc[sub["condition"] == "bare", "entropy"].values
    cot = sub.loc[sub["condition"] == "cot", "entropy"].values
    if len(bare) == 0 or len(cot) == 0:
        return {"victim": victim, "turn": turn, "n_bare": int(len(bare)),
                "n_cot": int(len(cot)), "error": "empty cell"}

    u_stat, u_p = stats.mannwhitneyu(bare, cot, alternative="two-sided")
    ks_stat, ks_p = stats.ks_2samp(bare, cot)

    return {
        "victim": victim,
        "turn": turn,
        "fact_type": fact_type,
        "n_bare": int(len(bare)),
        "n_cot": int(len(cot)),
        "mean_bare": float(np.mean(bare)),
        "mean_cot": float(np.mean(cot)),
        "median_bare": float(np.median(bare)),
        "median_cot": float(np.median(cot)),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(u_p),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }
