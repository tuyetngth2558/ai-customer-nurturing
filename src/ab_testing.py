"""
A/B Testing Framework
=====================
Statistical hypothesis testing for 3 core AI features:
  1. Sentiment-triggered notification timing
  2. Next-Best-Action (ML) vs rule-based recommendations
  3. GenAI chatbot response vs scripted template

Uses two-proportion z-test for conversion metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class ABTestResult:
    feature_name: str
    control_rate: float
    variant_rate: float
    uplift_pct:   float
    p_value:      float
    significant:  bool
    confidence:   float
    sample_size_control: int
    sample_size_variant: int
    recommendation: str


def two_proportion_ztest(
    successes_control: int,
    n_control: int,
    successes_variant: int,
    n_variant: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Two-proportion z-test for comparing conversion rates.

    H0: p_control == p_variant
    H1: p_variant > p_control  (one-tailed)

    Returns: (z_stat, p_value)
    """
    p_c = successes_control / n_control
    p_v = successes_variant / n_variant
    p_pool = (successes_control + successes_variant) / (n_control + n_variant)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_variant))
    z  = (p_v - p_c) / se if se > 0 else 0.0
    p  = 1 - stats.norm.cdf(z)  # one-tailed

    return z, p


def run_ab_test(
    feature_name: str,
    control_data: pd.DataFrame,
    variant_data: pd.DataFrame,
    metric_col: str,
    alpha: float = 0.05,
) -> ABTestResult:
    """
    Run A/B test for a given feature.

    Parameters
    ----------
    feature_name  : descriptive name of the feature tested
    control_data  : DataFrame for control group, must have `metric_col` (0/1)
    variant_data  : DataFrame for variant group
    metric_col    : binary outcome column (1 = converted / success)
    alpha         : significance level (default 0.05)

    Returns
    -------
    ABTestResult dataclass
    """
    n_c = len(control_data)
    n_v = len(variant_data)
    s_c = control_data[metric_col].sum()
    s_v = variant_data[metric_col].sum()

    p_c = s_c / n_c
    p_v = s_v / n_v
    uplift = (p_v - p_c) / p_c * 100 if p_c > 0 else 0

    _, p_value = two_proportion_ztest(s_c, n_c, s_v, n_v, alpha)
    significant = p_value < alpha

    recommendation = (
        f"✅ SHIP VARIANT — statistically significant uplift of {uplift:+.1f}% (p={p_value:.3f})"
        if significant
        else f"⚠️  INCONCLUSIVE — uplift {uplift:+.1f}% not significant (p={p_value:.3f}). "
             f"Continue test or increase sample size."
    )

    return ABTestResult(
        feature_name=feature_name,
        control_rate=round(p_c, 4),
        variant_rate=round(p_v, 4),
        uplift_pct=round(uplift, 2),
        p_value=round(p_value, 4),
        significant=significant,
        confidence=round((1 - p_value) * 100, 1),
        sample_size_control=n_c,
        sample_size_variant=n_v,
        recommendation=recommendation,
    )


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate required sample size per group using power analysis.

    Parameters
    ----------
    baseline_rate : current conversion rate (e.g., 0.12 = 12%)
    mde           : minimum detectable effect (e.g., 0.03 = 3pp lift)
    alpha         : Type I error rate
    power         : statistical power (1 - Type II error)

    Returns
    -------
    n : sample size per group
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    effect_size = abs(p2 - p1) / np.sqrt((p1 + p2) / 2 * (1 - (p1 + p2) / 2))

    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta  = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))


def summary_report(results: list[ABTestResult]) -> pd.DataFrame:
    """Generate a summary DataFrame of all A/B test results."""
    rows = []
    for r in results:
        rows.append({
            "Feature":       r.feature_name,
            "Control Rate":  f"{r.control_rate:.1%}",
            "Variant Rate":  f"{r.variant_rate:.1%}",
            "Uplift":        f"{r.uplift_pct:+.1f}%",
            "p-value":       r.p_value,
            "Significant":   "✅ Yes" if r.significant else "❌ No",
            "Confidence":    f"{r.confidence}%",
            "n Control":     r.sample_size_control,
            "n Variant":     r.sample_size_variant,
        })
    return pd.DataFrame(rows)


# ── Simulate all 3 A/B Tests ──────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def make_group(n, rate):
        return pd.DataFrame({"converted": rng.binomial(1, rate, n)})

    # Feature 1: Sentiment-triggered notification timing
    r1 = run_ab_test(
        "Sentiment-Triggered Notification Timing",
        control_data=make_group(1200, 0.18),   # 18% open rate baseline
        variant_data=make_group(1200, 0.205),  # +14% relative uplift
        metric_col="converted",
    )

    # Feature 2: ML Next-Best-Action vs rule-based
    r2 = run_ab_test(
        "ML Next-Best-Action Recommendation",
        control_data=make_group(900, 0.09),    # 9% CTR baseline
        variant_data=make_group(900, 0.110),   # +22% relative uplift
        metric_col="converted",
    )

    # Feature 3: GenAI chatbot vs scripted template
    r3 = run_ab_test(
        "GenAI Chatbot vs Scripted Template",
        control_data=make_group(600, 0.55),    # 55% CSAT baseline
        variant_data=make_group(600, 0.72),    # +31% relative uplift
        metric_col="converted",
    )

    results = [r1, r2, r3]
    df = summary_report(results)
    print("\n📊 A/B TEST SUMMARY REPORT")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    for r in results:
        print(f"\n🔬 {r.feature_name}")
        print(f"   {r.recommendation}")

    # Sample size calculation for future test
    n_needed = required_sample_size(baseline_rate=0.12, mde=0.02, power=0.80)
    print(f"\n📐 Required sample size for next test (baseline 12%, MDE 2pp): {n_needed:,} per group")
