"""
Statistical significance testing for SpikeRPGNN experiments.

Implements:
  - Benjamini-Hochberg FDR correction (q < 0.05)
  - Paired significance test (Shapiro-Wilk normality → t-test or Wilcoxon)
  - Cohen's d effect size
  - Full statistical report generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from scipy import stats


def benjamini_hochberg(
    p_values: List[float],
    q: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Args:
        p_values: List of raw p-values.
        q: False discovery rate threshold (default 0.05).

    Returns:
        Tuple of:
          - rejected: List of booleans indicating which hypotheses are rejected.
          - adjusted_p: List of BH-adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    # Sort p-values with index tracking
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Compute adjusted p-values
    adjusted = [0.0] * n
    for rank_i, (orig_idx, p_val) in enumerate(indexed):
        rank = rank_i + 1  # 1-indexed
        adjusted_p = p_val * n / rank
        adjusted[orig_idx] = adjusted_p

    # Enforce monotonicity (step-up procedure)
    # Process from largest rank to smallest
    sorted_by_rank = sorted(indexed, key=lambda x: x[1], reverse=True)
    min_so_far = 1.0
    for rank_i, (orig_idx, p_val) in enumerate(sorted_by_rank):
        adjusted_p = adjusted[orig_idx]
        min_so_far = min(min_so_far, adjusted_p)
        adjusted[orig_idx] = min(min_so_far, 1.0)

    # Determine rejections
    rejected = [adj_p < q for adj_p in adjusted]

    return rejected, adjusted


def cohens_d(
    group_a: List[float],
    group_b: List[float],
) -> float:
    """Compute Cohen's d effect size between two groups.

    Args:
        group_a: First group of measurements.
        group_b: Second group of measurements.

    Returns:
        Cohen's d (positive means group_a > group_b).
    """
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)

    n_a = len(a)
    n_b = len(b)

    if n_a < 2 or n_b < 2:
        return 0.0

    mean_diff = a.mean() - b.mean()

    # Pooled standard deviation
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float(mean_diff / pooled_std)


def paired_significance_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Perform paired significance test between two groups.

    First tests normality with Shapiro-Wilk, then chooses
    paired t-test (if normal) or Wilcoxon signed-rank test.

    Args:
        scores_a: First group of accuracy scores.
        scores_b: Second group of accuracy scores.
        alpha: Significance level.

    Returns:
        Dict with:
          - 'p_value': Raw p-value.
          - 'test_used': 't-test' or 'wilcoxon'.
          - 'delta': Mean difference (a - b).
          - 'cohens_d': Effect size.
          - 'significant': Whether p < alpha.
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return {
            "p_value": 1.0,
            "test_used": "none",
            "delta": 0.0,
            "cohens_d": 0.0,
            "significant": False,
        }

    # Test normality of differences
    diffs = a - b
    if len(diffs) >= 3:
        _, shapiro_p = stats.shapiro(diffs)
        normal = shapiro_p > alpha
    else:
        normal = False

    delta = float(a.mean() - b.mean())
    effect = cohens_d(scores_a, scores_b)

    if normal:
        # Paired t-test
        _, p_value = stats.ttest_rel(a, b)
        test_used = "t-test"
    else:
        # Wilcoxon signed-rank test
        try:
            _, p_value = stats.wilcoxon(a, b)
        except ValueError:
            # All differences are zero
            p_value = 1.0
        test_used = "wilcoxon"

    return {
        "p_value": float(p_value),
        "test_used": test_used,
        "delta": delta,
        "cohens_d": effect,
        "significant": p_value < alpha,
    }


def run_statistical_tests(
    results_dict: Dict[str, Dict[str, List[float]]],
    baseline_key: str = "full",
    q: float = 0.05,
) -> Dict[str, Dict]:
    """Run full statistical comparison of ablation variants vs baseline.

    Args:
        results_dict: Mapping variant_name -> {dataset_shot -> [acc_per_run]}.
        baseline_key: Name of the baseline variant (default "full").
        q: FDR threshold for Benjamini-Hochberg correction.

    Returns:
        Dict with per-comparison statistics and BH-corrected results.
    """
    if baseline_key not in results_dict:
        return {}

    baseline_results = results_dict[baseline_key]
    all_comparisons = {}
    all_p_values = []

    for variant, variant_results in results_dict.items():
        if variant == baseline_key:
            continue

        comparisons = {}
        for key in variant_results:
            if key in baseline_results:
                scores_a = baseline_results[key]
                scores_b = variant_results[key]
                if isinstance(scores_a, dict):
                    scores_a = scores_a.get("accs", [scores_a.get("mean_acc", 0)])
                    scores_b = scores_b.get("accs", [scores_b.get("mean_acc", 0)])
                test_result = paired_significance_test(scores_a, scores_b)
                comparisons[key] = test_result
                all_p_values.append(test_result["p_value"])

        all_comparisons[variant] = comparisons

    # BH correction across all comparisons
    if all_p_values:
        rejected, adjusted_p = benjamini_hochberg(all_p_values, q=q)
        # Map adjusted p-values back to comparisons
        p_idx = 0
        for variant in all_comparisons:
            for key in all_comparisons[variant]:
                all_comparisons[variant][key]["bh_adjusted_p"] = adjusted_p[p_idx]
                all_comparisons[variant][key]["bh_rejected"] = rejected[p_idx]
                p_idx += 1

    return all_comparisons
