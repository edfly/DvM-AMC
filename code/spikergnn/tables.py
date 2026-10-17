"""
LaTeX table generation for SpikeRPGNN paper.

Generates publication-ready LaTeX tables for:
  - Table 2: Accuracy comparison (mean ± std)
  - Table 3: Ablation study results
  - Table 4: Energy estimation
  - Statistical significance table
"""

import numpy as np
from typing import Dict, List, Any, Optional


def _fmt_mean_std(mean: float, std: float) -> str:
    """Format mean ± std for LaTeX table cell.

    Args:
        mean: Mean value.
        std: Standard deviation.

    Returns:
        Formatted string like "84.3 ± 2.1".
    """
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def format_accuracy_table(
    results: Dict[str, Dict[str, Any]],
    datasets: Optional[List[str]] = None,
    shots: Optional[List[int]] = None,
) -> str:
    """Generate LaTeX Table 2: accuracy comparison across datasets.

    Args:
        results: Mapping variant_name -> {dataset_shot -> {mean_acc, std_acc, ...}}.
        datasets: List of dataset names (default: common 5).
        shots: List of shot values (default: [1, 5]).

    Returns:
        LaTeX table string.
    """
    if datasets is None:
        datasets = ["MUTAG", "PROTEINS", "DD", "ENZYMES", "REDDIT-M5K"]
    if shots is None:
        shots = [1, 5]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{Few-shot graph classification accuracy (\\%). Mean $\\pm$ Std over 5 runs.}")
    lines.append("  \\label{tab:accuracy}")
    lines.append("  \\begin{tabular}{l" + "cc" * len(datasets) + "}")
    lines.append("    \\toprule")

    # Header row
    header = "    Method"
    for ds in datasets:
        for s in shots:
            header += f" & {ds} {s}-shot"
    header += " \\\\"
    lines.append(header)
    lines.append("    \\midrule")

    # Data rows
    for variant_name, variant_data in results.items():
        row = f"    {variant_name}"
        for ds in datasets:
            for s in shots:
                key = f"{ds}_{s}shot"
                if key in variant_data:
                    entry = variant_data[key]
                    if isinstance(entry, dict):
                        mean = entry.get("mean_acc", 0)
                        std = entry.get("std_acc", 0)
                    else:
                        mean, std = entry, 0
                    row += f" & {_fmt_mean_std(mean * 100, std * 100)}"
                else:
                    row += " & ---"
        row += " \\\\"
        lines.append(row)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def format_ablation_table(
    ablation_results: Dict[str, Dict[str, Any]],
    datasets: Optional[List[str]] = None,
    shots: Optional[List[int]] = None,
) -> str:
    """Generate LaTeX Table 3: ablation study results.

    Args:
        ablation_results: Mapping variant_name -> {dataset_shot -> {mean_acc, ...}}.
        datasets: List of dataset names.
        shots: List of shot values.

    Returns:
        LaTeX table string.
    """
    if datasets is None:
        datasets = ["MUTAG", "PROTEINS", "DD", "ENZYMES", "REDDIT-M5K"]
    if shots is None:
        shots = [1, 5]

    from spikergnn.ablation import ABLATION_LABELS

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{Ablation study results (\\%). $\\Delta$ = Full - Ablation.}")
    lines.append("  \\label{tab:ablation}")
    lines.append("  \\begin{tabular}{l" + "cc" * len(datasets) + "}")
    lines.append("    \\toprule")

    header = "    Variant"
    for ds in datasets:
        for s in shots:
            header += f" & {ds} {s}-shot"
    header += " \\\\"
    lines.append(header)
    lines.append("    \\midrule")

    # Get baseline (full) for delta computation
    baseline = ablation_results.get("full", {})

    for variant_name, variant_data in ablation_results.items():
        label = ABLATION_LABELS.get(variant_name, variant_name)
        row = f"    {label}"
        for ds in datasets:
            for s in shots:
                key = f"{ds}_{s}shot"
                if key in variant_data:
                    entry = variant_data[key]
                    if isinstance(entry, dict):
                        mean = entry.get("mean_acc", 0)
                    else:
                        mean = entry
                    row += f" & {mean * 100:.1f}"
                else:
                    row += " & ---"
        row += " \\\\"
        lines.append(row)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def format_energy_table(
    energy_results: Dict[str, Dict[str, float]],
) -> str:
    """Generate LaTeX Table 4: energy estimation.

    Args:
        energy_results: Mapping variant_name -> {e_snn_mJ, e_ann_mJ, reduction_pct, ...}.

    Returns:
        LaTeX table string.
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{Energy estimation: SNN vs ANN operations.}")
    lines.append("  \\label{tab:energy}")
    lines.append("  \\begin{tabular}{lrrr}")
    lines.append("    \\toprule")
    lines.append("    Variant & $E_{\\mathrm{SNN}}$ (mJ) & $E_{\\mathrm{ANN}}$ (mJ) & Reduction (\\%) \\\\")
    lines.append("    \\midrule")

    for variant_name, data in energy_results.items():
        e_snn = data.get("e_snn_mJ", 0)
        e_ann = data.get("e_ann_mJ", 0)
        reduction = data.get("reduction_pct", 0)
        lines.append(
            f"    {variant_name} & {e_snn:.4f} & {e_ann:.4f} & {reduction:.1f} \\\\"
        )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def format_stat_table(
    stats_results: Dict[str, Dict[str, Dict]],
) -> str:
    """Generate statistical significance LaTeX table.

    Args:
        stats_results: Output of run_statistical_tests().

    Returns:
        LaTeX table string.
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{Statistical significance tests (vs Full SpikeRPGNN). BH-corrected $p < 0.05$.}")
    lines.append("  \\label{tab:stats}")
    lines.append("  \\begin{tabular}{llrrrl}")
    lines.append("    \\toprule")
    lines.append("    Variant & Dataset & $\\Delta$ Acc (\\%) & $p$-value & Cohen's $d$ & Sig. \\\\")
    lines.append("    \\midrule")

    for variant, comparisons in stats_results.items():
        for key, data in comparisons.items():
            delta = data.get("delta", 0) * 100
            p_val = data.get("p_value", 1)
            cd = data.get("cohens_d", 0)
            bh_rejected = data.get("bh_rejected", False)
            sig_str = "$\\checkmark$" if bh_rejected else ""
            lines.append(
                f"    {variant} & {key} & {delta:+.1f} & {p_val:.4f} & {cd:.2f} & {sig_str} \\\\"
            )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_results_table(
    results: Dict[str, Dict[str, Any]],
    energy_results: Optional[Dict[str, Dict[str, float]]] = None,
    stats_results: Optional[Dict[str, Dict[str, Dict]]] = None,
    datasets: Optional[List[str]] = None,
    shots: Optional[List[int]] = None,
) -> str:
    """Generate a complete results report combining all four table formats.

    Composes the accuracy table, ablation table, energy table, and
    statistical significance table into a single LaTeX document.

    Args:
        results: Mapping variant_name -> {dataset_shot -> {mean_acc, std_acc, ...}}.
        energy_results: Mapping variant_name -> {e_snn_mJ, e_ann_mJ, reduction_pct, ...}.
        stats_results: Output of run_statistical_tests().
        datasets: List of dataset names (default: common 5).
        shots: List of shot values (default: [1, 5]).

    Returns:
        Combined LaTeX string with all four tables.
    """
    sections = []

    # Table 2: Accuracy comparison
    sections.append(format_accuracy_table(results, datasets, shots))

    # Table 3: Ablation study
    sections.append(format_ablation_table(results, datasets, shots))

    # Table 4: Energy estimation (optional)
    if energy_results is not None:
        sections.append(format_energy_table(energy_results))

    # Statistical significance table (optional)
    if stats_results is not None:
        sections.append(format_stat_table(stats_results))

    return "\n\n".join(sections)
