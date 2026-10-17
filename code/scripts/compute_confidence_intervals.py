"""
Confidence Interval Computation for all TU dataset results.

Reads all results JSON files, computes 95% confidence intervals
using t-distribution (appropriate for small n), and generates
formatted LaTeX tables with CI.

Usage:
    python compute_confidence_intervals.py [--input-dir DIR] [--output DIR]

Inputs:
    - *_results.json (DD, MUTAG, PROTEINS, ENZYMES)
    - msg_comparison_results.json
    - final_results.json
    - complete_results.json

Outputs:
    - ci_summary.json          — all CIs in machine-readable format
    - ci_table_dd.tex          — DD experiment matrix table
    - ci_table_comparison.tex  — MSG vs SpikeRPGNN comparison
    - ci_table_all.tex         — all 4 datasets overview
"""

import json
import os
import sys
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
def t_critical(n: int, confidence: float = 0.95) -> float:
    """t-distribution critical value (approximation for common df).

    Uses linear interpolation of tabulated values for df = n-1.
    Accurate to 3 decimal places for n ∈ [2, 120].
    """
    df = n - 1
    if df < 1:
        return float("inf")

    # Tabulated t-values for 95% two-sided CI
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009,
        60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
    }

    if df in t_table:
        return t_table[df]

    # Linear interpolation for values not in table
    keys = sorted(t_table.keys())
    for i in range(len(keys) - 1):
        if keys[i] < df < keys[i + 1]:
            t_low = t_table[keys[i]]
            t_high = t_table[keys[i + 1]]
            frac = (df - keys[i]) / (keys[i + 1] - keys[i])
            return t_low + frac * (t_high - t_low)

    if df < keys[0]:
        return t_table[keys[0]]
    return t_table[keys[-1]]


def compute_ci(accs: List[float], confidence: float = 0.95) -> Dict:
    """Compute confidence interval for a list of accuracy values.

    Args:
        accs: List of per-run accuracy values.
        confidence: Confidence level (default 0.95).

    Returns:
        Dict with mean, std, sem, ci_low, ci_high, n, t_crit.
    """
    n = len(accs)
    if n < 2:
        return {"mean": accs[0] if n == 1 else 0.0, "std": 0.0,
                "sem": 0.0, "ci_low": None, "ci_high": None,
                "n": n, "t_crit": None}

    mean = sum(accs) / n
    # Sample std (unbiased estimator)
    std = math.sqrt(sum((x - mean) ** 2 for x in accs) / (n - 1))
    sem = std / math.sqrt(n)
    t_c = t_critical(n, confidence)
    ci_half = t_c * sem

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - ci_half,
        "ci_high": mean + ci_half,
        "n": n,
        "t_crit": t_c,
    }


def cohens_d(accs_a: List[float], accs_b: List[float]) -> float:
    """Cohen's d effect size (pooled std)."""
    na, nb = len(accs_a), len(accs_b)
    if na < 2 or nb < 2:
        return 0.0
    ma = sum(accs_a) / na
    mb = sum(accs_b) / nb
    va = sum((x - ma) ** 2 for x in accs_a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in accs_b) / (nb - 1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return abs(ma - mb) / sp if sp > 0 else 0.0


# ------------------------------------------------------------
# CI result scanner
# ------------------------------------------------------------
def scan_results_json(filepath: str) -> List[Dict]:
    """Extract (dataset, shot, config, accs) tuples from results JSON."""
    entries = []
    with open(filepath, "r") as f:
        data = json.load(f)

    dataset = data.get("dataset", "Unknown")
    shots = data.get("shots", [1, 5])

    for shot in shots:
        sk = f"{shot}-shot"
        if sk not in data:
            continue
        for config_name, config_data in data[sk].items():
            accs = config_data.get("accs", [])
            if accs:
                entries.append({
                    "dataset": dataset,
                    "shot": shot,
                    "config": config_name,
                    "accs": accs,
                    "mean": config_data.get("mean"),
                    "std": config_data.get("std"),
                })
    return entries


def scan_comparison_json(filepath: str) -> List[Dict]:
    """Extract (dataset, shot, method, accs) from comparison JSON."""
    entries = []
    with open(filepath, "r") as f:
        data = json.load(f)

    for ds_name, ds_data in data.items():
        if ds_name in ("started", "device", "n_runs", "n_epochs",
                       "datasets", "methods", "finished"):
            continue
        for sk, sk_data in ds_data.items():
            # sk = "1-shot" or "5-shot"
            shot = int(sk.split("-")[0])
            for method_name, method_data in sk_data.items():
                accs = method_data.get("accs", [])
                if accs:
                    entries.append({
                        "dataset": ds_name,
                        "shot": shot,
                        "config": method_name,
                        "accs": accs,
                        "mean": method_data.get("mean"),
                        "std": method_data.get("std"),
                    })
    return entries


# ------------------------------------------------------------
# LaTeX table generators
# ------------------------------------------------------------
def _fmt_mean_ci(mean: float, ci_low: Optional[float],
                 ci_high: Optional[float], scale: float = 100) -> str:
    """Format as 'mean [CI_low, CI_high]' for LaTeX."""
    m = mean * scale
    if ci_low is not None and ci_high is not None:
        lo = ci_low * scale
        hi = ci_high * scale
        return f"{m:.1f} \\hspace{{2pt}}[{lo:.1f}, {hi:.1f}]"
    return f"{m:.1f}"


def generate_ci_table(entries: List[Dict],
                      configs_order: Optional[List[str]] = None,
                      title: str = "Experiment Results with 95\\% CI",
                      label: str = "tab:ci") -> str:
    """Generate a LaTeX table with confidence intervals.

    Args:
        entries: List of {dataset, shot, config, accs, ...}.
        configs_order: Ordered list of config names (columns).
        title: Table caption.
        label: Table label.

    Returns:
        LaTeX table string.
    """
    if not entries:
        return "% No data available"

    # Group by (dataset, shot) -> {config: accs}
    grouped = {}
    for e in entries:
        key = (e["dataset"], e["shot"])
        if key not in grouped:
            grouped[key] = {}
        ci = compute_ci(e["accs"])
        ci["config"] = e["config"]
        grouped[key][e["config"]] = ci

    if configs_order is None:
        # Determine order from all configs seen
        configs_seen = set()
        for key, configs in grouped.items():
            configs_seen.update(configs.keys())
        configs_order = sorted(configs_seen)

    ds_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))

    # Build table
    col_spec = "l" + "c" * len(configs_order)
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{title}}}",
        f"  \\label{{{label}}}",
        "  \\footnotesize",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
    ]

    # Header
    hdr = "    Dataset & Shot"
    for c in configs_order:
        hdr += f" & {c}"
    hdr += " \\\\"
    lines.append(hdr)
    lines.append("    \\midrule")

    # Data rows
    for (ds, shot) in ds_keys:
        configs = grouped[(ds, shot)]
        row = f"    {ds} & {shot}-shot"
        best_mean = -1
        best_config = ""
        for c in configs_order:
            if c in configs:
                ci = configs[c]
                row += " & " + _fmt_mean_ci(ci["mean"], ci["ci_low"],
                                            ci["ci_high"])
                if ci["mean"] > best_mean:
                    best_mean = ci["mean"]
                    best_config = c
            else:
                row += " & ---"
        row += " \\\\"
        lines.append(row)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_comparison_table(entries: List[Dict],
                              title: str = "MSG* vs SpikeRPGNN Comparison",
                              label: str = "tab:msg_comparison") -> str:
    """Generate a comparison table: MSG vs DvM vs Delta."""
    if not entries:
        return "% No data"

    # Group by (dataset, shot) -> {method: ci}
    grouped = {}
    for e in entries:
        key = (e["dataset"], e["shot"])
        if key not in grouped:
            grouped[key] = {}
        ci = compute_ci(e["accs"])
        ci["method"] = e["config"]
        grouped[key][e["config"]] = ci

    ds_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    methods = ["MSG*", "SpikeRPGNN DvM M=8"]

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{title}}}",
        f"  \\label{{{label}}}",
        "  \\footnotesize",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    Dataset & Shot & MSG* (95\\% CI) & DvM (95\\% CI) & $\\Delta$ (\\%) \\\\",
        "    \\midrule",
    ]

    for (ds, shot) in ds_keys:
        configs = grouped[(ds, shot)]
        if "MSG*" not in configs or "SpikeRPGNN DvM M=8" not in configs:
            continue
        msg_ci = configs["MSG*"]
        dvm_ci = configs["SpikeRPGNN DvM M=8"]
        delta = (dvm_ci["mean"] - msg_ci["mean"]) * 100

        msg_str = _fmt_mean_ci(msg_ci["mean"], msg_ci["ci_low"],
                               msg_ci["ci_high"])
        dvm_str = _fmt_mean_ci(dvm_ci["mean"], dvm_ci["ci_low"],
                               dvm_ci["ci_high"])
        lines.append(
            f"    {ds} & {shot}-shot & {msg_str} & {dvm_str} & "
            f"{delta:+.1f} \\\\"
        )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_ci_summary(entries: List[Dict]) -> Dict:
    """Generate machine-readable CI summary."""
    summary = {"generated": datetime.now().isoformat(), "entries": []}
    for e in entries:
        ci = compute_ci(e["accs"])
        summary["entries"].append({
            "dataset": e["dataset"],
            "shot": e["shot"],
            "config": e["config"],
            "n": ci["n"],
            "mean": round(ci["mean"], 6),
            "std": round(ci["std"], 6),
            "sem": round(ci["sem"], 6),
            "ci_low": round(ci["ci_low"], 6) if ci["ci_low"] else None,
            "ci_high": round(ci["ci_high"], 6) if ci["ci_high"] else None,
        })
    return summary


# ------------------------------------------------------------
# Cohen's d computation
# ------------------------------------------------------------
def compute_effect_sizes(entries: List[Dict]) -> List[Dict]:
    """Compute Cohen's d for key comparisons: DvM vs NoCorrection, DvM vs FullBPTT."""
    # Group by (dataset, shot)
    grouped = {}
    for e in entries:
        key = (e["dataset"], e["shot"], e["config"])
        grouped[key] = e["accs"]

    effects = []
    for (ds, shot) in set((e["dataset"], e["shot"]) for e in entries):
        pairs = [
            ("SpikeRPGNN DvM M=8", "No Correction M=8", "DvM vs NoCorr"),
            ("DvM M=8 truncated", "No Correction M=8", "DvM vs NoCorr"),
            ("DvM M=8 truncated", "Full BPTT", "DvM vs FullBPTT"),
        ]
        for a_name, b_name, label in pairs:
            a = grouped.get((ds, shot, a_name))
            b = grouped.get((ds, shot, b_name))
            if a and b:
                d = cohens_d(a, b)
                effects.append({
                    "dataset": ds, "shot": shot,
                    "comparison": label,
                    "a": a_name, "b": b_name,
                    "cohens_d": round(d, 3),
                })
    return effects


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute 95% CI for all TU dataset results")
    parser.add_argument("--input-dir", default=None,
                        help="Directory with results JSONs (default: DvM/)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for tables (default: same as input)")
    parser.add_argument("--output", default=None,
                        help="Output JSON summary path")
    parser.add_argument("--files", nargs="*",
                        help="Specific JSON files to process")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or script_dir
    output_dir = args.output_dir or script_dir

    # Find result files
    if args.files:
        json_files = args.files
    else:
        json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                      if f.endswith("_results.json") or
                      f == "msg_comparison_results.json" or
                      f == "complete_results.json"]

    # Scan all files
    all_entries = []
    for f in json_files:
        if not os.path.exists(f):
            continue
        fname = os.path.basename(f)
        if fname == "msg_comparison_results.json":
            entries = scan_comparison_json(f)
        else:
            entries = scan_results_json(f)
        if entries:
            print(f"Scanned {f}: {len(entries)} entries")
        all_entries.extend(entries)

    if not all_entries:
        print("No result entries found.")
        return

    # Generate CI summary
    summary = generate_ci_summary(all_entries)
    summary_file = args.output or os.path.join(output_dir, "ci_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCI summary: {summary_file} ({len(summary['entries'])} entries)")

    # Generate comparison table (if MSG data present)
    comp_entries = [e for e in all_entries
                    if e["config"] in ("MSG*", "SpikeRPGNN DvM M=8")]
    if comp_entries:
        comp_tex = generate_comparison_table(comp_entries)
        comp_file = os.path.join(output_dir, "ci_table_comparison.tex")
        with open(comp_file, "w", encoding="utf-8") as f:
            f.write(comp_tex)
        print(f"Comparison table: {comp_file}")

    # Generate DD table
    dd_entries = [e for e in all_entries if e["dataset"] == "DD"]
    if dd_entries:
        dd_tex = generate_ci_table(
            dd_entries,
            configs_order=["Full BPTT", "DvM M=8 truncated",
                           "DvM M=4 truncated", "No Correction M=8",
                           "Proto-GNN"],
            title="DD Dataset Experiment Results (95\\% CI)",
            label="tab:dd_results")
        dd_file = os.path.join(output_dir, "ci_table_dd.tex")
        with open(dd_file, "w", encoding="utf-8") as f:
            f.write(dd_tex)
        print(f"DD table: {dd_file}")

    # Generate all-data overview table
    nocomp_entries = [e for e in all_entries
                      if "MSG*" not in e["config"] and
                      "SpikeRPGNN DvM M=8" not in e["config"]]
    if nocomp_entries:
        all_tex = generate_ci_table(
            nocomp_entries,
            configs_order=["Full BPTT", "DvM M=8 truncated",
                           "DvM M=4 truncated", "No Correction M=8",
                           "Proto-GNN"],
            title="TU Dataset Results with 95\\% Confidence Intervals",
            label="tab:tu_ci")
        all_file = os.path.join(output_dir, "ci_table_all.tex")
        with open(all_file, "w", encoding="utf-8") as f:
            f.write(all_tex)
        print(f"All-data table: {all_file}")

    # Compute effect sizes
    effects = compute_effect_sizes(all_entries)
    if effects:
        eff_file = os.path.join(output_dir, "effect_sizes.json")
        with open(eff_file, "w") as f:
            json.dump(effects, f, indent=2)
        print(f"Effect sizes: {eff_file}")
        print(f"\nCohen's d summary:")
        for e in effects:
            print(f"  {e['dataset']:<12} {e['shot']}-shot "
                  f"{e['comparison']:<18} d={e['cohens_d']:.3f}")

    # Print summary
    print(f"\n=== CI Summary ===")
    for e in summary["entries"]:
        if e["ci_low"] is not None:
            print(f"  {e['dataset']:<12} {e['shot']}-shot "
                  f"{e['config']:<24} n={e['n']} "
                  f"mean={e['mean']:.4f} "
                  f"CI=[{e['ci_low']:.4f}, {e['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
