"""
Generate LaTeX tables directly from results_v3/*.json.

Every number that reaches the paper is produced here from the raw JSON -- no
value is ever transcribed by hand, so the "paper number vs. data file" check is
true by construction rather than by inspection.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(_HERE, "results_v3")
SENTINEL_TOL = 1e-3
STABILITY_NORM = 1e4


def main():
    d = json.load(open(os.path.join(V3, "gradient_error_matrix.json"),
                       encoding="utf-8"))
    rows = [r for r in d["rows"] if r["variant"] == "amc_truncated"]
    by_unit = defaultdict(dict)
    for r in rows:
        by_unit[(r["dataset"], r["seed"], r["T"])][r["M"]] = r

    stable = []
    for k, mp in by_unit.items():
        s = mp.get(k[2])
        if s and s["rel_err"] <= SENTINEL_TOL and s["ref_grad_norm"] <= STABILITY_NORM:
            stable.append(k)

    # ---- Table: truncation error at T=16 (the paper's operating point) ----
    T = 16
    Ms = [2, 4, 6, 8, 12]
    datasets = ["MUTAG", "PROTEINS", "NCI1", "DD"]
    agg = defaultdict(lambda: defaultdict(list))
    for k in stable:
        if k[2] != T:
            continue
        for M in Ms:
            if M in by_unit[k]:
                agg[k[0]][M].append(by_unit[k][M]["rel_err"])

    print("% ---- TABLE: measured truncation error (group A) ----")
    print("\\begin{table}[htbp]")
    print("\\caption{Measured gradient error of truncated AMC against full BPTT "
          "on the real SpikeRPGNN model ($T{=}16$, identical weights and batch; "
          "mean$\\pm$std over seeds).}")
    print("\\label{tab:truncation_error}")
    print("\\centering\\footnotesize")
    print("\\begin{tabular}{@{}lccccc@{}}")
    print("\\toprule")
    print("Dataset & " + " & ".join(f"$M{{=}}{m}$" for m in Ms) + " \\\\")
    print("\\midrule")
    for ds in datasets:
        cells = []
        for M in Ms:
            v = agg[ds].get(M, [])
            if not v:
                cells.append("n/a")
            else:
                m_, s_ = statistics.mean(v), (statistics.stdev(v)
                                              if len(v) > 1 else 0.0)
                cells.append(f"{m_:.3f}$\\pm${s_:.3f}")
        print(f"{ds} & " + " & ".join(cells) + " \\\\")
    print("\\midrule")
    closed = [1 - 0.9 ** (T - m) for m in Ms]
    print("$1-\\rho^{T-M}$ & " + " & ".join(f"{c:.3f}" for c in closed) + " \\\\")
    n_by_ds = {ds: len(agg[ds].get(Ms[0], [])) for ds in datasets}
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print(f"% seeds per dataset: {n_by_ds}")
    print()

    # ---- Table: metric-correction ablation (group B) ----
    by = defaultdict(dict)
    for r in d["rows"]:
        by[(r["dataset"], r["seed"], r["T"], r["M"])][r["variant"]] = r

    win = defaultdict(list)
    full = defaultdict(list)
    for k, v in by.items():
        a, n = v.get("amc_truncated"), v.get("no_metric_correction")
        if not (a and n):
            continue
        tgt = full if k[3] == k[2] else win
        tgt[(k[0], k[2])].append(abs(a["cos"] - n["cos"]))

    print("% ---- TABLE: metric-correction ablation (group B) ----")
    print("\\begin{table}[htbp]")
    print("\\caption{Effect of the analytic metric correction on the truncated "
          "gradient: $|\\cos_{\\text{AMC}}-\\cos_{\\text{no-corr}}|$ against the "
          "same full-BPTT reference (median over seeds).}")
    print("\\label{tab:mc_ablation}")
    print("\\centering\\footnotesize")
    print("\\begin{tabular}{@{}lccc@{}}")
    print("\\toprule")
    print("Dataset & $T$ & $M<T$ (window excludes transient) & $M=T$ \\\\")
    print("\\midrule")
    for ds in datasets:
        for TT in (8, 16, 24):
            a = win.get((ds, TT), [])
            b = full.get((ds, TT), [])
            if not a and not b:
                continue
            ca = f"{statistics.median(a):.5f}" if a else "n/a"
            cb = f"{statistics.median(b):.5f}" if b else "n/a"
            print(f"{ds} & {TT} & {ca} & {cb} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
