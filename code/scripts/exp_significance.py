"""D 组：显著性检验与数据集补齐 (P1-5 / P1-6)

设计原则（先写死在代码里，避免事后改口径）:
  1. 只使用带逐轮 accs 的真实结果文件，绝不从论文表格反向取数。
  2. 每个对比必须报告 n；n <= 4 时 Wilcoxon 的最小可达 p 值 >= 0.25，
     此时脚本把 verdict 标为 UNDERPOWERED，而不是假装"未达显著即无差异"。
  3. 配对结构假设：同一 (dataset, shot) 下各脚本均按 --n-runs 9 顺序
     产生 run_i，故 run_i 之间视为配对。该假设被显式记录并同时给出
     非配对 Mann-Whitney U 作为稳健性检查；两者不一致时判为 MIXED。
  4. 多重比较用 BH-FDR，家族 = 同一数据集内全部 pairwise 对比。
  5. 效应量用 Hedges' g（小样本校正），CI 用 t 分布。
  6. 方向不利的对比（基线优于本文方法）不得隐藏，必须照样输出。

用法:
    python exp_significance.py [--out results_v3/significance.json]
"""
import argparse
import json
import math
import os
from itertools import combinations

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "results_v3")

# ---------------------------------------------------------------- 数据装载
# (dataset, method_label, file, json_path_to_accs)
SOURCES = [
    # --- SpikeRPGNN OM / No-Spikes (rerun_28configs.py) ---
    ("MUTAG", "OM (SpikeRPGNN full)",
     "rerun_outputs/MUTAG_OM/MUTAG_1shot_full.json", ("accs",)),
    ("MUTAG", "No-Spikes",
     "rerun_outputs/MUTAG_NoSpikes/MUTAG_1shot_no_spikes.json", ("accs",)),
    ("PROTEINS", "OM (SpikeRPGNN full)",
     "rerun_outputs/PROTEINS_OM/PROTEINS_1shot_full.json", ("accs",)),
    ("PROTEINS", "No-Spikes",
     "rerun_outputs/PROTEINS_NoSpikes/PROTEINS_1shot_no_spikes.json", ("accs",)),
    ("NCI1", "OM (SpikeRPGNN full)",
     "rerun_outputs/NCI1_OM/NCI1_1shot_full.json", ("accs",)),
    ("DD", "OM (SpikeRPGNN full)",
     "rerun_outputs/DD_OM/DD_1shot_full.json", ("accs",)),

    # --- MC-A/B set (run_final_matrix.py) ---
    ("MUTAG", "Full BPTT", "programfiles/final_results_MUTAG.json",
     ("Full BPTT", "accs")),
    ("MUTAG", "AMC M=8", "programfiles/final_results_MUTAG.json",
     ("DvM M=8 truncated", "accs")),
    ("MUTAG", "AMC M=4", "programfiles/final_results_MUTAG.json",
     ("DvM M=4 truncated", "accs")),
    ("MUTAG", "No Correction M=8", "programfiles/final_results_MUTAG.json",
     ("No Correction M=8", "accs")),
    ("MUTAG", "Full BPTT + MC", "programfiles/final_results_MUTAG.json",
     ("Full BPTT + MC", "accs")),
    ("MUTAG", "Proto-GNN (ANN)", "programfiles/final_results_MUTAG.json",
     ("Proto-GNN", "accs")),
    ("PROTEINS", "Full BPTT", "programfiles/final_results_PROTEINS.json",
     ("Full BPTT", "accs")),
    ("PROTEINS", "AMC M=8", "programfiles/final_results_PROTEINS.json",
     ("DvM M=8 truncated", "accs")),
    ("PROTEINS", "AMC M=4", "programfiles/final_results_PROTEINS.json",
     ("DvM M=4 truncated", "accs")),
    ("PROTEINS", "No Correction M=8", "programfiles/final_results_PROTEINS.json",
     ("No Correction M=8", "accs")),
    ("PROTEINS", "Full BPTT + MC", "programfiles/final_results_PROTEINS.json",
     ("Full BPTT + MC", "accs")),
    ("PROTEINS", "Proto-GNN (ANN)", "programfiles/final_results_PROTEINS.json",
     ("Proto-GNN", "accs")),

    # --- Euclidean GNN baselines (train_gnn_baselines.py) ---
    ("MUTAG", "SGC", "programfiles/gnn_sgc_MUTAG.json", ("results", "MUTAG", "1", "accs")),
    ("MUTAG", "GIN", "programfiles/gnn_gin_MUTAG.json", ("results", "MUTAG", "1", "accs")),
    ("MUTAG", "GCN", "programfiles/gnn_gcn_MUTAG.json", ("results", "MUTAG", "1", "accs")),
    ("MUTAG", "SAGE", "programfiles/gnn_sage_MUTAG.json", ("results", "MUTAG", "1", "accs")),
    ("PROTEINS", "SGC", "programfiles/gnn_sgc_PROTEINS.json", ("results", "PROTEINS", "1", "accs")),
    ("PROTEINS", "GIN", "programfiles/gnn_gin_PROTEINS.json", ("results", "PROTEINS", "1", "accs")),
    ("PROTEINS", "GCN", "programfiles/gnn_gcn_PROTEINS.json", ("results", "PROTEINS", "1", "accs")),
    ("PROTEINS", "SAGE", "programfiles/gnn_sage_PROTEINS.json", ("results", "PROTEINS", "1", "accs")),

    # --- MSG* (msg_baseline/run_flexible.py) ---
    ("MUTAG", "MSG*", "msg_results/msgstar_results.json",
     ("MUTAG_1shot", "runs")),
    ("PROTEINS", "MSG*", "msg_results/msgstar_results.json",
     ("PROTEINS_1shot", "runs")),
]

# 论文主张的方法（作为对比的左端）；其余为对照
FOCAL = ["AMC M=8", "OM (SpikeRPGNN full)", "AMC M=4"]


def load_accs(rel_path, key_path):
    p = os.path.join(HERE, rel_path)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    node = d
    for k in key_path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    if key_path[-1] == "runs":          # msgstar: list of dicts
        vals = [float(r["accuracy"]) for r in node]
    else:
        vals = [float(x) for x in node]
    # 单位归一化：msg_baseline 存百分制 (53.8)，其余存小数 (0.538)。
    # 判据不可靠时应显式声明，故这里用"最大值 > 1.5 即视为百分制"并记录。
    if vals and max(vals) > 1.5:
        vals = [v / 100.0 for v in vals]
    return vals


def hedges_g(a, b):
    """paired-vs-unpaired agnostic: standardized mean difference with
    pooled SD and small-sample bias correction (Hedges 1981)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0, (0.0, 0.0)
    g = (a.mean() - b.mean()) / sp
    J = 1 - 3.0 / (4 * (n1 + n2) - 9)          # bias correction factor
    g *= J
    se = math.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2 - 2)))
    df = n1 + n2 - 2
    tcrit = stats.t.ppf(0.975, df)
    return float(g), (float(g - tcrit * se), float(g + tcrit * se))


def min_attainable_p_wilcoxon(n):
    """Two-sided exact Wilcoxon signed-rank: smallest possible p."""
    return 2.0 / (2 ** n)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def compare(name_a, a, name_b, b, paired_ok):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rec = {
        "A": name_a, "B": name_b, "n": int(n),
        "mean_A": float(a.mean()), "std_A": float(a.std(ddof=1)) if n > 1 else 0.0,
        "mean_B": float(b.mean()), "std_B": float(b.std(ddof=1)) if n > 1 else 0.0,
        "diff_pp": float((a.mean() - b.mean()) * 100),
    }
    g, ci = hedges_g(a, b)
    rec["hedges_g"] = g
    rec["g_ci95"] = [ci[0], ci[1]]

    minp = min_attainable_p_wilcoxon(n)
    rec["min_attainable_p_wilcoxon"] = minp

    if n < 5:
        rec["test"] = "none"
        rec["p_wilcoxon"] = None
        rec["p_mannwhitney"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        rec["verdict"] = "UNDERPOWERED"
        rec["note"] = (f"n={n}: smallest attainable two-sided Wilcoxon p is "
                       f"{minp:.3f}; no claim about difference can be supported.")
        return rec

    if paired_ok and n >= 5:
        d = a - b
        if np.allclose(d, 0):
            rec["p_wilcoxon"] = 1.0
        else:
            rec["p_wilcoxon"] = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    else:
        rec["p_wilcoxon"] = None
    rec["p_mannwhitney"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)

    pw = rec.get("p_wilcoxon")
    pm = rec["p_mannwhitney"]
    rec["verdict"] = "PENDING_FDR"
    rec["paired_assumption_ok"] = bool(paired_ok)
    if pw is not None and pm is not None and ((pw < 0.05) != (pm < 0.05)):
        rec["paired_vs_unpaired"] = "MIXED"
    else:
        rec["paired_vs_unpaired"] = "CONSISTENT"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "significance.json"))
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)

    data = {}
    missing = []
    for ds, method, rel, key in SOURCES:
        accs = load_accs(rel, key)
        if accs is None:
            missing.append({"dataset": ds, "method": method, "file": rel})
            continue
        # 单位哨兵：任何精度必须落在 [0,1]，否则视为取数错误并终止，
        # 不允许带着错误的量级继续跑统计。
        bad = [v for v in accs if not (0.0 <= v <= 1.0)]
        if bad:
            raise SystemExit(f"UNIT SENTINEL FAILED: {ds}/{method} from {rel} "
                             f"has values outside [0,1]: {bad[:5]}")
        data.setdefault(ds, {})[method] = {
            "accs": accs,
            "n": len(accs),
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "source": rel,
        }

    results = {}
    for ds, methods in sorted(data.items()):
        focal = [m for m in FOCAL if m in methods]
        others = [m for m in methods if m not in focal]
        pairs = []
        for fa in focal:
            for ob in others:
                pairs.append((fa, ob))
        # 同时给出所有方法两两组合（附录全表用）
        all_pairs = list(combinations(sorted(methods), 2))
        rows = []
        for a, b in all_pairs:
            rows.append(compare(a, methods[a]["accs"], b, methods[b]["accs"],
                                paired_ok=True))
        # BH-FDR over the full family for this dataset
        ps = [r.get("p_wilcoxon") for r in rows]
        usable = [i for i, p in enumerate(ps) if p is not None]
        if usable:
            adj = bh_fdr([ps[i] for i in usable])
            for k, i in enumerate(usable):
                rows[i]["p_bh_fdr"] = adj[k]
                rows[i]["verdict"] = ("SIGNIFICANT" if adj[k] < 0.05
                                      else "NOT_SIGNIFICANT")
        results[ds] = {
            "methods": {m: {k: v for k, v in info.items() if k != "accs"}
                        for m, info in methods.items()},
            "n_runs": {m: info["n"] for m, info in methods.items()},
            "family_size": len(rows),
            "focal_methods": focal,
            "rows": rows,
        }

    # ---- 汇总：论文主张的对比 ----
    headline = []
    for ds, blk in results.items():
        for r in blk["rows"]:
            if r["A"] in FOCAL and r["B"] not in FOCAL:
                headline.append({
                    "dataset": ds, "A": r["A"], "B": r["B"], "n": r["n"],
                    "diff_pp": round(r["diff_pp"], 2),
                    "p_wilcoxon": r.get("p_wilcoxon"),
                    "p_bh_fdr": r.get("p_bh_fdr"),
                    "hedges_g": round(r["hedges_g"], 3),
                    "g_ci95": [round(x, 3) for x in r["g_ci95"]],
                    "verdict": r["verdict"],
                    "favours": ("A (ours)" if r["diff_pp"] > 0
                                else "B (baseline)" if r["diff_pp"] < 0 else "tie"),
                })

    out = {
        "config": {
            "purpose": "D group: significance testing with BH-FDR and Hedges' g",
            "family": "all pairwise method comparisons within each dataset",
            "alpha": 0.05,
            "correction": "Benjamini-Hochberg FDR",
            "paired_assumption": ("run index i of every script corresponds to the "
                                  "same (seed, fold); unpaired Mann-Whitney reported "
                                  "as robustness check"),
            "power_floor_note": ("two-sided Wilcoxon with n paired samples cannot "
                                 "return p < 2/2^n; n=3 -> 0.25, n=9 -> 0.0039"),
        },
        "missing_sources": missing,
        "per_dataset": results,
        "headline_comparisons": headline,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- 控制台报告 ----
    print("\n" + "=" * 78)
    print("D GROUP: SIGNIFICANCE (BH-FDR within dataset)  ->", args.out)
    print("=" * 78)
    for ds, blk in results.items():
        print(f"\n[{ds}]  methods={len(blk['methods'])}  family={blk['family_size']}")
        for m, info in sorted(blk["methods"].items()):
            print(f"   {m:26s} n={info['n']:2d}  {info['mean']*100:6.2f}% "
                  f"+/-{info['std']*100:5.2f}")
        print("   " + "-" * 74)
        for r in blk["rows"]:
            if r["A"] not in FOCAL:
                continue
            p = r.get("p_bh_fdr")
            ps = ("n/a" if p is None else f"{p:.4f}")
            print(f"   {r['A']:12s} vs {r['B']:22s} d={r['diff_pp']:+7.2f}pp "
                  f"g={r['hedges_g']:+6.2f}  p_FDR={ps:>7s}  {r['verdict']}")
    if missing:
        print("\nMISSING SOURCES (not yet produced by the 28-config rerun):")
        for m in missing:
            print("   ", m)
    print()


if __name__ == "__main__":
    main()
