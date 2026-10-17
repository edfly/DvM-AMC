"""口径审计：把每一个 (dataset, method) 测量值连同它的出处摊开。

为什么需要这个脚本
------------------
同一标签 "AMC M=8 on MUTAG" 在本项目里至少有四个互不相同的数值，分别来自
不同脚本、不同 n、不同训练预算、以及修复前后不同版本的实现。论文若择一
报告而不披露其余，就是选择性报告。此脚本把所有来源并列，冲突自动标红，
并产出可直接进附录的 LaTeX 表。

用法
----
    python make_provenance_table.py
    python make_provenance_table.py --tex post/provenance_table.tex
"""
import argparse
import hashlib
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))

# 手工登记的出处。每一项 = (dataset, method_label, file, path_in_json, script,
#                          n, epochs, notes)
# n / epochs 若为 None 则由文件内元数据推断。
ENTRIES = [
    # ---- SpikeRPGNN full BPTT / AMC (run_final_matrix.py) ----
    ("MUTAG", "Full BPTT", "final_results.json", ("Full BPTT",),
     "run_final_matrix.py", 3, 50, "修复前实现 (2026-07), CPU"),
    ("MUTAG", "AMC M=8", "final_results.json", ("DvM M=8 truncated",),
     "run_final_matrix.py", 3, 50, "修复前实现 (2026-07), CPU"),
    ("MUTAG", "AMC M=4", "final_results.json", ("DvM M=4 truncated",),
     "run_final_matrix.py", 3, 50, "修复前实现 (2026-07), CPU"),
    ("MUTAG", "No Correction M=8", "final_results.json", ("No Correction M=8",),
     "run_final_matrix.py", 3, 50, "修复前实现 (2026-07), CPU"),
    ("MUTAG", "Full BPTT + MC", "final_results.json", ("Full BPTT + MC",),
     "run_final_matrix.py", 3, 50, "修复前实现 (2026-07), CPU"),
    ("MUTAG", "Proto-GNN (ANN)", "final_results.json", ("Proto-GNN",),
     "run_final_matrix.py", 3, 50, "修复前实现 (2026-07), CPU"),

    ("MUTAG", "Full BPTT", "programfiles/final_results_MUTAG.json", ("Full BPTT",),
     "run_final_matrix.py", 9, 50, "修复后实现 (2026-09), CUDA"),
    ("MUTAG", "AMC M=8", "programfiles/final_results_MUTAG.json", ("DvM M=8 truncated",),
     "run_final_matrix.py", 9, 50, "修复后实现 (2026-09), CUDA"),
    ("MUTAG", "AMC M=4", "programfiles/final_results_MUTAG.json", ("DvM M=4 truncated",),
     "run_final_matrix.py", 9, 50, "修复后实现 (2026-09), CUDA"),
    ("MUTAG", "No Correction M=8", "programfiles/final_results_MUTAG.json",
     ("No Correction M=8",), "run_final_matrix.py", 9, 50,
     "修复后实现 (2026-09), CUDA"),
    ("MUTAG", "Full BPTT + MC", "programfiles/final_results_MUTAG.json",
     ("Full BPTT + MC",), "run_final_matrix.py", 9, 50,
     "修复后实现 (2026-09), CUDA"),
    ("MUTAG", "Proto-GNN (ANN)", "programfiles/final_results_MUTAG.json",
     ("Proto-GNN",), "run_final_matrix.py", 9, 50,
     "修复后实现 (2026-09), CUDA"),

    ("PROTEINS", "Full BPTT", "programfiles/final_results_PROTEINS.json", ("Full BPTT",),
     "run_final_matrix.py", 9, 50, "修复后实现 (2026-09), CUDA"),
    ("PROTEINS", "AMC M=8", "programfiles/final_results_PROTEINS.json",
     ("DvM M=8 truncated",), "run_final_matrix.py", 9, 50,
     "修复后实现 (2026-09), CUDA"),
    ("PROTEINS", "AMC M=4", "programfiles/final_results_PROTEINS.json",
     ("DvM M=4 truncated",), "run_final_matrix.py", 9, 50,
     "修复后实现 (2026-09), CUDA"),
    ("PROTEINS", "No Correction M=8", "programfiles/final_results_PROTEINS.json",
     ("No Correction M=8",), "run_final_matrix.py", 9, 50,
     "修复后实现 (2026-09), CUDA"),
    ("PROTEINS", "Full BPTT + MC", "programfiles/final_results_PROTEINS.json",
     ("Full BPTT + MC",), "run_final_matrix.py", 9, 50,
     "修复后实现 (2026-09), CUDA"),
    ("PROTEINS", "Proto-GNN (ANN)", "programfiles/final_results_PROTEINS.json",
     ("Proto-GNN",), "run_final_matrix.py", 9, 50,
     "修复后实现 (2026-09), CUDA"),

    # ---- 四阶段脚本（论文 +27.3pp 的来源）----
    ("MUTAG", "AMC M=8", "msg_four_stage_comparison.json",
     ("stage1_convergence", "SpikeRPGNN DvM M=8"),
     "four_stage_msg_comparison.py", 3, 50,
     "episodes_per_epoch=10, test_episodes=30; 论文 +27.3pp 的来源"),
    ("MUTAG", "MSG*", "msg_four_stage_comparison.json",
     ("stage1_convergence", "MSG*"),
     "four_stage_msg_comparison.py", 3, 50, "hidden_dim=32 (3,424 参数)"),

    # ---- 28 配置重跑 ----
    ("MUTAG", "OM (ablation=full)", "rerun_outputs/MUTAG_OM/MUTAG_1shot_full.json",
     (), "spikergnn.run_experiment (rerun_28configs.py)", 9, None,
     "与 run_final_matrix 不同的训练预算"),
    ("MUTAG", "No-Spikes", "rerun_outputs/MUTAG_NoSpikes/MUTAG_1shot_no_spikes.json",
     (), "spikergnn.run_experiment", 9, None, ""),
    ("PROTEINS", "OM (ablation=full)",
     "rerun_outputs/PROTEINS_OM/PROTEINS_1shot_full.json", (),
     "spikergnn.run_experiment", 9, None, ""),
    ("PROTEINS", "No-Spikes",
     "rerun_outputs/PROTEINS_NoSpikes/PROTEINS_1shot_no_spikes.json", (),
     "spikergnn.run_experiment", 9, None, ""),

    # ---- MSG* / GNN 基线 ----
    ("MUTAG", "MSG*", "msg_results/msgstar_results.json", ("MUTAG_1shot",),
     "msg_baseline.run_flexible", 9, None, "百分制存储, 已归一化"),
    ("PROTEINS", "MSG*", "msg_results/msgstar_results.json", ("PROTEINS_1shot",),
     "msg_baseline.run_flexible", 9, None, ""),
    ("MUTAG", "GIN", "programfiles/gnn_gin_MUTAG.json", ("results", "MUTAG", "1"),
     "train_gnn_baselines.py", 9, None, ""),
    ("MUTAG", "GCN", "programfiles/gnn_gcn_MUTAG.json", ("results", "MUTAG", "1"),
     "train_gnn_baselines.py", 9, None, ""),
    ("MUTAG", "SAGE", "programfiles/gnn_sage_MUTAG.json", ("results", "MUTAG", "1"),
     "train_gnn_baselines.py", 9, None, ""),
    ("MUTAG", "SGC", "programfiles/gnn_sgc_MUTAG.json", ("results", "MUTAG", "1"),
     "train_gnn_baselines.py", 9, None, ""),
    ("PROTEINS", "GIN", "programfiles/gnn_gin_PROTEINS.json",
     ("results", "PROTEINS", "1"), "train_gnn_baselines.py", 9, None, ""),
    ("PROTEINS", "GCN", "programfiles/gnn_gcn_PROTEINS.json",
     ("results", "PROTEINS", "1"), "train_gnn_baselines.py", 9, None, ""),
    ("PROTEINS", "SAGE", "programfiles/gnn_sage_PROTEINS.json",
     ("results", "PROTEINS", "1"), "train_gnn_baselines.py", 9, None, ""),
    ("PROTEINS", "SGC", "programfiles/gnn_sgc_PROTEINS.json",
     ("results", "PROTEINS", "1"), "train_gnn_baselines.py", 9, None, ""),
]


def _dig(d, path):
    node = d
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _accs_of(node):
    if node is None:
        return None, None, None
    if isinstance(node, dict):
        if "accs" in node:
            a = [float(x) for x in node["accs"]]
            m = node.get("mean")
            s = node.get("std")
        elif "runs" in node:
            a = [float(r["accuracy"]) for r in node["runs"]]
            m, s = None, None
        else:
            return None, None, None
    else:
        return None, None, None
    if a and max(a) > 1.5:
        a = [x / 100.0 for x in a]
        if m is not None and m > 1.5:
            m /= 100.0
        if s is not None and s > 1.5:
            s /= 100.0
    n = len(a)
    if m is None:
        m = sum(a) / n
    if s is None:
        s = (sum((x - m) ** 2 for x in a) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return a, float(m), float(s)


def file_stamp(rel):
    p = os.path.join(_HERE, rel)
    if not os.path.exists(p):
        return None, None
    import datetime
    st = os.stat(p)
    return (datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d"),
            hashlib.sha1(open(p, "rb").read()).hexdigest()[:10])


def build():
    rows = []
    for ds, method, rel, path, script, n_exp, ep, note in ENTRIES:
        p = os.path.join(_HERE, rel)
        if not os.path.exists(p):
            rows.append({"dataset": ds, "method": method, "file": rel,
                         "missing": True, "note": note})
            continue
        with open(p) as f:
            d = json.load(f)
        node = _dig(d, path) if path else d
        accs, mean, std = _accs_of(node)
        if accs is None:
            rows.append({"dataset": ds, "method": method, "file": rel,
                         "missing": True, "note": note + " (字段未找到)"})
            continue
        date, sha = file_stamp(rel)
        rows.append({
            "dataset": ds, "method": method, "file": rel, "script": script,
            "n": len(accs), "n_declared": n_exp,
            "epochs": ep if ep is not None else d.get("n_epochs"),
            "device": d.get("device"),
            "mean_pct": round(mean * 100, 2), "std_pct": round(std * 100, 2),
            "date": date, "sha1_10": sha, "note": note,
            "accs": [round(a, 4) for a in accs],
        })

    # ---- 冲突检测：同一 (dataset, method) 出现不同 mean ----
    groups = {}
    for r in rows:
        if r.get("missing"):
            continue
        groups.setdefault((r["dataset"], r["method"]), []).append(r)
    conflicts = []
    for k, rs in groups.items():
        vals = {r["mean_pct"] for r in rs}
        if len(vals) > 1:
            conflicts.append({
                "dataset": k[0], "method": k[1],
                "spread_pp": round(max(vals) - min(vals), 2),
                "values": sorted([
                    {"mean_pct": r["mean_pct"], "n": r["n"], "file": r["file"],
                     "date": r["date"], "note": r["note"]} for r in rs],
                    key=lambda x: -x["mean_pct"]),
            })
    conflicts.sort(key=lambda c: -c["spread_pp"])
    return rows, conflicts


def to_tex(rows, conflicts):
    L = []
    L.append("% AUTO-GENERATED by make_provenance_table.py -- do not hand-edit")
    L.append("\\begin{tabular}{lllrrccl}")
    L.append("\\hline")
    L.append("Dataset & Method & Source file & $n$ & Epochs & Acc.(\\%) & "
             "Date & Note\\\\")
    L.append("\\hline")
    for r in rows:
        if r.get("missing"):
            continue
        L.append(f"{r['dataset']} & {r['method']} & \\small {r['file']} & "
                 f"{r['n']} & {r['epochs']} & {r['mean_pct']:.2f}$\\pm$"
                 f"{r['std_pct']:.2f} & {r['date']} & \\small {r['note']} \\\\")
    L.append("\\hline")
    L.append("\\end{tabular}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "results_v3",
                                                  "provenance.json"))
    ap.add_argument("--tex", default=None)
    args = ap.parse_args()

    rows, conflicts = build()
    out = {"rows": rows, "conflicts": conflicts,
           "n_conflicts": len(conflicts)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print("PROVENANCE AUDIT ->", args.out)
    print("=" * 78)
    for r in rows:
        if r.get("missing"):
            print(f"  MISSING  {r['dataset']:9s} {r['method']:18s} {r['file']}")
            continue
        print(f"  {r['dataset']:9s} {r['method']:18s} {r['mean_pct']:6.2f}"
              f"±{r['std_pct']:5.2f}  n={r['n']:2d}  {r['date']}  {r['file']}")
    print(f"\nCONFLICTS: {len(conflicts)}")
    for c in conflicts:
        print(f"\n  [{c['dataset']} / {c['method']}] spread = "
              f"{c['spread_pp']:.2f} pp")
        for v in c["values"]:
            print(f"     {v['mean_pct']:6.2f}%  n={v['n']:2d}  "
                  f"{v['date']}  {v['file']}   {v['note']}")

    if args.tex:
        with open(args.tex, "w", encoding="utf-8") as f:
            f.write(to_tex(rows, conflicts))
        print("\nLaTeX ->", args.tex)


if __name__ == "__main__":
    main()
