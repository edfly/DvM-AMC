"""
summarize_matrix.py — 汇总 DvM 全 32 配置矩阵，并自动纳入方案2 的 5-shot 补跑。

读取源：
  OM        : rerun_outputs/<DS>_OM/<DS>_{shot}shot_full.json        {mean_acc,std_acc,accs,energy}
  No-Spikes : rerun_outputs/<DS>_NoSpikes/<DS>_{shot}shot_no_spikes.json
  MSG       : msg_results/msgstar_results.json                      {<DS>_1shot:{runs:[{accuracy}]}, ...}
  MC-A/B    : programfiles/final_results_<DS>.json                  {<variant>:{accs,mean,std}}  (1-shot)
  GNN       : programfiles/gnn_<m>_<DS>.json                        {results:<DS>:{'1':{...},'5':{...}}}

5-shot（rerun_om_5shot.py 产出 <DS>_5shot_*.json）一旦落盘即自动并入 OM / No-Spikes 行；
尚未完成时该项显示为 None / "—"，不会误报。

输出：
  matrix_summary.json  — 全结构化结果 (machine-checkable)
  matrix_summary.md    — 可读报告 (按数据集分表)
"""
import json, os, statistics

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
GNN = {"SGC": "sgc", "GCN": "gcn", "GIN": "gin", "SAGE": "sage"}
MC_VARIANTS = ["Full BPTT", "DvM M=8 truncated", "DvM M=4 truncated",
               "No Correction M=8", "Full BPTT + MC", "Proto-GNN"]
SHOTS = [1, 5]


def _mn(iterable):
    xs = [float(x) for x in iterable]
    return round(statistics.fmean(xs), 4), round(statistics.pstdev(xs), 4), len(xs)


def load_om_ns(ds, kind, shot):
    """Load an OM / No-Spikes cell.

    No-Spikes has TWO on-disk locations:
      <DS>_NoSpikes/       legacy — produced by the BROKEN no_spikes branch
                           (short-circuited recurrence -> NaN -> exact 50.00%).
                           Kept as evidence, never used for the tables.
      <DS>_NoSpikesFixed/  corrected code (recurrence retained, continuous
                           activation). Preferred whenever present.
    """
    if kind == "OM":
        fn = f"{ds}_{shot}shot_full.json"
        subs = [f"{ds}_OM"]
    else:
        # ONLY the corrected directory. The legacy <DS>_NoSpikes/ holds invalid
        # numbers (the broken branch diverged to NaN -> a constant 50.00%); it is
        # deliberately NOT a fallback so stale data can never leak into a table.
        fn = f"{ds}_{shot}shot_no_spikes.json"
        subs = [f"{ds}_NoSpikesFixed"]
    for sub in subs:
        p = os.path.join(ROOT, "rerun_outputs", sub, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        mean, std, n = _mn(d["accs"])
        rec = {"shot": shot, "mean": mean, "std": std, "n": n,
               "source": os.path.relpath(p, ROOT).replace("\\", "/")}
        if "energy" in d:
            rec["energy"] = d["energy"]
        return rec
    return None


def load_msg(ds, shot):
    p = os.path.join(ROOT, "msg_results", "msgstar_results.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    key = f"{ds}_{shot}shot"
    if key not in d:
        return None
    # MSG 的 accuracy 是百分比形式(0-100)，其余为小数(0-1)，统一转小数。
    accs = [r["accuracy"] / 100.0 for r in d[key]["runs"]]
    mean, std, n = _mn(accs)
    return {"shot": shot, "mean": mean, "std": std, "n": n}


def load_mc(ds):
    p = os.path.join(ROOT, "programfiles", f"final_results_{ds}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for v in MC_VARIANTS:
        if v in d:
            out[v] = {"mean": round(d[v]["mean"], 4), "std": round(d[v]["std"], 4),
                      "n": len(d[v]["accs"])}
    return out


def load_gnn(ds, shot):
    out = {}
    for method, mlow in GNN.items():
        p = os.path.join(ROOT, "programfiles", f"gnn_{mlow}_{ds}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        res = d.get("results", {}).get(ds)
        if res and str(shot) in res:
            r = res[str(shot)]
            out[method] = {"shot": shot, "mean": round(r["mean"], 4),
                           "std": round(r["std"], 4), "n": len(r["accs"])}
    return out


summary = {}
for ds in DATASETS:
    summary[ds] = {
        "OM": {f"{s}-shot": load_om_ns(ds, "OM", s) for s in SHOTS},
        "No-Spikes": {f"{s}-shot": load_om_ns(ds, "No-Spikes", s) for s in SHOTS},
        "MSG": {f"{s}-shot": load_msg(ds, s) for s in SHOTS},
        "MC-A/B": load_mc(ds),
        "GNN": {f"{s}-shot": load_gnn(ds, s) for s in SHOTS},
    }

# 完整性检查（1-shot 必须齐；5-shot 允许缺失=方案2 未完成）
missing_required, missing_5shot = [], []
for ds in DATASETS:
    for k in ("OM", "No-Spikes", "MSG"):
        if summary[ds][k]["1-shot"] is None:
            missing_required.append(f"{ds}/{k}/1-shot")
        if summary[ds][k]["5-shot"] is None and k != "MSG":
            missing_5shot.append(f"{ds}/{k}/5-shot")
    for s in SHOTS:
        if summary[ds]["GNN"][f"{s}-shot"] is None or not summary[ds]["GNN"][f"{s}-shot"]:
            missing_required.append(f"{ds}/GNN/{s}-shot")
    if summary[ds]["MC-A/B"] is None:
        missing_required.append(f"{ds}/MC-A/B")

with open(os.path.join(ROOT, "matrix_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ---- markdown 报告 ----
lines = ["# DvM 全 32 配置矩阵 — 真实实验结果汇总（含方案2 5-shot）", "",
         "全部数字由源 JSON 机器提取，未人工录入。", ""]
lines.append("> OM/No-Spikes 的 5-shot 由补跑脚本（`rerun_fixed_supplement.py`）产出，落盘后自动并入；未完成处显示为 —。")
lines.append("> **No-Spikes 已修正**：旧 `no_spikes` 分支短路过递归 → 训练发散到 NaN → 恒 50.00%（实现 bug，"
             "原文件保留于 `<DS>_NoSpikes/` 作证据）；表格读取修正版 `<DS>_NoSpikesFixed/`（保留递归、脉冲换连续激活）。")
lines.append("> ⚠ 非脉冲变体走稠密路径，其能量字段退化（0 个脉冲 ⇒「能效↓100%」无意义），不作为能效证据使用。")
lines.append("")

def fmt(rec):
    if not rec:
        return "—"
    return f"{rec['mean']*100:.2f} ± {rec['std']*100:.2f} (n={rec['n']})"

for ds in DATASETS:
    s = summary[ds]
    lines.append(f"## {ds}")
    lines.append("")
    for k in ("OM", "No-Spikes"):
        r1, r5 = s[k]["1-shot"], s[k]["5-shot"]
        extra = ""
        if k == "OM" and r1 and "energy" in r1:
            e = r1["energy"]
            extra = f"  | 能效↓{e['reduction_pct']:.2f}% (SNN {e['e_snn_mJ']:.3e} vs ANN {e['e_ann_mJ']:.3e} mJ)"
        if k == "No-Spikes" and r1 and r1.get("source"):
            extra = f"  | 源 {r1['source']}"
        lines.append(f"- **{k}**: 1-shot {fmt(r1)}{extra}  |  5-shot {fmt(r5)}")
    lines.append(f"- **MSG**: 1-shot {fmt(s['MSG']['1-shot'])}  |  5-shot {fmt(s['MSG']['5-shot'])}")
    lines.append("")
    lines.append("**MC-A/B 变体 (1-shot):**")
    lines.append("")
    lines.append("| 变体 | mean ± std | n |")
    lines.append("|---|---|---|")
    for v, rec in (s["MC-A/B"] or {}).items():
        lines.append(f"| {v} | {rec['mean']*100:.2f} ± {rec['std']*100:.2f} | {rec['n']} |")
    lines.append("")
    lines.append("**GNN 基线:**")
    lines.append("")
    lines.append("| 方法 | 1-shot | 5-shot |")
    lines.append("|---|---|---|")
    for method in GNN:
        g1 = s["GNN"]["1-shot"].get(method) if s["GNN"]["1-shot"] else None
        g5 = s["GNN"]["5-shot"].get(method) if s["GNN"]["5-shot"] else None
        lines.append(f"| {method} | {fmt(g1)} | {fmt(g5)} |")
    lines.append("")

lines.append("## 完整性")
lines.append("")
lines.append(f"- 必需项（1-shot 全表）缺失：{'无' if not missing_required else ', '.join(missing_required)}")
lines.append(f"- 方案2 5-shot 待补（OM/No-Spikes）：{len(missing_5shot)}/8 未落盘" +
             ("" if not missing_5shot else " → " + ", ".join(missing_5shot)))
lines.append("")

with open(os.path.join(ROOT, "matrix_summary.md"), "w") as f:
    f.write("\n".join(lines))

n5 = sum(1 for ds in DATASETS for k in ("OM", "No-Spikes") if summary[ds][k]["5-shot"])
print("SUMMARY WRITTEN.")
print("required missing =", missing_required if missing_required else "none")
print(f"plan-2 5-shot landed: {n5}/8")
print("OM 5-shot means:", {ds: (round(summary[ds]['OM']['5-shot']['mean']*100, 2)
                               if summary[ds]['OM']['5-shot'] else None) for ds in DATASETS})
