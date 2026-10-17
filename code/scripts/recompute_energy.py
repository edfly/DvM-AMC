"""Recompute the energy table with two accounting scopes (P0-C).

Scope A ("stmp_only", as originally reported): snn_ops from the matrix JSONs
  = STMP LIF spikes only  (model.count_spikes() covers stmp only).
Scope B ("full", conservative): snn_ops = STMP + MSPE, where the MSPE spike
  count is measured on the eval protocol (50 test episodes x support+query
  forwards) with an instrumented untrained model -- the MSPE rate is a
  structural constant (clamp-saturated; verified at exactly 50.0% over 12
  training epochs on MUTAG/NCI1 and at init on all 4 datasets, see
  programfiles/diagnostics/probe_mspe_rate.py).

ANN side is unchanged (params x T dense ops).

Output: rerun_outputs/energy_recount.json + matrix_energy_recount.md
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "rerun_outputs")
SNN_PJ, ANN_PJ = 0.9, 4.6

DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]


def main():
    rows = []
    for ds in DATASETS:
        mat = json.load(open(os.path.join(OUT, f"{ds}_OM", f"{ds}_1shot_full.json")))
        probe = json.load(open(os.path.join(OUT, "diagnostics", f"mspe_rate_{ds}.json")))
        e = mat["energy"]
        stmp_ops = e["snn_ops"]
        ann_ops = e["ann_ops"]
        mspe_ops = int(round(probe["mspe_ops_per_eval"]))
        full_ops = stmp_ops + mspe_ops

        def eng(ops):
            e_snn = ops * SNN_PJ * 1e-9
            e_ann = ann_ops * ANN_PJ * 1e-9
            return e_snn, (1 - e_snn / e_ann) * 100

        e_stmp, red_stmp = eng(stmp_ops)
        e_full, red_full = eng(full_ops)
        rows.append({
            "dataset": ds,
            "acc_pct": mat["mean_acc"] * 100,
            "stmp_ops": stmp_ops,
            "mspe_ops_est": mspe_ops,
            "mspe_fire_pct": probe["mspe_fire_pct"],
            "full_ops": full_ops,
            "ann_ops": ann_ops,
            "e_snn_stmp_mJ": e_stmp, "reduction_stmp_pct": red_stmp,
            "e_snn_full_mJ": e_full, "reduction_full_pct": red_full,
        })

    out = {"scope_A": "STMP-only (as originally reported)",
           "scope_B": "STMP + MSPE (measured eval-protocol estimate)",
           "rows": rows}
    with open(os.path.join(OUT, "energy_recount.json"), "w") as f:
        json.dump(out, f, indent=1)

    lines = [
        "# 能耗口径重算（P0-C）：两套口径并列",
        "",
        "| 数据集 | acc | STMP ops | +MSPE ops | 全量 ops | ANN ops | 降幅(STMP-only) | 降幅(全量) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['acc_pct']:.2f}% | {r['stmp_ops']:,} "
            f"| +{r['mspe_ops_est']:,} | {r['full_ops']:,} | {r['ann_ops']:,} "
            f"| {r['reduction_stmp_pct']:.2f}% | **{r['reduction_full_pct']:.2f}%** |"
        )
    lines += [
        "",
        "- **口径 A（原上报）**：`snn_ops = STMP LIF 脉冲数`（`model.count_spikes()` 只统计 stmp）。",
        "- **口径 B（全量，保守）**：`snn_ops += MSPE LIF 脉冲数`；MSPE 在 eval 协议上实测"
        "（50 test episodes × support+query 前向，未训练模型 + 结构性钉死率），"
        "钉死率已在 MUTAG/NCI1 训练 12 epoch 中逐 epoch 验证恰为 50.0%、4 数据集 init 探针 46.97–50.00%。",
        "- ANN 侧不变（params × T 稠密 ops）。**论文主张应以口径 B 为准**（口径 A 是乐观下界）。",
    ]
    with open(os.path.join(ROOT, "matrix_energy_recount.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
