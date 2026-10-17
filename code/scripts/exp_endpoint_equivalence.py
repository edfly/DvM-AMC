"""端点一次性乘法 vs 逐步连乘 —— 决定性等价性实证。

背景
----
用户要求把度量校正从「每步乘 λ_old²/λ_new²」改成「窗口端点一次性乘
λ_起点²/λ_终点²」来"救回精度"。但沿窗口连乘的代数恒等式表明两者在数学上
精确相等（telescoping）：∏ λ_{t-1}²/λ_t² = λ_起点²/λ_终点²。

本脚本要回答的唯一问题：
  逐步版算出的梯度爆炸（实测 ‖g_MC‖/‖g_NoCorr‖ ≈ 5e8）是否来自
  float32 连乘的舍入/溢出？若是，端点版（直接除）会显著更小 → 能救精度；
  若否（逐步版与端点版的"真值"一致），则爆炸是 Riemannian 几何真实的，
  端点一次性只是把同一爆炸更精确复算一遍 → 救不回精度。

做法（不动训练逻辑，只记录每步 λ）
--------------------------------------
1. 训练一个 truncated(M) + MC 模型若干 epoch，使嵌入离开原点；
2. 设 DVM_CAPTURE=1，在固定 batch 上 forward，记录每步的
   λ_进 (state 进入 step t) 与 λ_出 (state 离开 step t)；
3. 对保留窗口 [T-M, T-1] 内每一步 t 算 factor_t = (λ_进[t]/λ_出[t])²；
   逐步连乘 = ∏ factor_t；端点比 = (λ_进[T-M]/λ_出[T-1])²；
4. 逐节点比较两者（rel_err）并打印端点比的量级分布。

判定
----
  - 若 rel_err(逐步, 端点) 普遍 ≤ 1e-4 且端点比量级与逐步连乘量级一致
    → 两者代数等价，端点修复是 no-op，梯度爆炸是几何真实的，
      **不能救回精度**（判 NO_RECOVERY）。
  - 若端点比量级远小于逐步连乘（差 > 10× 且逐步连乘出现溢出/NaN）
    → float32 连乘确为假象，端点修复可能救回精度（判 MAY_RECOVER）。
"""
import os
import sys
import json
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PF = os.path.join(_HERE, "programfiles")
for _p in (_PF, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 必须在导入 stmp 之前打开捕获开关
os.environ["DVM_CAPTURE"] = "1"

import torch
import torch.nn.functional as F
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_cfg(T, M, mc):
    return SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, T=T, tau=20.0, ablation="full",
        use_metric_correction=mc, lr=1e-3, weight_decay=5e-4,
        episodes_per_epoch=20, test_episodes=50,
        use_dvm=True, dvm_mode="truncated", dv_m_window=M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_HERE, "results_v3",
                                                  "endpoint_equivalence.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    ds = TUGraphDataset(name=args.dataset, tu_name=args.dataset,
                        train_ratio=0.8, seed=args.seed)
    in_dim = max(ds.num_features, 1)
    sampler = EpisodeSampler(dataset=ds, n_way=2, k_shot=1, q_query=5, seed=args.seed)

    model = create_spikergnn(make_cfg(args.T, args.M, True), in_dim=in_dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # 训练若干 epoch，使嵌入离开原点（否则 λ 全≈2，看不出差异）
    model.train()
    for ep in range(args.epochs):
        for _ in range(20):
            epd = sampler.sample_episode(mode="train")
            s_b, s_l, q_b, q_l = batch_episode_data(epd, DEV)
            logits = model(s_b.x, s_b.edge_index, s_b.batch,
                           q_b.x, q_b.edge_index, q_b.batch, s_l, 2)
            loss = F.cross_entropy(logits, q_l)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # 固定 batch forward，记录每步 λ
    model.train()
    probe = sampler.sample_episode(mode="train")
    s_b, s_l, q_b, q_l = batch_episode_data(probe, DEV)
    model(s_b.x, s_b.edge_index, s_b.batch, q_b.x, q_b.edge_index, q_b.batch, s_l, 2)

    lam_pre = [t.detach().float().cpu() for t in model.spikergnn.stmp._captured_lam_pre] \
        if hasattr(model, "spikergnn") else None
    # 兼容不同的属性路径
    if lam_pre is None:
        for mod in model.modules():
            if hasattr(mod, "_captured_lam_pre") and mod._captured_lam_pre:
                lam_pre = [t.float().cpu() for t in mod._captured_lam_pre]
                lam_post = [t.float().cpu() for t in mod._captured_lam_post]
                break

    T = len(lam_pre)
    M = args.M
    start = max(0, T - M)  # 保留窗口起点 = step 索引 T-M

    # 逐节点：逐步连乘 (float32, 模拟代码) vs 端点比 (float64, 精确参考)
    lam_pre_t = torch.stack(lam_pre[start:], dim=0)   # (win, N)
    lam_post_t = torch.stack(lam_post[start:], dim=0) # (win, N)
    lam_pre_f = lam_pre_t.double()
    lam_post_f = lam_post_t.double()

    # 逐步连乘（float32，与代码一致）
    per_step = torch.ones(lam_pre_t.shape[1])
    for t in range(lam_pre_t.shape[0]):
        f = (lam_pre_t[t] / lam_post_t[t].clamp(min=1e-5)) ** 2
        per_step = per_step * f
    # 端点比（float64 精确）
    endpoint = (lam_pre_f[0] / lam_post_f[-1].clamp(min=1e-12)) ** 2

    rel = (per_step.double() - endpoint).abs() / endpoint.abs().clamp(min=1e-12)
    out = {
        "config": {"dataset": args.dataset, "seed": args.seed,
                   "epochs": args.epochs, "T": T, "M": M, "device": DEV},
        "n_nodes": int(endpoint.numel()),
        "per_step_vs_endpoint_rel_err": {
            "median": float(rel.median()),
            "p95": float(torch.quantile(rel, 0.95)),
            "max": float(rel.max()),
        },
        "endpoint_ratio_magnitude": {
            "min": float(endpoint.min()),
            "median": float(endpoint.median()),
            "p95": float(torch.quantile(endpoint, 0.95)),
            "p99": float(torch.quantile(endpoint, 0.99)),
            "max": float(endpoint.max()),
        },
        "per_step_product_magnitude": {
            "min": float(per_step.min()),
            "median": float(per_step.median()),
            "max": float(per_step.max()),
        },
        "verdict": ("NO_RECOVERY (逐步连乘 == 端点比，爆炸是几何真实的，"
                    "端点修复为 no-op)" if rel.max() < 1e-2
                    else "MAY_RECOVER (逐步连乘与端点比量级不一致，"
                         "float32 连乘可能为真象)"),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
