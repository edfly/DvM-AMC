"""
diagnose_no_spikes.py — 判定 no_spikes 恒等 50% 是"真实现象"还是"实现 artifact"。

假设（静态代码分析得出）：stmp.forward 在 no_spikes 模式下 spike_mask 恒为 1
(因 neurons.no_spikes 分支返回 spikes=ones)，于是 p_hyp 每步被覆盖为 RGC 输出
z_hyp_new；随后 lif.no_spikes 分支再算 p_final = projx(p_hyp + 0.1*x_hyp_mean)，
而 x_hyp_mean == z_hyp_new 本身 → p_final ≈ projx(1.1 * z_hyp_new)。
双曲点被反复 1.1x 放大后投影回 Poincaré 球，可能把表示压到边界、类间塌缩，
最终 logits 恒定 → 预测恒定 → 精度 = 多数类比例(~50%)。

本脚本在 CPU、MUTAG、少量 epoch 上直接测量：
  1. 训练/测试 acc 轨迹（训练是否有效）
  2. z_hyp 范数分布（是否逼近 1 = 边界塌缩）
  3. 类内/类间表示距离（是否有判别信息）
  4. logits 是否对全部 query 恒定
并对比 full 模式作对照。
"""
import os, sys, math

_THIS = os.path.dirname(os.path.abspath(__file__))
_PF = os.path.dirname(_THIS)          # programfiles
sys.path.insert(0, _PF)

import torch
import torch.optim as optim

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed, compute_accuracy
from spikergnn.config import get_spikergnn_config
from spikergnn.model import create_spikergnn
from spikergnn.train import train_one_epoch, evaluate


def build(ablation, ds="MUTAG", shot=1, run_id=0, epochs=15, seed_base=0):
    cfg = get_spikergnn_config(ablation=ablation)
    cfg.n_runs = 1
    cfg.epochs = epochs
    ds_cfg = DATASET_CONFIGS[ds]
    seed = seed_base + run_id * 100 + shot * 10
    set_seed(seed)
    dataset = TUGraphDataset(name=ds, tu_name=ds_cfg["tu_name"],
                             train_ratio=cfg.train_ratio, seed=seed)
    sampler = EpisodeSampler(dataset=dataset, n_way=ds_cfg["n_way"],
                             k_shot=shot, q_query=cfg.q_query, seed=seed)
    model = create_spikergnn(cfg, in_dim=max(dataset.num_features, 1)).to("cpu")
    return cfg, dataset, sampler, model


def probe_representations(model, sampler, n_ep=8):
    """采样若干 test episode，测量 z_hyp 范数、类内/类间距离、logits 恒定性。"""
    model.eval()
    hyp_norms, within, between, logit_spread, preds = [], [], [], [], []
    with torch.no_grad():
        for _ in range(n_ep):
            ep = sampler.sample_episode(mode="test")
            sb, sl, qb, ql = batch_episode_data(ep, device="cpu")
            model.stmp.lif.reset_membrane()
            sz = model.encode(sb.x, sb.edge_index, sb.batch)
            qz = model.encode(qb.x, qb.edge_index, qb.batch)
            # 双曲范数（Poincaré，应 <1；≈1 表示边界塌缩）
            zh = qz[0]
            hyp_norms.append(zh.norm(dim=-1).mean().item())
            # 类内 / 类间欧氏距离（用合并表示）
            zh_n = zh.norm(dim=-1)
            classes = ql.unique()
            for c in classes:
                m = ql == c
                if m.sum() > 1:
                    within.append((zh[m] - zh[m].mean(0)).norm(dim=-1).mean().item())
            if len(classes) > 1:
                for i in range(len(classes)):
                    for j in range(i + 1, len(classes)):
                        ci, cj = classes[i], classes[j]
                        between.append((zh[ql == ci].mean(0) - zh[ql == cj].mean(0)).norm().item())
            logits = model.classify(qz, sz, sl, sampler.n_way)
            # logits 每行的极差（0 = 完全恒定，无判别）
            logit_spread.append((logits.max(dim=1).values - logits.min(dim=1).values).mean().item())
            preds.append(logits.argmax(dim=1).float().mean().item())  # 预测为正类的比例
    def _m(x):
        return float(torch.tensor(x).mean()) if x else float("nan")
    return {
        "hyp_norm_mean": _m(hyp_norms),
        "within_class_dist": _m(within),
        "between_class_dist": _m(between),
        "logit_row_spread": _m(logit_spread),      # → 0 表示无判别
        "pred_pos_rate_mean": _m(preds),           # → 0 或 1 表示恒定预测
        "pred_pos_rate_std": float(torch.tensor(preds).std()) if len(preds) > 1 else float("nan"),
    }


def run(ablation):
    print(f"\n{'='*64}\n ABLATION = {ablation}\n{'='*64}")
    cfg, dataset, sampler, model = build(ablation)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        tm = train_one_epoch(model, sampler, opt, cfg, "cpu")
        if (ep + 1) % 5 == 0 or ep == 0:
            em = evaluate(model, sampler, cfg, "cpu")
            print(f"  epoch {ep+1:2d}: train_acc={tm['mean_acc']:.4f} "
                  f"loss={tm['mean_loss']:.4f} test_acc={em['mean_acc']:.4f} "
                  f"spikes={tm['spike_count']}")
    probe = probe_representations(model, sampler)
    print("  --- representation probe ---")
    for k, v in probe.items():
        print(f"    {k:22s} = {v:.6f}" if v == v else f"    {k:22s} = nan")
    final = evaluate(model, sampler, cfg, "cpu")
    print(f"  FINAL test_acc = {final['mean_acc']:.4f}")
    return probe, final


if __name__ == "__main__":
    for abl in ["no_spikes", "full"]:
        run(abl)
