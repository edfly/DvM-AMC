"""P2 诊断：检查 DvM (Differentiation-via-Manifold) 是否过度截断梯度流
对比 use_dvm=True vs False 下，各层（早期/中期/晚期）的平均梯度范数。
若 DvM 开启导致早期层梯度大幅缩小 → 梯度截断阻碍训练。
"""
import torch, sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikergnn'))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from spikergnn.train import EpisodeSampler, TUGraphDataset, set_seed, batch_episode_data
import torch.nn as nn

DEVICE = 'cpu'
N_WAY, K_SHOT, Q_QUERY = 2, 1, 5


def grad_flow_for(use_dvm, n_episodes=8):
    set_seed(42)
    config = SpikeRPGNNConfig(hidden_dim=64, d_h=32, d_e=32, T=8,
                              u_threshold=1.0, use_dvm=use_dvm, lambda_sr=0.01,
                              ablation='full')
    model = SpikeRPGNN(config, in_dim=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TUGraphDataset(name='ENZYMES', tu_name='ENZYMES')
    sampler = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                            q_query=Q_QUERY, seed=42)

    # 目标模块
    modules = {
        'mspe (earliest)': list(model.mspe.parameters()),
        'rgc_layers (mid)': [p for lyr in model.stmp.rgc_layers for p in lyr.parameters()],
        'lif.W (late)': list(model.stmp.lif.W.parameters()),
        'gdc (latest)': list(model.gdc.parameters()),
    }

    sums = {k: 0.0 for k in modules}
    counts = {k: 0 for k in modules}
    any_nan = False
    total_norm_sum = 0.0

    for ep in range(n_episodes):
        episode = sampler.sample_episode(mode='train')
        sb, sl, qb, ql = batch_episode_data(episode, device=DEVICE)
        optimizer.zero_grad()
        sz = model.encode(sb.x, sb.edge_index, sb.batch)
        qz = model.encode(qb.x, qb.edge_index, qb.batch)
        logits = model.classify(qz, sz, sl, sampler.n_way)
        loss = nn.functional.cross_entropy(logits, ql)
        loss.backward()
        for k, params in modules.items():
            gs = [p.grad.norm().item() for p in params if p.grad is not None]
            if gs:
                sums[k] += float(np.mean(gs)); counts[k] += 1
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
        total_norm_sum += gn.item()
        if not torch.isfinite(loss).all():
            any_nan = True

    return {
        'per_module_mean_grad': {k: sums[k] / max(counts[k], 1) for k in modules},
        'total_grad_norm_mean': total_norm_sum / n_episodes,
        'any_nan_or_inf': any_nan,
    }


def main():
    print("[P2] 运行 use_dvm=True ...", flush=True)
    on = grad_flow_for(True)
    print("[P2] 运行 use_dvm=False ...", flush=True)
    off = grad_flow_for(False)

    ratios = {}
    for k in on['per_module_mean_grad']:
        v_on = on['per_module_mean_grad'][k]
        v_off = off['per_module_mean_grad'][k]
        ratios[k] = v_on / v_off if v_off > 1e-12 else float('inf')

    result = {
        'dvm_on': on,
        'dvm_off': off,
        'grad_ratio_on_div_off': ratios,
    }
    with open('spikergnn_results/diag_p2.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n=== P2 梯度流诊断结果 ===", flush=True)
    print(f"{'模块':<20} {'DvM=ON':>14} {'DvM=OFF':>14} {'ratio(ON/OFF)':>14}", flush=True)
    for k in on['per_module_mean_grad']:
        print(f"{k:<20} {on['per_module_mean_grad'][k]:>14.3e} "
              f"{off['per_module_mean_grad'][k]:>14.3e} {ratios[k]:>14.3e}", flush=True)
    print(f"{'TOTAL':<20} {on['total_grad_norm_mean']:>14.3e} "
          f"{off['total_grad_norm_mean']:>14.3e} "
          f"{on['total_grad_norm_mean']/off['total_grad_norm_mean']:>14.3e}", flush=True)
    print(f"\nNaN/Inf (DvM=ON): {on['any_nan_or_inf']} | (DvM=OFF): {off['any_nan_or_inf']}", flush=True)
    print("[P2] 完成，结果已保存: spikergnn_results/diag_p2.json", flush=True)


if __name__ == '__main__':
    main()
