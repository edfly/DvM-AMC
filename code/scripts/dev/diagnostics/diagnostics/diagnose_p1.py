"""P1 诊断：
  (a) full(full SNN) vs no_spikes(无脉冲) 在相同训练下的性能对比
      → 回答脉冲动力学是否拖累性能
  (b) 学习率扫描 lr ∈ {1e-4, 1e-3, 5e-3}
      → 排除学习率不当导致不收敛
"""
import torch, sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikergnn'))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from spikergnn.train import EpisodeSampler, TUGraphDataset, set_seed, batch_episode_data
import torch.nn as nn

DEVICE = 'cpu'
N_WAY, K_SHOT, Q_QUERY = 2, 1, 5


def run_phase(ablation, lr, n_epochs, seed=42, use_dvm=True):
    set_seed(seed)
    config = SpikeRPGNNConfig(hidden_dim=64, d_h=32, d_e=32, T=8,
                              u_threshold=1.0, use_dvm=use_dvm, lambda_sr=0.01,
                              ablation=ablation)
    model = SpikeRPGNN(config, in_dim=3).to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999), weight_decay=1e-5)
    dataset = TUGraphDataset(name='ENZYMES', tu_name='ENZYMES')
    sampler = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                            q_query=Q_QUERY, seed=seed)
    diag = {'epoch': [], 'loss': [], 'acc': [], 'spikes': []}
    for epoch in range(n_epochs):
        model.train()
        tot_loss, tot_acc, tot_spk = 0.0, 0.0, 0
        n = config.episodes_per_epoch
        for ep in range(n):
            episode = sampler.sample_episode(mode='train')
            sb, sl, qb, ql = batch_episode_data(episode, device=DEVICE)
            optimizer.zero_grad()
            sz = model.encode(sb.x, sb.edge_index, sb.batch)
            qz = model.encode(qb.x, qb.edge_index, qb.batch)
            logits = model.classify(qz, sz, sl, sampler.n_way)
            loss = nn.functional.cross_entropy(logits, ql)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            acc = (logits.detach().argmax(-1) == ql).float().mean().item()
            tot_loss += loss.item(); tot_acc += acc; tot_spk += model.count_spikes()
        diag['epoch'].append(epoch + 1)
        diag['loss'].append(tot_loss / n)
        diag['acc'].append(tot_acc / n)
        diag['spikes'].append(tot_spk)
        print(f"[{ablation}/lr={lr}] Ep {epoch+1}/{n_epochs}: loss={diag['loss'][-1]:.4f} "
              f"acc={diag['acc'][-1]:.4f} spk={tot_spk}", flush=True)
    return config, diag


def main():
    results = {}
    # (a) full vs no_spikes @ lr=1e-3, 30 ep
    _, results['full_lr1e-3'] = run_phase('full', 1e-3, 30)
    _, results['no_spikes_lr1e-3'] = run_phase('no_spikes', 1e-3, 30)
    # (b) lr sweep on full, 20 ep each
    _, results['full_lr1e-4'] = run_phase('full', 1e-4, 20)
    _, results['full_lr5e-3'] = run_phase('full', 5e-3, 20)

    with open('spikergnn_results/diag_p1.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 对比图
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'full_lr1e-3': 'b', 'no_spikes_lr1e-3': 'r',
              'full_lr1e-4': 'g', 'full_lr5e-3': 'm'}
    for key, c in colors.items():
        d = results[key]
        ax[0].plot(d['epoch'], d['loss'], color=c, marker='o', label=key)
        ax[1].plot(d['epoch'], d['acc'], color=c, marker='s', label=key)
        ax[2].plot(d['epoch'], d['spikes'], color=c, marker='^', label=key)
    ax[0].set_title('P1: Loss 对比'); ax[1].set_title('P1: Accuracy 对比 (random=0.167)'); ax[2].set_title('P1: Spikes 对比')
    for a in ax:
        a.axhline(0.1667, color='k', ls='--', alpha=0.5)
        a.legend(fontsize=8); a.set_xlabel('epoch')
    plt.tight_layout()
    plt.savefig('spikergnn_results/diag_p1.png', dpi=100)
    print("[P1] 完成，结果已保存: spikergnn_results/diag_p1.json / .png", flush=True)


if __name__ == '__main__':
    main()
