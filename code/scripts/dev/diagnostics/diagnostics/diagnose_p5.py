"""
P5 编码器修复诊断：
  在 ENZYMES 5-way 1-shot 上扫描节点预处理 MLP 维度 node_pre_dim ∈ {0,16,32,64}，
  并加一个组合变体 pre_dim=64 + num_layers=4。
  目标验证假设：ENZYMES 失败根因是 3-dim 特征贫乏 → 预 MLP 把特征抬升后，
  编码器 1-NN 与 5-way eval 应明显回升，且末轮 spike 数从 0/接近 0 复活。

所有 run 用默认 128 维配置，15 epoch，episodes_per_epoch=40，test=40 episodes。
结果写入 spikergnn_results/diag_p5.json，图写入 spikergnn_results/diag_p5.png。
"""
import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from torch_geometric.data import Batch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from spikergnn.train import train_one_epoch
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data

DEVICE = 'cpu'


def knn1_euc(Z, Y):
    d = torch.cdist(Z, Z); N = Z.shape[0]; c = 0
    for i in range(N):
        dd = d[i].clone(); dd[i] = 1e9; j = int(dd.argmin())
        if int(Y[j]) == int(Y[i]): c += 1
    return c / N


def poincare_pairwise(Z):
    Z = Z.double() + 1e-8
    x2 = (Z ** 2).sum(-1)
    num = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    den = (1 - x2[:, None]) * (1 - x2[None, :])
    val = (1 + 2 * num / (den + 1e-12)).clamp(min=1 + 1e-6)
    return torch.acosh(val)


def offdiag_mean(D):
    n = D.shape[0]; mask = ~torch.eye(n, dtype=bool)
    return D[mask].mean().item()


def make_cfg(overrides):
    base = dict(hidden_dim=128, num_layers=3, use_dvm=True, ablation='full',
                seed_base=42, episodes_per_epoch=40, test_episodes=40)
    ck = dict(base)
    dh = overrides.pop('d_h', None); de = overrides.pop('d_e', None)
    if dh is not None and de is not None:
        ck['d_h'] = dh; ck['d_e'] = de
    elif dh is not None:
        ck['d_h'] = dh; ck['d_e'] = 128 - dh
    elif de is not None:
        ck['d_e'] = de; ck['d_h'] = 128 - dh
    ck.update(overrides)
    cfg = SpikeRPGNNConfig(**ck)
    cfg.episodes_per_epoch = 40; cfg.test_episodes = 40
    return cfg


def run_exp(dataset, n_way, cfg_kwargs, k_shot=1, epochs=15, seed=42):
    cfg = make_cfg(dict(cfg_kwargs))
    in_dim = max(dataset.num_features, 1)
    M = SpikeRPGNN(cfg, in_dim=in_dim).to(DEVICE)
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot, q_query=5, seed=seed)
    opt = torch.optim.Adam(M.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tl, ta, tsc = [], [], []
    for ep in range(epochs):
        tm = train_one_epoch(M, sampler, opt, cfg, DEVICE)
        tl.append(tm['mean_loss']); ta.append(tm['mean_acc']); tsc.append(tm['spike_count'])
    M.eval(); correct = 0; total = 0
    for _ in range(40):
        ep = sampler.sample_episode(mode='test')
        sb, sl, qb, ql = batch_episode_data(ep, device=DEVICE)
        M.stmp.lif.reset_membrane()
        sz = M.encode(sb.x, sb.edge_index, sb.batch)
        qz = M.encode(qb.x, qb.edge_index, qb.batch)
        logits = M.classify(qz, sz, sl, n_way)
        pred = logits.argmax(1)
        correct += int((pred == ql).sum()); total += int(ql.shape[0])
    eval_acc = correct / total
    graphs = [g for g, _ in dataset.train_graphs]
    labels = [l for _, l in dataset.train_graphs]
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        z = M.encode(batch.x.to(DEVICE), batch.edge_index.to(DEVICE), batch.batch.to(DEVICE))
    zh, ze = z
    Y = torch.tensor(labels)
    knn_e = knn1_euc(ze, Y)
    zh_t = M.manifold.hyp.logmap0(zh)
    knn_h = knn1_euc(zh_t, Y)
    dH = offdiag_mean(poincare_pairwise(zh))
    dE = offdiag_mean(torch.cdist(ze, ze))
    hnorm = float(zh.norm(dim=1).mean())
    ennorm = float(ze.norm(dim=1).mean())
    return dict(n_way=n_way, cfg=cfg_kwargs, train_loss=tl, train_acc=ta,
                spike_last=int(tsc[-1]), spike_first=int(tsc[0]),
                eval_acc=float(eval_acc), random=1.0 / n_way,
                knn1_euc=float(knn_e), knn1_hyp=float(knn_h),
                dH_offdiag=dH, dE_offdiag=dE, hyp_norm_mean=hnorm, euc_norm_mean=ennorm)


def main():
    DS = TUGraphDataset(name='ENZYMES', tu_name='ENZYMES', train_ratio=0.8, seed=42)
    print(f"[ENZYMES] train={len(DS.train_graphs)} classes={DS.num_classes} feats={DS.num_features}")
    results = []

    variants = {
        'baseline(pre=0)': dict(node_pre_dim=0),
        'pre=16': dict(node_pre_dim=16),
        'pre=32': dict(node_pre_dim=32),
        'pre=64': dict(node_pre_dim=64),
        'pre=64+nl4': dict(node_pre_dim=64, num_layers=4),
    }
    for name, kw in variants.items():
        print(f"-- P5 {name}  kw={kw}")
        try:
            r = run_exp(DS, n_way=5, cfg_kwargs=kw, epochs=15)
            r['variant'] = name
            results.append(r)
            print(f"   eval={r['eval_acc']:.3f} (rand=0.2) knn1_euc={r['knn1_euc']:.3f} "
                  f"dH={r['dH_offdiag']:.3f} dE={r['dE_offdiag']:.3f} "
                  f"spike0={r['spike_first']} spikeLast={r['spike_last']}")
        except Exception as e:
            print("ERR", e); traceback.print_exc()
            results.append({'variant': name, 'error': str(e)})

    with open('spikergnn_results/diag_p5.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("saved diag_p5.json")
    plot(results)


def plot(results):
    good = [r for r in results if 'eval_acc' in r]
    names = [r['variant'] for r in good]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    ax = axes[0, 0]
    ax.bar(range(len(names)), [r['eval_acc'] for r in good], color='#4caf50')
    ax.axhline(0.2, ls='--', c='gray', label='rand=0.2')
    ax.axhline(0.357, ls=':', c='orange', label='3-way best P4')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('P5: 5-way eval by pre_dim'); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(range(len(names)), [r['knn1_euc'] for r in good], color='#2196f3')
    ax.axhline(0.5, ls=':', c='gray', label='0.5')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('P5: encoder embed 1-NN (euc)'); ax.legend(fontsize=8)

    ax = axes[0, 2]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, [r['dH_offdiag'] for r in good], w, label='dH_offdiag')
    ax.bar(x + w/2, [r['dE_offdiag'] for r in good], w, label='dE_offdiag')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('P5: hyperbolic vs euclidean spread'); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for r in good:
        ax.plot(r['train_loss'], label=r['variant'])
    ax.axhline(-np.log(1/5), ls=':', c='red', label='-ln(1/5)')
    ax.set_title('P5: train loss'); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.bar(range(len(names)), [r['spike_last'] for r in good], color='#9c27b0')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('P5: last-epoch spike count'); ax.set_ylabel('spikes')

    ax = axes[1, 2]
    ax.bar(range(len(names)), [r['knn1_hyp'] for r in good], color='#ff9800')
    ax.axhline(0.5, ls=':', c='gray')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('P5: embed 1-NN (hyp tangent)')

    plt.suptitle('P5 encoder repair: node pre-MLP sweep on ENZYMES 5-way', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('spikergnn_results/diag_p5.png', dpi=120)
    print("saved diag_p5.png")


if __name__ == '__main__':
    main()
