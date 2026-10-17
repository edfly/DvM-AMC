"""
P4 全套诊断（合并脚本）：
  P4a: ENZYMES 双曲退化确认 —— 5-way 默认配置下测 dH_offdiag / dE_offdiag / hyp_norm，
       看双曲分量是否同 MUTAG 一样退化（dH << dE）。
  P4b: 3/4/5-way 扩展性扫描 —— 判断失败是纯 N-way 容量问题，还是数据本身难。
  P4c: 5-way 修复尝试 —— 高曲率 c_init、低/高曲率、提高双曲占比 (d_h=64/d_e=64 + gate=0.8)，
       比较 1-NN 与 5-way eval 是否回升。

所有 run 用默认 128 维配置，15 epoch，episodes_per_epoch=40，test=40 episodes。
结果写入 spikergnn_results/diag_p4.json，图写入 spikergnn_results/diag_p4.png。
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
    x2 = (Z ** 2).sum(-1)                         # (N,)
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
        ck['d_e'] = de; ck['d_h'] = 128 - de
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
    tl, ta = [], []
    for ep in range(epochs):
        tm = train_one_epoch(M, sampler, opt, cfg, DEVICE)
        tl.append(tm['mean_loss']); ta.append(tm['mean_acc'])
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
                eval_acc=float(eval_acc), random=1.0 / n_way,
                knn1_euc=float(knn_e), knn1_hyp=float(knn_h),
                dH_offdiag=dH, dE_offdiag=dE, hyp_norm_mean=hnorm, euc_norm_mean=ennorm)


def main():
    DS = TUGraphDataset(name='ENZYMES', tu_name='ENZYMES', train_ratio=0.8, seed=42)
    print(f"[ENZYMES] train={len(DS.train_graphs)} classes={DS.num_classes} feats={DS.num_features}")
    results = {'p4b': [], 'p4c': []}

    # ---------- P4b: 3/4/5-way 扩展性 ----------
    for nw in [3, 4, 5]:
        print(f"-- P4b n_way={nw}")
        try:
            r = run_exp(DS, n_way=nw, cfg_kwargs={}, epochs=15)
            results['p4b'].append(r)
            print(f"   eval={r['eval_acc']:.3f} (rand={r['random']:.3f}) "
                  f"knn1_euc={r['knn1_euc']:.3f} dH={r['dH_offdiag']:.3f} "
                  f"dE={r['dE_offdiag']:.3f} hypnorm={r['hyp_norm_mean']:.3f}")
        except Exception as e:
            print("ERR", e); traceback.print_exc()
            results['p4b'].append({'n_way': nw, 'error': str(e)})

    # ---------- P4c: 5-way 修复尝试 ----------
    variants = {
        'base(dh32/de96/c1)': {},
        'c=5.0': {'c_init': 5.0},
        'c=0.3': {'c_init': 0.3},
        'dh64/de64/gate0.8': {'d_h': 64, 'd_e': 64, 'gate_init': 0.8},
    }
    for name, kw in variants.items():
        print(f"-- P4c {name}")
        try:
            r = run_exp(DS, n_way=5, cfg_kwargs=kw, epochs=15)
            r['variant'] = name
            results['p4c'].append(r)
            print(f"   eval={r['eval_acc']:.3f} (rand=0.2) knn1_euc={r['knn1_euc']:.3f} "
                  f"dH={r['dH_offdiag']:.3f} dE={r['dE_offdiag']:.3f}")
        except Exception as e:
            print("ERR", e); traceback.print_exc()
            results['p4c'].append({'variant': name, 'error': str(e)})

    with open('spikergnn_results/diag_p4.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("saved diag_p4.json")
    plot(results)


def plot(results):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    b = [r for r in results['p4b'] if 'eval_acc' in r]
    # P4b eval
    ax = axes[0, 0]
    ax.bar([str(r['n_way']) for r in b], [r['eval_acc'] for r in b], color='#4caf50')
    for r in b:
        ax.axhline(r['random'], ls='--', c='gray')
    ax.set_title('P4b: eval acc vs n_way'); ax.set_xlabel('n_way')

    # P4b loss
    ax = axes[0, 1]
    for r in b:
        ax.plot(r['train_loss'], label=f"{r['n_way']}-way")
    ax.axhline(-np.log(1/5), ls=':', c='red', label='-ln(1/5)')
    ax.set_title('P4b: train loss'); ax.legend(fontsize=8)

    # P4b knn1
    ax = axes[0, 2]
    ax.bar([str(r['n_way']) for r in b], [r['knn1_euc'] for r in b], color='#2196f3')
    ax.axhline(0.5, ls=':', c='gray'); ax.set_title('P4b: embed 1-NN (euc)')

    c = [r for r in results['p4c'] if 'eval_acc' in r]
    names = [r['variant'] for r in c]
    # P4c eval
    ax = axes[1, 0]
    ax.bar(range(len(names)), [r['eval_acc'] for r in c], color='#f44336')
    ax.axhline(0.2, ls='--', c='gray', label='rand=0.2')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('P4c: 5-way eval by variant'); ax.legend(fontsize=8)

    # P4c dH vs dE
    ax = axes[1, 1]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, [r['dH_offdiag'] for r in c], w, label='dH_offdiag')
    ax.bar(x + w/2, [r['dE_offdiag'] for r in c], w, label='dE_offdiag')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('P4c: hyperbolic vs euclidean spread'); ax.legend(fontsize=8)

    # P4c knn1
    ax = axes[1, 2]
    ax.bar(range(len(names)), [r['knn1_euc'] for r in c], color='#9c27b0')
    ax.axhline(0.5, ls=':', c='gray')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title('P4c: embed 1-NN (euc)')

    plt.suptitle('P4 diagnosis: ENZYMES 5-way failure root cause', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('spikergnn_results/diag_p4.png', dpi=120)
    print("saved diag_p4.png")


if __name__ == '__main__':
    main()
