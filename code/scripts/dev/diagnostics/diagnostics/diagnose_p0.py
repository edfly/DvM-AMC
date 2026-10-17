"""P0 诊断：监控 LIF 神经元内部状态在训练中的变化轨迹
监控指标（每个 epoch 末采样）：
  - u_th_learnable：可学习阈值
  - W_norm：LIF 输入权重矩阵范数
  - w_input_pre：pre-clamp 输入幅度 mean(||W·LN(I_t)||/sqrt(d))
  - drive_ratio = w_input_pre / u_th （< 0.6 表示神经元欠驱动，脉冲稀少）
  - spike_count, loss, acc
"""
import torch, sys, os, math, json, time
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
N_EPOCHS = 30
N_WAY, K_SHOT, Q_QUERY = 2, 1, 5


def build():
    set_seed(42)
    config = SpikeRPGNNConfig(hidden_dim=64, d_h=32, d_e=32, T=8,
                              u_threshold=1.0, use_dvm=True, lambda_sr=0.01,
                              ablation='full')
    model = SpikeRPGNN(config, in_dim=3).to(DEVICE)
    return config, model


def main():
    config, model = build()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 betas=(0.9, 0.999), weight_decay=1e-5)
    dataset = TUGraphDataset(name='ENZYMES', tu_name='ENZYMES')
    sampler = EpisodeSampler(dataset, n_way=N_WAY, k_shot=K_SHOT,
                            q_query=Q_QUERY, seed=42)

    diag = {'epoch': [], 'loss': [], 'acc': [], 'spikes': [],
            'u_th': [], 'W_norm': [], 'w_input_pre': [], 'drive_ratio': [],
            'clamp_active': []}
    pre_mags = []

    def hook(module, inp, out):
        x_seq = inp[0]
        It = x_seq[:, 0, :]
        In = module.input_ln(It)
        w = module.W(In)
        mag = (w.norm(dim=-1) / math.sqrt(module.d_total)).mean().item()
        pre_mags.append(mag)

    handle = model.stmp.lif.register_forward_hook(hook)

    for epoch in range(N_EPOCHS):
        model.train()
        tot_loss, tot_acc, tot_spk = 0.0, 0.0, 0
        n = config.episodes_per_epoch
        pre_mags.clear()
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
            tot_loss += loss.item()
            tot_acc += acc
            tot_spk += model.count_spikes()
        diag['epoch'].append(epoch + 1)
        diag['loss'].append(tot_loss / n)
        diag['acc'].append(tot_acc / n)
        diag['spikes'].append(tot_spk)
        u_th = model.stmp.lif.u_th_learnable.item()
        w_norm = model.stmp.lif.W.weight.norm().item()
        w_pre = float(np.mean(pre_mags)) if pre_mags else 0.0
        diag['u_th'].append(u_th)
        diag['W_norm'].append(w_norm)
        diag['w_input_pre'].append(w_pre)
        diag['drive_ratio'].append(w_pre / u_th if u_th > 0 else 0.0)
        diag['clamp_active'].append(1 if w_pre > 0.6 * u_th else 0)
        print(f"[P0] Epoch {epoch+1}/{N_EPOCHS}: loss={diag['loss'][-1]:.4f} "
              f"acc={diag['acc'][-1]:.4f} spk={tot_spk} u_th={u_th:.4f} "
              f"W={w_norm:.4f} wpre={w_pre:.4f} drive={diag['drive_ratio'][-1]:.4f} "
              f"clamp={'Y' if diag['clamp_active'][-1] else 'N'}", flush=True)

    handle.remove()

    with open('spikergnn_results/diag_p0.json', 'w') as f:
        json.dump(diag, f, indent=2)

    # 绘图
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].plot(diag['epoch'], diag['u_th'], 'b-o', label='u_th')
    ax[0, 0].plot(diag['epoch'], diag['W_norm'], 'g-s', label='W_norm')
    ax[0, 0].set_title('LIF 参数演化')
    ax[0, 0].legend(); ax[0, 0].set_xlabel('epoch')

    ax[0, 1].plot(diag['epoch'], diag['w_input_pre'], 'r-o', label='w_input_pre (pre-clamp)')
    ax[0, 1].plot(diag['epoch'], diag['drive_ratio'], 'm-^', label='drive=u_th ratio')
    ax[0, 1].axhline(0.6, color='k', ls='--', label='clamp threshold (0.6)')
    ax[0, 1].set_title('输入驱动度 (drive<0.6 → 脉冲稀少)')
    ax[0, 1].legend(); ax[0, 1].set_xlabel('epoch')

    ax[1, 0].plot(diag['epoch'], diag['spikes'], 'c-o', label='spike_count')
    ax[1, 0].set_title('脉冲数 / epoch')
    ax[1, 0].legend(); ax[1, 0].set_xlabel('epoch')

    ax[1, 1].plot(diag['epoch'], diag['loss'], 'b-o', label='loss')
    ax[1, 1].plot(diag['epoch'], diag['acc'], 'r-s', label='acc (rand=0.167)')
    ax[1, 1].axhline(0.1667, color='k', ls='--', label='random baseline')
    ax[1, 1].set_title('训练 loss / acc')
    ax[1, 1].legend(); ax[1, 1].set_xlabel('epoch')

    plt.tight_layout()
    plt.savefig('spikergnn_results/diag_p0.png', dpi=100)
    print("[P0] 完成，结果已保存: spikergnn_results/diag_p0.json / .png", flush=True)


if __name__ == '__main__':
    main()
