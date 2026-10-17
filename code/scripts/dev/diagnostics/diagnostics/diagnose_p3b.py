"""
P3b：高 N-way 是否成立？ — ENZYMES 5-way 1-shot（默认 128 维）验证

对比 P3（2-way MUTAG 128 维 → 0.797）vs 真实实验（5-way ENZYMES 128 维 → 24%）。
若 5-way 仍失败，则根因在「高 N-way 原型分类 / 距离退化」，而非维度或脉冲。

度量：
  - 训练损失/精度曲线（15 epoch）
  - 最终 eval 精度（5-way 1-shot，50 episode 平均）
  - 全训练集图嵌入的 1-NN 精度（6 类的全局可分性）
  - 每类嵌入类内散布 vs 类间距离（separability ratio）
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from torch_geometric.data import Batch
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from spikergnn.train import train_one_epoch, evaluate
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler
from msg_baseline.utils import set_seed

DEVICE = 'cpu'
torch.manual_seed(0)

DS = TUGraphDataset(name='ENZYMES', tu_name='ENZYMES', train_ratio=0.8, seed=42)
IN_DIM = max(DS.num_features, 1)
N_WAY = 5
print(f"[ENZYMES] train_graphs={len(DS.train_graphs)}, classes={DS.num_classes}, features={IN_DIM}, n_way={N_WAY}")

# 全训练集图嵌入（用于 1-NN / 类分离）
graphs = [g for g, _ in DS.train_graphs]
labels = [l for _, l in DS.train_graphs]
batch = Batch.from_data_list(graphs)
X = batch.x.to(DEVICE); EI = batch.edge_index.to(DEVICE); BATCH = batch.batch.to(DEVICE)
Y = torch.tensor(labels, dtype=torch.long)
from collections import Counter
print(f"  类分布: {dict(Counter(labels))}")

MAN = None
cfg = SpikeRPGNNConfig(hidden_dim=128, num_layers=3, use_dvm=True, ablation='full')
print(f"config: hidden={cfg.hidden_dim} d_h={cfg.d_h} d_e={cfg.d_e} T={cfg.T}")
M = SpikeRPGNN(cfg, in_dim=IN_DIM).to(DEVICE)
MAN = M.manifold

def knn1_euc(Z, Y):
    d = torch.cdist(Z, Z); Nn = Z.shape[0]; c=0
    for i in range(Nn):
        dd = d[i].clone(); dd[i]=1e9; j=dd.argmin()
        if int(Y[j])==int(Y[i]): c+=1
    return c/Nn

def per_class_separability(z_euc, Y):
    # 类内散布（平均类内两两距离）/ 类间距离（平均类质心两两距离）
    classes = sorted(set(Y.tolist()))
    cents = []
    within = []
    for c in classes:
        zc = z_euc[Y==c]
        cents.append(zc.mean(0))
        if zc.shape[0] > 1:
            within.append(torch.cdist(zc, zc).mean().item())
    cents = torch.stack(cents)
    between = torch.cdist(cents, cents)
    off = between[~torch.eye(len(cents), dtype=bool)]
    return float(np.mean(within)), float(off.mean())

# ---------- 训练 ----------
sampler = EpisodeSampler(DS, n_way=N_WAY, k_shot=1, q_query=5, seed=42)
opt = torch.optim.Adam(M.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
train_acc_curve, train_loss_curve = [], []
for ep in range(15):
    tm = train_one_epoch(M, sampler, opt, cfg, DEVICE)
    train_loss_curve.append(tm['mean_loss']); train_acc_curve.append(tm['mean_acc'])
    if (ep+1)%3==0:
        print(f"  epoch {ep+1:2d}: loss={tm['mean_loss']:.4f} acc={tm['mean_acc']:.3f} spikes={tm['spike_count']}")

# ---------- eval（5-way）----------
M.eval()
correct=0; total=0
for _ in range(50):
    ep = sampler.sample_episode(mode='test')
    sb, sl, qb, ql = __import__('msg_baseline.data_utils', fromlist=['batch_episode_data']).batch_episode_data(ep, device=DEVICE)
    M.stmp.lif.reset_membrane()
    sz = M.encode(sb.x, sb.edge_index, sb.batch)
    qz = M.encode(qb.x, qb.edge_index, qb.batch)
    logits = M.classify(qz, sz, sl, N_WAY)
    pred = logits.argmax(1)
    correct += int((pred==ql).sum()); total += int(ql.shape[0])
eval_acc = correct/total

# ---------- 嵌入可分性 ----------
M.eval()
with torch.no_grad():
    z = M.encode(X, EI, BATCH)
zh, ze = z
knn_euc = knn1_euc(ze, Y)
knn_hyp = knn1_euc(MAN.hyp.logmap0(zh), Y)  # 在切空间欧式化后 1-NN
within, between = per_class_separability(ze, Y)

random_baseline = 1.0/N_WAY
results = {
    'dataset':'ENZYMES', 'n_way': N_WAY,
    'train_loss_curve': train_loss_curve,
    'train_acc_curve': train_acc_curve,
    'final_eval_acc': float(eval_acc),
    'random_baseline': float(random_baseline),
    'embed_knn1_euc': float(knn_euc),
    'embed_knn1_hyp_tangent': float(knn_hyp),
    'class_within_spread': within,
    'class_between_dist': between,
}
with open('spikergnn_results/diag_p3b.json','w') as f:
    json.dump(results, f, indent=2)

print("\n========== P3b 摘要（ENZYMES 5-way 1-shot）==========")
print(f"训练末损失={train_loss_curve[-1]:.4f}  训练精度={train_acc_curve[-1]:.3f}")
print(f"最终 5-way eval 精度 = {eval_acc:.4f}  (随机基线={random_baseline:.4f})")
print(f"全训练集嵌入 1-NN: euc={knn_euc:.3f} hyp_tangent={knn_hyp:.3f}")
print(f"类内散布={within:.3f}  类间距离={between:.3f}  ratio(内/间)={within/max(between,1e-6):.3f}")

# 绘图
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax=axes[0]
ax.plot(range(1,len(train_loss_curve)+1), train_loss_curve,'b-o',label='loss')
ax.axhline(-math.log(random_baseline),ls='--',c='r',label='-ln(1/N)')
ax.set_title('train loss'); ax.legend()
ax=axes[1]
ax.plot(range(1,len(train_acc_curve)+1), train_acc_curve,'g-o',label='train acc')
ax.axhline(random_baseline,ls='--',c='gray',label='random')
ax.axhline(eval_acc,ls='-',c='orange',label=f'eval={eval_acc:.2f}')
ax.set_title('train/eval acc'); ax.legend()
fig.suptitle('P3b: ENZYMES 5-way 1-shot (default 128-dim)', fontsize=12)
fig.tight_layout(); fig.savefig('spikergnn_results/diag_p3b.png', dpi=110)
print("已保存 diag_p3b.json / diag_p3b.png")
