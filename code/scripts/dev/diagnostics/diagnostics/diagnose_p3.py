"""
P3 诊断：表征流水线（为什么输出恒定均匀分布 = 损失恒为 ln(N)）

假设：图嵌入坍缩 -> 原型坍缩 -> 测地线距离全相等 -> logits 均匀 -> 损失 ln(N)。

测试：
  T0. 症状确认：真实 episode 的 logits 是否均匀（max-min≈0）、损失是否≈ln(N)
  T1. 图嵌入坍缩：全部训练图的 z_graph 的范数/方差/两两距离
  T2. 节点级坍缩：图内节点方差 vs 图间方差（区分「过平滑」与「全坍缩」）
  T3. 判别力定位：MSPE-only（p_init 图均值）vs 最终 z_graph 的 1-NN 分类精度
  T4. 训练前后对比：untrained vs trained(12ep) 的坍缩/分离度，看训练是否起效
  T5. 分量距离：episode 中 query→prototype 的 d_H, d_E 是否退化；门控 lam
  T6. 输入流动：常数特征图嵌入是否=真实特征图嵌入（输入是否真的进入）

输出：spikergnn_results/diag_p3.json + 两张图
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN
from spikergnn.train import train_one_epoch, evaluate
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler
from msg_baseline.utils import set_seed

DEVICE = 'cpu'
torch.manual_seed(0)
MAN = None  # 全局：SpikeRPGNN.manifold，供辅助函数使用

def build_model():
    cfg = SpikeRPGNNConfig(ablation='full')
    return cfg, SpikeRPGNN(cfg, in_dim=IN_DIM)

# ---------- 指标工具 ----------
def knn1_acc_hyp(hyp, Y):
    N = hyp.shape[0]
    from spikergnn.manifolds import ProductManifold
    d = torch.zeros(N, N)
    for i in range(N):
        # hyp.dist to all
        di = MAN.hyp.dist(hyp, hyp[i].unsqueeze(0).expand_as(hyp))
        d[i] = di
    correct = 0
    for i in range(N):
        dd = d[i].clone(); dd[i] = 1e9
        j = dd.argmin()
        if int(Y[j]) == int(Y[i]): correct += 1
    return correct / N

def fused_pairwise(z_hyp, z_euc, lam):
    N = z_hyp.shape[0]
    d = torch.zeros(N, N)
    for i in range(N):
        dH = MAN.hyp.dist(z_hyp, z_hyp[i].unsqueeze(0).expand_as(z_hyp))
        dE = (z_euc - z_euc[i].unsqueeze(0).expand_as(z_euc)).norm(dim=-1)
        d[i] = lam * dH + (1-lam) * dE
    return d

def knn1_acc(Z, Y):
    """Z: (N,d) euclidean. 1-NN accuracy."""
    N = Z.shape[0]
    d = torch.cdist(Z, Z)
    correct = 0
    for i in range(N):
        dd = d[i].clone(); dd[i] = 1e9
        j = dd.argmin()
        if int(Y[j]) == int(Y[i]): correct += 1
    return correct / N

def collapse_report(z_hyp, z_euc, name):
    # 范数
    n_h = z_hyp.norm(dim=1)
    n_e = z_euc.norm(dim=1)
    # 成对距离（逐行构造 NxN 矩阵；PoincareManifold.dist 为逐行距离）
    Nn = z_hyp.shape[0]
    dH_mat = torch.zeros(Nn, Nn)
    for i in range(Nn):
        dH_mat[i] = MAN.hyp.dist(z_hyp, z_hyp[i].unsqueeze(0).expand_as(z_hyp))
    dE_mat = torch.cdist(z_euc, z_euc)
    eye = ~torch.eye(Nn, dtype=bool)
    offH = dH_mat[eye]
    offE = dE_mat[eye]
    return {
        'name': name,
        'n_graphs': int(z_hyp.shape[0]),
        'hyp_norm_mean': float(n_h.mean()),
        'hyp_norm_std': float(n_h.std()),
        'euc_norm_mean': float(n_e.mean()),
        'euc_norm_std': float(n_e.std()),
        'dH_offdiag_mean': float(offH.mean()),
        'dH_offdiag_std': float(offH.std()),
        'dE_offdiag_mean': float(offE.mean()),
        'dE_offdiag_std': float(offE.std()),
        'dH_offdiag_max': float(offH.max()),
        'dE_offdiag_max': float(offE.max()),
        'dH_offdiag_min': float(offH.min()),
        'dE_offdiag_min': float(offE.min()),
    }

# ---------- 加载 MUTAG ----------
print("载入 MUTAG ...")
DS = TUGraphDataset(name='MUTAG', tu_name='MUTAG', train_ratio=0.8, seed=42)
IN_DIM = max(DS.num_features, 1)
graphs = [g for g, _ in DS.train_graphs]
labels = [l for _, l in DS.train_graphs]
label_set = sorted(set(labels))
print(f"  MUTAG train: {len(graphs)} graphs, classes={label_set}, features={IN_DIM}")

batch = Batch.from_data_list(graphs)
X = batch.x.to(DEVICE)
EI = batch.edge_index.to(DEVICE)
BATCH = batch.batch.to(DEVICE)
Y = torch.tensor(labels, dtype=torch.long)

# label 存在性：每个类样本数
from collections import Counter
cnt = Counter(labels)
print(f"  类分布: {dict(cnt)}")

# 节点级嵌入捕获（monkey-patch global_mean_pool）
# 注意：stmp.py 在 import 时已绑定名称，必须 patch spikergnn.stmp 命名空间
import spikergnn.stmp as stmp_mod
captured = {}
_orig_gmp = stmp_mod.global_mean_pool
def _hooked_gmp(x, batch_arg, *a, **k):
    captured['node_hyp'] = x
    return _orig_gmp(x, batch_arg, *a, **k)
stmp_mod.global_mean_pool = _hooked_gmp

def encode_full(model, Xt, EIt, Bt):
    captured.clear()
    model.eval()
    with torch.no_grad():
        z = model.encode(Xt, EIt, Bt)   # graph-level (hyp, euc)
        # 节点级来自 hook
        node_hyp = captured.get('node_hyp')
    return z, node_hyp

def mspe_graph_embedding(model, Xt, EIt, Bt):
    """仅用 MSPE 输出 p_init 的图均值（跳过 STMP），定位判别力丢失位置。"""
    model.eval()
    with torch.no_grad():
        _, p_init = model.mspe(Xt, EIt, Bt)   # node-level
        ph, pe = p_init
        gh = _orig_gmp(ph, Bt)
        ge = _orig_gmp(pe, Bt)
    return gh, ge

# 创建模型
cfg, M = build_model()
M = M.to(DEVICE)
MAN = M.manifold
print(f"模型创建完成: d_h={cfg.d_h}, d_e={cfg.d_e}, hidden={cfg.hidden_dim}, T={cfg.T}")

results = {'dataset': 'MUTAG', 'n_way_real_exp': 2}

# ================= T0: 症状确认 =================
print("\n=== T0 症状确认：真实 episode logits 均匀性 ===")
sampler = EpisodeSampler(DS, n_way=2, k_shot=1, q_query=5, seed=42)
losses = []
logit_ranges = []
entropies = []
N = M.gdc.lam
for _ in range(15):
    ep = sampler.sample_episode(mode='train')
    sb, sl, qb, ql = __import__('msg_baseline.data_utils', fromlist=['batch_episode_data']).batch_episode_data(ep, device=DEVICE)
    M.stmp.lif.reset_membrane()
    sz = M.encode(sb.x, sb.edge_index, sb.batch)
    qz = M.encode(qb.x, qb.edge_index, qb.batch)
    logits = M.classify(qz, sz, sl, 2)
    los = torch.nn.functional.cross_entropy(logits, ql)
    lr = float(logits.max(1)[0].min() - logits.min(1)[0].max())  # should be ~0 if uniform per-row? no
    # per-row range
    pr = (logits.max(1)[0] - logits.min(1)[0])
    logit_ranges.append(float(pr.mean()))
    # entropy
    p = torch.softmax(logits, 1)
    ent = -(p * (p+1e-9).log()).sum(1).mean()
    entropies.append(float(ent))
    losses.append(float(los))
results['T0'] = {
    'mean_cross_entropy': float(np.mean(losses)),
    'ln2': float(math.log(2)),
    'mean_per_row_logit_range': float(np.mean(logit_ranges)),
    'mean_entropy': float(np.mean(entropies)),
    'ln2_entropy': float(math.log(2)),
    'gate_lam_before_train': float(N),
}

# ================= T1/T2/T3: 未训练状态 =================
print("\n=== T1-T3 未训练状态嵌入分析 ===")
z, node_hyp = encode_full(M, X, EI, BATCH)
zh, ze = z
# 图级坍缩报告
rep_graph = collapse_report(zh, ze, 'untrained_final')

# 节点级：图内方差 vs 图间方差
within_var = []
for gi in range(BATCH.max().item()+1):
    mask = (BATCH == gi)
    nh = node_hyp[mask]
    within_var.append(float(nh.norm(dim=1).var()))
within_var = np.mean(within_var)
# 图间（图均值）方差
graph_mean_norm = zh.norm(dim=1).var().item()
node_var = float(within_var)
graph_var = float(graph_mean_norm)

# 1-NN 分离度
acc_euc = knn1_acc(ze, Y)
acc_hyp = knn1_acc_hyp(zh, Y)
# fused
lam_cur = M.gdc.lam
d_fused = fused_pairwise(zh, ze, lam_cur)
fz = torch.zeros(len(Y), len(Y))
# reuse: compute fused knn
Nn = ze.shape[0]
correct = 0
for i in range(Nn):
    dd = d_fused[i].clone(); dd[i] = 1e9
    j = dd.argmin()
    if int(Y[j]) == int(Y[i]): correct += 1
acc_fused = correct / Nn

# MSPE-only 判别力
mh, me = mspe_graph_embedding(M, X, EI, BATCH)
acc_mspe_euc = knn1_acc(me, Y)
acc_mspe_hyp = knn1_acc_hyp(mh, Y)

results['untrained'] = {
    'graph_collapse': rep_graph,
    'node_within_graph_norm_var': node_var,
    'graph_mean_norm_var': graph_var,
    'knn1_acc_euc': acc_euc,
    'knn1_acc_hyp': acc_hyp,
    'knn1_acc_fused': acc_fused,
    'mspe_knn1_acc_euc': acc_mspe_euc,
    'mspe_knn1_acc_hyp': acc_mspe_hyp,
    'gate_lam': float(lam_cur),
}

# ================= T4: 训练 12 epoch 后对比 =================
print("\n=== T4 训练 12 epoch（观察训练是否改善嵌入）===")
M.train()
opt = torch.optim.Adam(M.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
epoch_losses = []
for ep in range(12):
    tm = train_one_epoch(M, sampler, opt, cfg, DEVICE)
    epoch_losses.append(tm['mean_loss'])
    if (ep+1) % 3 == 0:
        print(f"  epoch {ep+1:2d}: loss={tm['mean_loss']:.4f} acc={tm['mean_acc']:.3f} spikes={tm['spike_count']}")
results['training_loss_curve'] = epoch_losses

# 训练后再分析
print("  重新提取嵌入（trained）...")
z2, node_hyp2 = encode_full(M, X, EI, BATCH)
zh2, ze2 = z2
rep_graph2 = collapse_report(zh2, ze2, 'trained_final')
within_var2 = []
for gi in range(BATCH.max().item()+1):
    mask = (BATCH == gi)
    nh = node_hyp2[mask]
    within_var2.append(float(nh.norm(dim=1).var()))
within_var2 = np.mean(within_var2)
graph_mean_norm2 = zh2.norm(dim=1).var().item()
acc_euc2 = knn1_acc(ze2, Y)
acc_hyp2 = knn1_acc_hyp(zh2, Y)
lam2 = M.gdc.lam
d_fused2 = fused_pairwise(zh2, ze2, lam2)
correct2 = 0
for i in range(Nn):
    dd = d_fused2[i].clone(); dd[i] = 1e9
    j = dd.argmin()
    if int(Y[j]) == int(Y[i]): correct2 += 1
acc_fused2 = correct2 / Nn
mh2, me2 = mspe_graph_embedding(M, X, EI, BATCH)
acc_mspe_euc2 = knn1_acc(me2, Y)

results['trained'] = {
    'graph_collapse': rep_graph2,
    'node_within_graph_norm_var': float(within_var2),
    'graph_mean_norm_var': float(graph_mean_norm2),
    'knn1_acc_euc': acc_euc2,
    'knn1_acc_hyp': acc_hyp2,
    'knn1_acc_fused': acc_fused2,
    'mspe_knn1_acc_euc': acc_mspe_euc2,
    'gate_lam': float(lam2),
}

# ================= T5: episode 分量距离退化 =================
print("\n=== T5 episode 内 query->prototype 分量距离 ===")
dH_list, dE_list = [], []
for _ in range(20):
    ep = sampler.sample_episode(mode='train')
    sb, sl, qb, ql = __import__('msg_baseline.data_utils', fromlist=['batch_episode_data']).batch_episode_data(ep, device=DEVICE)
    M.stmp.lif.reset_membrane()
    sz = M.encode(sb.x, sb.edge_index, sb.batch)
    qz = M.encode(qb.x, qb.edge_index, qb.batch)
    proto = M.fpc(sz, sl, 2)
    # 成对距离（按原型循环；manifold.dist 为逐行，需同形）
    Nc = proto[0].shape[0]
    dh_mat = torch.zeros(qz[0].shape[0], Nc)
    de_mat = torch.zeros(qz[0].shape[0], Nc)
    for c in range(Nc):
        ph = proto[0][c].unsqueeze(0).expand_as(qz[0])
        pe = proto[1][c].unsqueeze(0).expand_as(qz[1])
        dh, de = M.manifold.dist(qz, (ph, pe))
        dh_mat[:, c] = dh
        de_mat[:, c] = de
    dH_list.append(float(dh_mat.mean())); dE_list.append(float(de_mat.mean()))
results['T5'] = {
    'dH_mean': float(np.mean(dH_list)),
    'dH_std': float(np.std(dH_list)),
    'dE_mean': float(np.mean(dE_list)),
    'dE_std': float(np.std(dE_list)),
    'gate_lam': float(M.gdc.lam),
}

# ================= T6: 输入流动检查 =================
print("\n=== T6 输入流动检查：常数特征 vs 真实特征 ===")
X_const = torch.ones_like(X)
z_const, _ = encode_full(M, X_const, EI, BATCH)
zhc, zec = z_const
# 真实特征图嵌入 vs 常数特征图嵌入的距离
d_real_const = torch.norm(ze - zec, dim=1).mean().item()
d_real_const_h = MAN.hyp.dist(zh, zhc).mean().item()
results['T6'] = {
    'euc_dist_real_vs_const_graph_mean': float(d_real_const),
    'euc_norm_real': float(ze.norm(dim=1).mean()),
    'euc_norm_const': float(zec.norm(dim=1).mean()),
    'input_flows': bool(d_real_const > 1e-4),
}

# ---------- 保存 ----------
# ================= T7: DvM train/eval 嵌入不一致验证 =================
print("\n=== T7 train vs eval 模式分类精度对比（核心假设验证）===")
def eval_mode_acc(model, sampler, mode_flag):
    model.train() if mode_flag == 'train' else model.eval()
    correct = 0; total = 0
    with torch.set_grad_enabled(mode_flag == 'train'):
        for _ in range(30):
            ep = sampler.sample_episode(mode='train')
            sb, sl, qb, ql = __import__('msg_baseline.data_utils', fromlist=['batch_episode_data']).batch_episode_data(ep, device=DEVICE)
            model.stmp.lif.reset_membrane()
            sz = model.encode(sb.x, sb.edge_index, sb.batch)
            qz = model.encode(qb.x, qb.edge_index, qb.batch)
            logits = model.classify(qz, sz, sl, 2)
            pred = logits.argmax(1)
            correct += int((pred == ql).sum())
            total += int(ql.shape[0])
    return correct / total

acc_train_mode = eval_mode_acc(M, sampler, 'train')
acc_eval_mode = eval_mode_acc(M, sampler, 'eval')
results['T7'] = {
    'acc_train_mode_DvM_on': float(acc_train_mode),
    'acc_eval_mode_DvM_off': float(acc_eval_mode),
    'gap': float(acc_train_mode - acc_eval_mode),
    'note': 'train模式启用DvM，eval模式不启用：若差距大则证实train/eval嵌入不一致',
}
print(f"  train模式(含DvM) acc = {acc_train_mode:.3f}")
print(f"  eval模式(无DvM) acc  = {acc_eval_mode:.3f}")
print(f"  差距 = {acc_train_mode - acc_eval_mode:.3f}")

with open('spikergnn_results/diag_p3.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n已保存 diag_p3.json")

# ---------- 绘图 ----------
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
# (a) 图嵌入范数分布 before/after
ax = axes[0,0]
ax.hist(zh.norm(1).cpu().numpy() if zh.dim()==1 else zh.norm(dim=1).detach().cpu().numpy(), bins=20, alpha=0.5, label='untrained hyp')
ax.hist(zh2.norm(dim=1).detach().cpu().numpy(), bins=20, alpha=0.5, label='trained hyp')
ax.set_title('(a) Hyperbolic graph-embedding norm'); ax.legend()
# (b) euclidean norm
ax = axes[0,1]
ax.hist(ze.norm(dim=1).detach().cpu().numpy(), bins=20, alpha=0.5, label='untrained euc')
ax.hist(ze2.norm(dim=1).detach().cpu().numpy(), bins=20, alpha=0.5, label='trained euc')
ax.set_title('(b) Euclidean graph-embedding norm'); ax.legend()
# (c) 训练损失
ax = axes[1,0]
ax.plot(range(1, len(epoch_losses)+1), epoch_losses, 'b-o')
ax.axhline(math.log(2), ls='--', c='r', label='ln(2)')
ax.set_title('(c) Training loss (12 ep)'); ax.legend()
# (d) 1-NN 对比
ax = axes[1,1]
cats = ['euc', 'hyp', 'fused', 'mspe_euc']
unt = [acc_euc, acc_hyp, acc_fused, acc_mspe_euc]
tr = [acc_euc2, acc_hyp2, acc_fused2, acc_mspe_euc2]
x = np.arange(len(cats)); w=0.35
ax.bar(x-w/2, unt, w, label='untrained')
ax.bar(x+w/2, tr, w, label='trained')
ax.axhline(0.5, ls='--', c='gray', label='random=0.5')
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_title('(d) 1-NN accuracy on embeddings'); ax.legend()
fig.suptitle('P3: Representation-pipeline diagnosis (MUTAG 2-way)', fontsize=12)
fig.tight_layout()
fig.savefig('spikergnn_results/diag_p3.png', dpi=110)
print("已保存 diag_p3.png")

# 打印摘要
print("\n================ P3 摘要 ================")
print(f"T0 loss={results['T0']['mean_cross_entropy']:.4f} (ln2={results['T0']['ln2']:.4f}), "
      f"per-row logit range={results['T0']['mean_per_row_logit_range']:.4f}, "
      f"entropy={results['T0']['mean_entropy']:.4f}")
print(f"Untrained 1-NN: euc={acc_euc:.3f} hyp={acc_hyp:.3f} fused={acc_fused:.3f} | mspe_euc={acc_mspe_euc:.3f}")
print(f"Trained   1-NN: euc={acc_euc2:.3f} hyp={acc_hyp2:.3f} fused={acc_fused2:.3f} | mspe_euc={acc_mspe_euc2:.3f}")
print(f"Hyp norm: untrained={rep_graph['hyp_norm_mean']:.3f}±{rep_graph['hyp_norm_std']:.3f} "
      f"-> trained={rep_graph2['hyp_norm_mean']:.3f}±{rep_graph2['hyp_norm_std']:.3f}")
print(f"dH offdiag: untrained mean={rep_graph['dH_offdiag_mean']:.4f} max={rep_graph['dH_offdiag_max']:.4f} "
      f"-> trained mean={rep_graph2['dH_offdiag_mean']:.4f} max={rep_graph2['dH_offdiag_max']:.4f}")
print(f"dE offdiag: untrained mean={rep_graph['dE_offdiag_mean']:.4f} max={rep_graph['dE_offdiag_max']:.4f} "
      f"-> trained mean={rep_graph2['dE_offdiag_mean']:.4f} max={rep_graph2['dE_offdiag_max']:.4f}")
print(f"T6 input flows: {results['T6']['input_flows']} (dist real vs const euc={results['T6']['euc_dist_real_vs_const_graph_mean']:.4f})")
print(f"T5 dH={results['T5']['dH_mean']:.4f} dE={results['T5']['dE_mean']:.4f} lam={results['T5']['gate_lam']:.4f}")
