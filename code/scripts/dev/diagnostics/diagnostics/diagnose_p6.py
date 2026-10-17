"""
P6 — Oracle 对照实验：纯 GIN + 欧氏 ProtoNet（砍掉 SNN 与流形）

目的：仲裁 ENZYMES 5-way 失败的两大互斥假设：
  (X) 数据本身极难（连普通方法也学不动）
  (Y) 失败专属我们的 SNN+流形+RGC 架构（普通 GIN 能学而我们不能）

协议严格对齐真实实验：
  - 同一 ENZYMES 数据集 / TUGraphDataset(train_ratio=0.8, seed=42)
  - 同一 EpisodeSampler(5-way 1-shot, q_query=5, seed=42)
  - 同一 lr=1e-3, weight_decay=5e-4, CosineAnnealingLR(T_max=15, eta_min=1e-5)
  - 同一 hidden_dim=128, num_layers=3
  - 15 epoch 训练 + 50-episode eval + 全训练集 6-class 1-NN（欧氏）可分性

若 oracle eval >> 0.23 → 证明失败专属我们的架构 (Y)，转 P6b 查 RGC/梯度。
若 oracle eval ≈ 0.20     → 证明数据本身难 (X)，走 B 路线（ENZYMES 降级）。
"""
import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from msg_baseline.data_utils import (
    TUGraphDataset, EpisodeSampler, batch_episode_data
)

N_WAY = 5
K_SHOT = 1
Q_QUERY = 5
EPOCHS = 15
TRAIN_EPIS_PER_EPOCH = 50
EVAL_EPISODES = 50
HIDDEN = 128
NUM_LAYERS = 3
LR = 1e-3
WD = 5e-4
DEVICE = "cpu"
SEED = 42


class GINEncoder(nn.Module):
    """Standard GIN encoder -> euclidean graph embedding (no manifold, no SNN)."""
    def __init__(self, in_dim, hidden=HIDDEN, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden = hidden
        proj_in = max(in_dim, 1)

        # input lift: in -> hidden (with BN+ReLU) — matches a standard GIN stem
        self.input = nn.Sequential(
            nn.Linear(proj_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU()
        )

        # GIN conv blocks
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        self.act = nn.ReLU()
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, edge_index, batch):
        if x is None or x.numel() == 0:
            x = torch.ones(batch.numel(), 1, dtype=torch.float, device=edge_index.device)
        h = self.input(x)
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        h = self.out(h)                     # (N_nodes, hidden)
        g = global_mean_pool(h, batch)      # (num_graphs, hidden)
        return g


class EuclideanProto(nn.Module):
    """Standard Prototypical Network readout (euclidean squared distance)."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, sup_x, sup_ei, sup_b, sup_lbl, que_x, que_ei, que_b, n_way):
        z_s = self.encoder(sup_x, sup_ei, sup_b)   # (N_s, D)
        z_q = self.encoder(que_x, que_ei, que_b)   # (N_q, D)
        protos = []
        for c in range(n_way):
            mask = (sup_lbl == c)
            protos.append(z_s[mask].mean(dim=0))
        protos = torch.stack(protos, dim=0)         # (N_way, D)
        # squared euclidean dist: (N_q, N_way)
        dists = torch.cdist(z_q, protos, p=2).pow(2)
        logits = -dists
        return logits


@torch.no_grad()
def knn1_euc_fulltrain(model, dataset, device):
    """6-class leave-one-out 1-NN on full training set embeddings (euclidean)."""
    model.eval()
    from msg_baseline.data_utils import collate_graphs
    graphs = dataset.train_graphs          # list of (Data, label) tuples
    embs, labels = [], []
    bs = 32
    for i in range(0, len(graphs), bs):
        chunk = graphs[i:i+bs]
        batch = collate_graphs([g for g, _ in chunk]).to(device)
        z = model.encoder(batch.x, batch.edge_index, batch.batch)
        embs.append(z.cpu())
        labels.extend([lbl for _, lbl in chunk])
    Z = torch.cat(embs, dim=0)        # (N, D)
    Y = torch.tensor(labels)
    # leave-one-out 1-NN
    correct = 0
    for i in range(Z.shape[0]):
        d = (Z - Z[i]).pow(2).sum(dim=1)
        d[i] = 1e9                     # exclude self
        nn = d.argmin()
        if Y[nn].item() == Y[i].item():
            correct += 1
    return correct / Z.shape[0]


def train_one_epoch(model, sampler, optimizer):
    model.train()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for _ in range(TRAIN_EPIS_PER_EPOCH):
        ep = sampler.sample_episode("train")
        sb, sl, qb, ql = batch_episode_data(ep, DEVICE)
        logits = model(sb.x, sb.edge_index, sb.batch, sl, qb.x, qb.edge_index, qb.batch, N_WAY)
        loss = F.cross_entropy(logits, ql)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = logits.argmax(dim=1)
        tot_loss += loss.item()
        tot_acc += (pred == ql).float().mean().item()
        n += 1
    return tot_loss / n, tot_acc / n


@torch.no_grad()
def evaluate(model, sampler):
    model.eval()
    tot = 0.0
    for _ in range(EVAL_EPISODES):
        ep = sampler.sample_episode("test")
        sb, sl, qb, ql = batch_episode_data(ep, DEVICE)
        logits = model(sb.x, sb.edge_index, sb.batch, sl, qb.x, qb.edge_index, qb.batch, N_WAY)
        pred = logits.argmax(dim=1)
        tot += (pred == ql).float().mean().item()
    return tot / EVAL_EPISODES


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    DS = TUGraphDataset(name="ENZYMES", tu_name="ENZYMES", train_ratio=0.8, seed=42)
    in_dim = max(DS.num_features, 1)
    sampler = EpisodeSampler(DS, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY, seed=42)

    enc = GINEncoder(in_dim=in_dim)
    model = EuclideanProto(enc).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[P6] ENZYMES in_dim={in_dim} | GIN params={n_params}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    train_loss, train_acc, evals = [], [], []
    for ep in range(EPOCHS):
        tl, ta = train_one_epoch(model, sampler, optimizer)
        scheduler.step()
        ev = evaluate(model, sampler)
        train_loss.append(tl); train_acc.append(ta); evals.append(ev)
        print(f"[P6] ep{ep+1:02d} loss={tl:.4f} train_acc={ta:.3f} eval={ev:.3f} "
              f"(rand={1.0/N_WAY:.3f})", flush=True)

    knn1 = knn1_euc_fulltrain(model, DS, DEVICE)
    final_eval = evals[-1]
    result = {
        "method": "GIN + Euclidean ProtoNet (oracle, no SNN, no manifold)",
        "n_way": N_WAY, "k_shot": K_SHOT,
        "params": n_params,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "eval_acc": final_eval,
        "eval_curve": evals,
        "random": 1.0 / N_WAY,
        "knn1_euc_fulltrain_6class": knn1,
        # 对照：我们 SpikeRPGNN 在 P5/P4 的表现
        "spike_rpgnn_eval_best_5way": 0.235,
        "spike_rpgnn_knn1_euc_best": 0.410,
        "verdict": ("ORACLE WINS → failure is architecture-specific (Y)"
                    if final_eval > 0.35
                    else "ORACLE ALSO FAILS → data intrinsically hard (X)")
    }
    out = "spikergnn_results/diag_p6.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[P6] knn1_euc={knn1:.3f} final_eval={final_eval:.3f} -> {result['verdict']}", flush=True)
    print(f"[P6] saved {out}", flush=True)


if __name__ == "__main__":
    main()
